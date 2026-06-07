"""Standalone MCP server for the deep research agent.

Exposes research as an MCP tool that any client (ctx, Claude, etc.) can call.
Jobs run asynchronously in a thread pool. The server can run via stdio (default) or SSE (HTTP).
"""

import asyncio
import concurrent.futures
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

import sys
sys.path.insert(0, str(Path(__file__).parent))
from settings import Config
from harness import run as run_research, save_report

logger = logging.getLogger("deep-research-mcp")

# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResearchJob:
    job_id: str
    query: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    report_path: Optional[str] = None
    report_content: Optional[str] = None
    error: Optional[str] = None
    _task: Optional[asyncio.Task] = field(default=None, repr=False)
    _future: Optional[concurrent.futures.Future] = field(default=None, repr=False)


_jobs: dict[str, ResearchJob] = {}

# Shared thread pool for blocking research runs
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "deep-research",
    instructions=(
        "Deep Research MCP server (ReAct agent via LLM API). "
        "Start research jobs and retrieve results via job_id polling."
    ),
)


@mcp.tool()
def start_research(
    query: str,
    max_rounds: Optional[int] = None,
) -> str:
    """Start a deep research job. Returns a job_id for polling.

    Args:
        query: The research question or topic to investigate.
        max_rounds: Maximum tool-call rounds (default: 50).
    """
    job_id = str(uuid.uuid4())[:8]
    job = ResearchJob(
        job_id=job_id,
        query=query,
        status=JobStatus.PENDING,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _jobs[job_id] = job

    # Spawn background task
    task = asyncio.create_task(_run_research(job, max_rounds))
    job._task = task

    return json.dumps({
        "job_id": job_id,
        "status": job.status.value,
        "query": query,
        "message": f"Research started. Poll get_research_status('{job_id}') for progress.",
    }, indent=2)


async def _run_research(
    job: ResearchJob,
    max_rounds: Optional[int],
) -> None:
    """Background coroutine that runs research in a thread pool."""
    cfg = Config.from_env()
    if max_rounds:
        cfg.max_tool_rounds = max_rounds

    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc).isoformat()

    loop = asyncio.get_event_loop()
    try:
        report = await loop.run_in_executor(_thread_pool, run_research, job.query, cfg)

        # Save report
        path = save_report(job.query, report, cfg)
        job.report_path = str(path)
        job.report_content = report
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc).isoformat()

    except asyncio.CancelledError:
        job.status = JobStatus.CANCELLED
        job.error = "Cancelled by user"
        raise
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error = str(exc)
        logger.exception("Research job %s failed", job.job_id)
        # Save partial output if available
        try:
            partial = f"# Partial Research Report\n\n**Query:** {job.query}\n\n**Error:** {exc}\n\nThe research was interrupted."
            path = save_report(job.query + "_PARTIAL", partial, Config.from_env())
            job.report_path = str(path)
        except Exception:
            pass


@mcp.tool()
def get_research_status(job_id: str) -> str:
    """Check the status of a research job.

    Args:
        job_id: The job_id returned by start_research().
    """
    job = _jobs.get(job_id)
    if not job:
        return json.dumps({"error": f"Job {job_id} not found"}, indent=2)

    payload = {
        "job_id": job.job_id,
        "status": job.status.value,
        "query": job.query,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }

    if job.status == JobStatus.COMPLETED:
        payload["report_path"] = job.report_path
        if job.report_content:
            preview = job.report_content[:2000]
            if len(job.report_content) > 2000:
                preview += "\n\n... [truncated — use get_report() for full text]"
            payload["report_preview"] = preview

    elif job.status == JobStatus.FAILED:
        payload["error"] = job.error

    return json.dumps(payload, indent=2)


@mcp.tool()
def get_report(job_id: str) -> str:
    """Retrieve the full report for a completed research job.

    Args:
        job_id: The job_id returned by start_research().
    """
    job = _jobs.get(job_id)
    if not job:
        return json.dumps({"error": f"Job {job_id} not found"}, indent=2)
    if job.status != JobStatus.COMPLETED:
        return json.dumps({
            "error": f"Job {job_id} is not complete. Status: {job.status.value}",
        }, indent=2)

    return json.dumps({
        "job_id": job.job_id,
        "status": job.status.value,
        "report_path": job.report_path,
        "report": job.report_content,
    }, indent=2)


@mcp.tool()
def list_research_jobs() -> str:
    """List all research jobs and their statuses."""
    results = []
    for job in _jobs.values():
        q = job.query[:80] + "..." if len(job.query) > 80 else job.query
        results.append({
            "job_id": job.job_id,
            "status": job.status.value,
            "query": q,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        })
    return json.dumps({"jobs": results}, indent=2)


@mcp.tool()
def cancel_research(job_id: str) -> str:
    """Cancel a running research job.

    Args:
        job_id: The job_id to cancel.
    """
    job = _jobs.get(job_id)
    if not job:
        return json.dumps({"error": f"Job {job_id} not found"}, indent=2)
    if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
        return json.dumps({"error": f"Job {job_id} is already {job.status.value}"}, indent=2)

    if job._future:
        job._future.cancel()
    if job._task:
        job._task.cancel()
    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.now(timezone.utc).isoformat()
    return json.dumps({"job_id": job_id, "status": "cancelled"}, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Deep Research MCP Server")
    parser.add_argument("--sse", action="store_true", help="Run in SSE mode (HTTP)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address for SSE")
    parser.add_argument("--port", type=int, default=8001, help="Port for SSE (default: 8001)")
    args = parser.parse_args()

    if args.sse:
        import uvicorn
        logger.info("Deep Research MCP SSE on %s:%s", args.host, args.port)
        uvicorn.run(mcp.sse_app(), host=args.host, port=args.port, log_level="info")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
