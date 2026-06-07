---
title: Deep Research — ReAct Agent via LLM API
project: deep-research
type: protocol
tags: [deep-research, agent, deepseek, react, research]
status: active
created: 2026-06-06
updated: 2026-06-06
---

# Deep Research — ReAct Agent via LLM API

## Overview

A ReAct (Reasoning + Acting) deep research agent. Uses the Alibaba-NLP/DeepResearch inference stack: a multi-turn agent loop where the model drives web searches and page reading, then synthesizes findings into a structured report.

The model runs via any **OpenAI-compatible API** (DeepSeek, OpenRouter, OpenAI, etc.) — no local GPUs needed. Tools execute locally (Serper.dev for search, Jina.ai for page reading).

## Quickstart

```bash
cd ~/ai/projects/dev/deep-research
cp .env.example .env  # fill in keys
pip install -r requirements.txt

# Recommended: dual-pass (practitioner stories + structured framework)
python src/dual_pass.py "Your research question here"

# Quick single-pass with source quality weighting
python src/harness.py "Your research question here"

# Quick single-pass without quality weighting
ENABLE_SOURCE_QUALITY=false python src/harness.py "Your research question here"
```

Or run in dialog mode:
```bash
python src/dual_pass.py    # dual-pass interactive
python src/harness.py      # single-pass interactive
```

## Architecture

```
User Query ──▶ Skill (/deep-research) ──▶ harness.py
                                               │
                                       TongyiDeepResearchAgent
                                    (subclass of Alibaba's MultiTurnReactAgent)
                                               │
                              ┌────────────────┼────────────────┐
                              │                │                │
                        Serper.dev         Jina.ai        DeepSeek API
                        (web search)    (page reader)    (or any OpenAI-
                              │                │          compatible API)
                              ▼                ▼
                     src/tools/search.py  src/tools/visit.py
```

### How the ReAct loop works

1. **Send messages** (system prompt + user query) to the LLM
2. **Model responds** with either:
   - `<tool_call>{"name": "search", "arguments": {...}}</tool_call>` — execute tool locally
   - `<answer>final answer</answer>` — research complete
3. **Execute tool** (Serper search / Jina page read / Google Scholar)
4. **Append** `<tool_response>\n...results\n</tool_response>` to messages
5. **Repeat** until `<answer>` or max rounds
6. **Extract** answer from `<answer>` tags → save report

The tool loop is from Alibaba's verbatim `react_agent.py`. The only custom code is `call_server` in `harness.py` which points the model API call at the configured provider.

## Files

| File | Purpose | Origin |
|------|---------|--------|
| `src/harness.py` | Entry point — agent subclass, `run()` function | New code |
| `src/agent.py` | `MultiTurnReactAgent` — ReAct loop, tool dispatch, XML parsing | **Verbatim** from Alibaba `react_agent.py` |
| `src/prompts.py` | `SYSTEM_PROMPT` and `EXTRACTOR_PROMPT` | **Verbatim** from Alibaba `prompt.py` |
| `src/settings.py` | Config — API key, base URL, Serper, Jina | New code |
| `src/mcp_server.py` | MCP server — research as async jobs | Adapted |
| `src/tools/search.py` | Serper.dev web search | **Verbatim** from Alibaba `tool_search.py` |
| `src/tools/visit.py` | Jina.ai page reader + LLM summarization | **Verbatim** from Alibaba `tool_visit.py` |
| `src/tools/scholar.py` | Google Scholar via Serper.dev | **Verbatim** from Alibaba `tool_scholar.py` |
| `archived/` | Previous implementations preserved for reference | See `archived/README.md` |

## Configuration

Set in `.env`:

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `API_KEY` | Yes | — | LLM provider API key |
| `BASE_URL` | No | `https://api.deepseek.com/v1` | API endpoint |
| `MODEL` | No | `deepseek-v4-flash` | Model name |
| `SERPER_KEY_ID` | Yes | — | Web search via Serper.dev |
| `JINA_API_KEYS` | Yes | — | Page reading via Jina.ai |
| `SUMMARY_MODEL_NAME` | No | same as `MODEL` | LLM for page summarization |
| `MAX_TOOL_ROUNDS` | No | `50` | Max ReAct loop iterations |
| `OUTPUT_DIR` | No | `output` | Report output directory |

### Supported Providers

Any OpenAI-compatible API works. Just change `API_KEY`, `BASE_URL`, and `MODEL`:

- **DeepSeek**: `https://api.deepseek.com/v1`, model `deepseek-v4-flash`
- **OpenRouter**: `https://openrouter.ai/api/v1`, model varies
- **OpenAI**: `https://api.openai.com/v1`, model `gpt-4o` etc.

## MCP Server

```bash
# stdio mode (for Claude Desktop, Claude Code)
python src/mcp_server.py

# SSE mode (HTTP for remote clients)
python src/mcp_server.py --sse --port 8001
```

### MCP Tools

| Tool | Args | Returns |
|------|------|---------|
| `start_research` | `query`, `max_rounds` (optional) | `job_id` |
| `get_research_status` | `job_id` | status, progress, report preview |
| `get_report` | `job_id` | full report text |
| `list_research_jobs` | — | all jobs |
| `cancel_research` | `job_id` | confirmation |

### MCP Client Config (`.mcp.json`)

```json
{
  "mcpServers": {
    "deep-research": {
      "command": "python",
      "args": ["src/mcp_server.py"],
      "env": {
        "API_KEY": "sk-...",
        "BASE_URL": "https://api.deepseek.com/v1",
        "MODEL": "deepseek-v4-flash",
        "SERPER_KEY_ID": "...",
        "JINA_API_KEYS": "..."
      }
    }
  }
}
```

## Tools

### Search (`src/tools/search.py`)
- Serper.dev HTTPS API
- Supports batched queries (array of strings)
- Returns: title, URL, date, source, snippet
- Handles Chinese characters with CN-specific location params

### Visit (`src/tools/visit.py`)
- Jina.ai Reader API (`r.jina.ai/{url}`)
- LLM-based extraction: fetches raw page → truncates → summarizes via `EXTRACTOR_PROMPT`
- Returns structured JSON: `{rational, evidence, summary}`
- Graceful fallbacks on all failure modes

### Scholar (`src/tools/scholar.py`)
- Serper.dev `/scholar` endpoint
- Returns: title, year, publication info, citedBy count, snippet, PDF URL
- Supports batched queries

## Prompt Design (`src/prompts.py`)

Two prompts, both verbatim from the Alibaba repo:

### SYSTEM_PROMPT
- "You are a deep research assistant..."
- Tool definitions in XML `<tools>` tags
- `<tool_call>` / `<tool_response>` / `<answer>` protocol instructions
- Current date appended at runtime

### EXTRACTOR_PROMPT
- Used by `visit.py` for page content summarization
- Extracts: `{rational, evidence, summary}` from raw page content

## Running Research Sessions

```bash
cd ~/ai/projects/dev/deep-research
python src/harness.py "Your research question"
```

Reports save to `output/<sanitized_topic>_Report.md` by default. To preserve a session's output, copy it to a topic-named subfolder:

```bash
mkdir -p <topic-name>
cp output/*_Report.md <topic-name>/
```

Each topic folder should contain a CLAUDE.md documenting the query, results, and any follow-up work.

## Output

Reports saved to `output/<sanitized_topic>_Report.md` containing the model's full `<answer>` content.

## References

- Repo: [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)
- Paper: [Tongyi DeepResearch Technical Report (arXiv)](https://arxiv.org/pdf/2510.24701)
- Tools: [Serper.dev](https://serper.dev/) | [Jina.ai](https://jina.ai/)
