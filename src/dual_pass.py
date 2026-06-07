"""Dual-pass deep research runner.

Runs the research harness twice:
  - Pass A: without source quality weighting (practitioner stories, broad search)
  - Pass B: with source quality weighting (dashboard-guided, structured framework)

Then synthesizes both reports into a single combined report.
"""

import os
import sys
from pathlib import Path

from openai import OpenAI

from settings import Config
from harness import run

# Resolve output dir relative to project root (not CWD)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


# ── Synthesis ──────────────────────────────────────────────────────

SYNTHESIS_PROMPT = """You are a research synthesis editor. Merge these two research reports on the same topic into one comprehensive report.

**Report A** was produced by broad, unfiltered web search — it has rich practitioner stories, named examples, specific numbers, and first-hand accounts.

**Report B** was produced with source-quality guidance — it has a structured framework, categorized findings, and actionable principles.

Merge them into ONE report that has:
- The structured sections and actionable framework from Report B
- The real stories, named people, numbers, and concrete examples from Report A
- Inline integration — weave examples into the relevant framework sections as evidence, don't put them in a separate, disconnected section
- A "Real Examples Summary" table at the end if the content supports it
- The same professional, analytical tone as both source reports

Report A:
---8<---
{content_a}
---8<---

Report B:
---8<---
{content_b}
---8<---"""


def synthesize(report_a: str, report_b: str, cfg: Config) -> str:
    """Merge two reports into one using the LLM."""
    prompt = SYNTHESIS_PROMPT.format(content_a=report_a, content_b=report_b)

    client = OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout=300.0,
    )

    model = cfg.summary_model or cfg.model
    print(f"\n  Synthesizing... ", end="", flush=True)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=16000,
        )
        result = response.choices[0].message.content or ""
        print("✅")
        return result
    except Exception as e:
        print(f"❌ Synthesis failed: {e}")
        raise


# ── Dual-pass runner ──────────────────────────────────────────────

def _slug(text: str) -> str:
    """Short filesystem-safe slug (40 chars max)."""
    keep = []
    for ch in text[:40]:
        if ch.isalnum() or ch in " -_":
            keep.append(ch)
    return "".join(keep).strip().replace(" ", "_") or "research_report"


def _write_report(slug: str, suffix: str, content: str) -> Path:
    """Write report to output dir with unique filename."""
    path = OUTPUT_DIR / f"{slug}_{suffix}.md"
    path.write_text(content, encoding="utf-8")
    return path


def dual_pass(query: str, cfg: Config | None = None) -> str:
    """Run two-pass research and return the combined report path."""
    if cfg is None:
        cfg = Config.from_env()

    slug = _slug(query)

    # Ensure output dir exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Pass A: no source quality ──
    print(f"🔬 Dual-pass research: \"{query}\"")
    print(f"  Pass A (practitioner stories)... ", end="", flush=True)
    os.environ["ENABLE_SOURCE_QUALITY"] = "false"
    try:
        report_a = run(query, cfg)
        path_a = _write_report(slug, "PassA", report_a)
        print(f"✅ → {path_a.name}")
    except Exception as e:
        print(f"❌ Pass A failed: {e}")
        print("  Partial output may exist. Exiting without synthesis.")
        sys.exit(1)

    # ── Pass B: with source quality ──
    print(f"  Pass B (structured framework)... ", end="", flush=True)
    os.environ["ENABLE_SOURCE_QUALITY"] = "true"
    try:
        report_b = run(query, cfg)
        path_b = _write_report(slug, "PassB", report_b)
        print(f"✅ → {path_b.name}")
    except Exception as e:
        print(f"❌ Pass B failed: {e}")
        print("  Pass A output preserved. Exiting without synthesis.")
        sys.exit(1)

    # ── Restore default ──
    os.environ["ENABLE_SOURCE_QUALITY"] = "true"

    # ── Synthesis ──
    try:
        combined = synthesize(report_a, report_b, cfg)
        path_c = _write_report(slug, "Report", combined)
        print(f"\n📄 Combined report: {path_c}")
        print(f"   Pass A (intermediate): {path_a}")
        print(f"   Pass B (intermediate): {path_b}")
        return str(path_c)
    except Exception as e:
        print(f"\n❌ Synthesis failed: {e}")
        print("Both pass outputs preserved for manual merging:")
        print(f"  Pass A: {path_a}")
        print(f"  Pass B: {path_b}")
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────

def main():
    cfg = Config.from_env()
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None

    try:
        if query:
            dual_pass(query, cfg)
        else:
            print("\n🔬 Dual-Pass Research Mode (type 'q' to quit)\n")
            while True:
                query = input("Q: ").strip()
                if query.lower() == "q":
                    break
                if not query:
                    continue
                dual_pass(query, cfg)
    except Exception as exc:
        print(f"\n❌ Error: {exc}")
        raise


if __name__ == "__main__":
    main()
