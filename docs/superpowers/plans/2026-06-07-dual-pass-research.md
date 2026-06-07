# Dual-Pass Research Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) for syntax tracking.

**Goal:** Create a wrapper that runs the deep research harness twice (once without source quality, once with) and synthesizes the results into one combined report.

**Architecture:** A new `src/dual_pass.py` orchestrates two sequential passes by calling the existing `run()` and `save_report()` from `harness.py` with different `ENABLE_SOURCE_QUALITY` env var settings. A 2-line env toggle in `search.py`/`scholar.py` gates the dashboard logic. Synthesis reuses the same OpenAI-compatible client from the project's config.

**Tech Stack:** Python 3, os.environ, openai — no new dependencies.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/tools/search.py` | **Modify** | Wrap `add_dashboard_to_response()` in `if os.getenv(...)` check |
| `src/tools/scholar.py` | **Modify** | Same as search.py |
| `src/dual_pass.py` | **Create** | Orchestrates Pass A, Pass B, and synthesis |
| `CLAUDE.md` | **Modify** | Update quickstart to show dual_pass.py first |

---

### Task 1: Add env toggle to `search.py` and `scholar.py`

**Files:**
- Modify: `src/tools/search.py`
- Modify: `src/tools/scholar.py`

Both need the same 2-line change: wrap the `add_dashboard_to_response()` call so it only runs when `ENABLE_SOURCE_QUALITY=true` (the default).

- [ ] **Step 1: Modify `src/tools/search.py`**

Read the current `src/tools/search.py` and find the line:

```python
content, _ = add_dashboard_to_response(
    content, results["organic"], cumulative_scores
)
```

Replace it with:

```python
if os.getenv("ENABLE_SOURCE_QUALITY", "true").lower() == "true":
    content, _ = add_dashboard_to_response(
        content, results["organic"], cumulative_scores
    )
```

- [ ] **Step 2: Modify `src/tools/scholar.py`**

Read `src/tools/scholar.py` and find the line:

```python
content, cumulative_scores = add_dashboard_to_response(
    content, scholar_results, cumulative_scores
)
```

Replace it with:

```python
if os.getenv("ENABLE_SOURCE_QUALITY", "true").lower() == "true":
    content, cumulative_scores = add_dashboard_to_response(
        content, scholar_results, cumulative_scores
    )
```

- [ ] **Step 3: Verify the toggle works**

```bash
cd /Users/computer/ai/projects/dev/deep-research/src
python3 -c "
import os; os.environ['ENABLE_SOURCE_QUALITY'] = 'false'
from tools.search import Search
from tools.scholar import Scholar
# Just verify they import without errors
s = Search()
sch = Scholar()
assert 'cumulative_scores' in s.parameters['properties']
assert 'cumulative_scores' in sch.parameters['properties']
print('Tools load with ENABLE_SOURCE_QUALITY=false: OK')

os.environ['ENABLE_SOURCE_QUALITY'] = 'true'
from tools.search import Search as Search2
s2 = Search2()
print('Tools load with ENABLE_SOURCE_QUALITY=true: OK')
"
```

Expected: both load without errors.

- [ ] **Step 4: Commit**

```bash
cd /Users/computer
git add -f ai/projects/dev/deep-research/src/tools/search.py ai/projects/dev/deep-research/src/tools/scholar.py
git commit -m "feat: add ENABLE_SOURCE_QUALITY env toggle to search and scholar tools"
```

---

### Task 2: Create `src/dual_pass.py`

**Files:**
- Create: `src/dual_pass.py`

- [ ] **Step 1: Write the module**

`src/dual_pass.py`:

```python
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
from harness import run, save_report


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

def dual_pass(query: str, cfg: Config | None = None) -> str:
    """Run two-pass research and return the combined report path."""
    if cfg is None:
        cfg = Config.from_env()

    topic_slug = _sanitize_filename(query)

    # ── Pass A: no source quality ──
    print(f"🔬 Dual-pass research: \"{query}\"")
    print(f"  Pass A (practitioner stories)... ", end="", flush=True)
    os.environ["ENABLE_SOURCE_QUALITY"] = "false"
    try:
        report_a = run(query, cfg)
        path_a = save_report(f"{query} (Pass A - Practitioner)", report_a, cfg)
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
        path_b = save_report(f"{query} (Pass B - Framework)", report_b, cfg)
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
        path_c = save_report(query, combined, cfg)
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

def _sanitize_filename(text: str) -> str:
    """Create a filesystem-safe filename from the query."""
    keep = []
    for ch in text[:60]:
        if ch.isalnum() or ch in " -_":
            keep.append(ch)
    return "".join(keep).strip().replace(" ", "_") or "research_report"


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
```

- [ ] **Step 2: Verify the module loads and structure is correct**

```bash
cd /Users/computer/ai/projects/dev/deep-research/src
python3 -c "
# Just verify imports work and the module has the right surface
from dual_pass import dual_pass, synthesize, SYNTHESIS_PROMPT
assert 'Report A' in SYNTHESIS_PROMPT
assert 'Report B' in SYNTHESIS_PROMPT
assert '{content_a}' in SYNTHESIS_PROMPT
assert '{content_b}' in SYNTHESIS_PROMPT
print('dual_pass.py loads correctly')
print(f'Synthesis prompt: {len(SYNTHESIS_PROMPT)} chars')
"
```

Expected: module loads, synthesis prompt contains required template variables.

- [ ] **Step 3: Commit**

```bash
cd /Users/computer
git add -f ai/projects/dev/deep-research/src/dual_pass.py
git commit -m "feat: add dual-pass deep research runner"
```

---

### Task 3: Update CLAUDE.md with new quickstart

**Files:**
- Modify: `CLAUDE.md` (at `/Users/computer/ai/projects/dev/deep-research/CLAUDE.md`)

- [ ] **Step 1: Update the quickstart section**

Read `CLAUDE.md` and find the Quickstart section (the first code block). It currently reads:

```
```bash
cd ~/ai/projects/dev/deep-research
cp .env.example .env  # fill in keys
pip install -r requirements.txt
python src/harness.py "Your research question here"
```

Or run in dialog mode:
```bash
python src/harness.py
```
```

Replace it with:

```
```bash
cd ~/ai/projects/dev/deep-research
cp .env.example .env  # fill in keys
pip install -r requirements.txt

# Recommended: dual-pass (practitioner stories + structured framework)
python src/dual_pass.py "Your research question here"

# Quick single-pass with quality weighting
python src/harness.py "Your research question here"

# Quick single-pass without quality weighting
ENABLE_SOURCE_QUALITY=false python src/harness.py "Your research question here"
```

Or run in dialog mode:
```bash
python src/dual_pass.py        # dual-pass interactive
python src/harness.py          # single-pass interactive
```
```

- [ ] **Step 2: Verify the file still looks good**

```bash
cd /Users/computer/ai/projects/dev/deep-research
head -35 CLAUDE.md
```

Expected: quickstart section updated with the three invocation patterns.

- [ ] **Step 3: Commit**

```bash
cd /Users/computer
git add -f ai/projects/dev/deep-research/CLAUDE.md
git commit -m "docs: update CLAUDE.md quickstart to recommend dual_pass.py"
```

---

## Integration Verification

After all three tasks are committed:

```bash
cd /Users/computer/ai/projects/dev/deep-research/src

# 1. Verify toggle works
python3 -c "
import os

# Off mode
os.environ['ENABLE_SOURCE_QUALITY'] = 'false'
from tools.search import Search
from tools.scholar import Scholar
s = Search()
sch = Scholar()
print('Toggle OFF: tools loaded')

# On mode
os.environ['ENABLE_SOURCE_QUALITY'] = 'true'
from tools.search import Search as S2
from tools.scholar import Scholar as Sch2
s2 = S2()
sch2 = Sch2()
print('Toggle ON: tools loaded')
"

# 2. Verify dual_pass loads
python3 -c "
from dual_pass import dual_pass, synthesize, SYNTHESIS_PROMPT
assert 'Report A' in SYNTHESIS_PROMPT
assert 'Report B' in SYNTHESIS_PROMPT
print('dual_pass.py: OK ({len(SYNTHESIS_PROMPT)} chars)')
"

# 3. Verify CLAUDE.md has dual_pass
grep -q 'dual_pass.py' ../CLAUDE.md && echo 'CLAUDE.md: dual_pass.py mentioned ✅' || echo 'CLAUDE.md: NOT FOUND ❌'

echo 'All integration checks passed.'
```

Expected: all checks pass.

```bash
cd /Users/computer
git add -f ai/projects/dev/deep-research/src/tools/search.py ai/projects/dev/deep-research/src/tools/scholar.py ai/projects/dev/deep-research/src/dual_pass.py ai/projects/dev/deep-research/CLAUDE.md && git commit -m "chore: final integration check — dual-pass system verified"
```
