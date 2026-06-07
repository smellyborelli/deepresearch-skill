# Dual-Pass Research Runner

## Overview

A wrapper that runs the deep research harness twice — once without source quality weighting (broad, unfiltered search → practitioner stories and concrete examples), once with source quality weighting (dashboard-guided → structured framework and actionable principles) — then synthesizes both into a single combined report that has the evidence *and* the structure.

## Components

### 1. Env Toggle in `search.py` / `scholar.py`

A simple environment variable check wraps the `add_dashboard_to_response()` call in each tool so it can be skipped. No import-level gating — just a runtime check at the point of use.

```python
# In search.py, after building content string:
if os.getenv("ENABLE_SOURCE_QUALITY", "true").lower() == "true":
    content, cumulative_scores = add_dashboard_to_response(
        content, results["organic"], cumulative_scores
    )
```

Same pattern in `scholar.py`. When `ENABLE_SOURCE_QUALITY=false`:
- No source classification or scoring
- No dashboard appended to responses
- No cumulative_scores tracking
- The `cumulative_scores` param still exists in the tool definition but the model can pass `{}` and nothing happens
- The model behaves exactly like the original Alibaba code

When `true` (default): current behavior — dashboard, scoring, cumulative state-passing.

### 2. New file: `src/dual_pass.py`

A standalone script that orchestrates two passes and a synthesis step.

```
Pass A (ENABLE_SOURCE_QUALITY=false)  →  output/<query>_PassA.md
Pass B (ENABLE_SOURCE_QUALITY=true)   →  output/<query>_PassB.md
Synthesis (LLM merge prompt)          →  output/<query>_Report.md
```

**CLI interface:**
```bash
# Normal mode (same args as harness.py)
python src/dual_pass.py "What makes a good portfolio blog?"

# Interactive mode (no args, prompts for query)
python src/dual_pass.py
```

**Concurrent vs. sequential:** Pass A and Pass B must run sequentially, not in parallel, because they share the same API keys and rate limits. Pass A runs first, then Pass B.

**Progress output:**
```
🔬 Dual-pass research: "What makes a good portfolio blog?"
  Pass A (practitioner stories)... ──▶ output/..._PassA.md ✅
  Pass B (structured framework)... ──▶ output/..._PassB.md ✅
  Synthesizing...                   ──▶ output/..._Report.md ✅
```

**Recovery:** If a pass fails (API error, timeout), the script reports which pass failed and exits without trying the second pass. The partial output (if any) is preserved for debugging. The script does not attempt synthesis if either pass failed.

**Synthesis prompt:**

```
You are a research synthesis editor. Merge these two research reports on the
same topic into one comprehensive report.

**Report A** was produced by broad, unfiltered web search — it has rich
practitioner stories, named examples, specific numbers, and first-hand accounts.

**Report B** was produced with source-quality guidance — it has a structured
framework, categorized findings, and actionable principles.

Merge them into ONE report that has:
- The structured sections and actionable framework from Report B
- The real stories, named people, numbers, and concrete examples from Report A
- Inline integration — weave examples into the relevant framework sections as
  evidence, don't put them in a separate, disconnected section
- A "Real Examples Summary" table at the end if the content supports it
- The same professional, analytical tone as both source reports

Report A:
---8<---
{content_a}
---8<---

Report B:
---8<---
{content_b}
---8<---
```

The synthesis uses the same LLM client as the harness (OpenAI-compatible). Model is configurable via the same `SUMMARY_MODEL_NAME` env var, defaulting to the main model.

### 3. Skill / CLAUDE.md Update

The quickstart in the project CLAUDE.md updated to show dual_pass.py as the recommended runner:

```bash
# Recommended: dual-pass (practitioner stories + structured framework)
python src/dual_pass.py "your query"

# Quick single-pass with quality weighting
python src/harness.py "your query"

# Quick single-pass without quality weighting
ENABLE_SOURCE_QUALITY=false python src/harness.py "your query"
```

## Files Changed

| File | Action | Lines |
|------|--------|-------|
| `src/dual_pass.py` | **Create** | ~120 lines |
| `src/tools/search.py` | **Modify** | Wrap dashboard in env check (2 lines) |
| `src/tools/scholar.py` | **Modify** | Wrap dashboard in env check (2 lines) |
| `CLAUDE.md` | **Modify** | Update quickstart |

## Architecture

```
User Query
    │
    ▼
src/dual_pass.py
    │
    ├── Pass A: run(query, quality=false) ──▶ save PassA.md
    │       │
    │       └── harness → ReAct loop
    │               search.py (no dashboard)
    │               visit.py
    │               LLM → <answer>
    │
    ├── Pass B: run(query, quality=true) ───▶ save PassB.md
    │       │
    │       └── harness → ReAct loop
    │               search.py (with dashboard, cumulative state)
    │               scholar.py (with dashboard if used)
    │               visit.py
    │               LLM → <answer>
    │
    └── Synthesis: LLM merge ──────────────▶ save Report.md
            │
            └── OpenAI-compatible API
                    prompt: "Merge Report A into Report B's structure"
```

## Edge Cases

| Case | Behavior |
|------|----------|
| Pass A fails (API error) | Script exits, Pass A partial output preserved, no synthesis attempted |
| Pass B fails | Same — Pass A output preserved, no synthesis |
| Both succeed but synthesis fails | Both pass outputs still saved — user can manually merge |
| Query is very narrow (no practitioner stories available) | Pass A may have limited Tier 1-3 content. Synthesis will weight Report B more heavily. Still better than single-pass. |
| Query is very broad (tons of data) | Dual-pass will be slower and more expensive. User can fall back to single-pass harness. |
| `ENABLE_SOURCE_QUALITY` set in environment | dual_pass.py overrides it per-pass regardless of env setting |

## Cost

Approximately 2× a normal research run + one small synthesis call. The synthesis call is a single LLM completion with the two reports as context (typically 2-5k input tokens) — negligible compared to the research tokens.

## Future Considerations

- **Configurable pass order**: For quantitative topics (medical, scientific), running the quality-weighted pass first might produce better initial results. For qualitative topics (practitioner stories, "how to X"), the unfiltered pass first is better.
- **Three-pass mode**: Some topics benefit from a third pass targeting specific domains — e.g., academic-only, or practitioner-forum-only.
- **Streaming progress**: The current design blocks on each pass. A future version could stream partial synthesis results.
