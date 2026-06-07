# Source Quality Weighting for Deep Research

## Overview

Add a source quality scoring system to the deep research ReAct agent. After each tool call, the model receives a **Source Quality Dashboard** — a per-batch classification + cumulative running total — that helps it decide whether to continue researching or synthesize its findings. The score is advisory, not compulsory.

## Core Idea

The model already receives structured search results. We augment those results with a quality tier label and score, plus a cumulative dashboard appended to each tool response. The system prompt guides the model to use this dashboard as research depth feedback without making it a stress-inducing pass/fail gate.

## Source Tier Classification

Classification is rule-based (domain + URL pattern + snippet keyword matching) — zero API cost, zero latency impact.

| Tier | Name | Score | Signals |
|------|------|-------|---------|
| 1 | Peer-Reviewed / Primary Lit | 5 | `.edu`, `pubmed.ncbi.nlm.nih.gov`, `nature.com`, `arxiv.org`, `ieee.org`, `acm.org`, `sciencedirect.com`, `springer.com`, `wiley.com`, `plos.org`, `bmj.com`, `thelancet.com`, Scholar results with `citations > 0` or known papers |
| 2 | Scholarly / Gov / Research | 4 | `.gov`, `.mil`, `who.int`, `un.org`, `oecd.org`, `worldbank.org`, `rand.org`, `brookings.edu`, `.int`, `nber.org`, established research institutes, `.edu` pages that aren't peer-reviewed venues (general university pages) |
| 3 | Anecdotal / Interview / Practitioner | 3 | **First-hand accounts with concrete details.** Reddit threads with real methods/numbers, personal blog posts with specific data, forum discussions from practitioners, interview transcripts, personal narratives, `substack.com`, `medium.com` (first-person), `quora.com`, `/interview/`, `/ama/`, `/experience/`, oral histories. **Often the most valuable sources for practical, real-world research** — someone sharing what actually happened and how they did it often yields more actionable information than a sanitised journal article. |
| 4 | Blog / News / Journalism | 2 | All remaining `.com` / `.org` / news sites, general reporting, blog posts, industry analysis |
| 5 | Affiliate / Product / Spam | 0 (discard) | `amazon.com` / `amzn.to`, known affiliate domains, `/product/` / `/shop/` / `/buy/` / `/pricing/` in URL, sponsored content markers, low-quality content farms |

**Classification method** — `src/source_quality.py`:

1. Parse domain from URL
2. Check domain against a known-domain dictionary (fast O(1) hash lookups)
3. Check URL path for pattern matches (affiliate/shop signals → discard)
4. Check snippet for content markers (interview keywords, citation counts from Scholar)
5. Fall back to Tier 4 (Blog/News) for unrecognised `.com`/`.org`
6. Default to Tier 2 (general web) for unknown TLDs

## What Gets Scored

Every search result returned by `search.py` (Serper web) and `google_scholar.py` (Serper scholar) gets scored before the response text is built.

Scholar results default-scope higher because the endpoint *is* academic:
- Results with `citedBy > 0` → Tier 1
- Results from `.edu` domains → Tier 1 if publication venue matches
- Other Scholar results → Tier 2

## Dashboard Format

### Per-Batch Section (appended to each tool response)

```
## Source Quality Dashboard
| # | Title | Domain | Tier | Score |
|---|-------|--------|------|-------|
| 1 | Paper Title | arxiv.org | Peer-Reviewed | 5 |
| 2 | Interview | substack.com | Anecdotal | 3 |

This batch: 2 sources, avg score 4.0
```

### Cumulative Section (appended after per-batch)

```
## Cumulative Quality (all rounds)
Peer-Reviewed (×5):  3  (15 pts)
Scholarly/Gov (×4):  2  (8 pts)
Anecdotal (×3):     1  (3 pts)
Blog/News (×2):     2  (4 pts)
Discarded (×0):     0  (0 pts)
─────────────────────
Weighted total:     30 / 50  (60%)
Sources:            8
```

**"All rounds" means cumulative across the entire conversation.** To support this without shared state, we need a small change: the formatting code in `source_quality.py` accepts an optional `cumulative` dict parameter. Each tool response appends only its own cumulative data. The tool itself (`search.py`, `scholar.py`) receives a serialised cumulative tally passed via tool arguments.

**Wait — tools can't hold state across calls.** The cleaner approach: the system prompt tells the model to keep a running tally by reading the cumulative section from previous rounds. But that's fragile.

**Better approach:** Each tool call accepts an optional `cumulative_scores` parameter (JSON string: `{"tier1": 3, "tier2": 2, ...}`) in its arguments. The model passes forward what it last saw. The tool reads this, adds current batch scores, and returns the updated cumulative in its response. The model then passes *that* value in its next search/scholar call.

This is a **lightweight state-passing pattern** — the model isn't doing arithmetic, just passing through the JSON blob the tool gave it. The `cumulative_scores` field is always optional (defaults to empty if model forgets — no harm).

### System Prompt Guidance

Add to `SYSTEM_PROMPT`:

```
## Source Quality Dashboard

Each search and scholar response now includes a **Source Quality Dashboard**
showing the quality tier and score for each result, plus a running cumulative
score across all rounds.

Use this dashboard to assess whether you have sufficient high-quality sources:

- **Tier 1 (Peer-Reviewed, score 5)** — strongest evidence. Prioritise.
- **Tier 2 (Scholarly/Gov, score 4)** — strong institutional sources.
- **Tier 3 (Anecdotal/Interview, score 3)** — valuable first-hand accounts.
- **Tier 4 (Blog/News, score 2)** — general reporting, context.
- **Tier 5 (Affiliate/Product, score 0)** — discarded, do not cite.

**How to use the cumulative score:**
- If your weighted total is below 40% after several rounds, try more
  targeted searches (scholar, academic domains, interview sources).
- If your weighted total is above 70%, you likely have strong coverage.
- If two consecutive search rounds add no new Tier 1-2 sources, you may
  have exhausted high-quality search angles — proceed to answer.
- **A low score is NOT a failure.** Some topics lack peer-reviewed
  literature. If you've searched thoroughly, synthesise what you have.
  Your report can note the source landscape honestly.
```

### Cumulative score parameter on tools

Add an optional `cumulative_scores` field to both `search` and `google_scholar` tool definitions:

```python
"cumulative_scores": {
    "type": "object",
    "properties": {
        "tier1_count": {"type": "integer"},
        "tier2_count": {"type": "integer"},
        "tier3_count": {"type": "integer"},
        "tier4_count": {"type": "integer"},
        "tier1_score": {"type": "integer"},
        "tier2_score": {"type": "integer"},
        "tier3_score": {"type": "integer"},
        "tier4_score": {"type": "integer"},
        "total_possible": {"type": "integer"},
    },
    "description": "Cumulative quality scores from previous rounds. Pass this value from the last dashboard you received.",
}
```

The tool merges this with its current batch and returns the updated cumulative in its response.

## Files Changed

### New: `src/source_quality.py` (~120 lines)

```python
# Source classification and scoring engine

DOMAIN_TIERS = {
    # Tier 1 — Peer-Reviewed
    "pubmed.ncbi.nlm.nih.gov": 1, "arxiv.org": 1,
    "ieeexplore.ieee.org": 1, "dl.acm.org": 1,
    "sciencedirect.com": 1, "link.springer.com": 1,
    "onlinelibrary.wiley.com": 1, "journals.plos.org": 1,
    "nature.com": 1, "science.org": 1, "cell.com": 1,
    "thelancet.com": 1, "bmj.com": 1, "nejm.org": 1,
    
    # Tier 2 — Scholarly/Gov
    ".gov": 2, ".mil": 2, ".int": 2,
    "who.int": 2, "un.org": 2, "oecd.org": 2,
    "worldbank.org": 2, "rand.org": 2, "brookings.edu": 2,
    "nber.org": 2, "imf.org": 2,
}

TIER_SCORES = {1: 5, 2: 4, 3: 3, 4: 2, 5: 0}

def classify_source(url: str, snippet: str = "", cited_by: int = 0) -> dict:
    """Classify a source URL into tier 1-5 with score."""

def score_search_results(results: list) -> list:
    """Score each result in a search response list."""

def format_batch_dashboard(scored_results: list) -> str:
    """Format the per-batch quality table."""

def format_cumulative_dashboard(cum: dict) -> str:
    """Format the cumulative quality summary."""

def merge_cumulative(existing: dict, batch: list) -> dict:
    """Add batch scores to running cumulative tally."""
```

### Modified: `src/tools/search.py`

- `google_search_with_serp()`: after building web_snippets, classify and score each result, append per-batch + cumulative dashboards to response text
- Tool parameters: add optional `cumulative_scores` field
- `call()`: extract `cumulative_scores` from params if present, pass to formatting functions

### Modified: `src/tools/scholar.py`

- Same pattern as search.py: score Scholar results (citedBy > 0 → Tier 1), append dashboards
- Tool parameters: add optional `cumulative_scores` field

### Modified: `src/prompts.py`

- Add "Source Quality Dashboard" section to `SYSTEM_PROMPT` (guidance text above)
- Mention `cumulative_scores` parameter in relevant tool descriptions

### Modified: `src/agent.py`

- No changes needed — `custom_call_tool()` and `_run()` are generic; they already pass all tool args. The new fields flow through automatically.

## Score Is Not a Gate

Critical design constraint — the score must never trigger anxiety or hallucination:

1. **No minimum score requirement.** The model can answer at any time. The dashboard is information, not a pass/fail.
2. **Diminishing returns → stop signal.** Two consecutive search rounds with zero new Tier 1-2 sources means "you've probably exhausted the high-quality landscape." Proceed.
3. **Transparency in reporting.** If source quality is limited, the model can include a note: *"This topic draws primarily on practitioner interviews (Tier 3) and industry reporting (Tier 2) — limited peer-reviewed literature exists."*

## Edge Cases

| Case | Behavior |
|------|----------|
| No `cumulative_scores` passed | Cumulative shows "Round 1" — no historical data. Dashboard still works. |
| All results are Tier 4 | Cumulative score will be low. Model may try scholar endpoint. If still low after exhaustive search, answer with honest note. |
| Topic is about "best affiliate products" | Intentionally scores 0 — the model will correctly classify and may note the source landscape. This is correct behavior. |
| Mixed batch (some scored, some not) | Unscored results get Tier 4 default. Never leave a source unscored. |
| Model never passes cumulative forward | Cumulative stays per-round. Model loses the aggregate view but still sees each batch. No functional breakage. |

## Implementation Order

1. Create `src/source_quality.py` — classifier, scorer, dashboard formatters
2. Modify `src/tools/search.py` — integrate scoring + dashboard appendage
3. Modify `src/tools/scholar.py` — integrate scoring + dashboard appendage
4. Modify `src/prompts.py` — add Source Quality Dashboard system prompt section
5. Manual test with a few queries to verify scoring + dashboard rendering
6. Tune classification rules based on observed false positives/negatives

## Concerns, Caveats & Open Questions

### Tier scoring is not universal truth

The tier system encodes a *general* hierarchy of source authority, but it's inherently domain-dependent:

- **For a medical question**, Tier 1 (peer-reviewed RCTs) genuinely outranks everything.
- **For "how do people actually use X tool?"**, a Tier 3 Reddit thread with numbered steps and real screenshots is probably more useful than a Tier 2 government report.
- **For emerging topics** (new tech, current events), there may be zero Tier 1-2 sources available. The low score is a feature, not a bug — the dashboard honestly reflects the source landscape and the report can note that.

The score is a *descriptive* metric about what kinds of sources were used, not a *prescriptive* value judgment about research quality. A narrowly-scoped practical question with mostly Tier 3-4 sources and a high-quality synthesis of those sources is *good research*. A broad question with many Tier 1 sources but a shallow synthesis is *bad research*. The score only measures the first part.

### Hallucination risk

The primary risk: a model that's "trying to get the score up" starts fabricating sources. Mitigations:

1. **Nowhere in the system prompt does the model have agency over scores.** It does not assign scores. It only reads them. The scoring is done by `source_quality.py` on real search results. A model that can't write a score can't fake a score.
2. **The diminishing-returns gate is the safety valve.** Two consecutive zero-Tier-1-2 rounds is the most explicit "stop" signal. This catches the case where the model keeps searching fruitlessly.
3. **Honest-reporting guidance.** The prompt explicitly says: "If you've searched thoroughly, synthesise what you have. Your report can note the source landscape honestly." This normalises the "low score, but that's okay" outcome.

**Still, we should flag this as a monitoring priority in testing.** If the model starts doing extra search rounds after the cumulative score looks reasonable, that's a sign the score is causing anxiety. The prompt may need softening.

### The cumulative state-passing pattern is fragile

The `cumulative_scores` parameter being passed through tool arguments is clever but has failure modes:

- **Model forgets to include it** → dashboard shows "Round 1" again, no cumulative data. The model loses the aggregate view but each batch is still scored. No functional harm.
- **Model makes up a value** → possible if the model gets creative. The cumulative would show inconsistent numbers. The system prompt should emphasise "pass the value from the most recent dashboard verbatim."
- **JSON gets corrupted in transit** → tools should validate the `cumulative_scores` field and silently discard malformed values.

All these fail gracefully — the per-batch dashboard still works independently. Cumulative is a bonus, not a requirement.

### Source classification granularity

The domain-based classification is intentionally coarse for v1. Known limitations:

- **`.edu` is not automatically peer-reviewed.** A university press release, a professor's blog, and a departmental landing page all live on `.edu` but aren't peer-reviewed. The classifier currently lumps all `.edu` into Tier 1. A future version could refine this: check for `/news/`, `/blog/` in the path → downgrade to Tier 4; check for `/journal/`, `/pub/`, `/article/` → confirm Tier 1.
- **Medium/Substack cannot be reliably classified.** Some Substack newsletters are rigorous deep-dives with citations (should be Tier 2+); others are quick personal musings (Tier 3). We default to Tier 3 for these domains, which is a reasonable compromise.
- **Reddit is a noise signal.** Most Reddit content is Tier 3 (anecdotal), but some subreddits function as quasi-peer-reviewed communities (e.g., `/r/askscience`, `/r/academic*`). Detecting subreddit context from URL structure could be a future refinement.
- **Affiliate networks are a moving target.** New affiliate programs spawn daily. The domain blocklist will need periodic updates. Consider making it data-driven (a config file or data source).

## Future Versions / Ideas

### v2: Dynamic per-query tier weights

The fixed Tier 1-5 scoring doesn't account for the *type* of question being asked. Future idea:

- Before starting research, classify the query into a domain (medical, technical, practical, current-events, philosophical).
- Different domains get different tier weightings. For a practical "how do I X" query, Tier 3 (anecdotal/practitioner) could weight at 4 instead of 3. For a medical query, Tier 3 could drop to 2.
- This could be done by the model itself (a brief classification step before research begins) or by a lightweight keyword classifier.

This would make the cumulative score a more honest measure of "research depth" for each specific query type.

### v3: Post-research quality report

After the model produces its final answer, a separate pass (the model itself, or a brief script) could:

1. Scan the final report for in-text citations
2. Cross-reference each citation against the quality dashboard
3. Produce a **Source Quality Report** appended to the document:

```
## Source Quality

This report draws from 12 sources:
- Peer-Reviewed: 3 (25%)
- Scholarly/Gov: 2 (17%)
- Practitioner/Anecdotal: 5 (42%) ← heavy reliance on first-hand accounts
- Blog/News: 2 (17%)

Weighted quality score: 33/60 (55%)

Note: Limited peer-reviewed literature exists on this topic.
Research draws primarily on practitioner interviews and
industry reporting. Key claims are corroborated across
multiple first-hand accounts.
```

This turns the score from an internal model signal into an output quality indicator that adds transparency and credibility to reports.

### v4: Visit-page verification

Currently, `visit.py` doesn't score the page it reads — it just summarises it. Future idea:

- When the model visits a URL that was previously scored as a search result, the `visit` tool could look up that URL in its cumulative record and re-affirm (or downgrade) the tier based on actual page content.
- Example: a search result from `.edu` was classified Tier 1, but when visited, the page turns out to be a student blog post. The tier could be revised down.
- This would require the cumulative tally to include per-URL tier records, not just aggregate counts. More complex state-passing.

### v5: Adversarial source check

A dedicated round where the model explicitly asks: *"What's the weakest source in my cumulative dashboard? Do I have enough corroboration for the claims I'm about to make?"* This forces it to check its own work before committing to an answer. Could be prompted by the system prompt as a pre-answer checklist:

```
Before answering, briefly scan your source dashboard:
1. Do you have at least 2 sources for your main claims?
2. Are any critical claims backed only by Tier 4 sources?
3. Have you exhausted the obvious search angles?
```

This keeps the check internal to the model's reasoning (in `<think>` tags) and adds no new tool calls.

### v6: Top-K source preservation for report writing

Store the top-K (e.g., top 10-20) highest-scoring search results (title, URL, snippet, tier, score) in a dedicated context block that the model can reference during synthesis. Currently, the model must rely on what it remembers from its tool responses. A structured "Best Sources" block appended to the end of the ReAct loop (just before the final synthesis pass) would:

- Give the model a clean reference for citation and synthesis
- Surface high-quality sources it visited but may have forgotten
- Make the report writing pass more accurate and better-sourced

The block could be generated by `source_quality.py` at the start of each round and updated with each new batch. The model sees it as a tool response-like observation.

### v7: User-configurable tier profiles

Not all research needs peer-reviewed literature. A config setting could let the user specify source preferences:

```
# .env or CLI arg
RESEARCH_PROFILE=rapid        # prefer speed, accept Tier 3-4 sources
RESEARCH_PROFILE=academic     # require Tier 1-2, more rounds
RESEARCH_PROFILE=balanced     # default — current tier system
```

Or even a custom tier mapping passed at runtime:
```
# "I care about practitioner accounts more than academic papers"
CUSTOM_TIERS='{"1": 3, "2": 3, "3": 5, "4": 2}'
```

### v8: LLM-assisted classification fallback

The rule-based classifier will inevitably miss edge cases. A fallback: when a source domain isn't in the known list and the snippet is ambiguous, defer to a quick LLM call (the summary model) for classification:

```
Classify this source into one of:
1. Peer-reviewed academic publication
2. Government/institutional/research
3. Personal account / interview / practitioner
4. Blog / news / general reporting
5. Affiliate / product / low-quality

URL: example.com
Snippet: "..."
Respond with just the number.
```

This would catch things like niche academic journals, high-quality practitioner blogs, and obscure government agencies. Cost trade-off: each ambiguous source adds ~200 tokens and a brief LLM call. Only worth it if false classifications are a real problem in practice.

## Testing & Monitoring Concerns

### What to watch for in early testing

1. **Does the model over-search?** Count how many search rounds happen on a topic with low Tier 1-2 availability. If it keeps searching past the diminishing-returns gate, the prompt guidance isn't strong enough.
2. **Does the cumulative score ever decrease?** (It shouldn't — scores only accumulate.) If the dashboard shows a decreasing cumulative total, there's a bug in `merge_cumulative()`.
3. **Does the model ignore the dashboard entirely?** Some models may not engage with the quality information. This is fine — the dashboard is still appended as structured data, and the model naturally uses it even if it doesn't explicitly reference it. Check reports for diversity of source tiers.
4. **Does the per-batch table inflate context too much?** Each search result adds ~2 lines of dashboard. With ~10 results per batch and ~5-10 rounds, that's ~100-200 lines of dashboard text. At 10k+ total context, this is negligible. But if it becomes a concern, the dashboard could be summarised into a one-liner.

### What success looks like

- The model does 1-2 *more* search rounds on topics where it initially found only blog posts
- The model switches its search strategy (e.g., web search → scholar search) when dashboard shows low score
- The model stops naturally once it has diverse tier coverage
- The model produces a source-quality note in its report when the score is low
- No observable increase in hallucinated sources
