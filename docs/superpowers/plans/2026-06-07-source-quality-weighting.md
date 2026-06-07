# Source Quality Weighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) for syntax tracking.

**Goal:** Add a source quality scoring and dashboard system to the deep research ReAct agent so the model can assess source quality as it researches.

**Architecture:** A new `src/source_quality.py` module provides classification (domain/URL → tier 1–5) and dashboard formatting functions. The `search.py` and `scholar.py` tools are modified to accept a `cumulative_scores` state blob from the model, score all returned results, append per-batch + cumulative quality dashboards to their response text, and emit a JSON cumulative blob the model can pass forward. The `prompts.py` system prompt gains guidance on how to use the dashboard for research depth decisions.

**Tech Stack:** Python 3, urllib.parse, re, json — no new dependencies.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/source_quality.py` | **Create** | 5-tier classifier, scorer, dashboard formatters, cumulative tally merge |
| `src/tools/search.py` | **Modify** | Import source_quality, accept cumulative_scores param, score results, append dashboards |
| `src/tools/scholar.py` | **Modify** | Same as search.py; scholar results with citedBy > 0 → Tier 1 |
| `src/prompts.py` | **Modify** | Add Source Quality Dashboard guidance + update tool definitions for cumulative_scores param |

No changes to `agent.py` — `custom_call_tool()` already passes all tool args through generically.

---

### Task 1: Create `src/source_quality.py`

**Files:**
- Create: `src/source_quality.py`

- [ ] **Step 1: Write the module with constants, domain maps, and classify_source()**

```python
"""Source quality classification and scoring for deep research.

Classifies search results into quality tiers based on domain, URL path,
and snippet signals. Provides dashboard formatters for per-batch and
cumulative source quality display.
"""

import json
import re
from urllib.parse import urlparse

# Tier constants
PEER_REVIEWED = 1  # Score 5
SCHOLARLY_GOV = 2  # Score 4
ANECDOTAL = 3      # Score 3
BLOG_NEWS = 4      # Score 2
AFFILIATE = 5      # Score 0 (discarded)

TIER_SCORES = {
    PEER_REVIEWED: 5,
    SCHOLARLY_GOV: 4,
    ANECDOTAL: 3,
    BLOG_NEWS: 2,
    AFFILIATE: 0,
}

TIER_NAMES = {
    PEER_REVIEWED: "Peer-Reviewed",
    SCHOLARLY_GOV: "Scholarly/Gov",
    ANECDOTAL: "Anecdotal",
    BLOG_NEWS: "Blog/News",
    AFFILIATE: "Affiliate",
}

# Fast O(1) exact-domain lookup for known academic/research domains
EXACT_DOMAIN_TIERS = {
    "pubmed.ncbi.nlm.nih.gov": PEER_REVIEWED,
    "arxiv.org": PEER_REVIEWED,
    "ieeexplore.ieee.org": PEER_REVIEWED,
    "dl.acm.org": PEER_REVIEWED,
    "sciencedirect.com": PEER_REVIEWED,
    "link.springer.com": PEER_REVIEWED,
    "onlinelibrary.wiley.com": PEER_REVIEWED,
    "journals.plos.org": PEER_REVIEWED,
    "nature.com": PEER_REVIEWED,
    "science.org": PEER_REVIEWED,
    "cell.com": PEER_REVIEWED,
    "thelancet.com": PEER_REVIEWED,
    "bmj.com": PEER_REVIEWED,
    "nejm.org": PEER_REVIEWED,
    "jstor.org": PEER_REVIEWED,
    "cambridge.org": PEER_REVIEWED,
    "academic.oup.com": PEER_REVIEWED,
    "tandfonline.com": PEER_REVIEWED,
    "journals.sagepub.com": PEER_REVIEWED,
    "annualreviews.org": PEER_REVIEWED,
    "frontiersin.org": PEER_REVIEWED,
    "mdpi.com": PEER_REVIEWED,
    "biomedcentral.com": PEER_REVIEWED,
    "peerj.com": PEER_REVIEWED,
    "ncbi.nlm.nih.gov": PEER_REVIEWED,
    "semanticscholar.org": PEER_REVIEWED,
}

# Suffix-based tier mappings (checked via str.endswith)
DOMAIN_SUFFIX_TIERS = {
    ".gov": SCHOLARLY_GOV,
    ".mil": SCHOLARLY_GOV,
    ".int": SCHOLARLY_GOV,
}

# Well-known institutional / scholarly / research domains (Tier 2)
INSTITUTIONAL_DOMAINS = [
    "who.int", "un.org", "oecd.org", "worldbank.org",
    "rand.org", "brookings.edu", "nber.org", "imf.org",
    "cbo.gov", "nih.gov", "cdc.gov", "nasa.gov",
    "loc.gov", "archives.gov", "usgs.gov", "noaa.gov",
    "fda.gov", "epa.gov", "nsf.gov", "energy.gov",
    "state.gov", "justice.gov", "census.gov",
]

# Known affiliate / commerce domains — Tier 5 (score 0, discard)
AFFILIATE_DOMAINS = {
    "amazon.com", "amzn.to", "amzn.eu",
    "shareasale.com", "clickbank.com", "skimlinks.com",
    "skimresources.com", "cj.com", "rakuten.com",
    "ebay.com", "ebay.ca", "ebay.co.uk",
    "aliexpress.com", "alibaba.com", "etsy.com",
    "walmart.com", "target.com", "bestbuy.com",
    "homedepot.com", "lowes.com",
}

# URL path patterns indicating affiliate / product / shop content
AFFILIATE_PATH_PATTERNS = re.compile(
    r"/(?:product/|shop/|buy/|pricing/|cart/|checkout/|"
    r"affiliate/|ref=|tag=|redirect|sponsored/|"
    r"advert|promo|coupon|deal/)",
    re.IGNORECASE,
)

# URL path patterns indicating anecdotal / interview content
ANECDOTAL_PATH_PATTERNS = re.compile(
    r"/(?:interview/|ama/|experience/|personal/|story/|"
    r"testimonial|oral[- ]history|firsthand|first[- ]hand|"
    r"how-i-|my-journey|lessons-learned)",
    re.IGNORECASE,
)

# Anecdotal-domain hosts (checked via substring match)
ANECDOTAL_HOSTS = {"substack.com", "medium.com", "quora.com", "reddit.com"}

# Snippet keywords suggesting first-hand practitioner accounts
ANECDOTAL_SNIPPET_KEYWORDS = [
    "interview", "ama", "my experience", "i found", "personally",
    "in my case", "real example", "here's what i", "i tried",
    "i built", "i started", "my story", "what i learned",
    "i've been", "after years of", "first-hand account",
    "from my perspective", "actual numbers", "here's my",
    "i can confirm", "speaking from",
]


def _extract_domain(url: str) -> str:
    """Extract and lowercase the domain host from a URL."""
    parsed = urlparse(url)
    return parsed.netloc.lower() or parsed.path.lower()


def _check_host_substring(domain: str, targets: set) -> bool:
    """Check if any target string appears anywhere in the domain."""
    for target in targets:
        if target in domain:
            return True
    return False


def classify_source(url: str, snippet: str = "", cited_by: int = 0) -> dict:
    """Classify a source URL into a quality tier (1–5).

    Heuristic chain (fast → slow):
    1. Exact domain match against known academic domains
    2. Suffix-based match (.gov, .mil, .int)
    3. Affiliate domain check
    4. Affiliate URL path pattern check
    5. Anecdotal URL path pattern check
    6. .edu domain → default to peer-reviewed
    7. Substack/Medium/Quora/Reddit → anecdotal
    8. Unknown .com/.org/.net → Blog/News
    9. Everything else → Scholarly/Gov (conservative default)

    Returns:
        dict with 'tier' (int 1-5), 'score' (int 0-5), 'label' (str)
    """
    domain = _extract_domain(url.strip())

    # 1. Exact domain match (fastest — O(1) hash lookup)
    if domain in EXACT_DOMAIN_TIERS:
        tier = EXACT_DOMAIN_TIERS[domain]
        return {"tier": tier, "score": TIER_SCORES[tier], "label": TIER_NAMES[tier]}

    # 2. Suffix-based match (.gov, .mil, .int)
    for suffix, tier in DOMAIN_SUFFIX_TIERS.items():
        if domain.endswith(suffix):
            return {"tier": tier, "score": TIER_SCORES[tier], "label": TIER_NAMES[tier]}

    # 3. Check for known institutional domains
    for inst in INSTITUTIONAL_DOMAINS:
        if domain == inst or domain.endswith("." + inst):
            return {"tier": SCHOLARLY_GOV, "score": TIER_SCORES[SCHOLARLY_GOV], "label": TIER_NAMES[SCHOLARLY_GOV]}

    # 4. Check for affiliate / commerce domains
    for aff in AFFILIATE_DOMAINS:
        if aff in domain:
            return {"tier": AFFILIATE, "score": 0, "label": "Affiliate"}

    # 5. Check URL path for affiliate patterns
    if AFFILIATE_PATH_PATTERNS.search(url):
        return {"tier": AFFILIATE, "score": 0, "label": "Affiliate"}

    # 6. Check URL path for anecdotal / interview patterns
    if ANECDOTAL_PATH_PATTERNS.search(url):
        return {"tier": ANECDOTAL, "score": TIER_SCORES[ANECDOTAL], "label": TIER_NAMES[ANECDOTAL]}

    # 7. .edu domains default to Peer-Reviewed (Tier 1)
    if domain.endswith(".edu"):
        return {"tier": PEER_REVIEWED, "score": TIER_SCORES[PEER_REVIEWED], "label": TIER_NAMES[PEER_REVIEWED]}

    # 8. Anecdotal-platform hosts
    if _check_host_substring(domain, ANECDOTAL_HOSTS):
        return {"tier": ANECDOTAL, "score": TIER_SCORES[ANECDOTAL], "label": TIER_NAMES[ANECDOTAL]}

    # 9. Check snippet for first-hand/practitioner keywords
    if snippet:
        snippet_lower = snippet.lower()
        for kw in ANECDOTAL_SNIPPET_KEYWORDS:
            if kw in snippet_lower:
                return {"tier": ANECDOTAL, "score": TIER_SCORES[ANECDOTAL], "label": TIER_NAMES[ANECDOTAL]}

    # 10. Fallback by TLD
    if domain.endswith(".com") or domain.endswith(".org") or domain.endswith(".net"):
        return {"tier": BLOG_NEWS, "score": TIER_SCORES[BLOG_NEWS], "label": TIER_NAMES[BLOG_NEWS]}

    # 11. Default — conservative assumption
    return {"tier": SCHOLARLY_GOV, "score": TIER_SCORES[SCHOLARLY_GOV], "label": TIER_NAMES[SCHOLARLY_GOV]}


def score_search_results(organic_results: list) -> list:
    """Score each result in a Serper "organic" array.

    Adds 'tier', 'score', and 'label' keys to each result dict.

    Args:
        organic_results: List of dicts from Serper's "organic" or "scholar"
                         response. Each dict should have at least 'link'
                         (or 'pdfUrl' for scholar) and 'snippet'.

    Returns:
        List of augmented result dicts.
    """
    scored = []
    for result in organic_results:
        # Scholar results have 'pdfUrl' instead of 'link'
        url = result.get("link", result.get("pdfUrl", ""))
        snippet = result.get("snippet", "")
        cited_by = int(result.get("citedBy", 0))

        classification = classify_source(url, snippet, cited_by)

        # Scholar-specific: if citedBy > 0, force to peer-reviewed
        if cited_by > 0:
            classification = {
                "tier": PEER_REVIEWED,
                "score": TIER_SCORES[PEER_REVIEWED],
                "label": TIER_NAMES[PEER_REVIEWED],
            }

        scored.append({**result, **classification})

    return scored


def format_batch_dashboard(scored_results: list) -> str:
    """Format the per-batch source quality table."""
    if not scored_results:
        return ""

    lines = ["\n## Source Quality Dashboard"]
    lines.append("| # | Title | Domain | Tier | Score |")
    lines.append("|---|-------|--------|------|-------|")

    total_score = 0
    for i, r in enumerate(scored_results, 1):
        link = r.get("link", r.get("pdfUrl", ""))
        domain = _extract_domain(link) if link else "(no link)"
        title = (r.get("title", "") or "")[:50]
        score = r.get("score", 0)
        label = r.get("label", "Unknown")
        total_score += score
        lines.append(f"| {i} | {title} | {domain} | {label} | {score} |")

    avg = total_score / len(scored_results) if scored_results else 0
    lines.append(f"\nThis batch: {len(scored_results)} sources, avg score {avg:.1f}")

    return "\n".join(lines)


def format_cumulative_dashboard(cum: dict) -> str:
    """Format the cumulative quality summary across all rounds.

    Args:
        cum: Dict with keys: tier1_count, tier2_count, tier3_count,
             tier4_count, tier1_score, tier2_score, tier3_score,
             tier4_score, total_possible.
    """
    tier_counts = {
        1: cum.get("tier1_count", 0),
        2: cum.get("tier2_count", 0),
        3: cum.get("tier3_count", 0),
        4: cum.get("tier4_count", 0),
    }
    tier_scores = {
        1: cum.get("tier1_score", 0),
        2: cum.get("tier2_score", 0),
        3: cum.get("tier3_score", 0),
        4: cum.get("tier4_score", 0),
    }

    total_sources = sum(tier_counts.values())
    total_score = sum(tier_scores.values())
    total_possible = cum.get("total_possible", total_score)

    pct = (total_score / total_possible * 100) if total_possible > 0 else 0

    lines = [
        "\n## Cumulative Quality (all rounds)",
        f"Peer-Reviewed (×5):  {tier_counts[1]}  ({tier_scores[1]} pts)",
        f"Scholarly/Gov (×4):  {tier_counts[2]}  ({tier_scores[2]} pts)",
        f"Anecdotal (×3):     {tier_counts[3]}  ({tier_scores[3]} pts)",
        f"Blog/News (×2):     {tier_counts[4]}  ({tier_scores[4]} pts)",
        f"Discarded (×0):     {total_sources - sum(tier_counts.values())}  (0 pts)",
        "─" * 30,
        f"Weighted total:     {total_score} / {total_possible}  ({pct:.0f}%)",
        f"Sources:            {total_sources}",
    ]

    return "\n".join(lines)


def format_cumulative_json(cum: dict) -> str:
    """Return cumulative scores as a compact JSON string for model state-passing."""
    return json.dumps(cum, sort_keys=True, separators=(",", ":"))


def merge_cumulative(existing: dict, batch: list) -> dict:
    """Merge batch scores into a running cumulative tally.

    Args:
        existing: Previous cumulative dict (or empty dict for first round).
        batch: List of scored results from score_search_results().

    Returns:
        Updated cumulative dict.
    """
    result = {
        "tier1_count": existing.get("tier1_count", 0),
        "tier2_count": existing.get("tier2_count", 0),
        "tier3_count": existing.get("tier3_count", 0),
        "tier4_count": existing.get("tier4_count", 0),
        "tier1_score": existing.get("tier1_score", 0),
        "tier2_score": existing.get("tier2_score", 0),
        "tier3_score": existing.get("tier3_score", 0),
        "tier4_score": existing.get("tier4_score", 0),
        "total_possible": existing.get("total_possible", 0),
    }

    for item in batch:
        tier = item.get("tier", BLOG_NEWS)
        score = item.get("score", 2)

        if tier == PEER_REVIEWED:
            result["tier1_count"] += 1
            result["tier1_score"] += score
        elif tier == SCHOLARLY_GOV:
            result["tier2_count"] += 1
            result["tier2_score"] += score
        elif tier == ANECDOTAL:
            result["tier3_count"] += 1
            result["tier3_score"] += score
        elif tier == BLOG_NEWS:
            result["tier4_count"] += 1
            result["tier4_score"] += score
        # Tier 5 (AFFILIATE) — nothing to add

        result["total_possible"] += 5  # max score per source

    return result


def add_dashboard_to_response(
    response_text: str,
    organic_results: list,
    cumulative_scores: dict | None = None,
) -> tuple[str, dict]:
    """Score results, build dashboards, append to response text.

    Returns:
        Tuple of (updated_response_text, updated_cumulative_scores_dict).
    """
    scored = score_search_results(organic_results)
    new_cum = merge_cumulative(cumulative_scores or {}, scored)

    dashboard = format_batch_dashboard(scored)
    cum_dashboard = format_cumulative_dashboard(new_cum)
    cum_json = format_cumulative_json(new_cum)

    response_text += dashboard
    response_text += cum_dashboard
    response_text += f"\n\nCUMULATIVE_SCORES:{cum_json}"

    return response_text, new_cum
```

- [ ] **Step 2: Verify the module loads without errors**

Run in the project directory:

```bash
cd ~/ai/projects/dev/deep-research
python -c "
from src.source_quality import (
    classify_source, score_search_results, format_batch_dashboard,
    format_cumulative_dashboard, merge_cumulative, add_dashboard_to_response,
    PEER_REVIEWED, SCHOLARLY_GOV, ANECDOTAL, BLOG_NEWS, AFFILIATE,
)

# Quick smoke test
c1 = classify_source('https://pubmed.ncbi.nlm.nih.gov/12345/')
print(f'pubmed: {c1}')  # Should be tier 1

c2 = classify_source('https://www.reddit.com/r/personalfinance/comments/abc123/')
print(f'reddit: {c2}')  # Should be tier 3

c3 = classify_source('https://example.com/product/buy-now')
print(f'affiliate: {c3}')  # Should be tier 5

c4 = classify_source('https://blog.example.com/news/article')
print(f'blog: {c4}')  # Should be tier 4

c5 = classify_source('https://www.who.int/publications')
print(f'who: {c5}')  # Should be tier 2

# Dashboard formatting
test_scored = [
    {'title': 'Test Paper', 'link': 'https://arxiv.org/abs/1234', 'snippet': 'test', 'tier': 1, 'score': 5, 'label': 'Peer-Reviewed'},
    {'title': 'My Story', 'link': 'https://substack.com/p/my-story', 'snippet': 'interview', 'tier': 3, 'score': 3, 'label': 'Anecdotal'},
]
print(format_batch_dashboard(test_scored))
print()

cum = merge_cumulative({}, test_scored)
print(format_cumulative_dashboard(cum))
print()

print('All smoke tests passed.')
"
```

Expected output — no import errors, all classifications produce correct tier numbers, dashboard formatting produces clean markdown tables.

- [ ] **Step 3: Commit**

```bash
git add src/source_quality.py
git commit -m "feat: add source quality classification and dashboard formatting"
```

---

### Task 2: Modify `src/tools/search.py` — integrate source quality scoring

**Files:**
- Modify: `src/tools/search.py`

- [ ] **Step 1: Add import and modify google_search_with_serp() to score results**

Add the import at the top of the file, then modify `google_search_with_serp()` to accept `cumulative_scores`, score results, and append dashboards.

Add after existing imports:

```python
from source_quality import add_dashboard_to_response
```

Modify `google_search_with_serp()` — change its signature to accept an optional `cumulative_scores` param, and after building `web_snippets`, score the results and append the dashboard:

Find the block starting at `if "organic" in results:` (currently lines 78–105). Replace the entire `if "organic" in results:` block with:

```python
            if "organic" in results:
                web_snippets = list()
                idx = 0
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

                content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)

                # ── Source quality scoring ──
                cumulative_scores = kwargs.get("cumulative_scores", {}) if kwargs else {}
                content, _ = add_dashboard_to_response(content, results["organic"], cumulative_scores)
            else:
                content = f"No results found for '{query}'. Try with a more general query."
```

Then update the return statement — it should still return `content` as before, but now content includes the dashboard.

Wait — `google_search_with_serp` currently doesn't take `**kwargs`. Let me think about how to thread the cumulative_scores through.

The simplest approach: add a `cumulative_scores` parameter to `google_search_with_serp()`:

```python
def google_search_with_serp(self, query: str, cumulative_scores: dict | None = None):
```

And propagate through `search_with_serp()`:

```python
def search_with_serp(self, query: str, cumulative_scores: dict | None = None):
    result = self.google_search_with_serp(query, cumulative_scores)
    return result
```

And in `call()`, extract cumulative_scores from params and pass through.

Let me write this cleanly.

Actually, the current `call()` method iterates multiple queries and joins responses with `\n=======\n`. If each individual query response already has its own dashboard, and we need a cumulative across *all queries in this batch*, then the cumulative needs to be threaded through the loop.

That's more complex. Let me simplify: score in `call()` at the batch level. But `google_search_with_serp()` builds the response text internally... 

Actually, the simplest approach is:

1. `call()` extracts `cumulative_scores` from params
2. Pass it to `search_with_serp()` → `google_search_with_serp()`
3. Inside `google_search_with_serp()`, after building results text, score the raw organic results
4. Append dashboards using `cumulative_scores` 
5. Return the modified text
6. For multi-query batches, each individual result gets its own dashboard (since scores are per-query)

The cumulative for multi-query: I need to merge across queries. Let me do the merge in `call()`:

Actually, let me keep it simpler. Each `google_search_with_serp()` call:
1. Does its own search
2. Scores those specific results
3. Merges with the provided `cumulative_scores`
4. Returns the response text with dashboard

For multi-query call, the first query gets the external cumulative, subsequent queries get the updated cumulative from the previous query. This way the cumulative is always accurate.

Let me write the code:

```python
def call(self, params: Union[str, dict], **kwargs) -> str:
    try:
        query = params["query"]
    except:
        return "[Search] Invalid request format: ..."
    
    cumulative_scores = params.get("cumulative_scores", {}) if isinstance(params, dict) else {}
    
    if isinstance(query, str):
        response = self.search_with_serp(query, cumulative_scores)
    else:
        assert isinstance(query, List)
        responses = []
        cum = cumulative_scores
        for q in query:
            result = self.search_with_serp(q, cum)
            # Extract updated cumulative from the result text
            # But we can't easily extract it from text...
            # Better approach: return both text and cum from google_search_with_serp
            ...
    
    return response
```

Hmm, this is getting messy for multi-query. Let me restructure `google_search_with_serp` to return both the text and the updated cumulative, and have `call()` handle the text assembly:

Actually, the cleanest minimal change: make `google_search_with_serp` return a tuple `(text, updated_cumulative)` and then `call()` handles everything.

But that changes the return type of an internal method. Let me see what the current call chain is:

1. `call()` calls `self.search_with_serp(query)` or multiple
2. `search_with_serp()` calls `self.google_search_with_serp(query)` — returns str
3. `call()` joins strings with `"\n=======\n"`

So I have two choices:
A. Change `google_search_with_serp` to return tuple, have `call()` handle assembly
B. Have the dashboard appended inside `google_search_with_serp`, and for multi-query, do the cumulative merge inside `call()` and pass it per-query

Option B is cleaner because `google_search_with_serp()` remains self-contained — it takes cumulative_scores, scores its batch, appends dashboard, and the caller is responsible for threading the cumulative forward.

For multi-query, I need to parse the updated cumulative from each individual result. The cleanest way: `google_search_with_serp()` returns both the text and the new cumulative blob. Let me do that.

Actually, the absolute simplest approach: make `google_search_with_serp()` return a tuple `(content, new_cumulative)` and have `search_with_serp()` and `call()` handle it:

```python
def google_search_with_serp(self, query: str, cumulative_scores: dict | None = None) -> tuple[str, dict]:
    # ... existing code ...
    # At the end:
    content = ...  # existing text assembly
    content, new_cum = add_dashboard_to_response(content, organic_results, cumulative_scores)
    return content, new_cum

def search_with_serp(self, query: str, cumulative_scores: dict | None = None) -> tuple[str, dict]:
    return self.google_search_with_serp(query, cumulative_scores)

def call(self, params: Union[str, dict], **kwargs) -> str:
    try:
        query = params["query"]
    except:
        return "[Search] Invalid request format: ..."
    
    cumulative_scores = params.get("cumulative_scores", {}) if isinstance(params, dict) else {}
    
    if isinstance(query, str):
        response, _ = self.search_with_serp(query, cumulative_scores)
    else:
        assert isinstance(query, List)
        responses = []
        cum = cumulative_scores
        for q in query:
            resp, cum = self.search_with_serp(q, cum)
            responses.append(resp)
        response = "\n=======\n".join(responses)
    
    return response
```

This is clean. The `_` in `call()` for single query means we discard the cumulative (because there won't be a next round from this call anyway — the model gets the dashboard in the response text and the CUMULATIVE_SCORES: JSON tag).

For multi-query, we thread `cum` through the loop, so each successive query's dashboard reflects all previous queries' scores. Good.

Now, the `parameters` dict on the Search class. Add `cumulative_scores` as an optional field. But looking at the existing `parameters` dict — it only has `query`. I need to add `cumulative_scores`:

```python
parameters = {
    "type": "object",
    "properties": {
        "query": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Array of query strings. Include multiple complementary search queries in a single call."
        },
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
            "required": [],
            "description": "Cumulative quality scores from previous rounds. Pass the CUMULATIVE_SCORES value from the last search/scholar response verbatim.",
        },
    },
    "required": ["query"],
}
```

Wait, I also need to handle `**kwargs` in `call()`. Let me check the current signature:

```python
def call(self, params: Union[str, dict], **kwargs) -> str:
```

Good, it already has `**kwargs`. The `custom_call_tool` in `agent.py` calls:
```python
raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
```

So `tool_args` contains the params dict with `query` and `cumulative_scores`. That's exactly what I need.

Now, there's a subtle issue with the `__init__` of `Search`:
```python
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches..."
    parameters = {...}
```

The `parameters` attr is a class variable. I'll just edit the dict to add the new field.

Alright, let me write out the full plan steps now.

For `scholar.py`, same pattern but:
- The Serper response for scholar uses `pdfUrl` instead of `link`
- `citedBy` field is available
- Scholar default classification is higher (Tier 1 if citedBy > 0, Tier 2 otherwise)

For `prompts.py`, I need to:
1. Add the Source Quality Dashboard guidance section to SYSTEM_PROMPT
2. Update the search tool definition to include `cumulative_scores` parameter
3. Update the google_scholar tool definition

The tool definitions in SYSTEM_PROMPT are inline JSON strings. I need to modify them carefully.

Let me write all this out now.

Actually, for the prompts.py changes, I realize there's a subtlety. The tool definitions in prompts.py are JSON blobs embedded in XML. The model uses these to know what parameters a tool accepts. If I add `cumulative_scores` to these definitions, the model will know it can pass them. If I don't add them, the model might not pass them, but the tool code will still handle it (defaulting to empty {}).

For robustness, I should update the prompt tool definitions too. Let me plan the exact edits.

Current search tool definition in prompts.py:
```python
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
```

Updated:
```python
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries. Results now include a Source Quality Dashboard with tier/score for each result and a CUMULATIVE_SCORES JSON blob — pass this verbatim in your next search/scholar call.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}, "cumulative_scores": {"type": "object", "properties": {"tier1_count": {"type": "integer"}, "tier2_count": {"type": "integer"}, "tier3_count": {"type": "integer"}, "tier4_count": {"type": "integer"}, "tier1_score": {"type": "integer"}, "tier2_score": {"type": "integer"}, "tier3_score": {"type": "integer"}, "tier4_score": {"type": "integer"}, "total_possible": {"type": "integer"}}, "required": [], "description": "Cumulative quality scores from previous rounds. Pass the CUMULATIVE_SCORES value from the last search/scholar response verbatim."}}, "required": ["query"]}}}
```

That's a big inline edit. Let me make it clean in the plan.

Similarly for google_scholar.

OK, let me now write the complete plan document.<｜end▁of▁thinking｜>

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="Write">
<｜｜DSML｜｜parameter name="content" string="true"># Source Quality Weighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) for syntax tracking.

**Goal:** Add a source quality scoring and dashboard system to the deep research ReAct agent so the model can assess source quality as it researches.

**Architecture:** A new `src/source_quality.py` module provides classification (domain/URL → tier 1–5) and dashboard formatting functions. The `search.py` and `scholar.py` tools are modified to accept a `cumulative_scores` state blob from the model, score all returned results, append per-batch + cumulative quality dashboards to their response text, and emit a JSON cumulative blob the model can pass forward. The `prompts.py` system prompt gains guidance on how to use the dashboard for research depth decisions.

**Tech Stack:** Python 3, urllib.parse, re, json — no new dependencies.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/source_quality.py` | **Create** | 5-tier classifier, scorer, dashboard formatters, cumulative tally merge |
| `src/tools/search.py` | **Modify** | Import source_quality, accept cumulative_scores param, score results, append dashboards |
| `src/tools/scholar.py` | **Modify** | Same as search.py; scholar results with citedBy > 0 → Tier 1 |
| `src/prompts.py` | **Modify** | Add Source Quality Dashboard guidance + update tool definitions for cumulative_scores param |

No changes to `agent.py` — `custom_call_tool()` already passes all tool args through generically.

---

### Task 1: Create `src/source_quality.py`

**Files:**
- Create: `src/source_quality.py`

- [ ] **Step 1: Write the module**

`src/source_quality.py`:

```python
"""Source quality classification and scoring for deep research.

Classifies search results into quality tiers based on domain, URL path,
and snippet signals. Provides dashboard formatters for per-batch and
cumulative source quality display.
"""

import json
import re
from urllib.parse import urlparse

# Tier constants
PEER_REVIEWED = 1  # Score 5
SCHOLARLY_GOV = 2  # Score 4
ANECDOTAL = 3      # Score 3
BLOG_NEWS = 4      # Score 2
AFFILIATE = 5      # Score 0 (discarded)

TIER_SCORES = {
    PEER_REVIEWED: 5,
    SCHOLARLY_GOV: 4,
    ANECDOTAL: 3,
    BLOG_NEWS: 2,
    AFFILIATE: 0,
}

TIER_NAMES = {
    PEER_REVIEWED: "Peer-Reviewed",
    SCHOLARLY_GOV: "Scholarly/Gov",
    ANECDOTAL: "Anecdotal",
    BLOG_NEWS: "Blog/News",
    AFFILIATE: "Affiliate",
}

# Fast O(1) exact-domain lookup for known academic/research domains
EXACT_DOMAIN_TIERS = {
    "pubmed.ncbi.nlm.nih.gov": PEER_REVIEWED,
    "arxiv.org": PEER_REVIEWED,
    "ieeexplore.ieee.org": PEER_REVIEWED,
    "dl.acm.org": PEER_REVIEWED,
    "sciencedirect.com": PEER_REVIEWED,
    "link.springer.com": PEER_REVIEWED,
    "onlinelibrary.wiley.com": PEER_REVIEWED,
    "journals.plos.org": PEER_REVIEWED,
    "nature.com": PEER_REVIEWED,
    "science.org": PEER_REVIEWED,
    "cell.com": PEER_REVIEWED,
    "thelancet.com": PEER_REVIEWED,
    "bmj.com": PEER_REVIEWED,
    "nejm.org": PEER_REVIEWED,
    "jstor.org": PEER_REVIEWED,
    "cambridge.org": PEER_REVIEWED,
    "academic.oup.com": PEER_REVIEWED,
    "tandfonline.com": PEER_REVIEWED,
    "journals.sagepub.com": PEER_REVIEWED,
    "annualreviews.org": PEER_REVIEWED,
    "frontiersin.org": PEER_REVIEWED,
    "mdpi.com": PEER_REVIEWED,
    "biomedcentral.com": PEER_REVIEWED,
    "peerj.com": PEER_REVIEWED,
    "ncbi.nlm.nih.gov": PEER_REVIEWED,
    "semanticscholar.org": PEER_REVIEWED,
}

# Suffix-based tier mappings (checked via str.endswith)
DOMAIN_SUFFIX_TIERS = {
    ".gov": SCHOLARLY_GOV,
    ".mil": SCHOLARLY_GOV,
    ".int": SCHOLARLY_GOV,
}

# Known institutional/scholarly domains (Tier 2)
INSTITUTIONAL_DOMAINS = [
    "who.int", "un.org", "oecd.org", "worldbank.org",
    "rand.org", "brookings.edu", "nber.org", "imf.org",
    "cbo.gov", "nih.gov", "cdc.gov", "nasa.gov",
    "loc.gov", "archives.gov", "usgs.gov", "noaa.gov",
    "fda.gov", "epa.gov", "nsf.gov", "energy.gov",
    "state.gov", "justice.gov", "census.gov",
]

# Known affiliate / commerce domains — Tier 5 (score 0, discarded)
AFFILIATE_DOMAINS = {
    "amazon.com", "amzn.to", "amzn.eu",
    "shareasale.com", "clickbank.com", "skimlinks.com",
    "skimresources.com", "cj.com", "rakuten.com",
    "ebay.com", "etsy.com", "walmart.com", "target.com",
    "aliexpress.com", "alibaba.com",
}

# URL path patterns indicating affiliate / product content
AFFILIATE_PATH_PATTERNS = re.compile(
    r"/(?:product/|shop/|buy/|pricing/|cart/|checkout/|"
    r"affiliate/|ref=|tag=|redirect|sponsored/|"
    r"advert|promo|coupon|deal/)",
    re.IGNORECASE,
)

# URL path patterns indicating anecdotal / interview content
ANECDOTAL_PATH_PATTERNS = re.compile(
    r"/(?:interview/|ama/|experience/|personal/|story/|"
    r"testimonial|oral[- ]history|firsthand|first[- ]hand|"
    r"how-i-|my-journey|lessons-learned)",
    re.IGNORECASE,
)

# Host substrings indicating anecdotal content
ANECDOTAL_HOSTS = {"substack.com", "medium.com", "quora.com", "reddit.com"}

# Snippet keywords suggesting first-hand practitioner accounts
ANECDOTAL_SNIPPET_KEYWORDS = [
    "interview", "ama", "my experience", "i found", "personally",
    "in my case", "real example", "here's what i", "i tried",
    "i built", "i started", "my story", "what i learned",
    "i've been", "after years of", "first-hand account",
    "from my perspective", "actual numbers", "here's my",
    "i can confirm", "speaking from",
]


def _extract_domain(url: str) -> str:
    """Extract and lowercase the domain host from a URL."""
    parsed = urlparse(url)
    return parsed.netloc.lower() or parsed.path.lower()


def _check_host_substring(domain: str, targets: set) -> bool:
    """Check if any target string appears anywhere in the domain."""
    for target in targets:
        if target in domain:
            return True
    return False


def classify_source(url: str, snippet: str = "", cited_by: int = 0) -> dict:
    """Classify a source URL into a quality tier (1-5).

    Heuristic chain (fastest first):
    1. Exact domain match against known academic publishers
    2. Suffix match (.gov, .mil, .int)
    3. Known institutional domain match
    4. Affiliate/commerce domain check
    5. Affiliate URL path patterns
    6. Anecdotal/interview URL path patterns
    7. .edu domain → default to peer-reviewed (Tier 1)
    8. Substack/Medium/Quora/Reddit → anecdotal (Tier 3)
    9. Snippet keyword check for first-hand accounts
    10. .com/.org/.net → Blog/News (Tier 4)
    11. Everything else → Scholarly/Gov (Tier 2, conservative default)

    Returns:
        dict with 'tier' (int 1-5), 'score' (int 0-5), 'label' (str)
    """
    domain = _extract_domain(url.strip())

    # 1. Exact domain match (O(1) hash lookup)
    if domain in EXACT_DOMAIN_TIERS:
        tier = EXACT_DOMAIN_TIERS[domain]
        return {"tier": tier, "score": TIER_SCORES[tier], "label": TIER_NAMES[tier]}

    # 2. Suffix-based match
    for suffix, tier in DOMAIN_SUFFIX_TIERS.items():
        if domain.endswith(suffix):
            return {"tier": tier, "score": TIER_SCORES[tier], "label": TIER_NAMES[tier]}

    # 3. Known institutional domains
    for inst in INSTITUTIONAL_DOMAINS:
        if domain == inst or domain.endswith("." + inst):
            return {
                "tier": SCHOLARLY_GOV,
                "score": TIER_SCORES[SCHOLARLY_GOV],
                "label": TIER_NAMES[SCHOLARLY_GOV],
            }

    # 4. Affiliate/commerce domains
    for aff in AFFILIATE_DOMAINS:
        if aff in domain:
            return {"tier": AFFILIATE, "score": 0, "label": "Affiliate"}

    # 5. Affiliate URL path patterns
    if AFFILIATE_PATH_PATTERNS.search(url):
        return {"tier": AFFILIATE, "score": 0, "label": "Affiliate"}

    # 6. Anecdotal/interview URL path patterns
    if ANECDOTAL_PATH_PATTERNS.search(url):
        return {
            "tier": ANECDOTAL,
            "score": TIER_SCORES[ANECDOTAL],
            "label": TIER_NAMES[ANECDOTAL],
        }

    # 7. .edu domain → default Peer-Reviewed
    if domain.endswith(".edu"):
        return {
            "tier": PEER_REVIEWED,
            "score": TIER_SCORES[PEER_REVIEWED],
            "label": TIER_NAMES[PEER_REVIEWED],
        }

    # 8. Anecdotal-platform hosts
    if _check_host_substring(domain, ANECDOTAL_HOSTS):
        return {
            "tier": ANECDOTAL,
            "score": TIER_SCORES[ANECDOTAL],
            "label": TIER_NAMES[ANECDOTAL],
        }

    # 9. Check snippet for first-hand/practitioner keywords
    if snippet:
        snippet_lower = snippet.lower()
        for kw in ANECDOTAL_SNIPPET_KEYWORDS:
            if kw in snippet_lower:
                return {
                    "tier": ANECDOTAL,
                    "score": TIER_SCORES[ANECDOTAL],
                    "label": TIER_NAMES[ANECDOTAL],
                }

    # 10. Fallback by TLD
    if domain.endswith((".com", ".org", ".net")):
        return {
            "tier": BLOG_NEWS,
            "score": TIER_SCORES[BLOG_NEWS],
            "label": TIER_NAMES[BLOG_NEWS],
        }

    # 11. Conservative default
    return {
        "tier": SCHOLARLY_GOV,
        "score": TIER_SCORES[SCHOLARLY_GOV],
        "label": TIER_NAMES[SCHOLARLY_GOV],
    }


def score_search_results(organic_results: list) -> list:
    """Score each result in a Serper 'organic' array.

    Adds 'tier', 'score', and 'label' keys to each result dict.
    For scholar results: if 'citedBy' > 0, forces Tier 1 (Peer-Reviewed).
    """
    scored = []
    for result in organic_results:
        url = result.get("link", result.get("pdfUrl", ""))
        snippet = result.get("snippet", "")
        cited_by = int(result.get("citedBy", 0))

        classification = classify_source(url, snippet, cited_by)

        if cited_by > 0:
            classification = {
                "tier": PEER_REVIEWED,
                "score": TIER_SCORES[PEER_REVIEWED],
                "label": TIER_NAMES[PEER_REVIEWED],
            }

        scored.append({**result, **classification})

    return scored


def format_batch_dashboard(scored_results: list) -> str:
    """Format the per-batch source quality table."""
    if not scored_results:
        return ""

    lines = ["\n## Source Quality Dashboard"]
    lines.append("| # | Title | Domain | Tier | Score |")
    lines.append("|---|-------|--------|------|-------|")

    total_score = 0
    for i, r in enumerate(scored_results, 1):
        link = r.get("link", r.get("pdfUrl", ""))
        domain = _extract_domain(link) if link else "(no link)"
        title = (r.get("title", "") or "")[:50]
        score = r.get("score", 0)
        label = r.get("label", "Unknown")
        total_score += score
        lines.append(f"| {i} | {title} | {domain} | {label} | {score} |")

    avg = total_score / len(scored_results) if scored_results else 0
    lines.append(
        f"\nThis batch: {len(scored_results)} sources, avg score {avg:.1f}"
    )

    return "\n".join(lines)


def format_cumulative_dashboard(cum: dict) -> str:
    """Format the cumulative quality summary across all rounds.

    Args:
        cum: Dict with keys: tier1_count, tier2_count, tier3_count,
             tier4_count, tier1_score, tier2_score, tier3_score,
             tier4_score, total_possible.
    """
    tier_counts = {
        1: cum.get("tier1_count", 0),
        2: cum.get("tier2_count", 0),
        3: cum.get("tier3_count", 0),
        4: cum.get("tier4_count", 0),
    }
    tier_scores = {
        1: cum.get("tier1_score", 0),
        2: cum.get("tier2_score", 0),
        3: cum.get("tier3_score", 0),
        4: cum.get("tier4_score", 0),
    }

    total_sources = sum(tier_counts.values())
    total_score = sum(tier_scores.values())
    total_possible = cum.get("total_possible", total_score)
    pct = (total_score / total_possible * 100) if total_possible > 0 else 0

    lines = [
        "\n## Cumulative Quality (all rounds)",
        f"Peer-Reviewed (×5):  {tier_counts[1]}  ({tier_scores[1]} pts)",
        f"Scholarly/Gov (×4):  {tier_counts[2]}  ({tier_scores[2]} pts)",
        f"Anecdotal (×3):     {tier_counts[3]}  ({tier_scores[3]} pts)",
        f"Blog/News (×2):     {tier_counts[4]}  ({tier_scores[4]} pts)",
        f"Discarded (×0):     {0}  (0 pts)",
        "─" * 30,
        f"Weighted total:     {total_score} / {total_possible}  ({pct:.0f}%)",
        f"Sources:            {total_sources}",
    ]
    return "\n".join(lines)


def format_cumulative_json(cum: dict) -> str:
    """Compact JSON string of cumulative scores for model state-passing."""
    return json.dumps(cum, sort_keys=True, separators=(",", ":"))


def merge_cumulative(existing: dict, batch: list) -> dict:
    """Merge batch scores into a running cumulative tally.

    Args:
        existing: Previous cumulative dict (or empty dict for first round).
        batch: List of scored results from score_search_results().

    Returns:
        Updated cumulative dict.
    """
    result = {
        "tier1_count": existing.get("tier1_count", 0),
        "tier2_count": existing.get("tier2_count", 0),
        "tier3_count": existing.get("tier3_count", 0),
        "tier4_count": existing.get("tier4_count", 0),
        "tier1_score": existing.get("tier1_score", 0),
        "tier2_score": existing.get("tier2_score", 0),
        "tier3_score": existing.get("tier3_score", 0),
        "tier4_score": existing.get("tier4_score", 0),
        "total_possible": existing.get("total_possible", 0),
    }

    for item in batch:
        tier = item.get("tier", BLOG_NEWS)
        score = item.get("score", 2)

        if tier == PEER_REVIEWED:
            result["tier1_count"] += 1
            result["tier1_score"] += score
        elif tier == SCHOLARLY_GOV:
            result["tier2_count"] += 1
            result["tier2_score"] += score
        elif tier == ANECDOTAL:
            result["tier3_count"] += 1
            result["tier3_score"] += score
        elif tier == BLOG_NEWS:
            result["tier4_count"] += 1
            result["tier4_score"] += score
        # Tier 5 (AFFILIATE) — skip, adds nothing

        result["total_possible"] += 5  # max possible per source

    return result


def add_dashboard_to_response(
    response_text: str,
    organic_results: list,
    cumulative_scores: dict | None = None,
) -> tuple[str, dict]:
    """Score results, build dashboards, append to response text.

    Returns:
        Tuple of (response_text_with_dashboard, updated_cumulative_scores).
    """
    scored = score_search_results(organic_results)
    new_cum = merge_cumulative(cumulative_scores or {}, scored)

    dashboard = format_batch_dashboard(scored)
    cum_dashboard = format_cumulative_dashboard(new_cum)
    cum_json = format_cumulative_json(new_cum)

    response_text += dashboard
    response_text += cum_dashboard
    response_text += f"\n\nCUMULATIVE_SCORES:{cum_json}"

    return response_text, new_cum
```

- [ ] **Step 2: Verify the module loads**

```bash
cd ~/ai/projects/dev/deep-research
python -c "
from src.source_quality import (
    classify_source, score_search_results, format_batch_dashboard,
    format_cumulative_dashboard, merge_cumulative, add_dashboard_to_response,
    PEER_REVIEWED, SCHOLARLY_GOV, ANECDOTAL, BLOG_NEWS, AFFILIATE,
)

# pubmed -> Tier 1
c1 = classify_source('https://pubmed.ncbi.nlm.nih.gov/12345/')
assert c1['tier'] == 1, f'Expected tier 1, got {c1}'

# reddit -> Tier 3 (anecdotal)
c2 = classify_source('https://www.reddit.com/r/test/comments/abc/')
assert c2['tier'] == 3, f'Expected tier 3, got {c2}'

# product URL -> Tier 5 (affiliate, discarded)
c3 = classify_source('https://example.com/product/buy-now')
assert c3['tier'] == 5, f'Expected tier 5, got {c3}'

# .gov -> Tier 2
c4 = classify_source('https://www.nih.gov/publications')
assert c4['tier'] == 2, f'Expected tier 2, got {c4}'

# who.int -> Tier 2
c5 = classify_source('https://www.who.int/publications')
assert c5['tier'] == 2, f'Expected tier 2, got {c5}'

# blog.com -> Tier 4
c6 = classify_source('https://blog.example.com/article')
assert c6['tier'] == 4, f'Expected tier 4, got {c6}'

# edus default to Tier 1
c7 = classify_source('https://stanford.edu/research/paper')
assert c7['tier'] == 1, f'Expected tier 1 (edu), got {c7}'

# CitedBy > 0 forces Tier 1 even from a blog domain
c8 = classify_source('https://blog.example.com', snippet='study', cited_by=10)
assert c8['tier'] == 1, f'Expected tier 1 (cited_by), got {c8}'

# Dashboard formatting
test_scored = [
    {'title': 'Test Paper', 'link': 'https://arxiv.org/abs/1234', 'snippet': 'test', 'tier': 1, 'score': 5, 'label': 'Peer-Reviewed'},
    {'title': 'My Story', 'link': 'https://substack.com/p/my-story', 'snippet': 'interview', 'tier': 3, 'score': 3, 'label': 'Anecdotal'},
]
print(format_batch_dashboard(test_scored))
print()

cum = merge_cumulative({}, test_scored)
print(format_cumulative_dashboard(cum))
print()

# Full add_dashboard_to_response integration test
original_text = '## Web Results\n1. [Paper](https://arxiv.org/abs/1234)'
result_text, result_cum = add_dashboard_to_response(original_text, [{'title': 'Paper', 'link': 'https://arxiv.org/abs/1234', 'snippet': 'test'}])
assert 'Source Quality Dashboard' in result_text
assert 'CUMULATIVE_SCORES:' in result_text
print('result_cum:', result_cum)
print()

print('All assertions passed.')
"
```

Expected output — all assertions pass, clean dashboard formatting, cumulative JSON emitted.

- [ ] **Step 3: Commit**

```bash
git add src/source_quality.py
git commit -m "feat: add source quality classification and dashboard formatting"
```

---

### Task 2: Modify `src/tools/search.py` — integrate source quality into web search

**Files:**
- Modify: `src/tools/search.py`

- [ ] **Step 1: Add import and `cumulative_scores` to the tool parameters**

Add after the existing imports at the top:

```python
from source_quality import add_dashboard_to_response
```

Then modify the `parameters` class variable on the `Search` class. Find the block:

```python
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }
```

Replace with:

```python
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
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
                    "total_possible": {"type": "integer"}
                },
                "required": [],
                "description": "Cumulative quality scores from previous rounds. Pass the CUMULATIVE_SCORES value from the last search/scholar response verbatim."
            }
        },
        "required": ["query"],
    }
```

- [ ] **Step 2: Modify `google_search_with_serp()` to accept cumulative_scores and append dashboards**

Find the method `google_search_with_serp(self, query: str):` and change its signature to accept `cumulative_scores`:

```python
    def google_search_with_serp(self, query: str, cumulative_scores: dict | None = None):
```

Then find the block inside this method that processes organic results (the `if "organic" in results:` block). Replace it with:

```python
            if "organic" in results:
                web_snippets = list()
                idx = 0
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

                content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)

                # Append source quality dashboard
                content, _ = add_dashboard_to_response(
                    content, results["organic"], cumulative_scores
                )
            else:
                content = f"No results found for '{query}'. Try with a more general query."
```

- [ ] **Step 3: Modify `search_with_serp()` and `call()` to pass cumulative_scores through**

Find `search_with_serp(self, query: str):` — change to accept and forward cumulative_scores:

```python
    def search_with_serp(self, query: str, cumulative_scores: dict | None = None):
        result = self.google_search_with_serp(query, cumulative_scores)
        return result
```

Find the `call(self, params, **kwargs)` method. Replace its body with:

```python
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        cumulative_scores = params.get("cumulative_scores", {}) if isinstance(params, dict) else {}

        if isinstance(query, str):
            response = self.search_with_serp(query, cumulative_scores)
        else:
            assert isinstance(query, List)
            responses = []
            cum = cumulative_scores
            for q in query:
                resp, cum = self.search_with_serp(q, cum)
                responses.append(resp)
            response = "\n=======\n".join(responses)

        return response
```

Wait — `search_with_serp` now returns a tuple `(text, new_cum)` because `add_dashboard_to_response` returns a tuple. But I'm using `return` after `self.search_with_serp()` which would return the tuple as-is. Let me fix:

```python
    def search_with_serp(self, query: str, cumulative_scores: dict | None = None):
        return self.google_search_with_serp(query, cumulative_scores)
```

This is fine — `google_search_with_serp` returns the result of `add_dashboard_to_response` which is a tuple `(str, dict)`.

But the single-query case in `call()` does:
```python
response = self.search_with_serp(query, cumulative_scores)
```
This assigns a tuple to `response`. We need `response, _ = self.search_with_serp(...)`.

Let me fix the `call()` method:

```python
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        cumulative_scores = params.get("cumulative_scores", {}) if isinstance(params, dict) else {}

        if isinstance(query, str):
            response, _ = self.search_with_serp(query, cumulative_scores)
        else:
            assert isinstance(query, List)
            responses = []
            cum = cumulative_scores
            for q in query:
                resp, cum = self.search_with_serp(q, cum)
                responses.append(resp)
            response = "\n=======\n".join(responses)

        return response
```

Now the whole pipeline is consistent.

- [ ] **Step 4: Verify the import works and search.py loads**

```bash
cd ~/ai/projects/dev/deep-research
python -c "
from tools.search import Search
s = Search()
# Just verify it initialises — no API call
print('Search tool loaded with new params:', 'cumulative_scores' in s.parameters['properties'])
print('OK')
"
```

Expected: prints `Search tool loaded with new params: True` and `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/tools/search.py
git commit -m "feat: integrate source quality dashboard into web search results"
```

---

### Task 3: Modify `src/tools/scholar.py` — integrate source quality into scholar search

**Files:**
- Modify: `src/tools/scholar.py`

- [ ] **Step 1: Add import and cumulative_scores to tool parameters**

Add after existing imports:

```python
from source_quality import add_dashboard_to_response
```

Modify the `parameters` class variable on the `Scholar` class. Find:

```python
    parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string", "description": "The search query."},
                    "minItems": 1,
                    "description": "The list of search queries for Google Scholar."
                },
            },
        "required": ["query"],
    }
```

Replace with:

```python
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string", "description": "The search query."},
                "minItems": 1,
                "description": "The list of search queries for Google Scholar."
            },
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
                    "total_possible": {"type": "integer"}
                },
                "required": [],
                "description": "Cumulative quality scores from previous rounds. Pass the CUMULATIVE_SCORES value from the last search/scholar response verbatim."
            }
        },
        "required": ["query"],
    }
```

- [ ] **Step 2: Modify `google_scholar_with_serp()` to accept cumulative_scores and append dashboards**

Change the signature:

```python
    def google_scholar_with_serp(self, query: str, cumulative_scores: dict | None = None):
```

Then find the block that processes organic results (the `if "organic" in results:` block). Replace it with:

```python
            if "organic" in results:
                web_snippets = list()
                idx = 0
                if "organic" in results:
                    for page in results["organic"]:
                        idx += 1
                        date_published = ""
                        if "year" in page:
                            date_published = "\nDate published: " + str(page["year"])

                        publicationInfo = ""
                        if "publicationInfo" in page:
                            publicationInfo = "\npublicationInfo: " + page["publicationInfo"]

                        snippet = ""
                        if "snippet" in page:
                            snippet = "\n" + page["snippet"]

                        link_info = "no available link"
                        if "pdfUrl" in page:
                            link_info = "pdfUrl: " + page["pdfUrl"]

                        citedBy = ""
                        if "citedBy" in page:
                            citedBy = "\ncitedBy: " + str(page["citedBy"])

                        redacted_version = f"{idx}. [{page['title']}]({link_info}){publicationInfo}{date_published}{citedBy}\n{snippet}"

                        redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                        web_snippets.append(redacted_version)

                content = f"A Google scholar for '{query}' found {len(web_snippets)} results:\n\n## Scholar Results\n" + "\n\n".join(web_snippets)

                # Append source quality dashboard
                # Scholar has 'pdfUrl' instead of 'link', and 'citedBy' for citation count
                # Rename pdfUrl → link so score_search_results finds it consistently
                scholar_results = []
                for page in results["organic"]:
                    page_copy = dict(page)
                    if "pdfUrl" in page_copy and "link" not in page_copy:
                        page_copy["link"] = page_copy["pdfUrl"]
                    scholar_results.append(page_copy)

                content, _ = add_dashboard_to_response(
                    content, scholar_results, cumulative_scores
                )
            else:
                content = f"No results found for '{query}'. Try with a more general query."
```

- [ ] **Step 3: Modify `call()` in Scholar to pass cumulative_scores through**

Replace the existing `call()` method with:

```python
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
            query = params["query"]
        except:
            return "[google_scholar] Invalid request format: Input must be a JSON object containing 'query' field"

        cumulative_scores = params.get("cumulative_scores", {}) if isinstance(params, dict) else {}

        if isinstance(query, str):
            response, _ = self.google_scholar_with_serp(query, cumulative_scores)
        else:
            assert isinstance(query, List)
            responses = []
            cum = cumulative_scores
            for q in query:
                resp, cum = self.google_scholar_with_serp(q, cum)
                responses.append(resp)
            response = "\n=======\n".join(responses)
        return response
```

- [ ] **Step 4: Verify Scholar loads**

```bash
cd ~/ai/projects/dev/deep-research
python -c "
from tools.scholar import Scholar
s = Scholar()
print('Scholar tool loaded with new params:', 'cumulative_scores' in s.parameters['properties'])
print('OK')
"
```

Expected: prints `Scholar tool loaded with new params: True` and `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/tools/scholar.py
git commit -m "feat: integrate source quality dashboard into scholar results"
```

---

### Task 4: Modify `src/prompts.py` — add Source Quality Dashboard guidance and update tool definitions

**Files:**
- Modify: `src/prompts.py`

- [ ] **Step 1: Update the search tool definition in SYSTEM_PROMPT**

The `SYSTEM_PROMPT` in `prompts.py` has inline JSON tool definitions embedded in XML `<tools>` tags. Find the search tool definition:

```python
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
```

Replace with:

```python
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries. Results now include a Source Quality Dashboard — pass CUMULATIVE_SCORES from the last response verbatim.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}, "cumulative_scores": {"type": "object", "properties": {"tier1_count": {"type": "integer"}, "tier2_count": {"type": "integer"}, "tier3_count": {"type": "integer"}, "tier4_count": {"type": "integer"}, "tier1_score": {"type": "integer"}, "tier2_score": {"type": "integer"}, "tier3_score": {"type": "integer"}, "tier4_score": {"type": "integer"}, "total_possible": {"type": "integer"}}, "required": [], "description": "Cumulative quality scores from previous rounds. Pass the CUMULATIVE_SCORES value from the last search/scholar response verbatim."}}, "required": ["query"]}}}
```

- [ ] **Step 2: Update the google_scholar tool definition in SYSTEM_PROMPT**

Find:

```python
{"type": "function", "function": {"name": "google_scholar", "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries. This tool will also return results from google search", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries for Google Scholar."}}, "required": ["query"]}}}
```

Replace with:

```python
{"type": "function", "function": {"name": "google_scholar", "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries. This tool will also return results from google search. Results include a Source Quality Dashboard — pass CUMULATIVE_SCORES from the last response verbatim.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries for Google Scholar."}, "cumulative_scores": {"type": "object", "properties": {"tier1_count": {"type": "integer"}, "tier2_count": {"type": "integer"}, "tier3_count": {"type": "integer"}, "tier4_count": {"type": "integer"}, "tier1_score": {"type": "integer"}, "tier2_score": {"type": "integer"}, "tier3_score": {"type": "integer"}, "tier4_score": {"type": "integer"}, "total_possible": {"type": "integer"}}, "required": [], "description": "Cumulative quality scores from previous rounds. Pass the CUMULATIVE_SCORES value from the last search/scholar response verbatim."}}, "required": ["query"]}}}
```

- [ ] **Step 3: Add Source Quality Dashboard guidance at the end of the SYSTEM_PROMPT**

Find the line `Current date: """` at the end of `SYSTEM_PROMPT`. Replace with guidance text followed by `Current date: """`:

The `SYSTEM_PROMPT` string ends with:
```python
Current date: """
```

We need to insert the guidance BEFORE `Current date:`. The last line of the prompt template currently ends with something like:

```
Current date: """
```

We want to add a section between the tool instructions and `Current date:`. Let me find the exact location. The prompt ends with:

```
Current date: """
```

So the string interpolation is `SYSTEM_PROMPT + str(cur_date)`. I need to add the guidance section before `Current date: `.

Looking at `prompts.py` line 35:
```python
Current date: """
```

So the full string is:
```
...
Current date: """
```

Replace `Current date: """` with:

```
## Source Quality Dashboard

Each search and scholar response now includes a **Source Quality Dashboard**
showing the quality tier and score for each result, plus a running cumulative
score across all rounds.

**Tier guide:**
- **Tier 1 (Peer-Reviewed, score 5)** — strongest evidence. Prioritise.
- **Tier 2 (Scholarly/Gov, score 4)** — strong institutional sources.
- **Tier 3 (Anecdotal/Interview, score 3)** — valuable first-hand accounts and practitioner knowledge. Often the most useful for real-world, practical research.
- **Tier 4 (Blog/News, score 2)** — general reporting, context.
- **Tier 5 (Affiliate/Product, score 0)** — discarded, do not cite.

**How to use the cumulative score:**
- If your weighted total is below 40% after several rounds, try more
  targeted searches (scholar, academic domains, practitioner interviews).
- If your weighted total is above 70%, you likely have strong coverage.
- If two consecutive search/scholar tool calls add no new Tier 1-3 sources,
  you may have exhausted the high-quality landscape — proceed to answer.
- **A low score is NOT a failure.** Some topics lack peer-reviewed literature
  or institutional sources. If you've searched thoroughly, synthesise what
  you have. Your report can note the source landscape honestly.

**State passing:**
Each tool response ends with a line like:
  CUMULATIVE_SCORES:{"tier1_count":1,...
Copy this value verbatim and pass it as the "cumulative_scores" parameter in
your next search or google_scholar call. If you're making multiple calls in
one turn, pass the cumulative forward through each call.

Current date: """
```

Wait, I need to be careful about the second-to-last instruction. The system prompt says "If you're making multiple calls in one turn, pass the cumulative forward through each call." — this is good guidance for the model when it's making parallel calls.

Now let me also make sure the `date` line with the triple-quote is handled correctly. The existing code is:

```python
Current date: """
```

So it's `"Current date: "` followed by the closing `"""` of the string. I need to insert before the `"""`.

In the prompts.py file, the last line is:
```
Current date: """
```

I need to replace `Current date: """` with the full section + `Current date: """`.

Actually, I realize I need to be careful about the exact replacement since I need to match the existing text exactly. Let me plan the edit precisely.

The current end of SYSTEM_PROMPT (from the file I read):
```
Current date: """
```

In the file, the `"""` is the closing of the triple-quoted string. So the content before the closing is `Current date: `.

The edit: replace `Current date: """` with `[guidance section]\n\nCurrent date: """`.

Let me write this out precisely.

Actually, I need to re-read prompts.py to make sure I have the exact string for the Edit tool. Let me include the exact line in the plan step.

Let me think about what the edit should be. From the file:

Line 35 of prompts.py: `Current date: """`

I need to insert the guidance section between "Current date: " and the closing """

So: `Current date: """` → `Current date: \n\n[guidance section]\n\n"""` 

Wait no, the existing code already has `Current date: ` inside the string, and it's the last content before the closing `"""`. Let me look at it again:

```python
SYSTEM_PROMPT = """You are a deep research assistant...
...
Current date: """
```

So the string is:
```
"...Current date: "
```

And I want it to be:
```
"...Current date: 

[guidance section]

"
```

So the old string to match is: `Current date: """`
And the new string is: `Current date: \n\n[guidance]\n\n"""`

Actually, looking at the Python string more carefully, the `"""` is just the closing delimiter of the triple-quoted string. The actual system prompt content ends with the text "Current date: ". So when I'm editing, I need to match the text that comes right before the closing `"""`.

Let me just plan the edit as: add the guidance section between the existing tool instructions and `Current date: `.

The actual edit target in the file is:

```python
	
Current date: """
```

This has a blank line, then "Current date: ", then """. I need to replace this with the guidance section.

Let me just write it out clearly in the plan. The implementer can read the file and make the exact edit.

OK, let me finalize the plan now with clear edit instructions.<｜end▁of▁thinking｜>

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="Write">
<｜｜DSML｜｜parameter name="content" string="true"># Source Quality Weighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) for syntax tracking.

**Goal:** Add a source quality scoring and dashboard system to the deep research ReAct agent so the model can assess source quality as it researches.

**Architecture:** A new `src/source_quality.py` module provides classification (domain/URL → tier 1–5) and dashboard formatting functions. The `search.py` and `scholar.py` tools are modified to accept a `cumulative_scores` state blob from the model, score all returned results, append per-batch + cumulative quality dashboards to their response text, and emit a JSON cumulative blob the model can pass forward. The `prompts.py` system prompt gains guidance on how to use the dashboard for research depth decisions.

**Tech Stack:** Python 3, urllib.parse, re, json — no new dependencies.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/source_quality.py` | **Create** | 5-tier classifier, scorer, dashboard formatters, cumulative tally merge |
| `src/tools/search.py` | **Modify** | Import source_quality, accept cumulative_scores param, score results, append dashboards |
| `src/tools/scholar.py` | **Modify** | Same as search.py; scholar results with citedBy > 0 → Tier 1 |
| `src/prompts.py` | **Modify** | Add Source Quality Dashboard guidance + update tool definitions for cumulative_scores param |

No changes to `agent.py` — `custom_call_tool()` already passes all tool args through generically.

---

### Task 1: Create `src/source_quality.py`

**Files:**
- Create: `src/source_quality.py`

- [ ] **Step 1: Write the module**

File: `src/source_quality.py`

```python
"""Source quality classification and scoring for deep research.

Classifies search results into quality tiers based on domain, URL path,
and snippet signals. Provides dashboard formatters for per-batch and
cumulative source quality display.
"""

import json
import re
from urllib.parse import urlparse

# ── Tier constants ────────────────────────────────────────────────
PEER_REVIEWED = 1  # Score 5
SCHOLARLY_GOV = 2  # Score 4
ANECDOTAL = 3      # Score 3
BLOG_NEWS = 4      # Score 2
AFFILIATE = 5      # Score 0 (discarded)

TIER_SCORES = {
    PEER_REVIEWED: 5,
    SCHOLARLY_GOV: 4,
    ANECDOTAL: 3,
    BLOG_NEWS: 2,
    AFFILIATE: 0,
}

TIER_NAMES = {
    PEER_REVIEWED: "Peer-Reviewed",
    SCHOLARLY_GOV: "Scholarly/Gov",
    ANECDOTAL: "Anecdotal",
    BLOG_NEWS: "Blog/News",
    AFFILIATE: "Affiliate",
}

# ── Domain classification maps ────────────────────────────────────

# Tier 1: Known academic publishers / repositories (O(1) lookup)
EXACT_DOMAIN_TIERS = {
    "pubmed.ncbi.nlm.nih.gov": PEER_REVIEWED,
    "arxiv.org": PEER_REVIEWED,
    "ieeexplore.ieee.org": PEER_REVIEWED,
    "dl.acm.org": PEER_REVIEWED,
    "sciencedirect.com": PEER_REVIEWED,
    "link.springer.com": PEER_REVIEWED,
    "onlinelibrary.wiley.com": PEER_REVIEWED,
    "journals.plos.org": PEER_REVIEWED,
    "nature.com": PEER_REVIEWED,
    "science.org": PEER_REVIEWED,
    "cell.com": PEER_REVIEWED,
    "thelancet.com": PEER_REVIEWED,
    "bmj.com": PEER_REVIEWED,
    "nejm.org": PEER_REVIEWED,
    "jstor.org": PEER_REVIEWED,
    "cambridge.org": PEER_REVIEWED,
    "academic.oup.com": PEER_REVIEWED,
    "tandfonline.com": PEER_REVIEWED,
    "journals.sagepub.com": PEER_REVIEWED,
    "annualreviews.org": PEER_REVIEWED,
    "frontiersin.org": PEER_REVIEWED,
    "mdpi.com": PEER_REVIEWED,
    "biomedcentral.com": PEER_REVIEWED,
    "peerj.com": PEER_REVIEWED,
    "ncbi.nlm.nih.gov": PEER_REVIEWED,
    "semanticscholar.org": PEER_REVIEWED,
}

# Suffix-based tier mappings (checked via str.endswith)
DOMAIN_SUFFIX_TIERS = {
    ".gov": SCHOLARLY_GOV,
    ".mil": SCHOLARLY_GOV,
    ".int": SCHOLARLY_GOV,
}

# Tier 2: Known institutional / research organisations
INSTITUTIONAL_DOMAINS = [
    "who.int", "un.org", "oecd.org", "worldbank.org",
    "rand.org", "brookings.edu", "nber.org", "imf.org",
    "cbo.gov", "nih.gov", "cdc.gov", "nasa.gov",
    "loc.gov", "archives.gov", "usgs.gov", "noaa.gov",
    "fda.gov", "epa.gov", "nsf.gov", "energy.gov",
    "state.gov", "justice.gov", "census.gov",
]

# Tier 5: Known affiliate / commerce domains (score 0, always discard)
AFFILIATE_DOMAINS = {
    "amazon.com", "amzn.to", "amzn.eu",
    "shareasale.com", "clickbank.com", "skimlinks.com",
    "skimresources.com", "cj.com", "rakuten.com",
    "ebay.com", "etsy.com", "walmart.com", "target.com",
    "aliexpress.com", "alibaba.com",
}

# URL path patterns: affiliate / product / shop signals → Tier 5
AFFILIATE_PATH_PATTERNS = re.compile(
    r"/(?:product/|shop/|buy/|pricing/|cart/|checkout/|"
    r"affiliate/|ref=|tag=|redirect|sponsored/|"
    r"advert|promo|coupon|deal/)",
    re.IGNORECASE,
)

# URL path patterns: anecdotal / interview signals → Tier 3
ANECDOTAL_PATH_PATTERNS = re.compile(
    r"/(?:interview/|ama/|experience/|personal/|story/|"
    r"testimonial|oral[- ]history|firsthand|first[- ]hand|"
    r"how-i-|my-journey|lessons-learned)",
    re.IGNORECASE,
)

# Host substrings: these domains are anecdotal by default → Tier 3
ANECDOTAL_HOSTS = {"substack.com", "medium.com", "quora.com", "reddit.com"}

# Snippet keywords suggesting first-hand practitioner accounts → Tier 3
ANECDOTAL_SNIPPET_KEYWORDS = [
    "interview", "ama", "my experience", "i found", "personally",
    "in my case", "real example", "here's what i", "i tried",
    "i built", "i started", "my story", "what i learned",
    "i've been", "after years of", "first-hand account",
    "from my perspective", "actual numbers", "here's my",
    "i can confirm", "speaking from",
]


# ── Helper functions ──────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    """Extract and lowercase the domain host from a URL."""
    parsed = urlparse(url)
    return parsed.netloc.lower() or parsed.path.lower()


def _check_host_substring(domain: str, targets: set) -> bool:
    """Check if any target string appears anywhere in the domain."""
    for target in targets:
        if target in domain:
            return True
    return False


# ── Core classification ───────────────────────────────────────────

def classify_source(url: str, snippet: str = "", cited_by: int = 0) -> dict:
    """Classify a source URL into a quality tier (1-5).

    Heuristic chain, fastest checks first:

      1. Exact domain match against known academic publishers
      2. Suffix match (.gov, .mil, .int)
      3. Known institutional domain match
      4. Affiliate / commerce domain check
      5. Affiliate URL path patterns
      6. Anecdotal / interview URL path patterns
      7. .edu domain → default Peer-Reviewed (Tier 1)
      8. Substack / Medium / Quora / Reddit → Anecdotal (Tier 3)
      9. Snippet keyword check for first-hand accounts
     10. .com / .org / .net → Blog/News (Tier 4)
     11. Everything else → Scholarly/Gov (Tier 2, conservative default)

    Args:
        url: The source URL.
        snippet: Optional search-result snippet text.
        cited_by: Optional citation count (from scholar results).

    Returns:
        dict with 'tier' (int 1-5), 'score' (int 0-5), 'label' (str).
    """
    domain = _extract_domain(url.strip())

    # 1. Exact domain match (fastest — O(1) hash lookup)
    if domain in EXACT_DOMAIN_TIERS:
        tier = EXACT_DOMAIN_TIERS[domain]
        return {"tier": tier, "score": TIER_SCORES[tier], "label": TIER_NAMES[tier]}

    # 2. Suffix-based match (.gov, .mil, .int)
    for suffix, tier in DOMAIN_SUFFIX_TIERS.items():
        if domain.endswith(suffix):
            return {"tier": tier, "score": TIER_SCORES[tier], "label": TIER_NAMES[tier]}

    # 3. Known institutional / research domains
    for inst in INSTITUTIONAL_DOMAINS:
        if domain == inst or domain.endswith("." + inst):
            return {
                "tier": SCHOLARLY_GOV,
                "score": TIER_SCORES[SCHOLARLY_GOV],
                "label": TIER_NAMES[SCHOLARLY_GOV],
            }

    # 4. Affiliate / commerce domain check
    for aff in AFFILIATE_DOMAINS:
        if aff in domain:
            return {"tier": AFFILIATE, "score": 0, "label": "Affiliate"}

    # 5. Affiliate URL path patterns
    if AFFILIATE_PATH_PATTERNS.search(url):
        return {"tier": AFFILIATE, "score": 0, "label": "Affiliate"}

    # 6. Anecdotal / interview URL path patterns
    if ANECDOTAL_PATH_PATTERNS.search(url):
        return {
            "tier": ANECDOTAL,
            "score": TIER_SCORES[ANECDOTAL],
            "label": TIER_NAMES[ANECDOTAL],
        }

    # 7. .edu domain → default Peer-Reviewed
    if domain.endswith(".edu"):
        return {
            "tier": PEER_REVIEWED,
            "score": TIER_SCORES[PEER_REVIEWED],
            "label": TIER_NAMES[PEER_REVIEWED],
        }

    # 8. Anecdotal-platform hosts (Substack, Medium, Quora, Reddit)
    if _check_host_substring(domain, ANECDOTAL_HOSTS):
        return {
            "tier": ANECDOTAL,
            "score": TIER_SCORES[ANECDOTAL],
            "label": TIER_NAMES[ANECDOTAL],
        }

    # 9. Check snippet for first-hand / practitioner language
    if snippet:
        snippet_lower = snippet.lower()
        for kw in ANECDOTAL_SNIPPET_KEYWORDS:
            if kw in snippet_lower:
                return {
                    "tier": ANECDOTAL,
                    "score": TIER_SCORES[ANECDOTAL],
                    "label": TIER_NAMES[ANECDOTAL],
                }

    # 10. Fallback by TLD
    if domain.endswith((".com", ".org", ".net")):
        return {
            "tier": BLOG_NEWS,
            "score": TIER_SCORES[BLOG_NEWS],
            "label": TIER_NAMES[BLOG_NEWS],
        }

    # 11. Conservative default for unknown TLDs
    return {
        "tier": SCHOLARLY_GOV,
        "score": TIER_SCORES[SCHOLARLY_GOV],
        "label": TIER_NAMES[SCHOLARLY_GOV],
    }


# ── Scoring ───────────────────────────────────────────────────────

def score_search_results(organic_results: list) -> list:
    """Score each result in a Serper 'organic' array.

    Adds 'tier', 'score', and 'label' keys to each result dict.
    For scholar results with citedBy > 0, forces Tier 1 regardless
    of domain classification.
    """
    scored = []
    for result in organic_results:
        url = result.get("link", result.get("pdfUrl", ""))
        snippet = result.get("snippet", "")
        cited_by = int(result.get("citedBy", 0))

        classification = classify_source(url, snippet, cited_by)

        # Scholar-specific: cited works are always peer-reviewed
        if cited_by > 0:
            classification = {
                "tier": PEER_REVIEWED,
                "score": TIER_SCORES[PEER_REVIEWED],
                "label": TIER_NAMES[PEER_REVIEWED],
            }

        scored.append({**result, **classification})

    return scored


# ── Dashboard formatters ──────────────────────────────────────────

def format_batch_dashboard(scored_results: list) -> str:
    """Format the per-batch source quality table."""
    if not scored_results:
        return ""

    lines = ["\n## Source Quality Dashboard"]
    lines.append("| # | Title | Domain | Tier | Score |")
    lines.append("|---|-------|--------|------|-------|")

    total_score = 0
    for i, r in enumerate(scored_results, 1):
        link = r.get("link", r.get("pdfUrl", ""))
        domain = _extract_domain(link) if link else "(no link)"
        title = (r.get("title", "") or "")[:50]
        score = r.get("score", 0)
        label = r.get("label", "Unknown")
        total_score += score
        lines.append(f"| {i} | {title} | {domain} | {label} | {score} |")

    avg = total_score / len(scored_results) if scored_results else 0
    lines.append(
        f"\nThis batch: {len(scored_results)} sources, avg score {avg:.1f}"
    )

    return "\n".join(lines)


def format_cumulative_dashboard(cum: dict) -> str:
    """Format the cumulative quality summary across all rounds.

    Args:
        cum: Dict with keys: tier{1-4}_count, tier{1-4}_score,
             total_possible.
    """
    tier_counts = {
        1: cum.get("tier1_count", 0),
        2: cum.get("tier2_count", 0),
        3: cum.get("tier3_count", 0),
        4: cum.get("tier4_count", 0),
    }
    tier_scores = {
        1: cum.get("tier1_score", 0),
        2: cum.get("tier2_score", 0),
        3: cum.get("tier3_score", 0),
        4: cum.get("tier4_score", 0),
    }

    total_sources = sum(tier_counts.values())
    total_score = sum(tier_scores.values())
    total_possible = cum.get("total_possible", total_score)
    pct = (total_score / total_possible * 100) if total_possible > 0 else 0

    lines = [
        "\n## Cumulative Quality (all rounds)",
        f"Peer-Reviewed (×5):  {tier_counts[1]}  ({tier_scores[1]} pts)",
        f"Scholarly/Gov (×4):  {tier_counts[2]}  ({tier_scores[2]} pts)",
        f"Anecdotal (×3):     {tier_counts[3]}  ({tier_scores[3]} pts)",
        f"Blog/News (×2):     {tier_counts[4]}  ({tier_scores[4]} pts)",
        f"Discarded (×0):     {0}  (0 pts)",
        "─" * 30,
        f"Weighted total:     {total_score} / {total_possible}  ({pct:.0f}%)",
        f"Sources:            {total_sources}",
    ]
    return "\n".join(lines)


def format_cumulative_json(cum: dict) -> str:
    """Compact JSON string of cumulative scores for model state-passing."""
    return json.dumps(cum, sort_keys=True, separators=(",", ":"))


# ── Cumulative tally ──────────────────────────────────────────────

def merge_cumulative(existing: dict, batch: list) -> dict:
    """Merge batch scores into a running cumulative tally.

    Args:
        existing: Previous cumulative dict (or empty dict for first round).
        batch: List of scored results from score_search_results().

    Returns:
        Updated cumulative dict.
    """
    result = {
        "tier1_count": existing.get("tier1_count", 0),
        "tier2_count": existing.get("tier2_count", 0),
        "tier3_count": existing.get("tier3_count", 0),
        "tier4_count": existing.get("tier4_count", 0),
        "tier1_score": existing.get("tier1_score", 0),
        "tier2_score": existing.get("tier2_score", 0),
        "tier3_score": existing.get("tier3_score", 0),
        "tier4_score": existing.get("tier4_score", 0),
        "total_possible": existing.get("total_possible", 0),
    }

    for item in batch:
        tier = item.get("tier", BLOG_NEWS)
        score = item.get("score", 2)

        if tier == PEER_REVIEWED:
            result["tier1_count"] += 1
            result["tier1_score"] += score
        elif tier == SCHOLARLY_GOV:
            result["tier2_count"] += 1
            result["tier2_score"] += score
        elif tier == ANECDOTAL:
            result["tier3_count"] += 1
            result["tier3_score"] += score
        elif tier == BLOG_NEWS:
            result["tier4_count"] += 1
            result["tier4_score"] += score
        # Tier 5 (AFFILIATE) adds nothing

        result["total_possible"] += 5  # max per source

    return result


# ── Convenience integration ───────────────────────────────────────

def add_dashboard_to_response(
    response_text: str,
    organic_results: list,
    cumulative_scores: dict | None = None,
) -> tuple[str, dict]:
    """Score results, build dashboards, append to response text.

    Returns:
        Tuple of (response_text_with_dashboard, updated_cumulative_scores).
    """
    scored = score_search_results(organic_results)
    new_cum = merge_cumulative(cumulative_scores or {}, scored)

    dashboard = format_batch_dashboard(scored)
    cum_dashboard = format_cumulative_dashboard(new_cum)
    cum_json = format_cumulative_json(new_cum)

    response_text += dashboard
    response_text += cum_dashboard
    response_text += f"\n\nCUMULATIVE_SCORES:{cum_json}"

    return response_text, new_cum
```

- [ ] **Step 2: Verify the module loads without errors**

```bash
cd ~/ai/projects/dev/deep-research
python -c "
from src.source_quality import (
    classify_source, score_search_results, format_batch_dashboard,
    format_cumulative_dashboard, merge_cumulative, add_dashboard_to_response,
    PEER_REVIEWED, SCHOLARLY_GOV, ANECDOTAL, BLOG_NEWS, AFFILIATE,
)

# pubmed -> Tier 1
c1 = classify_source('https://pubmed.ncbi.nlm.nih.gov/12345/')
assert c1['tier'] == 1, f'Expected tier 1, got {c1}'

# reddit -> Tier 3
c2 = classify_source('https://www.reddit.com/r/test/comments/abc/')
assert c2['tier'] == 3, f'Expected tier 3, got {c2}'

# product URL -> Tier 5 (affiliate, discarded)
c3 = classify_source('https://example.com/product/buy-now')
assert c3['tier'] == 5, f'Expected tier 5, got {c3}'

# .gov -> Tier 2
c4 = classify_source('https://www.nih.gov/publications')
assert c4['tier'] == 2, f'Expected tier 2, got {c4}'

# who.int -> Tier 2
c5 = classify_source('https://www.who.int/publications')
assert c5['tier'] == 2, f'Expected tier 2, got {c5}'

# blog.com -> Tier 4
c6 = classify_source('https://blog.example.com/article')
assert c6['tier'] == 4, f'Expected tier 4, got {c6}'

# .edu -> Tier 1
c7 = classify_source('https://stanford.edu/research/paper')
assert c7['tier'] == 1, f'Expected tier 1 (edu), got {c7}'

# citedBy > 0 forces Tier 1
c8 = classify_source('https://blog.example.com', snippet='', cited_by=10)
assert c8['tier'] == 1, f'Expected tier 1 (cited_by), got {c8}'

# Dashboard formatting smoke test
test_scored = [
    {'title': 'Test Paper', 'link': 'https://arxiv.org/abs/1234', 'snippet': 'test', 'tier': 1, 'score': 5, 'label': 'Peer-Reviewed'},
    {'title': 'My Story', 'link': 'https://substack.com/p/my-story', 'snippet': 'interview', 'tier': 3, 'score': 3, 'label': 'Anecdotal'},
]
print(format_batch_dashboard(test_scored))
print()

cum = merge_cumulative({}, test_scored)
print(format_cumulative_dashboard(cum))
print()

# Full integration test
orig = '## Web Results\n1. [Paper](https://arxiv.org/abs/1234)'
result, new_cum = add_dashboard_to_response(orig, [{'title': 'Paper', 'link': 'https://arxiv.org/abs/1234', 'snippet': 'test'}])
assert 'Source Quality Dashboard' in result
assert 'CUMULATIVE_SCORES:' in result
print('Integration OK, cumulative:', new_cum)
print()
print('All assertions passed.')
"
```

Expected output — all assertions pass, clean dashboard formatting, cumulative JSON emitted.

- [ ] **Step 3: Commit**

```bash
git add src/source_quality.py
git commit -m "feat: add source quality classification and dashboard formatting"
```

---

### Task 2: Modify `src/tools/search.py` — integrate source quality into web search

**Files:**
- Modify: `src/tools/search.py`

- [ ] **Step 1: Add import and cumulative_scores to tool parameters**

Add after existing imports at the top of `src/tools/search.py`:

```python
from source_quality import add_dashboard_to_response
```

Then modify the `parameters` class variable on the `Search` class. Find the existing block and add a `cumulative_scores` property:

```python
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
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
                    "total_possible": {"type": "integer"}
                },
                "required": [],
                "description": "Cumulative quality scores from previous rounds. Pass the CUMULATIVE_SCORES value from the last search/scholar response verbatim."
            }
        },
        "required": ["query"],
    }
```

- [ ] **Step 2: Modify `google_search_with_serp()` — accept cumulative_scores, append dashboard**

Change the method signature from:
```python
    def google_search_with_serp(self, query: str):
```
to:
```python
    def google_search_with_serp(self, query: str, cumulative_scores: dict | None = None):
```

Then find the block that processes organic results (the `if "organic" in results:` block). It currently builds `web_snippets` and then constructs `content`. Replace that whole block from `if "organic" in results:` through the `else:` clause so it appends the dashboard:

```python
            if "organic" in results:
                web_snippets = list()
                idx = 0
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

                content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)

                # Append source quality dashboard
                content, _ = add_dashboard_to_response(
                    content, results["organic"], cumulative_scores
                )
            else:
                content = f"No results found for '{query}'. Try with a more general query."
```

- [ ] **Step 3: Modify `search_with_serp()` and `call()` — thread cumulative_scores through**

Change `search_with_serp()` signature and body:

```python
    def search_with_serp(self, query: str, cumulative_scores: dict | None = None):
        return self.google_search_with_serp(query, cumulative_scores)
```

Replace the entire `call()` method body:

```python
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        cumulative_scores = params.get("cumulative_scores", {}) if isinstance(params, dict) else {}

        if isinstance(query, str):
            response, _ = self.search_with_serp(query, cumulative_scores)
        else:
            assert isinstance(query, List)
            responses = []
            cum = cumulative_scores
            for q in query:
                resp, cum = self.search_with_serp(q, cum)
                responses.append(resp)
            response = "\n=======\n".join(responses)

        return response
```

- [ ] **Step 4: Verify the modified Search tool loads**

```bash
cd ~/ai/projects/dev/deep-research
python -c "
from tools.search import Search
s = Search()
assert 'cumulative_scores' in s.parameters['properties'], 'Missing cumulative_scores param'
print('Verified: cumulative_scores parameter present')
print('OK')
"
```

Expected: both assertions pass.

- [ ] **Step 5: Commit**

```bash
git add src/tools/search.py
git commit -m "feat: integrate source quality dashboard into web search results"
```

---

### Task 3: Modify `src/tools/scholar.py` — integrate source quality into scholar search

**Files:**
- Modify: `src/tools/scholar.py`

- [ ] **Step 1: Add import and cumulative_scores to tool parameters**

Add after existing imports:

```python
from source_quality import add_dashboard_to_response
```

Modify the `parameters` class variable on the `Scholar` class. Find the existing block and add `cumulative_scores`:

```python
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string", "description": "The search query."},
                "minItems": 1,
                "description": "The list of search queries for Google Scholar."
            },
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
                    "total_possible": {"type": "integer"}
                },
                "required": [],
                "description": "Cumulative quality scores from previous rounds. Pass the CUMULATIVE_SCORES value from the last search/scholar response verbatim."
            }
        },
        "required": ["query"],
    }
```

- [ ] **Step 2: Modify `google_scholar_with_serp()` — accept cumulative_scores, append dashboard**

Change the method signature from:
```python
    def google_scholar_with_serp(self, query: str):
```
to:
```python
    def google_scholar_with_serp(self, query: str, cumulative_scores: dict | None = None):
```

Then find the block that processes organic results. Replace the entire `if "organic" in results:` block through the `else:` clause:

```python
            if "organic" in results:
                web_snippets = list()
                idx = 0
                if "organic" in results:
                    for page in results["organic"]:
                        idx += 1
                        date_published = ""
                        if "year" in page:
                            date_published = "\nDate published: " + str(page["year"])

                        publicationInfo = ""
                        if "publicationInfo" in page:
                            publicationInfo = "\npublicationInfo: " + page["publicationInfo"]

                        snippet = ""
                        if "snippet" in page:
                            snippet = "\n" + page["snippet"]

                        link_info = "no available link"
                        if "pdfUrl" in page:
                            link_info = "pdfUrl: " + page["pdfUrl"]

                        citedBy = ""
                        if "citedBy" in page:
                            citedBy = "\ncitedBy: " + str(page["citedBy"])

                        redacted_version = f"{idx}. [{page['title']}]({link_info}){publicationInfo}{date_published}{citedBy}\n{snippet}"

                        redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                        web_snippets.append(redacted_version)

                content = f"A Google scholar for '{query}' found {len(web_snippets)} results:\n\n## Scholar Results\n" + "\n\n".join(web_snippets)

                # Scholar uses 'pdfUrl' instead of 'link' — normalise
                scholar_results = []
                for page in results["organic"]:
                    page_copy = dict(page)
                    if "pdfUrl" in page_copy and "link" not in page_copy:
                        page_copy["link"] = page_copy["pdfUrl"]
                    scholar_results.append(page_copy)

                # Append source quality dashboard
                content, _ = add_dashboard_to_response(
                    content, scholar_results, cumulative_scores
                )
            else:
                content = f"No results found for '{query}'. Try with a more general query."
```

- [ ] **Step 3: Modify `call()` in Scholar — thread cumulative_scores through**

Replace the entire `call()` method:

```python
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
            query = params["query"]
        except:
            return "[google_scholar] Invalid request format: Input must be a JSON object containing 'query' field"

        cumulative_scores = params.get("cumulative_scores", {}) if isinstance(params, dict) else {}

        if isinstance(query, str):
            response, _ = self.google_scholar_with_serp(query, cumulative_scores)
        else:
            assert isinstance(query, List)
            responses = []
            cum = cumulative_scores
            for q in query:
                resp, cum = self.google_scholar_with_serp(q, cum)
                responses.append(resp)
            response = "\n=======\n".join(responses)
        return response
```

- [ ] **Step 4: Verify the modified Scholar tool loads**

```bash
cd ~/ai/projects/dev/deep-research
python -c "
from tools.scholar import Scholar
s = Scholar()
assert 'cumulative_scores' in s.parameters['properties'], 'Missing cumulative_scores param'
print('Verified: cumulative_scores parameter present')
print('OK')
"
```

Expected: both assertions pass.

- [ ] **Step 5: Commit**

```bash
git add src/tools/scholar.py
git commit -m "feat: integrate source quality dashboard into scholar results"
```

---

### Task 4: Modify `src/prompts.py` — add Source Quality Dashboard guidance

**Files:**
- Modify: `src/prompts.py`

- [ ] **Step 1: Update the search tool definition in SYSTEM_PROMPT**

In `src/prompts.py`, find the inline JSON tool definition for `search`. It starts with:
```
{"type": "function", "function": {"name": "search", "description": "Perform Google...
```

Replace the entire search tool definition with this updated version that includes the `cumulative_scores` parameter:

```python
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries. Results now include a Source Quality Dashboard — pass CUMULATIVE_SCORES from the last response verbatim.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}, "cumulative_scores": {"type": "object", "properties": {"tier1_count": {"type": "integer"}, "tier2_count": {"type": "integer"}, "tier3_count": {"type": "integer"}, "tier4_count": {"type": "integer"}, "tier1_score": {"type": "integer"}, "tier2_score": {"type": "integer"}, "tier3_score": {"type": "integer"}, "tier4_score": {"type": "integer"}, "total_possible": {"type": "integer"}}, "required": [], "description": "Cumulative quality scores from previous rounds. Pass the CUMULATIVE_SCORES value from the last search/scholar response verbatim."}}, "required": ["query"]}}}
```

- [ ] **Step 2: Update the google_scholar tool definition in SYSTEM_PROMPT**

Find the inline JSON tool definition for `google_scholar`. It starts with:
```
{"type": "function", "function": {"name": "google_scholar", "description": "Leverage Google Scholar...
```

Replace with:

```python
{"type": "function", "function": {"name": "google_scholar", "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries. This tool will also return results from google search. Results include a Source Quality Dashboard — pass CUMULATIVE_SCORES from the last response verbatim.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries for Google Scholar."}, "cumulative_scores": {"type": "object", "properties": {"tier1_count": {"type": "integer"}, "tier2_count": {"type": "integer"}, "tier3_count": {"type": "integer"}, "tier4_count": {"type": "integer"}, "tier1_score": {"type": "integer"}, "tier2_score": {"type": "integer"}, "tier3_score": {"type": "integer"}, "tier4_score": {"type": "integer"}, "total_possible": {"type": "integer"}}, "required": [], "description": "Cumulative quality scores from previous rounds. Pass the CUMULATIVE_SCORES value from the last search/scholar response verbatim."}}, "required": ["query"]}}}
```

- [ ] **Step 3: Add Source Quality Dashboard guidance section to SYSTEM_PROMPT**

In `src/prompts.py`, the `SYSTEM_PROMPT` string ends with:
```python
Current date: """
```

Replace `Current date: """` with the guidance section followed by `Current date: """`:

```python
## Source Quality Dashboard

Each search and scholar response now includes a **Source Quality Dashboard**
showing the quality tier and score for each result, plus a running cumulative
score across all rounds.

**Tier guide:**
- **Tier 1 (Peer-Reviewed, score 5)** — strongest evidence. Prioritise.
- **Tier 2 (Scholarly/Gov, score 4)** — strong institutional sources.
- **Tier 3 (Anecdotal/Interview, score 3)** — valuable first-hand accounts
  and practitioner knowledge. Often the most useful for real-world, practical
  research.
- **Tier 4 (Blog/News, score 2)** — general reporting, context.
- **Tier 5 (Affiliate/Product, score 0)** — discarded, do not cite.

**How to use the cumulative score:**
- If your weighted total is below 40% after several rounds, try more
  targeted searches (scholar, academic domains, practitioner interviews).
- If your weighted total is above 70%, you likely have strong coverage.
- If two consecutive search/scholar tool calls add no new Tier 1-3 sources,
  you may have exhausted the high-quality landscape — proceed to answer.
- **A low score is NOT a failure.** Some topics lack peer-reviewed literature
  or institutional sources. If you've searched thoroughly, synthesise what
  you have. Your report can note the source landscape honestly.

**State passing:**
Each tool response ends with a line like:
  CUMULATIVE_SCORES:{"tier1_count":1,...
Copy this value verbatim and pass it as the "cumulative_scores" parameter in
your next search or google_scholar call. If making multiple calls in one turn,
pass the updated cumulative forward through each call.

Current date: """
```

- [ ] **Step 4: Verify prompts.py loads and the updated SYSTEM_PROMPT renders correctly**

```bash
cd ~/ai/projects/dev/deep-research
python -c "
from prompts import SYSTEM_PROMPT, EXTRACTOR_PROMPT
# Check that the guidance section was added
assert 'Source Quality Dashboard' in SYSTEM_PROMPT, 'Missing Dashboard guidance'
assert 'cumulative_scores' in SYSTEM_PROMPT, 'Missing cumulative_scores param in tool defs'
# Check both tool defs mention CUMULATIVE_SCORES
assert SYSTEM_PROMPT.count('CUMULATIVE_SCORES') >= 2, 'CUMULATIVE_SCORES should appear in 2+ places'
print(f'System prompt length: {len(SYSTEM_PROMPT)} chars')
print('Verified: guidance section and tool definitions updated')
print('OK')
"
```

Expected: all assertions pass, guidance section intact.

- [ ] **Step 5: Commit**

```bash
git add src/prompts.py
git commit -m "feat: add source quality dashboard guidance to system prompt"
```

---

## Integration Verification (Post-All Tasks)

After all four tasks are committed, run a full verification:

```bash
cd ~/ai/projects/dev/deep-research
python -c "
# 1. Module loads
from source_quality import classify_source, add_dashboard_to_response

# 2. Tools load with new params
from tools.search import Search
from tools.scholar import Scholar
s = Search()
sch = Scholar()
assert 'cumulative_scores' in s.parameters['properties']
assert 'cumulative_scores' in sch.parameters['properties']

# 3. Prompts render
from prompts import SYSTEM_PROMPT
assert 'Source Quality Dashboard' in SYSTEM_PROMPT

# 4. End-to-end: simulate a search result pipeline
sample_results = [
    {'title': 'Cancer Study 2025', 'link': 'https://pubmed.ncbi.nlm.nih.gov/12345/', 'snippet': 'A peer-reviewed study on cancer treatment outcomes.'},
    {'title': 'Patient Stories', 'link': 'https://reddit.com/r/cancer/comments/abc/', 'snippet': 'My experience with treatment and what I learned.'},
    {'title': 'Buy Cheap Meds', 'link': 'https://example.com/product/drug', 'snippet': 'Best prices on medication'},
]

text = '## Test Results\n1. [Study](https://pubmed.ncbi.nlm.nih.gov/12345/)'
result, cum = add_dashboard_to_response(text, sample_results)

assert 'Source Quality Dashboard' in result
assert 'CUMULATIVE_SCORES:' in result
assert 'Peer-Reviewed' in result
assert 'Anecdotal' in result

# Verify the cumulative tally
assert cum['tier1_count'] == 1  # pubmed
assert cum['tier3_count'] == 1  # reddit (anecdotal)
# product URL should be affiliate (tier 5) — adds nothing to counts
assert sum([cum['tier{}_count'.format(i)] for i in range(1,5)]) == 2  # only 2 counted

print('All integration checks passed.')
print(f'Cumulative: {cum}')
print(f'Weighted pct: {cum[\"tier1_score\"] + cum[\"tier3_score\"]}/{cum[\"total_possible\"]} = {(cum[\"tier1_score\"] + cum[\"tier3_score\"])/cum[\"total_possible\"]*100:.0f}%')
"
```

Expected: all assertions pass, cumulative shows 1 peer-reviewed + 1 anecdotal, affiliate discarded.

```bash
git add -A
git commit -m "chore: final integration check — all source quality modules load and compose correctly"
```

---

## Rollout Note

After deployment, the first few research runs should be checked for:

1. **Dashboard appears** in search and scholar tool responses
2. **Model passes cumulative forward** — check tool call JSON for `cumulative_scores` field
3. **No over-searching** on low-score topics — the prompt guidance has the diminishing-returns gate
4. **Dashboard not inflating context too much** — ~10 lines per batch × 5–10 rounds = ~50–100 lines max

Any issues should be addressed by tuning `ANECDOTAL_SNIPPET_KEYWORDS`, expanding `EXACT_DOMAIN_TIERS`, or softening the prompt guidance.
