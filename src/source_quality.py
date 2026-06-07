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

# URL path patterns: affiliate / product / shop signals -> Tier 5
AFFILIATE_PATH_PATTERNS = re.compile(
    r"/(?:product/|shop/|buy/|pricing/|cart/|checkout/|"
    r"affiliate/|ref=|tag=|redirect|sponsored/|"
    r"advert|promo|coupon|deal/)",
    re.IGNORECASE,
)

# URL path patterns: anecdotal / interview signals -> Tier 3
ANECDOTAL_PATH_PATTERNS = re.compile(
    r"/(?:interview/|ama/|experience/|personal/|story/|"
    r"testimonial|oral[- ]history|firsthand|first[- ]hand|"
    r"how-i-|my-journey|lessons-learned)",
    re.IGNORECASE,
)

# Host substrings: these domains are anecdotal by default -> Tier 3
ANECDOTAL_HOSTS = {"substack.com", "medium.com", "quora.com", "reddit.com"}

# Snippet keywords suggesting first-hand practitioner accounts -> Tier 3
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

      0. cited_by > 0 -> force Peer-Reviewed (scholar override)
      1. Exact domain match against known academic publishers
      2. Suffix match (.gov, .mil, .int)
      3. Known institutional domain match
      4. Affiliate / commerce domain check
      5. Affiliate URL path patterns
      6. Anecdotal / interview URL path patterns
      7. .edu domain -> default Peer-Reviewed (Tier 1)
      8. Substack / Medium / Quora / Reddit -> Anecdotal (Tier 3)
      9. Snippet keyword check for first-hand accounts
     10. .com / .org / .net -> Blog/News (Tier 4)
     11. Everything else -> Scholarly/Gov (Tier 2, conservative default)

    Args:
        url: The source URL.
        snippet: Optional search-result snippet text.
        cited_by: Optional citation count (from scholar results).

    Returns:
        dict with 'tier' (int 1-5), 'score' (int 0-5), 'label' (str).
    """
    domain = _extract_domain(url.strip())

    # 0. cited_by > 0 forces Peer-Reviewed (scholar override, fastest check)
    if cited_by > 0:
        return {"tier": PEER_REVIEWED, "score": TIER_SCORES[PEER_REVIEWED], "label": TIER_NAMES[PEER_REVIEWED]}

    # 1. Exact domain match (fastest -- O(1) hash lookup)
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

    # 7. .edu domain -> default Peer-Reviewed
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
    citedBy > 0 forces Tier 1 via classify_source's pre-check.
    """
    scored = []
    for result in organic_results:
        url = result.get("link", result.get("pdfUrl", ""))
        snippet = result.get("snippet", "")
        cited_by = int(result.get("citedBy", 0))

        classification = classify_source(url, snippet, cited_by)

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
        f"Peer-Reviewed (x5):  {tier_counts[1]}  ({tier_scores[1]} pts)",
        f"Scholarly/Gov (x4):  {tier_counts[2]}  ({tier_scores[2]} pts)",
        f"Anecdotal (x3):     {tier_counts[3]}  ({tier_scores[3]} pts)",
        f"Blog/News (x2):     {tier_counts[4]}  ({tier_scores[4]} pts)",
        f"Discarded (x0):     {0}  (0 pts)",
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
