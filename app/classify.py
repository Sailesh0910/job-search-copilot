"""
Sponsorship and work-mode classification, shared between the Spark batch job
(notebooks/ingest_jobs_spark.py, applies these as column expressions across
thousands of rows) and the live single-posting path (app/job_broker.py, when
you fetch fresh postings on demand and can't reasonably start a Spark job for
20 rows).

The regex patterns live here once. Spark wraps them in F.col(...).rlike(...);
the plain functions below use re.search(...) on one string at a time. Same
patterns, two callers — not two versions of the logic to keep in sync.

Adzuna exposes neither signal as a structured field, so this is rule-based
text classification, not verified employer data. Downstream consumers (the
agent, the UI) must present it as a signal to confirm, never a fact.
"""

import re

# Negative check runs BEFORE positive in classify_sponsorship — a posting
# saying "we do not provide H1B sponsorship" contains the word "sponsorship"
# and would be misread as positive if checked second.
SPONSORSHIP_NEGATIVE_PATTERN = (
    r"(no (visa )?sponsorship|not (be )?(able to )?sponsor|"
    r"unable to sponsor|without (visa )?sponsorship|"
    r"do(es)? not (provide|offer) sponsorship|"
    r"must be (legally )?authorized to work.{0,40}without sponsorship|"
    r"no h-?1b|us citizens? only|citizenship (is )?required|"
    r"security clearance required)"
)

SPONSORSHIP_POSITIVE_PATTERN = (
    r"(will sponsor|visa sponsorship (is )?available|sponsorship (is )?(available|provided|offered)|"
    r"h-?1b (sponsorship|transfer|candidates? welcome)|"
    r"opt|cpt|stem opt|ead|green card|"
    r"we sponsor|open to sponsor|sponsor(ship)? for the right candidate)"
)

# Hybrid checked BEFORE remote — "hybrid remote" and "remote 2 days a week"
# both contain "remote" but aren't fully-remote roles.
WORK_MODE_HYBRID_PATTERN = r"(hybrid|[0-9] days? (per week )?(in|at) (the )?office|partially remote|flexible work arrangement)"
WORK_MODE_REMOTE_PATTERN = r"(fully remote|100% remote|work from home|remote[- ]first|telecommute|\bremote\b)"
WORK_MODE_ONSITE_PATTERN = r"(on[- ]?site|in[- ]?office|in[- ]person|no remote|must relocate)"


def classify_sponsorship(text: str) -> str:
    """Returns 'mentions_sponsorship', 'no_sponsorship_stated', or 'not_mentioned'."""
    text = (text or "").lower()
    if re.search(SPONSORSHIP_NEGATIVE_PATTERN, text):
        return "no_sponsorship_stated"
    if re.search(SPONSORSHIP_POSITIVE_PATTERN, text):
        return "mentions_sponsorship"
    return "not_mentioned"


def classify_work_mode(text: str) -> str:
    """Returns 'hybrid', 'remote', 'onsite', or 'not_mentioned'."""
    text = (text or "").lower()
    if re.search(WORK_MODE_HYBRID_PATTERN, text):
        return "hybrid"
    if re.search(WORK_MODE_REMOTE_PATTERN, text):
        return "remote"
    if re.search(WORK_MODE_ONSITE_PATTERN, text):
        return "onsite"
    return "not_mentioned"


def classify_posting(posting: dict) -> dict:
    """
    Classifies one posting dict in place-ish — returns a new dict with
    sponsorship_signal and work_mode_signal added, based on title + description
    + location (location only matters for work mode, e.g. a listing literally
    titled "Remote").
    """
    text = " ".join([
        posting.get("title", ""),
        posting.get("description", ""),
        posting.get("location", ""),
    ])
    return {
        **posting,
        "sponsorship_signal": classify_sponsorship(text),
        "work_mode_signal": classify_work_mode(text),
    }