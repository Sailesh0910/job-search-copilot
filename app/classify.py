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
#
# The "do not provide/offer/support ... sponsorship" branch tolerates 0-2
# words in between (e.g. "does not offer visa sponsorship" or "will not
# provide H1B sponsorship") — real postings almost always have a qualifier
# like "visa" in there, so a pattern requiring the words to be adjacent
# misses most real instances of this phrasing.
SPONSORSHIP_NEGATIVE_PATTERN = (
    r"(no (visa )?sponsorship|"
    r"without (visa )?sponsorship|"
    r"(not|cannot|can'?t|won'?t|will not|unable to)\s+(be\s+)?(able to\s+)?sponsor\b|"
    r"(do(es)? not|cannot|can'?t|won'?t|will not|not able to|unable to)\s+"
    r"(currently\s+)?(provide|offer|support)(\s+\w+){0,2}?\s+sponsorship|"
    r"sponsorship\s+(is\s+)?not\s+(currently\s+)?available|"
    r"(must be|candidate must be)\s+(legally\s+)?authorized to work.{0,60}without sponsorship|"
    r"no h-?1b|us citizens? only|citizenship (is )?required|"
    r"security clearance required)"
)

# Short acronyms (OPT, CPT, EAD) are wrapped in \b so they only match as
# standalone words — unbounded, they matched inside ordinary words like
# "optimize", "adopt", "lead", and "deadline", which made almost every
# posting look like it mentioned sponsorship.
SPONSORSHIP_POSITIVE_PATTERN = (
    r"(will sponsor|visa sponsorship (is )?available|sponsorship (is )?(available|provided|offered)|"
    r"h-?1b (sponsorship|transfer|candidates? welcome)|"
    r"\bopt\b|\bcpt\b|\bstem[- ]opt\b|\bead\b|green card|"
    r"we sponsor|open to sponsor|sponsor(ship)? for the right candidate)"
)

# Hybrid checked BEFORE onsite/remote — "hybrid remote" and "remote 2 days a
# week" both contain "remote" but aren't fully-remote roles.
#
# Onsite is checked BEFORE the bare-remote fallback — WORK_MODE_ONSITE_PATTERN
# already contains "no remote", but a posting like "on-site. No remote." was
# still being classified as remote because the plain \bremote\b check ran
# first and matched before the onsite check ever got a chance.
WORK_MODE_HYBRID_PATTERN = r"(hybrid|[0-9] days? (per week )?(in|at) (the )?office|partially remote|flexible work arrangement)"
WORK_MODE_ONSITE_PATTERN = r"(on[- ]?site|in[- ]?office|in[- ]person|no remote|not remote|must relocate)"
WORK_MODE_REMOTE_PATTERN = r"(fully remote|100% remote|work from home|remote[- ]first|telecommute|\bremote\b)"


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
    if re.search(WORK_MODE_ONSITE_PATTERN, text):
        return "onsite"
    if re.search(WORK_MODE_REMOTE_PATTERN, text):
        return "remote"
    return "not_mentioned"


def classify_posting(posting: dict) -> dict:
    """
    Classifies one posting dict in place-ish — returns a new dict with
    sponsorship_signal and work_mode_signal added.

    Sponsorship is judged on title + description only; work mode also
    includes location (e.g. a listing literally titled "Remote"). This
    mirrors exactly what the Spark column expressions in
    notebooks/ingest_jobs_spark.py do, so the same posting gets the same
    labels regardless of which ingestion path classified it.
    """
    title = posting.get("title", "")
    description = posting.get("description", "")
    location = posting.get("location", "")

    sponsorship_text = " ".join([title, description])
    work_mode_text = " ".join([title, description, location])

    return {
        **posting,
        "sponsorship_signal": classify_sponsorship(sponsorship_text),
        "work_mode_signal": classify_work_mode(work_mode_text),
    }