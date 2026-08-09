"""
classify.py is pure string-in, string-out logic — the single most testable
file in the app, and the one the original audit found the most real bugs in
(unbounded acronym matches, a negative-sponsorship pattern that missed most
real phrasings, and a word-mode ordering bug). Every case below is either a
real phrasing that was previously misclassified, or a guard against
regressing the fix.
"""

import pytest

import classify


@pytest.mark.parametrize("text", [
    "We are unable to provide visa sponsorship at this time.",
    "This position does not offer visa sponsorship.",
    "We do not provide visa sponsorship for this role.",
    "Applicants must be authorized to work in the US without sponsorship.",
    "No visa sponsorship available.",
    "Sponsorship is not available for this position.",
    "We will not sponsor visas for this role.",
    "We can't sponsor a visa for this position.",
    "We are not able to sponsor visas.",
    "Sorry, we cannot sponsor work visas.",
    "No H1B sponsorship.",
    "US citizens only.",
])
def test_classify_sponsorship_negative_phrasings(text):
    assert classify.classify_sponsorship(text) == "no_sponsorship_stated"


@pytest.mark.parametrize("text", [
    "Senior Data Engineer. You will optimize ETL pipelines, lead a team, "
    "and meet deadlines already planned.",
    "This role is optional overtime, adopt new tools, headcount is growing, "
    "reading docs required.",
    "Build dashboards in Tableau. Great benefits.",
    "",
])
def test_classify_sponsorship_benign_words_not_false_positive(text):
    """These previously matched the old unbounded 'opt'/'ead' patterns
    (optimize, optional, adopt, lead, deadline, already, headcount,
    reading) purely as substrings — almost every posting would have been
    misclassified as mentioning sponsorship."""
    assert classify.classify_sponsorship(text) == "not_mentioned"


@pytest.mark.parametrize("text", [
    "We will sponsor H1B for the right candidate.",
    "Candidates on F1 OPT or STEM OPT are welcome to apply.",
    "Green card holders preferred but will sponsor.",
    "Students with valid CPT or EAD are welcome.",
    "H-1B transfer candidates welcome.",
])
def test_classify_sponsorship_positive_phrasings(text):
    assert classify.classify_sponsorship(text) == "mentions_sponsorship"


def test_classify_sponsorship_negative_checked_before_positive():
    """A posting saying it does NOT sponsor still contains the word
    'sponsorship' — the negative check must win."""
    text = "We do not provide visa sponsorship, though we do offer relocation."
    assert classify.classify_sponsorship(text) == "no_sponsorship_stated"


@pytest.mark.parametrize("text,expected", [
    ("This role is on-site in Austin. No remote.", "onsite"),
    ("Candidate must relocate. In-office 5 days.", "onsite"),
    ("Fully remote position.", "remote"),
    ("100% remote, work from home.", "remote"),
    ("Hybrid, 3 days in office.", "hybrid"),
    ("This is not a remote position, on-site only.", "onsite"),
    ("Great benefits and a competitive salary.", "not_mentioned"),
    ("", "not_mentioned"),
])
def test_classify_work_mode(text, expected):
    assert classify.classify_work_mode(text) == expected


def test_classify_posting_splits_sponsorship_and_work_mode_text():
    """Mirrors what the Spark job's column expressions do: sponsorship is
    judged on title + description only; work mode also includes location
    (e.g. a listing literally titled/located 'Remote'). Keeping this in sync
    is what makes the same posting get the same labels regardless of which
    ingestion path (Spark batch vs. live single-fetch) classified it."""
    posting = {"title": "Remote Data Engineer", "description": "Great benefits.", "location": "Remote"}
    result = classify.classify_posting(posting)
    assert result["work_mode_signal"] == "remote"
    assert result["sponsorship_signal"] == "not_mentioned"
    assert result["title"] == "Remote Data Engineer"  # original fields preserved


def test_classify_posting_handles_missing_fields():
    result = classify.classify_posting({})
    assert result["sponsorship_signal"] == "not_mentioned"
    assert result["work_mode_signal"] == "not_mentioned"
