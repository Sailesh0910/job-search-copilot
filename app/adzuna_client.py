"""
Adzuna Job Search API Client

Fetches job postings from Adzuna and normalizes them into a document shape
matching the job_postings table schema.

Adzuna auth: two values, app_id and app_key, passed as query parameters on
every request (not an Authorization header, that's Adzuna's own design,
not a workaround).

Adzuna pagination: results come back in pages (results_per_page, capped at
a reasonable size by Adzuna itself), with the page number embedded directly
in the URL path, e.g. /search/1, /search/2.
"""

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class AdzunaClient:
    """Client for fetching and normalizing job postings from the Adzuna API."""

    BASE_URL = "https://api.adzuna.com/v1/api/jobs"

    def __init__(self, app_id: str, app_key: str, country: str = "us"):
        """
        Args:
            app_id: Your Adzuna application ID.
            app_key: Your Adzuna application key.
            country: Two-letter country code Adzuna expects in the URL path
                     (e.g. "us", "gb"). Defaults to "us".
        """
        self.app_id = app_id
        self.app_key = app_key
        self.country = country
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # HTTP helper — same retry/backoff shape as weather_client.py, for the
    # same reason: a single dropped connection shouldn't kill the whole sync.
    # ------------------------------------------------------------------

    def _make_request(self, page: int, params: Dict, max_retries: int = 3) -> Dict:
        url = f"{self.BASE_URL}/{self.country}/search/{page}"
        request_params = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "content-type": "application/json",
            **params,
        }

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=request_params, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    # ------------------------------------------------------------------
    # Normalize a single raw Adzuna result into our job_postings shape.
    # Every field access uses .get() with a fallback, same defensive habit
    # as weather_client.py, since we don't control Adzuna's response and a
    # missing field shouldn't crash the whole sync over one bad posting.
    # ------------------------------------------------------------------

    def _normalize_posting(self, raw: Dict) -> Dict:
        title = raw.get("title", "")
        company = raw.get("company", {}).get("display_name", "")
        location = raw.get("location", {}).get("display_name", "")

        posting_id = raw.get("id")
        if not posting_id:
            # Fallback dedup key if Adzuna ever omits an id: hash the title
            # + company + location, so the same posting won't be inserted twice.
            fallback = f"{title}-{company}-{location}"
            posting_id = f"ADZUNA-{hashlib.md5(fallback.encode()).hexdigest()}"

        created = raw.get("created")
        posted_at = (
            datetime.fromisoformat(created.replace("Z", "+00:00"))
            if created
            else datetime.now(timezone.utc)
        )

        return {
            "id": str(posting_id),
            "title": title or "Untitled posting",
            "company": company or "Unknown",
            "location": location or "Unknown",
            "description": raw.get("description", ""),
            # Adzuna returns salary as a float (e.g. 90000.0); the Spark
            # ingestion schema declares these columns IntegerType, so a raw
            # float here would fail createDataFrame's type check.
            "salary_min": self._to_int(raw.get("salary_min")),
            "salary_max": self._to_int(raw.get("salary_max")),
            "url": raw.get("redirect_url"),
            "posted_at": posted_at,
            "synced_at": datetime.now(timezone.utc),
        }

    @staticmethod
    def _to_int(value) -> Optional[int]:
        """Coerces Adzuna's salary float (or None) to an int, rounding rather
        than truncating so a value like 89999.6 doesn't undercount by a
        dollar."""
        return round(value) if value is not None else None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def search(
        self,
        what: str,
        where: Optional[str] = None,
        results_per_page: int = 20,
        max_pages: int = 1,
        sort_by: str = "date",
        max_days_old: Optional[int] = None,
        salary_min: Optional[int] = None,
        **extra_params,
    ) -> List[Dict]:
        """
        Searches Adzuna for postings and returns normalized job_postings dicts.

        Args:
            what: Keywords to search for (e.g. "backend engineer").
            where: Location filter (e.g. "Raleigh"). Optional, omit for
                   nationwide results.
            results_per_page: Postings per page, Adzuna caps this reasonably.
            max_pages: How many pages to fetch, so results_per_page * max_pages
                       is roughly your total posting count for this search.
            sort_by: Adzuna's own sort — 'date', 'relevance', or 'salary'.
                     Defaults to 'date' so a sync pulls the freshest postings
                     first, which matters when you're only fetching a page or two
                     out of thousands of matches.
            max_days_old: Only postings newer than this many days. Useful for
                          incremental syncs that shouldn't re-pull stale listings.
            salary_min: Server-side salary floor. Filtering here rather than
                        after the fetch means you don't waste your free-tier
                        quota on postings you'd discard anyway.
            **extra_params: Anything else Adzuna accepts, e.g. full_time=1.

        Note: Adzuna has no sponsorship or remote-work filter. Those signals are
        derived heuristically from posting text during the Spark ingest step.
        """
        postings: List[Dict] = []

        params = {
            "what": what,
            "results_per_page": results_per_page,
            "sort_by": sort_by,
            **extra_params,
        }
        if where:
            params["where"] = where
        if max_days_old is not None:
            params["max_days_old"] = max_days_old
        if salary_min is not None:
            params["salary_min"] = salary_min

        for page in range(1, max_pages + 1):
            try:
                data = self._make_request(page, params)
            except requests.exceptions.RequestException as e:
                logger.warning("Adzuna request failed on page %d: %s", page, e)
                break

            results = data.get("results", [])
            if not results:
                # No more results, stop paging early instead of requesting
                # empty pages Adzuna will never fill.
                break

            postings.extend(self._normalize_posting(r) for r in results)

        return postings


if __name__ == "__main__":
    import os

    client = AdzunaClient(
        app_id=os.environ["ADZUNA_APP_ID"],
        app_key=os.environ["ADZUNA_APP_KEY"],
    )
    jobs = client.search(what="backend engineer", where="Raleigh", max_pages=1)
    print(f"Fetched {len(jobs)} postings")
    if jobs:
        print(f"First: {jobs[0]['title']} at {jobs[0]['company']}")