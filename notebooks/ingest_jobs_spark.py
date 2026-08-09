"""
Spark ingestion: Adzuna API -> Delta table (job_postings_raw)

This is the raw landing zone. Postings are fetched, normalized, classified,
and appended to Delta. Nothing here touches Lakebase — that's a separate
step, so a database problem can't cost us the API fetch, and so we can
re-run classification or embedding later without re-hitting Adzuna's
rate-limited free tier.

Why Spark for a few hundred rows: at this volume a plain Python script would
be just as fast. The pattern matters more than the current data size — the
same code scales to millions of postings without changing, because the
classification runs as distributed column expressions rather than a Python
loop over rows.

Run in a Databricks notebook, or as a Job task pointed at this file.
"""

import os
import sys

# lakebase.py, adzuna_client.py, embeddings.py live in ../app relative to this
# notebook's location, so add that to the import path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, TimestampType
)

from adzuna_client import AdzunaClient

import classify
import lakebase

DELTA_TABLE = "job_postings_raw"

DEFAULT_ROLES = "data engineer, backend engineer, software engineer"

RESULTS_PER_PAGE = 50
MAX_PAGES = 2
MAX_DAYS_OLD = 30


def get_search_params():
    """
    Reads roles/location from Databricks notebook widgets, so a run is
    configured by typing into a box at the top of the notebook (or passing
    Job parameters), not by editing this file.

    Falls back to defaults when dbutils isn't available, e.g. running this
    file directly for a quick local syntax/logic check outside Databricks.
    """
    try:
        dbutils.widgets.text("roles", DEFAULT_ROLES, "Roles to search (comma-separated)")
        dbutils.widgets.text("location", "", "Location filter (optional, blank = nationwide)")
        roles_raw = dbutils.widgets.get("roles")
        location_raw = dbutils.widgets.get("location").strip()
    except NameError:
        roles_raw = DEFAULT_ROLES
        location_raw = ""

    roles = [r.strip() for r in roles_raw.split(",") if r.strip()]
    where = location_raw or None
    return [{"what": role, "where": where} for role in roles]


# Explicit schema rather than letting Spark infer it. Inference reads the data
# twice and can guess wrong when a column is all-null in one batch — an explicit
# schema means the Delta table's shape never shifts between runs.
POSTING_SCHEMA = StructType([
    StructField("id", StringType(), False),
    StructField("title", StringType(), True),
    StructField("company", StringType(), True),
    StructField("location", StringType(), True),
    StructField("description", StringType(), True),
    StructField("salary_min", IntegerType(), True),
    StructField("salary_max", IntegerType(), True),
    StructField("url", StringType(), True),
    StructField("posted_at", TimestampType(), True),
    StructField("synced_at", TimestampType(), True),
])


def fetch_postings():
    """Pulls postings from Adzuna across all configured searches."""
    client = AdzunaClient(
        app_id=os.environ["ADZUNA_APP_ID"],
        app_key=os.environ["ADZUNA_APP_KEY"],
    )

    searches = get_search_params()
    print(f"  Roles: {[s['what'] for s in searches]}")

    all_postings = []
    for search in searches:
        postings = client.search(
            what=search["what"],
            where=search.get("where"),
            results_per_page=RESULTS_PER_PAGE,
            max_pages=MAX_PAGES,
            sort_by="date",
            max_days_old=MAX_DAYS_OLD,
        )
        print(f"  '{search['what']}': {len(postings)} postings")
        all_postings.extend(postings)

    return all_postings


def classify_sponsorship(df):
    """
    Derives a sponsorship signal from posting text.

    Adzuna exposes no structured sponsorship field — almost no job API does —
    so this is rule-based classification over the description.

    Three-way, deliberately:
      - no_sponsorship_stated : the posting explicitly rules it out
      - mentions_sponsorship  : the posting explicitly discusses sponsoring
      - not_mentioned         : silence, which is the majority case

    Patterns come from classify.py, shared with the live single-posting path
    in job_broker.py — same regex, applied here as Spark column expressions
    instead of Python's re.search, not a second copy of the logic.

    The negative check runs FIRST because a posting saying "we do not provide
    H1B sponsorship" contains the word "sponsorship" and would otherwise be
    misread as positive. Order matters here.

    This is a heuristic over free text, not a verified employer statement.
    Downstream, the agent must present it as a signal to confirm, never a fact.
    """
    text = F.lower(F.concat_ws(" ", F.col("title"), F.col("description")))

    return df.withColumn(
        "sponsorship_signal",
        F.when(text.rlike(classify.SPONSORSHIP_NEGATIVE_PATTERN), F.lit("no_sponsorship_stated"))
         .when(text.rlike(classify.SPONSORSHIP_POSITIVE_PATTERN), F.lit("mentions_sponsorship"))
         .otherwise(F.lit("not_mentioned"))
    )


def classify_work_mode(df):
    """
    Derives remote/hybrid/onsite from posting text.

    Same situation as sponsorship: Adzuna has no structured field for it, and
    patterns come from classify.py rather than being duplicated here.

    Hybrid is checked BEFORE remote, because "hybrid remote" and
    "remote 2 days a week" both contain "remote" but are not fully remote
    roles. Checking remote first would swallow every hybrid posting.
    """
    text = F.lower(F.concat_ws(" ", F.col("title"), F.col("description"), F.col("location")))

    return df.withColumn(
        "work_mode_signal",
        F.when(text.rlike(classify.WORK_MODE_HYBRID_PATTERN), F.lit("hybrid"))
         .when(text.rlike(classify.WORK_MODE_REMOTE_PATTERN), F.lit("remote"))
         .when(text.rlike(classify.WORK_MODE_ONSITE_PATTERN), F.lit("onsite"))
         .otherwise(F.lit("not_mentioned"))
    )


def clean_postings(df):
    """Normalization that has to happen before anything downstream trusts the data."""
    return (
        df
        # Adzuna descriptions contain HTML entities and tags from the original listing.
        .withColumn("description", F.regexp_replace(F.col("description"), r"<[^>]+>", " "))
        .withColumn("description", F.regexp_replace(F.col("description"), r"&[a-z]+;", " "))
        .withColumn("description", F.trim(F.regexp_replace(F.col("description"), r"\s+", " ")))
        # A posting with no description is useless to us — it's what gets embedded.
        .filter(F.col("description").isNotNull() & (F.length(F.col("description")) > 0))
        # The same posting can match several of our searches; keep one row per id.
        .dropDuplicates(["id"])
    )


def _run(run):
    spark = SparkSession.builder.appName("adzuna-job-ingest").getOrCreate()

    print("=" * 70)
    print("ADZUNA -> DELTA INGESTION")
    print("=" * 70)

    print("\nFetching from Adzuna...")
    raw_postings = fetch_postings()
    print(f"  Total fetched: {len(raw_postings)}")

    if not raw_postings:
        print("\nNothing fetched. Stopping.")
        return

    df = spark.createDataFrame(raw_postings, schema=POSTING_SCHEMA)

    print("\nCleaning and classifying...")
    # Each of these returns a new DataFrame rather than mutating — Spark
    # transformations are lazy, so nothing has actually run yet at this point.
    # The whole chain executes once, optimized as a single plan, on write below.
    df = clean_postings(df)
    df = classify_sponsorship(df)
    df = classify_work_mode(df)

    df = df.withColumn("ingested_at", F.current_timestamp())

    # This .write is the ACTION that triggers everything above to actually run.
    print(f"\nWriting to Delta table '{DELTA_TABLE}'...")
    (
        df.write
          .format("delta")
          .mode("append")
          .option("mergeSchema", "true")
          .saveAsTable(DELTA_TABLE)
    )

    run.rows = df.count()

    total = spark.table(DELTA_TABLE).count()
    print(f"  Wrote {df.count()} rows. Table now holds {total} rows total.")

    print("\nClassification breakdown for this batch:")
    df.groupBy("sponsorship_signal").count().show(truncate=False)
    df.groupBy("work_mode_signal").count().show(truncate=False)

    print("=" * 70)
    print("Done. Next: run load_jobs_to_lakebase.py")
    print("=" * 70)


def main(**kwargs):
    """Runs the job, recording start/finish/failure in job_copilot.pipeline_runs."""
    with lakebase.track_run("ingest_jobs_spark") as run:
        _run(run, **kwargs)


if __name__ == "__main__":
    main()