# Databricks notebook source
"""
Delta -> Lakebase loader.

Second step of the batch pipeline. Reads postings from the Delta raw landing
zone and upserts them into Lakebase's operational tables. Embedding is a
separate step (ingest_jobs_embeddings.py).

Why this is separate from the Spark job:
  - Delta stays the source of truth for raw ingested data. If this step fails,
    or the classifiers need re-running, we reprocess from Delta rather than
    re-hitting Adzuna's rate-limited free tier.
  - Spark writes to Delta in bulk; Lakebase wants row-level upserts through
    psycopg2. Different shapes, different failure modes.
  - spark.write.jdbc against Lakebase is explicitly unsupported in this
    environment, so psycopg2 is the only viable write path regardless.

Run in a Databricks notebook, or as a Job task, after ingest_jobs_spark.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

from pyspark.sql import SparkSession, Window, functions as F

import lakebase

DELTA_TABLE = "job_postings_raw"


def read_postings_from_delta(spark, only_new=True):
    """
    Reads postings out of Delta.

    Deduplication: the Spark job appends on every run, and Adzuna re-serves the
    same posting across days, so one id can appear in several batches. The
    window function keeps only the most recent version of each — the standard
    "latest record per key" pattern, done without a self-join.

    only_new: when True, skips postings already in Lakebase, so repeat runs
    don't re-upsert thousands of unchanged rows.
    """
    df = spark.table(DELTA_TABLE)

    window = Window.partitionBy("id").orderBy(F.col("ingested_at").desc())
    df = (
        df.withColumn("_rn", F.row_number().over(window))
          .filter(F.col("_rn") == 1)
          .drop("_rn")
    )

    postings = [row.asDict() for row in df.collect()]

    if only_new:
        existing = lakebase.get_existing_posting_ids()
        postings = [p for p in postings if p["id"] not in existing]

    return postings


def _run(run, only_new=True):
    spark = SparkSession.builder.appName("delta-to-lakebase-load").getOrCreate()

    print("=" * 70)
    print("DELTA -> LAKEBASE LOAD")
    print("=" * 70)

    lakebase.ensure_schema()

    print(f"\nReading from Delta table '{DELTA_TABLE}'...")
    postings = read_postings_from_delta(spark, only_new=only_new)
    print(f"  {len(postings)} postings to load")

    if not postings:
        print("\nNothing new to load.")
        print("Next: run ingest_jobs_embeddings.py")
        return

    written = lakebase.upsert_job_postings(postings)
    run.rows = written

    print("\n" + "=" * 70)
    print(f"Done. Upserted {written} postings into Lakebase.")
    print("Next: run ingest_jobs_embeddings.py")
    print("=" * 70)


def main(**kwargs):
    """Runs the job, recording start/finish/failure in job_copilot.pipeline_runs."""
    with lakebase.track_run("load_jobs_to_lakebase") as run:
        _run(run, **kwargs)


if __name__ == "__main__":
    main()