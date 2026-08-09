"""
Shared configuration constants.

Deliberately dependency-free (stdlib only) so importing it never pulls in
psycopg2, sentence-transformers, or the Databricks SDK. lakebase.py and
embeddings.py both need EMBEDDING_DIM/EMBEDDING_MODEL_NAME to agree with each
other and with schema.sql's vector(768) columns; previously each file
hardcoded its own copy, which is exactly the kind of drift that already bit
this project once (see ensure_schema's docstring).
"""

import os

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

COVER_LETTER_MODEL = os.environ.get("COVER_LETTER_MODEL", "databricks-claude-sonnet-4-5")
