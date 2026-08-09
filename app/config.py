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

COVER_LETTER_MODEL = os.environ.get("COVER_LETTER_MODEL", "system.ai.llama-4-maverick")

# The Agent Bricks Supervisor Agent's own serving endpoint, queried directly
# by the /chat web UI. It handles its own tool-calling (including MCP calls
# back into this app) internally, so this is just the client side of that
# conversation, unrelated to COVER_LETTER_MODEL above.
SUPERVISOR_AGENT_ENDPOINT = os.environ.get("SUPERVISOR_AGENT_ENDPOINT", "mas-180b0199-endpoint")
