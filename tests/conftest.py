"""
Makes app/ importable as a flat module namespace (import classify, import
lakebase, from config import ...), matching how the app's own files import
each other and how notebooks/ already adds app/ to sys.path.

Nothing in this test suite touches a real Postgres database, makes a real
HTTP call to Adzuna, or downloads the real sentence-transformers model —
everything external is faked at the seams (see individual test files), so
the suite runs the same with no credentials configured at all.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
