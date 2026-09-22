"""
Seeded web scraping for the ten PLD languages.

  seeds     — distinctive keywords per language, mined from PLD / halohalo /
              BantayWika text by contrast against the other languages
  search    — pluggable discovery backends; Tavily is the default, FineWeb-2
              and a direct URL list are retained
  fetch     — polite fetching (robots.txt, per-host rate limit) + trafilatura
              main-text extraction
  dedup     — exact (content hash) and near-duplicate (MinHash LSH) filtering
  pipeline  — search → fetch → LID gate → clean → dedup → FineWeb-schema
              parquet shards with a per-URL manifest, resumable
  voice     — YouTube discovery seeded by the same keywords; CC-licensed audio
              download with provenance sidecars

Everything a row carries is the FineWeb-compatible schema halohalo already
uses, plus `lid_score`, `lid_agreement`, `lid_models_agree`, `query` and
`backend` so every document can be audited or excised by source later.
"""

from .search import BACKENDS, DEFAULT_BACKEND, MissingCredential, get_backend

__all__ = ["BACKENDS", "DEFAULT_BACKEND", "MissingCredential", "get_backend"]
