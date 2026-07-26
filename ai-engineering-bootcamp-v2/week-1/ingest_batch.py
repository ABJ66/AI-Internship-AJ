#!/usr/bin/env python3
"""
Batch-ingest text documents via POST /ingest.

Looks for Northwind sample docs under ./northwind/ or ./sample_docs/northwind/.
If none are found, ingests insertable_cardiac_monitor.txt as a fallback.

Usage (server must be running on port 8000):
    python ingest_batch.py

Optional:
    INGEST_BASE_URL=http://127.0.0.1:8000 python ingest_batch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

WORKDIR = Path(__file__).resolve().parent
BASE_URL = __import__("os").getenv("INGEST_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

NORTHWIND_DIRS = [
    WORKDIR / "northwind",
    WORKDIR / "sample_docs" / "northwind",
    WORKDIR / "docs" / "northwind",
]

FALLBACK_FILES = [
    WORKDIR / "insertable_cardiac_monitor.txt",
]


def discover_files() -> list[Path]:
    """Find Northwind .txt/.md files, or fall back to the cardiac monitor doc."""

    for directory in NORTHWIND_DIRS:
        if directory.is_dir():
            files = sorted(
                p
                for p in directory.iterdir()
                if p.is_file() and p.suffix.lower() in {".txt", ".md"}
            )
            if files:
                print(f"Found {len(files)} Northwind file(s) in {directory}")
                return files

    print("No Northwind sample docs found — using fallback document(s).")
    return [p for p in FALLBACK_FILES if p.is_file()]


def stable_document_id(path: Path) -> str:
    """Stable ID from filename stem (no extension)."""

    return path.stem.lower().replace(" ", "_")


def post_ingest(client: httpx.Client, path: Path) -> int:
    """Read a file and POST it to /ingest. Returns chunks_indexed."""

    text = path.read_text(encoding="utf-8")
    document_id = stable_document_id(path)
    payload = {
        "document_id": document_id,
        "text": text,
        "source": path.name,
    }

    response = client.post(f"{BASE_URL}/ingest", json=payload, timeout=300.0)
    response.raise_for_status()
    data = response.json()
    return int(data["chunks_indexed"])


def fetch_total_vectors(client: httpx.Client) -> int | None:
    """Read total_vector_count from GET /debug/pinecone."""

    response = client.get(f"{BASE_URL}/debug/pinecone", timeout=30.0)
    response.raise_for_status()
    stats = response.json().get("index_stats") or {}
    total = stats.get("total_vector_count")
    return int(total) if total is not None else None


def main() -> int:
    files = discover_files()
    if not files:
        print("ERROR: No documents to ingest.", file=sys.stderr)
        return 1

    print(f"Ingest API: {BASE_URL}/ingest\n")

    batch_total = 0
    with httpx.Client() as client:
        for path in files:
            document_id = stable_document_id(path)
            try:
                chunks = post_ingest(client, path)
                batch_total += chunks
                print(f"  {path.name:40}  document_id={document_id:30}  chunks={chunks}")
            except httpx.HTTPStatusError as exc:
                print(f"  FAILED {path.name}: HTTP {exc.response.status_code} — {exc.response.text}")
                return 1
            except httpx.HTTPError as exc:
                print(f"  FAILED {path.name}: {exc}")
                print(f"\nIs the server running?  uvicorn main:app --reload")
                return 1

        print(f"\nBatch total chunks indexed this run: {batch_total}")

        try:
            store_total = fetch_total_vectors(client)
            if store_total is not None:
                print(f"Total vectors in Pinecone index:   {store_total}")
            else:
                print("Could not read total_vector_count from /debug/pinecone")
        except httpx.HTTPError as exc:
            print(f"Could not fetch /debug/pinecone: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
