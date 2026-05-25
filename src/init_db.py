"""
Initialize the SQLite database using the project schema.

Usage:
    python init_db.py --db path\to\occlusion.db
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT_DIR / "occlusion.db"
DEFAULT_SCHEMA_PATH = ROOT_DIR / "Aggregated_dataset_db" / "db_schema.sql"


def resolve_schema_path(schema_path: Path) -> Path:
    if schema_path.exists():
        return schema_path
    fallback = ROOT_DIR / "Aggregated_dataset_db" / schema_path.name
    if fallback.exists():
        return fallback
    return schema_path


def init_db(db_path: Path, schema_path: Path) -> None:
    """Create or update the database schema."""
    schema_path = resolve_schema_path(schema_path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        with schema_path.open("r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize SQLite database.")
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=str(DEFAULT_SCHEMA_PATH),
        help="Path to the SQL schema file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_db(Path(args.db), Path(args.schema))
    print(f"SQLite database ready: {args.db}")


if __name__ == "__main__":
    main()
