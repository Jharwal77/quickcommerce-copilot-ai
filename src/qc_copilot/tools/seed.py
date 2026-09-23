"""Load the generated catalog into the tools database.

Usage:
    python -m qc_copilot.tools.seed
    python -m qc_copilot.tools.seed --sqlite data/darkstore.sqlite
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qc_copilot.config import get_settings
from qc_copilot.tools.repository import (
    MySQLRepository,
    SqliteRepository,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--sqlite",
        type=Path,
        help="Seed a SQLite file instead of the configured MySQL database.",
    )

    args = parser.parse_args()
    settings = get_settings()

    if args.sqlite:
        repo = SqliteRepository(args.sqlite)
        target = str(args.sqlite)

    else:
        repo = MySQLRepository(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            database=settings.mysql_database,
        )
        target = f"mysql://{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"

    try:
        count = repo.seed(settings.catalog_dir)
    finally:
        repo.close()

    print(f"seeded {count} rows into {target}")


if __name__ == "__main__":
    main()
