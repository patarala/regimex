"""Join accepted analysis IDs to operator-side queue and child timings."""

import argparse
from pathlib import Path

import psycopg

from regimex.application.load_timings import collect_timings
from regimex.infrastructure.database import DatabaseRepository
from regimex.service_settings import ServiceSettings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, type=Path, help="JSONL of phase and analysis_id")
    parser.add_argument("--worker-log", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    settings = ServiceSettings.from_env()
    repository = DatabaseRepository(
        settings.database_dsn.get_secret_value(), schema=settings.database_schema
    )
    return 0 if collect_timings(repository, args.jobs, args.worker_log, args.output) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except psycopg.Error:
        raise SystemExit(
            "Unable to read queue timing events from the configured database"
        ) from None
