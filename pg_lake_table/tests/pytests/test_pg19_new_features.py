"""Regression tests for PostgreSQL 19 commands/features interacting with pg_lake.

PostgreSQL 19 introduces new statements that target regular tables (e.g.
REPACK).  pg_lake managed iceberg tables (CREATE TABLE ... USING iceberg) and
pg_lake foreign tables are backed by the pg_lake FDW, so these heap-rewrite
commands must either be rejected with a meaningful error or be a harmless
no-op that does not corrupt the table.  These tests pin that behaviour so a
future PostgreSQL or pg_lake change cannot silently regress it.

Gated to PG19+ because the statements do not parse on older servers.
"""

import pytest
from utils_pytest import *

PG19 = 190000


def _skip_if_not_pg19(conn):
    if get_pg_version_num(conn) < PG19:
        pytest.skip("PG19-only commands (REPACK) require PostgreSQL 19+")


def _assert_rejected_or_harmless(conn, relation, repack_sql, expected_count):
    """REPACK must either raise a meaningful error or leave the data intact.

    pg_lake tables are foreign tables under the hood, so PostgreSQL core is
    expected to reject REPACK on them.  We accept a clean error *or* a no-op
    that preserves the row count; we reject silent data loss/corruption and
    crashes.
    """
    error = run_command(repack_sql, conn, raise_error=False)
    conn.rollback()

    if error:
        # Must be a clean, intelligible error -- not an internal crash.
        lowered = error.lower()
        assert any(
            kw in lowered
            for kw in (
                "foreign table",
                "not a table",
                "cannot",
                "not supported",
                "is not",
            )
        ), f"REPACK on {relation} raised an unclear error: {error!r}"
    else:
        # If it "succeeded", the table must still be readable and intact.
        rows = run_query(f"SELECT count(*) AS n FROM {relation}", conn)
        conn.commit()
        assert (
            int(rows[0]["n"]) == expected_count
        ), f"REPACK on {relation} changed the row count unexpectedly"

    # The server must still be alive and the table still queryable.
    rows = run_query(f"SELECT count(*) AS n FROM {relation}", conn)
    conn.commit()
    assert int(rows[0]["n"]) == expected_count


def test_repack_managed_iceberg_table(pg_conn, s3, extension, with_default_location):
    """REPACK on a managed iceberg table must not corrupt it or crash PG."""
    _skip_if_not_pg19(pg_conn)

    run_command("DROP TABLE IF EXISTS pg19_repack_iceberg", pg_conn)
    run_command(
        "CREATE TABLE pg19_repack_iceberg (id int) USING iceberg",
        pg_conn,
    )
    run_command(
        "INSERT INTO pg19_repack_iceberg SELECT generate_series(1, 100)",
        pg_conn,
    )
    pg_conn.commit()

    _assert_rejected_or_harmless(
        pg_conn, "pg19_repack_iceberg", "REPACK pg19_repack_iceberg", 100
    )

    run_command("DROP TABLE IF EXISTS pg19_repack_iceberg", pg_conn)
    pg_conn.commit()


def test_repack_foreign_table(pg_conn, s3, extension):
    """REPACK on a pg_lake foreign table must not corrupt it or crash PG."""
    _skip_if_not_pg19(pg_conn)

    url = f"s3://{TEST_BUCKET}/test_pg19_repack/data.parquet"
    run_command(
        f"COPY (SELECT s AS id FROM generate_series(1, 100) s) TO '{url}'",
        pg_conn,
    )
    run_command("DROP FOREIGN TABLE IF EXISTS pg19_repack_fdw", pg_conn)
    run_command(
        f"""
        CREATE FOREIGN TABLE pg19_repack_fdw (id int)
        SERVER pg_lake OPTIONS (format 'parquet', path '{url}')
        """,
        pg_conn,
    )
    pg_conn.commit()

    _assert_rejected_or_harmless(
        pg_conn, "pg19_repack_fdw", "REPACK pg19_repack_fdw", 100
    )

    run_command("DROP FOREIGN TABLE IF EXISTS pg19_repack_fdw", pg_conn)
    pg_conn.commit()
