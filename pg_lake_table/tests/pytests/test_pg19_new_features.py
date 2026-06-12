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


def test_group_by_all_managed_iceberg(pg_conn, s3, extension, with_default_location):
    """PG19 GROUP BY ALL must work transparently over a managed iceberg table."""
    _skip_if_not_pg19(pg_conn)

    run_command("DROP TABLE IF EXISTS pg19_group_by_all", pg_conn)
    run_command(
        "CREATE TABLE pg19_group_by_all (g int, v int) USING iceberg",
        pg_conn,
    )
    # 3 groups (g = 0,1,2), 10 rows each.
    run_command(
        """
        INSERT INTO pg19_group_by_all
        SELECT s % 3, s FROM generate_series(1, 30) s
        """,
        pg_conn,
    )
    pg_conn.commit()

    rows = run_query(
        "SELECT g, count(*) AS n FROM pg19_group_by_all GROUP BY ALL ORDER BY g",
        pg_conn,
    )
    pg_conn.commit()

    assert [(int(r["g"]), int(r["n"])) for r in rows] == [(0, 10), (1, 10), (2, 10)]

    run_command("DROP TABLE IF EXISTS pg19_group_by_all", pg_conn)
    pg_conn.commit()


def test_on_conflict_do_select_managed_iceberg(
    pg_conn, s3, extension, with_default_location
):
    """PG19 INSERT ... ON CONFLICT DO SELECT must error meaningfully (no unique
    index support on managed iceberg tables) rather than crash."""
    _skip_if_not_pg19(pg_conn)

    run_command("DROP TABLE IF EXISTS pg19_on_conflict", pg_conn)
    run_command(
        "CREATE TABLE pg19_on_conflict (id int, v text) USING iceberg",
        pg_conn,
    )
    run_command("INSERT INTO pg19_on_conflict VALUES (1, 'a')", pg_conn)
    pg_conn.commit()

    # DO SELECT requires a RETURNING clause; include it so we actually reach
    # the conflict-arbiter resolution against the managed iceberg table.
    error = run_command(
        """
        INSERT INTO pg19_on_conflict VALUES (1, 'b')
        ON CONFLICT (id) DO SELECT RETURNING *
        """,
        pg_conn,
        raise_error=False,
    )
    pg_conn.rollback()

    # Managed iceberg tables have no unique index / arbiter, so this must be
    # rejected with a clear error, not an internal failure.
    assert error, "Expected ON CONFLICT DO SELECT to be rejected"
    lowered = error.lower()
    assert any(
        kw in lowered
        for kw in (
            "no unique",
            "constraint",
            "on conflict",
            "not supported",
            "cannot",
            "unique or exclusion",
            "arbiter",
        )
    ), f"ON CONFLICT DO SELECT raised an unclear error: {error!r}"

    # Server still alive and table intact.
    rows = run_query("SELECT count(*) AS n FROM pg19_on_conflict", pg_conn)
    pg_conn.commit()
    assert int(rows[0]["n"]) == 1

    run_command("DROP TABLE IF EXISTS pg19_on_conflict", pg_conn)
    pg_conn.commit()
