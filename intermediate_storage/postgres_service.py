import os
import json
import hashlib
import random
from datetime import datetime
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, request, jsonify
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


class TrainingStore:
    """
    PostgreSQL-backed storage for training-session data.

    Design goals:
    - Support multiple processed outputs for the same table subset.
    - Support parallel API requests better than SQLite.
    - Keep reset/delete operations simple after a training session.
    - Keep the code readable and easy to extend.

    Data model:
    - db_registry: one row per db_id, stores the full table list.
    - split_group: one row per unique table subset for that db_id.
    - processed_result: many rows per split_group, one per output generated.

    Important note about temporary data:
    - The tables below are created as UNLOGGED tables.
    - UNLOGGED tables are faster to write, but they are not crash-safe.
    - This is usually acceptable for temporary training-session data.
    """

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 10):
        # Connection pool avoids opening a new PostgreSQL connection for every request.
        self.pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            open=True,
        )
        # self._delete_all_tables()
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        """
        Create tables and indexes if they do not exist.

        UNLOGGED is used because training data is temporary and we want faster writes.
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE UNLOGGED TABLE IF NOT EXISTS db_registry (
                        db_id TEXT PRIMARY KEY,
                        all_tables JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)

                cur.execute("""
                    CREATE UNLOGGED TABLE IF NOT EXISTS split_group (
                        split_id BIGSERIAL PRIMARY KEY,
                        db_id TEXT NOT NULL REFERENCES db_registry(db_id) ON DELETE CASCADE,
                        split_key TEXT NOT NULL,
                        tables_subset JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (db_id, split_key)
                    )
                """)
                cur.execute("""
                    CREATE UNLOGGED TABLE IF NOT EXISTS question (
                        question_id BIGSERIAL PRIMARY KEY,
                        db_id TEXT NOT NULL REFERENCES db_registry(db_id) ON DELETE CASCADE,
                        question_text TEXT NOT NULL,
                        question_key TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (db_id, question_key)
                    )
                """)

                cur.execute("""
                    CREATE UNLOGGED TABLE IF NOT EXISTS processed_result (
                        result_id BIGSERIAL PRIMARY KEY,
                        question_id BIGINT NOT NULL REFERENCES question(question_id) ON DELETE CASCADE,
                        split_id BIGINT NOT NULL REFERENCES split_group(split_id) ON DELETE CASCADE,
                        version_tag TEXT,
                        score DOUBLE PRECISION,
                        processed_info JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_split_group_db_id
                    ON split_group (db_id)
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_processed_result_split_id
                    ON processed_result (split_id)
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_processed_result_created_at
                    ON processed_result (split_id, created_at DESC)
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_processed_result_score
                    ON processed_result (split_id, score DESC NULLS LAST, created_at DESC)
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_question_db_id
                    ON question (db_id)
                """)
                # import pdb; pdb.set_trace()
                # cur.execute("""SELECT * FROM processed_result""").description
                # cur.execute("""SELECT * FROM question""").description
                # cur.execute("""SELECT * FROM question""").description
                # cur.execute("DROP TABLE IF EXISTS processed_result;")
                # cur.execute("DROP TABLE IF EXISTS question;")
                # cur.execute("DROP TABLE IF EXISTS split_group;")
                # cur.execute("DROP TABLE IF EXISTS db_registry;")
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_processed_result_question_id
                    ON processed_result (result_id, question_id, created_at DESC)
                """)

            conn.commit()
    def _delete_all_tables(self) -> None:
        print("Delete all data")
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS processed_result CASCADE;")
                cur.execute("DROP TABLE IF EXISTS question CASCADE;")
                cur.execute("DROP TABLE IF EXISTS split_group CASCADE;")
                cur.execute("DROP TABLE IF EXISTS db_registry CASCADE;")
            conn.commit()
    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _normalize_tables(self, tables: list[str]) -> list[str]:
        """
        Remove duplicates and sort names so table order becomes stable.

        Example:
            ["tbl3", "tbl1", "tbl1"] -> ["tbl1", "tbl3"]
        """
        return sorted(set([ttt.lower() for ttt in tables]))

    def _make_split_key(self, db_id: str, tables_subset: list[str]) -> str:
        """
        Build a stable key for a table subset.

        This makes these two subsets equivalent:
            ["tbl1", "tbl3"]
            ["tbl3", "tbl1"]
        """
        normalized = self._normalize_tables(tables_subset)
        raw = f"{db_id}::{'|'.join(normalized)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _make_question_key(self, question_text: str) -> str:
        normalized = " ".join(question_text.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def register_db(self, db_id: str, all_tables: list[str]) -> None:
        """
        Register or update the full list of tables for one db_id.
        """
        normalized = self._normalize_tables(all_tables)

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                # cur.execute("""
                #     INSERT INTO db_registry (db_id, all_tables, created_at)
                #     VALUES (%s, %s::jsonb, NOW())
                #     ON CONFLICT (db_id)
                #     DO UPDATE SET all_tables = EXCLUDED.all_tables
                # """, (db_id, json.dumps(normalized)))
                cur.execute("""
                    INSERT INTO db_registry (db_id, all_tables, created_at)
                    VALUES (%s, %s::jsonb, NOW())
                    ON CONFLICT (db_id)
                    DO NOTHING;
                """, (db_id, json.dumps(normalized)))
            conn.commit()

    def register_question(
            self,
            db_id: str,
            question_text: str,
            metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        question_text = question_text.strip()

        if not question_text:
            raise ValueError("question_text must not be empty")

        question_key = self._make_question_key(question_text)
        metadata = metadata or {}

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                # cur.execute("""
                #     INSERT INTO question (
                #         db_id, question_text,  question_key, metadata, created_at
                #     )
                #     VALUES (%s, %s, %s, %s::jsonb, NOW())
                #     ON CONFLICT (db_id, question_key)
                #     DO UPDATE SET
                #         question_text = EXCLUDED.question_text,
                #         metadata = EXCLUDED.metadata
                #     RETURNING question_id
                # """, (
                #     db_id, question_text, question_key, json.dumps(metadata),
                # ))
                cur.execute("""
                            INSERT INTO question (
                                db_id, question_text,  question_key, metadata, created_at
                            )
                            VALUES (%s, %s, %s, %s::jsonb, NOW())
                            ON CONFLICT (db_id, question_key)
                            DO NOTHING RETURNING question_id;
                        """, (
                    db_id, question_text, question_key, json.dumps(metadata),
                ))
                # BUG31072026 : if using DO NOTHING this return nothing None if question exist
                # print("register question ")
                # question_id = cur.fetchone()["question_id"] # if want question_id need use DO UPDATE

            conn.commit()
            # return question_id # BUG31072026 so do not return anything

    def store_processed_result(
        self,
        db_id: str,
        question_text: str,
        tables_subset: list[str],
        processed_info: Any,
        version_tag: Optional[str] = None,
        score: Optional[float] = None,
    ) -> int:
        """
        Store one processed result for one split.

        Behavior:
        - Create the split if it does not already exist.
        - Always insert a new processed_result row.
        - Return the new result_id.
        """
        normalized_subset = self._normalize_tables(tables_subset)
        split_key = self._make_split_key(db_id, normalized_subset)
        question_key = self._make_question_key(question_text)

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                # Lock the db_registry row so validation and insert happen consistently.

                cur.execute("""
                    SELECT question_id
                    FROM question
                    WHERE question_key = %s AND db_id = %s AND question_text = %s
                    FOR KEY SHARE
                """, (question_key, db_id, question_text))

                question_row = cur.fetchone()

                if question_row is None:
                    raise ValueError(
                        f"question_id '{question_key}' does not exist for db_id '{db_id}'"
                    )
                question_id = question_row["question_id"]
                cur.execute("""
                    SELECT all_tables
                    FROM db_registry
                    WHERE db_id = %s
                    FOR UPDATE
                """, (db_id,))
                db_row = cur.fetchone()

                if db_row is None:
                    raise ValueError(f"db_id '{db_id}' is not registered")

                full_tables = set([t.lower() for t in db_row["all_tables"]])
                if not set(normalized_subset).issubset(full_tables):
                    raise ValueError("tables_subset contains table not found in db_id")

                # Create split if missing.
                cur.execute("""
                    INSERT INTO split_group (db_id, split_key, tables_subset, created_at)
                    VALUES (%s, %s, %s::jsonb, NOW())
                    ON CONFLICT (db_id, split_key) DO NOTHING
                """, (db_id, split_key, json.dumps(normalized_subset)))

                # Fetch the split_id whether it was newly inserted or already existed.
                cur.execute("""
                    SELECT split_id
                    FROM split_group
                    WHERE db_id = %s AND split_key = %s
                """, (db_id, split_key))
                split_row = cur.fetchone()
                split_id = split_row["split_id"]

                # Insert one processed output for this split.
                cur.execute("""
                    INSERT INTO processed_result (question_id, split_id, version_tag, score, processed_info, created_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
                    RETURNING result_id
                """, (question_id, split_id, version_tag, score, json.dumps(processed_info)))

                result_id = cur.fetchone()["result_id"]
            conn.commit()
            return result_id

    def store_many_parallel(self, items: list[dict[str, Any]], max_workers: int = 8) -> dict[str, Any]:
        """
        Store many results in parallel using a thread pool.

        PostgreSQL handles concurrent writes much better than SQLite,
        so this method is useful when many training workers finish at similar times.
        """
        results: list[dict[str, Any]] = []

        def worker(index: int, item: dict[str, Any]) -> dict[str, Any]:
            try:
                result_id = self.store_processed_result(
                    db_id=item["db_id"],
                    question_text=item["question_text"],
                    tables_subset=item["tables_subset"],
                    processed_info=item["processed_info"],
                    version_tag=item.get("version_tag"),
                    score=item.get("score"),
                )
                return {"index": index, "ok": True, "result_id": result_id}
            except Exception as exc:
                return {"index": index, "ok": False, "error": str(exc)}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, i, item) for i, item in enumerate(items)]
            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda x: x["index"])
        return {
            "total": len(items),
            "success": sum(1 for x in results if x["ok"]),
            "failed": sum(1 for x in results if not x["ok"]),
            "items": results,
        }

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_results_for_split(self, db_id: str, tables_subset: list[str]) -> list[dict[str, Any]]:
        """
        Return all processed results for one split.
        Newest results appear first.
        """
        normalized_subset = self._normalize_tables(tables_subset)
        split_key = self._make_split_key(db_id, normalized_subset)

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        pr.result_id,
                        pr.version_tag,
                        pr.score,
                        pr.processed_info,
                        pr.created_at
                    FROM split_group sg
                    JOIN processed_result pr ON pr.split_id = sg.split_id
                    WHERE sg.db_id = %s AND sg.split_key = %s
                    ORDER BY pr.created_at DESC
                """, (db_id, split_key))
                rows = cur.fetchall()

        return [
            {
                "result_id": row["result_id"],
                "version_tag": row["version_tag"],
                "score": row["score"],
                "processed_info": row["processed_info"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]

    def get_results_for_question(
            self,
            db_id: str,
            question_text: str,
    ) -> list[dict[str, Any]]:
        question_key = self._make_question_key(question_text)

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        q.question_id,
                        q.question_text,
                        sg.split_id,
                        sg.tables_subset,
                        pr.result_id,
                        pr.version_tag,
                        pr.score,
                        pr.processed_info,
                        pr.created_at
                    FROM question q
                    JOIN processed_result pr
                        ON pr.question_id = q.question_id
                    JOIN split_group sg
                        ON sg.split_id = pr.split_id
                    WHERE q.db_id = %s
                      AND q.question_key = %s AND q.question_text = %s
                    ORDER BY pr.created_at DESC
                """, (db_id, question_key, question_text))

                rows = cur.fetchall()

        return [
            {
                "question_id": row["question_id"],
                "question_text": row["question_text"],
                "split_id": row["split_id"],
                "tables_subset": row["tables_subset"],
                "result_id": row["result_id"],
                "version_tag": row["version_tag"],
                "score": row["score"],
                "processed_info": row["processed_info"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]

    def _get_question_id(
            self,
            cur,
            db_id: str,
            question_text: str,
    ) -> int:
        """
        Find one existing question for a db_id.

        Raises ValueError if db_id or question_text is unknown.
        """
        normalized_question = question_text.strip()

        if not normalized_question:
            raise ValueError("question_text must not be empty")

        question_key = self._make_question_key(normalized_question)

        cur.execute("""
            SELECT q.question_id
            FROM question q
            JOIN db_registry d ON d.db_id = q.db_id
            WHERE q.db_id = %s
              AND q.question_key = %s
              AND q.question_text = %s
        """, (db_id, question_key, question_text))

        row = cur.fetchone()

        if row is None:
            raise ValueError(
                f"question not found for db_id '{db_id}'"
            )

        return row["question_id"]

    def _get_split_table_candidates(
            self,
            cur,
            db_id: str,
            question_id: int,
            prefer: str,
    ) -> list[dict[str, Any]]:
        """
        Get exactly one processed_result per split for one question.

        prefer:
        - latest: newest output per split
        - best_score: highest-score output per split
        - random: random output per split
        """
        if prefer not in {"latest", "best_score", "random"}:
            raise ValueError(
                "prefer must be one of: latest, best_score, random"
            )

        if prefer == "latest":
            order_by = """
                pr.split_id,
                pr.created_at DESC,
                pr.result_id DESC
            """
        elif prefer == "best_score":
            order_by = """
                pr.split_id,
                pr.score DESC NULLS LAST,
                pr.created_at DESC,
                pr.result_id DESC
            """
        else:
            order_by = """
                pr.split_id,
                RANDOM()
            """

        cur.execute(f"""
            SELECT DISTINCT ON (pr.split_id)
                sg.split_id,
                sg.tables_subset,
                pr.result_id,
                pr.version_tag,
                pr.score,
                pr.processed_info,
                pr.created_at
            FROM processed_result pr
            JOIN split_group sg
                ON sg.split_id = pr.split_id
            WHERE sg.db_id = %s
              AND pr.question_id = %s
            ORDER BY {order_by}
        """, (db_id, question_id))

        rows = cur.fetchall()

        return [
            {
                "split_id": row["split_id"],
                "tables": set(row["tables_subset"]),
                "result_id": row["result_id"],
                "version_tag": row["version_tag"],
                "score": row["score"],
                "processed_info": row["processed_info"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]

    def get_cover(self, db_id: str, question_text: str, prefer: str = "latest") -> Optional[list[dict[str, Any]]]:
        """
        Return one valid exact cover for db_id.

        Exact cover means:
        - chosen split subsets do not overlap
        - together they cover the full table set exactly once

        prefer:
        - latest: use the newest processed result for each split
        - best_score: use the highest-scoring result for each split

        https://arxiv.org/pdf/0910.0460
        https://arxiv.org/abs/1809.06506
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT all_tables
                    FROM db_registry
                    WHERE db_id = %s
                """, (db_id,))
                db_row = cur.fetchone()

                if db_row is None:
                    raise ValueError(f"db_id '{db_id}' is not registered")

                target_tables = set(db_row["all_tables"])
                question_id = self._get_question_id(
                    cur=cur,
                    db_id=db_id,
                    question_text=question_text,
                )

                candidates = self._get_split_table_candidates(
                    cur=cur,
                    db_id=db_id,
                    question_id=question_id,
                    prefer=prefer,
                )
        # Reject invalid candidate splits defensively.
        candidates = [
            candidate
            for candidate in candidates
            if candidate["tables"]
               and candidate["tables"].issubset(target_tables)
        ]

        # Build reverse index: table -> candidate splits that contain that table.
        by_table: dict[str, list[dict[str, Any]]] = {t: [] for t in target_tables}
        for candidate in candidates:
            for table_name in candidate["tables"]:
                by_table[table_name].append(candidate)

        # If any table has no candidate split, there is no exact-cover answer.
        for table_name in target_tables:
            if not by_table[table_name]:
                return None
            else:
                random.shuffle(by_table[table_name])
                # import pdb;pdb.set_trace()
        """
        Core để make sure cover tất cả table trong db_id
        """
        def backtrack(covered: set[str], chosen: list[dict[str, Any]]) -> Optional[list[dict[str, Any]]]:
            """
            Backtracking exact-cover search.

            Heuristic:
            choose the uncovered table that has the fewest candidate splits.
            This usually reduces search time.
            """
            if covered == target_tables:
                # Convert sets into JSON-friendly sorted lists.
                return [
                    {
                        **item,
                        "tables": sorted(item["tables"]),
                    }
                    for item in chosen
                ]

            remaining = target_tables - covered
            """
            Lấy từ split ít table nhất. Méo phải, theo số candidate ít nhất thì có
            Cái cách này thì đảm bảo lấy đủ cover, nhưng nó lại k được random cho lắm nhỉ  
            """
            # import pdb;pdb.set_trace()
            # current_table = min(remaining, key=lambda t: len(by_table[t]))
            current_table = random.choice(list(remaining))

            for candidate in by_table[current_table]:
                # Skip overlapping subsets.
                if candidate["tables"] & covered:
                    continue

                result = backtrack(
                    covered | candidate["tables"],
                    chosen + [candidate],
                )
                if result is not None:
                    return result

            return None

        return backtrack(set(), [])

    def get_random(self, db_id: str, question_text: str, prefer: str = "latest", max_groups: Optional[int] = None, seed: Optional[int] = None) -> list[dict[str, Any]]:
        """
        Return a random non-overlapping set of split results for one db_id.

        Difference from get_cover:
        - the result does not need to cover every table
        - the function picks a random valid subset of non-overlapping groups

        prefer:
        - 'latest'     -> pick newest processed result per split
        - 'best_score' -> pick highest score per split

        max_groups:
        - optional limit on how many groups to return

        seed:
        - optional random seed for reproducible results
        """
        # import random
        if max_groups is not None and max_groups <= 0:
            raise ValueError("max_groups must be greater than 0")

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""SELECT all_tables FROM db_registry  WHERE db_id = %s """, (db_id,))
                db_row = cur.fetchone()

                if db_row is None:
                    raise ValueError(f"db_id '{db_id}' is not registered")

                """
                Lấy tất cả các split_id có thể
                Tức là lấy các cách split table thành subset có thể có 
                Vd như db_id = sport có table [id, name1,sportname,player]
                thì có thể split thành các tables_subset như [id,name1], [sportname,player], [id,name1,sportname], [name1,sportname,player]
                Lấy tất cả split_id của các cách split đó 
                """
                target_tables = set(db_row["all_tables"])
                question_id = self._get_question_id(
                    cur=cur,
                    db_id=db_id,
                    question_text=question_text,
                )

                candidates = self._get_split_table_candidates(
                    cur=cur,
                    db_id=db_id,
                    question_id=question_id,
                    prefer=prefer,
                )
        # Ignore malformed or out-of-registry subsets.
        candidates = [
            candidate
            for candidate in candidates
            if candidate["tables"]
               and candidate["tables"].issubset(target_tables)
        ]
        rng = random.Random(seed)
        rng.shuffle(candidates)

        chosen: list[dict[str, Any]] = []
        covered: set[str] = set()

        for candidate in candidates:
            if candidate["tables"] & covered:
                continue

            chosen.append({**candidate, "tables": sorted(candidate["tables"])})
            covered |= candidate["tables"]

            if max_groups is not None and len(chosen) >= max_groups:
                break

        return chosen

    # ------------------------------------------------------------------
    # Delete / reset operations
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Delete all training data but keep the schema.
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                # TRUNCATE is fast and good for session cleanup.
                cur.execute("TRUNCATE TABLE processed_result, split_group, db_registry RESTART IDENTITY CASCADE")
            conn.commit()

    def reset_db_id(self, db_id: str) -> None:
        """
        Delete data for one db_id only.
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM db_registry WHERE db_id = %s", (db_id,))
            conn.commit()

    def close(self) -> None:
        """
        Close the PostgreSQL connection pool.
        """
        self.pool.close()
    def cleanup_old_processed_results(self, retention_hours: float = 2.0,delete_empty_splits: bool = False) -> dict[str, Any]:
        """
        Delete processed_result records older than retention_hours.

        Example:
            store.cleanup_old_processed_results(2)    # delete older than 2 hours
            store.cleanup_old_processed_results(1.5)  # delete older than 90 minutes

        Returns:
            A dictionary containing the cleanup statistics.

        """
        if retention_hours <= 0:
            raise ValueError("retention_hours must be greater than 0")

        # PostgreSQL accepts an interval as text, for example: '2 hours'.
        retention_interval = f"{retention_hours} hours"

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM processed_result
                    WHERE created_at < NOW() - %s::interval
                    """,
                    (retention_interval,),
                )

                # deleted_count = cur.rowcount
                # CALL returns one row because the procedure has deleted_count
                # as an INOUT parameter.
                # row = cur.fetchone()
                # deleted_results = int(row["deleted_count"]) if row else 0
                deleted_results = cur.rowcount

                deleted_empty_splits = 0

                if delete_empty_splits:
                    cur.execute("""
                                DELETE FROM split_group sg
                                WHERE NOT EXISTS (
                                    SELECT 1
                                    FROM processed_result pr
                                    WHERE pr.split_id = sg.split_id
                                )
                            """)
                    deleted_empty_splits = cur.rowcount

            conn.commit()
        return {
            "retention_hours": retention_hours,
            "deleted_results": deleted_results,
            "deleted_empty_splits": deleted_empty_splits,
        }

    def cleanup_empty_splits(self) -> int:
        """
        Delete split groups that no longer have any processed result.
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM split_group sg
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM processed_result pr
                        WHERE pr.split_id = sg.split_id
                    )
                """)
                deleted_count = cur.rowcount

            conn.commit()

        return deleted_count

# ----------------------------------------------------------------------
# Flask service
# ----------------------------------------------------------------------


def build_app() -> Flask:
    """
    Create and configure the Flask application.
    """
    app = Flask(__name__)

    # Build PostgreSQL DSN from environment variables.
    # Example:
    #   postgresql://training_user:training_password@localhost:5432/training_db
    db_user = os.getenv("POSTGRES_USER", "training_user")
    db_password = os.getenv("POSTGRES_PASSWORD", "training_password")
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    # db_port = os.getenv("POSTGRES_PORT", "5432")
    db_port = os.getenv("POSTGRES_PORT", "5434")
    db_name = os.getenv("POSTGRES_DB", "training_db")
    pg_pool_min = int(os.getenv("PG_POOL_MIN", "1"))
    pg_pool_max = int(os.getenv("PG_POOL_MAX", "32"))

    dsn = os.getenv(
        "DATABASE_URL",
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
    )

    store = TrainingStore(
        dsn=dsn,
        min_size=pg_pool_min,
        max_size=pg_pool_max,
    )

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "message": "service is running"})

    @app.post("/register_db")
    def register_db():
        """
        Register one db_id and its full table list.
        """
        try:
            data = request.get_json(force=True)
            store.register_db(data["db_id"], data["all_tables"])
            return jsonify({
                "ok": True,
                "db_id": data["db_id"],
                "all_tables": sorted(set(data["all_tables"])),
            })
        except Exception as exc:
            print(str(exc))
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/register_question")
    def register_question():
        try:
            data = request.get_json(force=True)

            store.register_question(
                db_id=data["db_id"],
                question_text=data["question_text"],
                metadata=data.get("metadata"),
            )

            return jsonify({
                "ok": True,
                "question_text": data["question_text"],
            })

        except Exception as exc:
            print(str(exc))
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/store_result")
    def store_result():
        """
        Store one processed result.
        """
        # print("store_result")
        try:
            data = request.get_json(force=True)
            # print(data)
            result_id = store.store_processed_result(
                db_id=data["db_id"],
                question_text=data["question_text"],
                tables_subset=data["tables_subset"],
                processed_info=data["processed_info"],
                version_tag=data.get("version_tag"),
                score=data.get("score"),
            )
            return jsonify({"ok": True, "result_id": result_id})
        except Exception as exc:
            print(str(exc))
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/store_many_parallel")
    def store_many_parallel():
        """
        Store many results in one request using a thread pool.
        """
        try:
            data = request.get_json(force=True)
            items = data["items"]
            max_workers = int(data.get("max_workers", 8))
            result = store.store_many_parallel(items, max_workers=max_workers)
            return jsonify({"ok": True, "batch": result})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/get_results")
    def get_results():
        """
        Get all results for one split.
        """
        try:
            data = request.get_json(force=True)
            results = store.get_results_for_split(
                db_id=data["db_id"],
                tables_subset=data["tables_subset"],
            )
            return jsonify({"ok": True, "count": len(results), "results": results})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/get_cover")
    def get_cover():
        """
        Get one exact-cover result for a db_id.
        """
        try:
            data = request.get_json(force=True)
            cover = store.get_cover(
                db_id=data["db_id"],
                question_text=data["question_text"],
                prefer=data.get("prefer", "random"),
            )
            return jsonify({"ok": True, "cover": cover})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/get_random")
    def get_random():
        try:
            data = request.get_json(force=True)

            cover = store.get_random(
                db_id=data["db_id"],
                question_text=data["question_text"],
                prefer=data.get("prefer", "random"),
                max_groups=data.get("max_groups"),
                seed=data.get("seed"),
            )

            return jsonify({
                "ok": True,
                "count": len(cover),
                "cover": cover,
            })

        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/reset")
    def reset():
        """
        Delete all training data.
        """
        try:
            store.reset()
            return jsonify({"ok": True, "message": "all data deleted"})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/reset_db_id")
    def reset_db_id():
        """
        Delete training data for one db_id only.
        """
        try:
            data = request.get_json(force=True)
            store.reset_db_id(data["db_id"])
            return jsonify({"ok": True, "db_id": data["db_id"], "message": "db_id data deleted"})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/cleanup_old_results")
    def cleanup_old_results():
        """
        Delete old processed_result records.

        Request body:
        {
            "retention_hours": 2,
            "delete_empty_splits": true
        }

        retention_hours:
            - Optional
            - Default is 2 hours
            - Must be greater than 0

        delete_empty_splits:
            - Optional
            - Default is false
            - If true, also delete split_group records that have no results left
        """
        try:
            # Read JSON body. If body is empty, use an empty dictionary.
            data = request.get_json(silent=True) or {}

            # Read optional values from the request.
            retention_hours = float(data.get("retention_hours", 2))
            delete_empty_splits = bool(data.get("delete_empty_splits", False))

            # Validate input before calling PostgreSQL.
            if retention_hours <= 0:
                return jsonify({
                    "ok": False,
                    "error": "retention_hours must be greater than 0"
                }), 400

            # Call the function in TrainingStore.
            # It runs the PostgreSQL cleanup procedure.
            result = store.cleanup_old_processed_results(
                retention_hours=retention_hours,
                delete_empty_splits=delete_empty_splits,
            )

            # Example result:
            # {
            #     "retention_hours": 2.0,
            #     "deleted_results": 15,
            #     "deleted_empty_splits": 3
            # }
            return jsonify({
                "ok": True,
                **result
            }), 200

        except ValueError as exc:
            # Use HTTP 400 for invalid user input.
            return jsonify({
                "ok": False,
                "error": str(exc)
            }), 400

        except Exception as exc:
            # Use HTTP 500 for unexpected server/database errors.
            return jsonify({
                "ok": False,
                "error": f"cleanup failed: {exc}"
            }), 500
    @app.get("/example_env")
    def example_env():
        """
        Return example environment variables for quick setup.
        """
        return jsonify({
            "POSTGRES_HOST": db_host,
            "POSTGRES_PORT": db_port,
            "POSTGRES_DB": db_name,
            "POSTGRES_USER": db_user,
            "POSTGRES_PASSWORD": db_password,
            "PG_POOL_MIN": pg_pool_min,
            "PG_POOL_MAX": pg_pool_max,
        })

    return app


app = build_app()


if __name__ == "__main__":
    # Flask dev server is fine for local testing.
    # For a more production-like run, use waitress or gunicorn.
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
"""

# conda activate llmnew
flask --app postgres_service run --debug --host=0.0.0.0 --port=5000
sudo ufw allow 5000

POSTGRES_PORT=5434 /home/tampm/miniconda3/envs/llmnew/bin/python3.1 /home/tampm/miniconda3/envs/llmnew/bin/flask --app postgres_service run --debug --host=0.0.0.0 --port=5000
"""