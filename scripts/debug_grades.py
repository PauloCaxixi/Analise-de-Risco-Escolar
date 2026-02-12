from __future__ import annotations

import json
from app.db import get_conn

with get_conn() as conn:
    # pega um aluno que tenha grade
    r = conn.execute(
        """
        SELECT student_id, reference_year, LENGTH(payload_json) AS n
        FROM grades
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()

    print("Última grade:", dict(r) if r else None)

    if r:
        row = conn.execute(
            """
            SELECT payload_json
            FROM grades
            WHERE student_id = ? AND reference_year = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (r["student_id"], r["reference_year"]),
        ).fetchone()

        payload = json.loads(row["payload_json"])
        print("Total de features:", len(payload))
        # imprime algumas chaves
        print("Amostra keys:", list(payload.keys())[:20])
