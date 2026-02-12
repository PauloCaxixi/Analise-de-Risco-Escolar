from __future__ import annotations

from app.db import get_conn
import json
from typing import Any, Dict, List, Optional

from app.db import get_conn


# =========================
# Students
# =========================
def create_student(
    full_name: str,
    student_code: Optional[str] = None,
    birth_date: Optional[str] = None,
    grade_level: Optional[str] = None,
    class_name: Optional[str] = None,
) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO students (student_code, full_name, birth_date, grade_level, class_name)
            VALUES (?, ?, ?, ?, ?)
            """,
            (student_code, full_name, birth_date, grade_level, class_name),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_students(active_only: bool = True) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        if active_only:
            rows = conn.execute("SELECT * FROM students WHERE is_active = 1 ORDER BY full_name").fetchall()
        else:
            rows = conn.execute("SELECT * FROM students ORDER BY full_name").fetchall()
    return [dict(r) for r in rows]


def get_student(student_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    return dict(row) if row else None


# =========================
# Grades (features armazenadas)
# =========================
def upsert_grades(
    student_id: int,
    reference_year: str,
    features: Dict[str, Any],
) -> int:
    payload_json = json.dumps(features, ensure_ascii=False)

    with get_conn() as conn:
        cur = conn.cursor()
        # estratégia simples: sempre inserir um novo snapshot
        cur.execute(
            """
            INSERT INTO grades (student_id, reference_year, payload_json)
            VALUES (?, ?, ?)
            """,
            (student_id, reference_year, payload_json),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_latest_grades(student_id: int, reference_year: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT payload_json
            FROM grades
            WHERE student_id = ? AND reference_year = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (student_id, reference_year),
        ).fetchone()

    if not row:
        return None
    return json.loads(row["payload_json"])


# =========================
# Predictions
# =========================
def create_prediction(
    student_id: int,
    reference_year: str,
    risk_score: float,
    risk_label: str,
    model_version: str,
    input_features: Dict[str, Any],
) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions (student_id, reference_year, risk_score, risk_label, model_version, input_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                reference_year,
                float(risk_score),
                risk_label,
                model_version,
                json.dumps(input_features, ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_predictions(student_id: int) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM predictions
            WHERE student_id = ?
            ORDER BY created_at DESC
            """,
            (student_id,),
        ).fetchall()
    return [dict(r) for r in rows]




def list_students_overview(limit: int = 200, q: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Lista alunos com defasagem REAL por ano (ground_truth) e última previsão (não-ground_truth).
    - limit: pagina simples (primeiros N) para não travar a UI com 1661 linhas.
    - q: filtro por nome ou RA (student_code).
    """
    params: list[Any] = []
    where = ""
    if q and q.strip():
        where = "WHERE (s.full_name LIKE ? OR s.student_code LIKE ?)"
        like = f"%{q.strip()}%"
        params.extend([like, like])

    params.append(int(limit))

    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT
                s.id,
                s.student_code,
                s.full_name,

                MAX(CASE WHEN p.reference_year='PEDE2022' AND p.model_version='ground_truth' THEN p.risk_score END) AS def_2022,
                MAX(CASE WHEN p.reference_year='PEDE2023' AND p.model_version='ground_truth' THEN p.risk_score END) AS def_2023,
                MAX(CASE WHEN p.reference_year='PEDE2024' AND p.model_version='ground_truth' THEN p.risk_score END) AS def_2024,

                MAX(CASE WHEN p.model_version!='ground_truth' THEN p.risk_score END) AS last_pred,
                MAX(CASE WHEN p.model_version!='ground_truth' THEN p.created_at END) AS last_pred_at

            FROM students s
            LEFT JOIN predictions p ON p.student_id = s.id
            {where}
            GROUP BY s.id, s.student_code, s.full_name
            ORDER BY s.full_name
            LIMIT ?
            """,
            params,
        ).fetchall()

    return [dict(r) for r in rows]



def get_student_panel(student_id: int) -> dict:
    """
    Painel completo do aluno:
    - dados
    - defasagem real por ano
    - histórico de previsões
    """
    with get_conn() as conn:
        student = conn.execute(
            "SELECT * FROM students WHERE id = ?",
            (student_id,),
        ).fetchone()

        if not student:
            raise ValueError("Aluno não encontrado")

        history = conn.execute(
            """
            SELECT reference_year, risk_score, risk_label, model_version, created_at
            FROM predictions
            WHERE student_id = ?
            ORDER BY created_at DESC
            """,
            (student_id,),
        ).fetchall()

    return {
        "student": dict(student),
        "history": [dict(h) for h in history],
    }




def get_student_year_features(student_id: int, reference_year: str) -> Optional[dict[str, Any]]:
    """
    Busca o último snapshot de features por (aluno, ano).
    NÃO depende de created_at (nem sempre existe); usa id DESC.
    """
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT payload_json
            FROM grades
            WHERE student_id = ? AND reference_year = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (int(student_id), str(reference_year)),
        ).fetchone()

    if not row:
        return None

    raw = row["payload_json"]
    if raw is None or str(raw).strip() == "":
        return None

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return None
    except Exception:
        return None



def list_students_for_select(limit: int = 5000) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, student_code, full_name
            FROM students
            ORDER BY full_name
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]

    import json
from typing import Any, Optional

def get_latest_year_with_grades(student_id: int) -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT reference_year
            FROM grades
            WHERE student_id = ?
            GROUP BY reference_year
            ORDER BY
              CASE reference_year
                WHEN 'PEDE2024' THEN 3
                WHEN 'PEDE2023' THEN 2
                WHEN 'PEDE2022' THEN 1
                ELSE 0
              END DESC
            LIMIT 1
            """,
            (int(student_id),),
        ).fetchone()
    return str(row["reference_year"]) if row else None


def get_real_defasagem(student_id: int, reference_year: str) -> Optional[float]:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT risk_score
            FROM predictions
            WHERE student_id = ? AND reference_year = ? AND model_version = 'ground_truth'
            ORDER BY id DESC
            LIMIT 1
            """,
            (int(student_id), str(reference_year)),
        ).fetchone()
    return float(row["risk_score"]) if row else None


def create_prediction_record(
    student_id: int,
    reference_year: str,
    risk_score: float,
    risk_label: str,
    model_version: str,
    input_features: dict[str, Any],
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO predictions
              (student_id, reference_year, risk_score, risk_label, model_version, input_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(student_id),
                str(reference_year),
                float(risk_score),
                str(risk_label),
                str(model_version),
                json.dumps(input_features, ensure_ascii=False),
            ),
        )
        conn.commit()

