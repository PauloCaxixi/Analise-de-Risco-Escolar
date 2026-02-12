from __future__ import annotations

from math import e
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeSerializer
from werkzeug.security import check_password_hash, generate_password_hash

import json

from app.services.predictor import predict
from app.schemas.predict import PredictRequest
from app.services.model_loader import load_metadata

from app.repositories import (
    create_prediction,
    get_latest_grades,
    upsert_grades,
    create_student,
    get_latest_year_with_grades,
    get_real_defasagem,
    create_prediction_record,
    get_student,
    list_predictions,
    list_students,
    list_students_overview,
    list_students_for_select,
    get_student_year_features,
    get_student_panel, list_students, list_predictions
)
from app.services.predictor import predict
from app.schemas.predict import PredictRequest
from app.services.model_loader import load_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "platform.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

router = APIRouter(include_in_schema=False)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        cur.execute("SELECT COUNT(*) AS n FROM users")
        n = int(cur.fetchone()["n"])
        if n == 0:
            cur.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("admin", generate_password_hash("admin123"), "admin"),
            )
        conn.commit()


def _serializer(secret_key: str) -> URLSafeSerializer:
    return URLSafeSerializer(secret_key, salt="platform-session")


def _get_user_from_cookie(request: Request, secret_key: str) -> Optional[Dict[str, Any]]:
    raw = request.cookies.get("session")
    if not raw:
        return None
    try:
        data = _serializer(secret_key).loads(raw)
        if isinstance(data, dict) and "username" in data:
            return data
        return None
    except Exception:
        return None


def _set_session_cookie(resp: RedirectResponse, secret_key: str, user: Dict[str, Any]) -> None:
    token = _serializer(secret_key).dumps(user)
    resp.set_cookie("session", token, httponly=True, samesite="lax")


def _clear_session_cookie(resp: RedirectResponse) -> None:
    resp.delete_cookie("session")


@router.get("/login", response_class=HTMLResponse)
def login(request: Request):
    return TEMPLATES.TemplateResponse("login.html", {"request": request, "error": None})


@router.post("/login")
def login_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    username = username.strip()

    with _conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    if not row or not int(row["is_active"]) or not check_password_hash(row["password_hash"], password):
        return TEMPLATES.TemplateResponse(
            "login.html",
            {"request": request, "error": "Credenciais inválidas."},
            status_code=401,
        )

    resp = RedirectResponse(url="/", status_code=302)
    _set_session_cookie(resp, request.app.state.secret_key, {"username": row["username"], "role": row["role"]})
    return resp


@router.get("/logout")
def logout(request: Request):
    resp = RedirectResponse(url="/login", status_code=302)
    _clear_session_cookie(resp)
    return resp


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    user = _get_user_from_cookie(request, request.app.state.secret_key)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    return TEMPLATES.TemplateResponse(
        "index.html",
        {"request": request, "user": user, "active": "home"},
    )



@router.get("/students", response_class=HTMLResponse)
def students_page(request: Request, q: str = "", limit: int = 200):
    user = _get_user_from_cookie(request, request.app.state.secret_key)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    limit = max(50, min(int(limit), 500))
    students = list_students_overview(limit=limit, q=q)

    return TEMPLATES.TemplateResponse(
        "students.html",
        {
            "request": request,
            "user": user,
            "active": "students",
            "students": students,
            "q": q,
            "limit": limit,
        },
    )





@router.post("/students/create")
def students_create(
    request: Request,
    full_name: str = Form(...),
    student_code: str = Form(""),
    birth_date: str = Form(""),
    grade_level: str = Form(""),
    class_name: str = Form(""),
):
    user = _get_user_from_cookie(request, request.app.state.secret_key)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    try:
        create_student(
            full_name=full_name.strip(),
            student_code=student_code.strip() or None,
            birth_date=birth_date.strip() or None,
            grade_level=grade_level.strip() or None,
            class_name=class_name.strip() or None,
        )
        return RedirectResponse(url="/students", status_code=302)
    except Exception as e:
        students = list_students(active_only=True)
        return TEMPLATES.TemplateResponse(
            "students.html",
            {"request": request, "user": user, "students": students, "error": str(e)},
            status_code=400,
        )


@router.get("/students/{student_id}", response_class=HTMLResponse)
def student_panel(request: Request, student_id: int):
    user = _get_user_from_cookie(request, request.app.state.secret_key)
    if not user:
        return RedirectResponse("/login", 302)

    panel = get_student_panel(student_id)
    return TEMPLATES.TemplateResponse(
        "student_panel.html",
        {
            "request": request,
            "user": user,
            "active": "students",
            "panel": panel,
        },
    )


@router.get("/grades", response_class=HTMLResponse)
def grades_page(request: Request):
    user = _get_user_from_cookie(request, request.app.state.secret_key)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    students = list_students(active_only=True)
    sample_json = json.dumps({"IDADE_ALUNO_2020": 12, "IAN_2022": 7.5, "IDA_2022": 6.0, "IEG_2022": 8.0}, ensure_ascii=False, indent=2)
    return TEMPLATES.TemplateResponse(
        "grades.html",
        {"request": request, "user": user, "active": "grades", "students": students, "error": None, "ok": None, "sample_json": sample_json},
    )


@router.post("/grades/save")
def grades_save(
    request: Request,
    student_id: int = Form(...),
    reference_year: str = Form(...),
    features_json: str = Form(...),
):
    user = _get_user_from_cookie(request, request.app.state.secret_key)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    students = list_students(active_only=True)

    try:
        features = json.loads(features_json)
        if not isinstance(features, dict):
            raise ValueError("features_json deve ser um objeto JSON (dicionário).")

        upsert_grades(student_id=student_id, reference_year=reference_year, features=features)

        return TEMPLATES.TemplateResponse(
            "grades.html",
            {
                "request": request,
                "user": user,
                "active": "grades",
                "students": students,
                "error": None,
                "ok": "Snapshot salvo com sucesso.",
                "sample_json": features_json,
            },
        )
    except Exception as e:
        sample_json = features_json or json.dumps({"IDADE_ALUNO_2020": 12}, ensure_ascii=False, indent=2)
        return TEMPLATES.TemplateResponse(
            "grades.html",
            {"request": request, "user": user, "active": "grades", "students": students, "error": str(e), "ok": None, "sample_json": sample_json},
            status_code=400,
        )


@router.get("/predictions", response_class=HTMLResponse)
def predictions_page(request: Request):
    user = _get_user_from_cookie(request, request.app.state.secret_key)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    try:
        students = list_students_for_select(limit=5000)
    except Exception as e:
        students = []
        return TEMPLATES.TemplateResponse(
            "predictions.html",
            {
                "request": request,
                "user": user,
                "active": "predictions",
                "students": students,
                "error": f"Falha ao carregar lista de alunos: {e}",
                "result": None,
            },
            status_code=500,
        )

    return TEMPLATES.TemplateResponse(
        "predictions.html",
        {
            "request": request,
            "user": user,
            "active": "predictions",
            "students": students,
            "error": None,
            "result": None,
        },
    )


@router.post("/predictions/run", response_class=HTMLResponse)
def predictions_run(request: Request, student_id: int = Form(...)):
    user = _get_user_from_cookie(request, request.app.state.secret_key)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    students = list_students_for_select(limit=5000)

    try:
        ref = get_latest_year_with_grades(student_id)
        if not ref:
            raise ValueError("Aluno não possui notas/features importadas.")

        feats = get_student_year_features(student_id, ref)
        if not feats:
            raise ValueError(f"Não há features para o ano {ref}.")

        # ML prediction
        out = predict(PredictRequest(features=feats))

        # Observações pedagógicas (não-ML)
        destaque_ieg = feats.get("Destaque IEG")
        if destaque_ieg is None:
            # algumas bases vêm com variações de nome
            for k in feats.keys():
                if "destaque" in str(k).casefold() and "ieg" in str(k).casefold():
                    destaque_ieg = feats.get(k)
                    break


        md = load_metadata()
        model_version = str(md.get("model_version", out.get("model_version", "unknown")))

        real_def = get_real_defasagem(student_id, ref)

        create_prediction_record(
            student_id=student_id,
            reference_year=ref,
            risk_score=float(out["risk_probability"]),   # <-- salva probabilidade como score
            risk_label=str(out["risk_level"]),           # <-- salva nível como label
            model_version=model_version,
            input_features=feats,
        )

        # delta não faz sentido entre probabilidade e defasagem -> remove
        result = {
            "reference_year": ref,
            "model_version": model_version,
            "risk_probability": float(out["risk_probability"]),
            "risk_level": str(out["risk_level"]),
            "status": str(out["status"]),
            "recommended_action": str(out["recommended_action"]),
            "real_defasagem": real_def,
            "feature_coverage": out.get("feature_coverage"),
            "features_filled": out.get("features_filled"),
            "features_expected": out.get("features_expected"),
            "destaque_ieg": destaque_ieg,
        }



        return TEMPLATES.TemplateResponse(
            "predictions.html",
            {"request": request, "user": user, "active": "predictions", "students": students, "error": None, "result": result},
        )

    except Exception as e:
        return TEMPLATES.TemplateResponse(
            "predictions.html",
            {"request": request, "user": user, "active": "predictions", "students": students, "error": str(e), "result": None},
            status_code=400,
        )


@router.get("/grades", response_class=HTMLResponse)
def grades_page(request: Request, student_id: int = 0, reference_year: str = "PEDE2024"):
    user = _get_user_from_cookie(request, request.app.state.secret_key)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    students = list_students_for_select(limit=5000)

    selected_student_id = int(student_id) if str(student_id).isdigit() else 0
    features = None
    error = None

    if selected_student_id:
        try:
            features = get_student_year_features(selected_student_id, reference_year)
            if features is None:
                error = f"Sem snapshot: student_id={selected_student_id} / ano={reference_year}. (Verifique se existe grade importada.)"

        except Exception as e:
            error = str(e)

    return TEMPLATES.TemplateResponse(
        "grades.html",
        {
            "request": request,
            "user": user,
            "active": "grades",
            "students": students,
            "selected_student_id": selected_student_id,
            "reference_year": reference_year,
            "features": features,
            "error": error,
        },
    )
