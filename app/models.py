from __future__ import annotations

from app.db import get_conn


def init_schema() -> None:
    with get_conn() as conn:
        cur = conn.cursor()

        # Usuários (já existia no portal)
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

        # Alunos
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_code TEXT UNIQUE,
                full_name TEXT NOT NULL,
                birth_date TEXT,
                grade_level TEXT,
                class_name TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )

        # Notas / métricas por período (por ano/aba)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS grades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                reference_year TEXT NOT NULL,        -- PEDE2022/PEDE2023/PEDE2024 ou ano letivo
                payload_json TEXT NOT NULL,          -- features completas (JSON)
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
            )
            """
        )

        # Histórico de previsões
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                reference_year TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_label TEXT NOT NULL,
                model_version TEXT NOT NULL,
                input_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
            )
            """
        )

        conn.commit()
