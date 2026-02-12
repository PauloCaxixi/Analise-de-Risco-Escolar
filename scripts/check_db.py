from app.db import DB_PATH, get_conn

print("DB_PATH =", DB_PATH)

with get_conn() as conn:
    n_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    n_grades = conn.execute("SELECT COUNT(*) FROM grades").fetchone()[0]
    n_preds = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

print("students =", n_students)
print("grades =", n_grades)
print("predictions =", n_preds)
