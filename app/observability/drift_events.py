import json
from datetime import datetime
from pathlib import Path
from typing import Dict

DRIFT_LOG_PATH = Path(__file__).resolve().parents[2] / "data" / "processed"
DRIFT_LOG_PATH.mkdir(parents=True, exist_ok=True)


def log_drift_event(drift_report: Dict[str, float]) -> None:
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "drift": drift_report,
    }

    file_path = DRIFT_LOG_PATH / "drift_events.jsonl"

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
