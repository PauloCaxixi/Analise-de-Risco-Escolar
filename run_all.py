from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import urllib.request
import urllib.error


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_ROOT / "app" / "model" / "model.joblib"
EXCEL_FILE = PROJECT_ROOT / "data" / "raw" / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000
    api_url: str = "http://127.0.0.1:8000"
    wait_timeout_sec: int = 45
    run_dashboard: bool = False  # opcional
    dashboard_port: int = 8501


def _run(cmd: list[str], *, cwd: Path, env: Optional[dict[str, str]] = None) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env or os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _http_get(url: str, timeout: int = 5) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return int(resp.status), body


def _http_post_json(url: str, payload: dict[str, Any], timeout: int = 10) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return int(resp.status), body


def _wait_for_api(settings: Settings) -> None:
    deadline = time.time() + settings.wait_timeout_sec
    last_err: Optional[str] = None

    while time.time() < deadline:
        try:
            status, _ = _http_get(f"{settings.api_url}/health", timeout=3)
            if status == 200:
                return
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(0.8)

    raise TimeoutError(f"API não respondeu em {settings.wait_timeout_sec}s. Último erro: {last_err}")


def _tail_process_output(proc: subprocess.Popen, *, prefix: str, max_lines: int = 80) -> None:
    if proc.stdout is None:
        return

    lines = []
    for _ in range(max_lines):
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line.rstrip("\n"))
    if lines:
        print(f"\n--- {prefix} (últimas linhas) ---")
        for ln in lines[-max_lines:]:
            print(ln)


def _terminate_process(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is not None:
        return

    try:
        if sys.platform.startswith("win"):
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            time.sleep(1.0)
            proc.terminate()
        else:
            proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        proc.kill()


def main() -> int:
    settings = Settings(
        host="127.0.0.1",
        port=8000,
        api_url="http://127.0.0.1:8000",
        run_dashboard=False,  # mude para True se quiser iniciar Streamlit junto
    )

    if not EXCEL_FILE.exists():
        print(f"ERRO: Excel não encontrado em: {EXCEL_FILE}")
        print("Coloque o arquivo em data/raw/ com o nome exatamente igual.")
        return 2

    # 1) Treino (somente se não existir modelo)
    if not MODEL_FILE.exists():
        print("Modelo não encontrado. Treinando...")
        train_cmd = [sys.executable, "-m", "src.train"]
        train_proc = _run(train_cmd, cwd=PROJECT_ROOT)
        train_rc = train_proc.wait()

        _tail_process_output(train_proc, prefix="TREINO")

        if train_rc != 0:
            print(f"ERRO: treino falhou (exit code {train_rc}).")
            return train_rc

        if not MODEL_FILE.exists():
            print("ERRO: treino concluiu, mas model.joblib não foi gerado.")
            return 3

        print("Treino concluído e modelo gerado.")
    else:
        print("Modelo já existe. Pulando treino.")

    # 2) Subir API
    print("Subindo API (uvicorn)...")
    uvicorn_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        settings.host,
        "--port",
        str(settings.port),
    ]

    env = os.environ.copy()
    # Necessário para CTRL_BREAK_EVENT no Windows quando Popen captura stdout
    if sys.platform.startswith("win"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        api_proc = subprocess.Popen(
            uvicorn_cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )
    else:
        api_proc = _run(uvicorn_cmd, cwd=PROJECT_ROOT, env=env)

    dashboard_proc: Optional[subprocess.Popen] = None
    try:
        _wait_for_api(settings)
        print(f"API OK: {settings.api_url}")

        # 3) Teste rápido do /predict
        print("Executando teste rápido do /predict...")
        sample_payload = {
            "features": {
                "IDADE_ALUNO_2020": 12,
                "IAN_2022": 7.5,
                "IDA_2022": 6.0,
                "IEG_2022": 8.0,
            }
        }

        status, body = _http_post_json(f"{settings.api_url}/predict", sample_payload, timeout=15)
        print(f"/predict status={status}")
        print("Resposta:", body)

        # 4) (Opcional) subir dashboard
        if settings.run_dashboard:
            print("Subindo dashboard Streamlit...")
            dashboard_cmd = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "dashboards/app.py",
                "--server.port",
                str(settings.dashboard_port),
                "--server.address",
                "127.0.0.1",
            ]
            dashboard_proc = _run(dashboard_cmd, cwd=PROJECT_ROOT, env=env)
            print(f"Dashboard: http://127.0.0.1:{settings.dashboard_port}")

        print("\nTudo OK.")
        print("Swagger: http://127.0.0.1:8000/docs")
        print("Para encerrar, pressione Ctrl+C.\n")

        # Mantém rodando
        while True:
            if api_proc.poll() is not None:
                _tail_process_output(api_proc, prefix="API")
                raise RuntimeError("API parou inesperadamente.")
            if dashboard_proc and dashboard_proc.poll() is not None:
                _tail_process_output(dashboard_proc, prefix="DASHBOARD")
                raise RuntimeError("Dashboard parou inesperadamente.")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nEncerrando...")
        return 0
    except urllib.error.HTTPError as e:
        print(f"ERRO HTTP: {e.code} {e.reason}")
        try:
            print(e.read().decode("utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            pass
        return 10
    except Exception as e:  # noqa: BLE001
        print(f"ERRO: {e}")
        _tail_process_output(api_proc, prefix="API")
        if dashboard_proc:
            _tail_process_output(dashboard_proc, prefix="DASHBOARD")
        return 1
    finally:
        if dashboard_proc:
            _terminate_process(dashboard_proc, "dashboard")
        _terminate_process(api_proc, "api")


if __name__ == "__main__":
    raise SystemExit(main())
