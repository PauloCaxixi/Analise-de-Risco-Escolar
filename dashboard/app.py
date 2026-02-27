from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =========================
# PROJECT ROOT BOOTSTRAP
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent  # TECH_5/dashboard/
REPO_ROOT = PROJECT_ROOT.parent                 # TECH_5/

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# =========================
# INTERNAL IMPORTS (src/*)
# =========================
from src.services.tendencia import calcular_tendencia
from src.features import coerce_numeric as _coerce_numeric_all
from src.features import standardize_columns as _standardize_columns
from src.features import calc_media_disciplinas as _calc_media_disciplinas
from src.features import standardize_columns 

import pandas as pd
from flask import Flask, abort, redirect, render_template, request, url_for, Response, flash, jsonify

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


# =========================
# CONFIG
# =========================
# =========================
# CONFIG
# =========================
# REPO_ROOT já foi definido no bootstrap (TECH_5/)
BASE_DIR = REPO_ROOT  # raiz real do projeto

DEFAULT_XLSX = BASE_DIR / "dashboard" / "data" / "raw" / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
DATA_XLSX_PATH = Path(os.environ.get("PEDE_XLSX_PATH", str(DEFAULT_XLSX)))
DEFAULT_SHEET = os.environ.get("PEDE_DEFAULT_SHEET", "PEDE2022")
AVAILABLE_SHEETS = ["PEDE2022", "PEDE2023", "PEDE2024"]

def _next_year_from_sheet(sheet: str) -> str:
    digits = "".join(ch for ch in sheet if ch.isdigit())
    if len(digits) == 4:
        return str(int(digits) + 1)
    return "próximo ano"

MODEL_DIR = BASE_DIR / "app" / "model"
MODEL_PATH = Path(os.environ.get("PEDE_MODEL_PATH", str(MODEL_DIR / "model.joblib")))
PREPROCESSOR_PATH = Path(os.environ.get("PEDE_PREPROCESSOR_PATH", str(MODEL_DIR / "preprocessor.joblib")))
METADATA_PATH = Path(os.environ.get("PEDE_METADATA_PATH", str(MODEL_DIR / "metadata.json")))

RISK_LABELS = ["Muito Alto", "Alto", "Médio", "Regular"]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PLANO_REFORCO_STORE = PROCESSED_DIR / "intervencoes_plano_reforco.csv"


# =========================
# APP
# =========================
TEMPLATES_DIR = str(REPO_ROOT / "templates")
STATIC_DIR = str(REPO_ROOT / "static")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = "super_chave_ultra_secreta_123"

# =========================
# TYPES
# =========================
@dataclass(frozen=True)
class AlunoRow:
    ra: str
    nome: str
    turma: str
    media: float
    media_classe: str
    risco: str


# =========================
# IO + NORMALIZATION
# =========================
# =========================
# IO + NORMALIZATION
# =========================


def _read_xlsx_sheet(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {xlsx_path}\n"
            f"Dica: coloque o XLSX em {BASE_DIR / 'data' / 'raw'} "
            f"ou defina PEDE_XLSX_PATH com o caminho completo."
        )
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine=None)
    except ValueError as exc:
        raise ValueError(f"Sheet inválida: {sheet_name}. Erro: {exc}") from exc
    if df.empty:
        raise ValueError(f"Sheet '{sheet_name}' está vazia.")
    return df


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Mantém compatibilidade com chamadas existentes no app.py,
    mas usa o coercion centralizado.
    """
    out = _coerce_numeric_all(df)

    # Respeita o parâmetro cols (não altera colunas não listadas)
    existing = [c for c in cols if c in out.columns]
    if not existing:
        return out

    # As colunas numéricas já foram convertidas; apenas garante dtype coerente
    for c in existing:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def _truthy(v: Any) -> bool:
    if v is None:
        return False
    s = str(v).strip().casefold()
    return s in {"1", "true", "sim", "yes", "y", "x", "ok"}


# =========================
# MODEL LOADING (OPTIONAL)
# =========================
def _load_model_bundle() -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]:
    """
    Retorna (preprocessor, model, metadata). Se não existir, devolve (None, None, {}).
    """
    metadata: Dict[str, Any] = {}
    if METADATA_PATH.exists():
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    if joblib is None:
        return None, None, metadata

    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        return None, None, metadata

    try:
        pre = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
        return pre, model, metadata
    except Exception:
        return None, None, metadata


def _predict_risk_with_model(
    df: pd.DataFrame,
    preprocessor: Any,
    model: Any,
    feature_cols: List[str],
) -> Tuple[pd.Series, pd.Series]:
    """
    Retorna (score, label). Score em [0, 1] se possível; label em {Muito Alto, Alto, Médio, Regular}.
    """
    x = df.reindex(columns=feature_cols)
    x_t = preprocessor.transform(x)

    # Classificação: tenta predict_proba; regressão: normaliza score
    score: Optional[pd.Series] = None
    try:
        proba = model.predict_proba(x_t)[:, 1]
        score = pd.Series(proba, index=df.index, dtype="float64")
    except Exception:
        yhat = model.predict(x_t)
        yhat_s = pd.Series(pd.to_numeric(yhat, errors="coerce"), index=df.index, dtype="float64")
        # Normaliza robusto para [0, 1]
        mn, mx = float(yhat_s.min(skipna=True)), float(yhat_s.max(skipna=True))
        if mx > mn:
            score = (yhat_s - mn) / (mx - mn)
        else:
            score = pd.Series(0.0, index=df.index, dtype="float64")

    s = score.fillna(0.0).clip(0.0, 1.0)

    def to_label(p: float) -> str:
        if p >= 0.85:
            return "Muito Alto"
        if p >= 0.70:
            return "Alto"
        if p >= 0.50:
            return "Médio"
        return "Regular"

    label = s.apply(to_label)
    return s, label


def _predict_risk_fallback(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Heurística (somente fallback) mais realista:
    - prioriza INDE 22 (0-10)
    - thresholds compatíveis com cenário educacional (6.0 é linha de corte comum)
    """
    out = df.copy()

    inde: pd.Series
    if "INDE 22" in out.columns:
        inde = pd.to_numeric(out["INDE 22"], errors="coerce")
    else:
        inde = pd.Series(index=out.index, dtype="float64")
        if {"Matem", "Portug", "Inglês"}.issubset(out.columns):
            inde = pd.concat(
                [
                    pd.to_numeric(out["Matem"], errors="coerce"),
                    pd.to_numeric(out["Portug"], errors="coerce"),
                    pd.to_numeric(out["Inglês"], errors="coerce"),
                ],
                axis=1,
            ).mean(axis=1, skipna=True)

    inde_f = inde.fillna(6.0).clip(0.0, 10.0)

    # Score de risco: nota baixa => risco alto (0..1)
    score = (1.0 - (inde_f / 10.0)).clip(0.0, 1.0)

    # Thresholds educativos (mais realistas que 0.70/0.85)
    # INDE < 4.0 => Muito Alto
    # INDE < 5.5 => Alto
    # INDE < 6.5 => Médio
    def to_label_from_inde(v: float) -> str:
        if v < 4.0:
            return "Muito Alto"
        if v < 5.5:
            return "Alto"
        if v < 6.5:
            return "Médio"
        return "Regular"

    label = inde_f.apply(to_label_from_inde)
    return score, label

# =========================
# BUSINESS LOGIC (DASH)
# =========================

def _media_class(media: float) -> str:
    if pd.isna(media):
        return "ok"
    if media < 6.0:
        return "bad"
    if media < 7.0:
        return "ok"
    return "good"


def _select_escolas(df: pd.DataFrame) -> List[str]:
    col = "Instituição de ensino"
    if col not in df.columns:
        return []

    series = df[col]
    if isinstance(series, pd.DataFrame):
        # Excel com colunas duplicadas (mesmo header) => pega a 1ª
        series = series.iloc[:, 0]

    escolas = series.dropna().astype(str).str.strip()
    escolas = escolas[escolas != ""].unique().tolist()
    escolas.sort()
    return escolas


def _apply_filters(df: pd.DataFrame, escola: Optional[str], q: Optional[str]) -> pd.DataFrame:
    out = df.copy()

    if escola and "Instituição de ensino" in out.columns:
        out = out[out["Instituição de ensino"].astype(str).str.strip() == escola]

    if q:
        qn = q.strip().casefold()
        if qn:
            mask = pd.Series(False, index=out.index)
            for c in ["Nome", "Turma", "Matem", "Portug", "Inglês"]:
                if c in out.columns:
                    mask = mask | out[c].astype(str).str.casefold().str.contains(qn, na=False)
            out = out[mask]

    return out


def _build_dashboard_context(df_raw: pd.DataFrame) -> Dict[str, Any]:
    df = _standardize_columns(df_raw)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = _coerce_numeric(df, ["Matem", "Portug", "Inglês", "INDE 22", "Pedra 20", "Pedra 21", "Pedra 22", "IEG", "IPS"])

    pre, model, metadata = _load_model_bundle()

    if pre is not None and model is not None:
        feature_cols = metadata.get("feature_columns")
        if not isinstance(feature_cols, list) or not feature_cols:
            # feature columns não declaradas => fallback
            score, risco = _predict_risk_fallback(df)
        else:
            score, risco = _predict_risk_with_model(df, pre, model, [str(c) for c in feature_cols])
    else:
        score, risco = _predict_risk_fallback(df)

    df = df.assign(_risk_score=score, _risk_label=risco)

    total = int(len(df))
    if total <= 0:
        return {
            "indicadores": {
                "alto_risco": {"total": 0, "percentual": 0},
                "medio_risco": {"total": 0, "percentual": 0},
                "regulares": {"total": 0, "percentual": 0},
                "total": 0,
            },
            "alunos_alto_risco": [],
            "disciplinas": {"matematica": 0, "portugues": 0, "ingles": 0},
            "tendencia": {"media": json.dumps([0.0, 0.0, 0.0]), "risco": json.dumps([0, 0, 0])},
            "acoes": {"plano_reforco": 0, "acompanhamento": 0, "reuniao_pais": 0},
            "proximos_prazos": [
                {"titulo": "Provas de Recuperação", "data": "A definir"},
                {"titulo": "Reuniões de Pais", "data": "A definir"},
                {"titulo": "Relatórios", "data": "A definir"},
            ],
            "alertas_count": 0,
            "media_geral": "0,0",
        }

    alto_mask = df["_risk_label"].isin(["Alto", "Muito Alto"])
    medio_mask = df["_risk_label"].isin(["Médio"])
    reg_mask = df["_risk_label"].isin(["Regular"])

    alto_total = int(alto_mask.sum())
    medio_total = int(medio_mask.sum())
    reg_total = int(reg_mask.sum())

    def pct(n: int) -> int:
        return int(round((n / total) * 100, 0)) if total else 0

    # Média geral (preferencialmente INDE 22)
    if "INDE 22" in df.columns and df["INDE 22"].notna().any():
        media_geral = float(df["INDE 22"].mean(skipna=True))
    else:
        media_disc = df.apply(_calc_media_disciplinas, axis=1)
        media_geral = float(pd.to_numeric(media_disc, errors="coerce").mean(skipna=True))

    media_geral_display = f"{media_geral:.1f}".replace(".", ",")

    # Tabela de alto risco
    table_df = df.loc[alto_mask].copy()
    table_df["_media"] = table_df.apply(_calc_media_disciplinas, axis=1)
    table_df["_media_classe"] = table_df["_media"].apply(_media_class)

    alunos_alto: List[AlunoRow] = []
    for _, r in table_df.sort_values(by="_risk_score", ascending=False).head(15).iterrows():
        ra = str(r.get("RA", "")).strip()
        nome = str(r.get("Nome", "")).strip()
        turma = str(r.get("Turma", "")).strip()
        media = float(r.get("_media")) if pd.notna(r.get("_media")) else float("nan")
        media_fmt = f"{media:.1f}".replace(".", ",") if pd.notna(media) else "-"
        alunos_alto.append(
            AlunoRow(
                ra=ra,
                nome=nome,
                turma=turma,
                media=float(media) if pd.notna(media) else float("nan"),
                media_classe=str(r.get("_media_classe", "ok")),
                risco=str(r.get("_risk_label", "Médio")),
            )
        )

    # Converte para formato do template
    alunos_alto_risco_payload: List[Dict[str, Any]] = []
    for a in alunos_alto:
        alunos_alto_risco_payload.append(
            {
                "ra": a.ra,
                "nome": a.nome,
                "turma": a.turma,
                "media": (f"{a.media:.1f}".replace(".", ",") if pd.notna(a.media) else "-"),
                "media_classe": a.media_classe,
                "risco": a.risco,
            }
        )

    # Filtrar por disciplina (nota < 6)
    def count_below(col: str) -> int:
        if col not in df.columns:
            return 0
        s = pd.to_numeric(df[col], errors="coerce")
        return int((s < 6.0).sum())

    disciplinas = {
        "matematica": count_below("Matem"),
        "portugues": count_below("Portug"),
        "ingles": count_below("Inglês"),
    }

    # Tendência (Pedra 20/21/22)
    pedra_cols = ["Pedra 20", "Pedra 21", "Pedra 22"]
    media_series: List[float] = []
    risco_series: List[int] = []

    for c in pedra_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            media_series.append(float(s.mean(skipna=True)) if s.notna().any() else 0.0)
            # Risco por "pedra": aproximação (pedra baixa => risco)
            risco_series.append(int((s <= 2).sum()))
        else:
            media_series.append(0.0)
            risco_series.append(0)

    tendencia = {
        "media": json.dumps([round(v, 2) for v in media_series]),
        "risco": json.dumps(risco_series),
    }

    # Ações recomendadas
    # 8.1 Plano de Reforço: alunos alto risco sem recuperação (Rec Av1..4 vazios)
    rec_cols = [c for c in ["Rec Av1", "Rec Av2", "Rec Av3", "Rec Av4"] if c in df.columns]
    if rec_cols:
        rec_any = df[rec_cols].astype(str).apply(lambda x: x.str.strip().replace({"nan": ""}), axis=0).agg(lambda col: col != "")
        # rec_any acima não é correto; vamos fazer linha a linha:
        rec_filled = df[rec_cols].astype(str).apply(lambda x: x.str.strip(), axis=1)
        rec_present = rec_filled.apply(lambda row: any((v and v.casefold() != "nan") for v in row.tolist()), axis=1)
        plano_reforco = int((alto_mask & (~rec_present)).sum())
    else:
        plano_reforco = 0

    # 8.2 Acompanhamento: "inativos" (critério OR)
    criterios: List[pd.Series] = []
    if "Ativo/ Inativo" in df.columns:
        criterios.append(df["Ativo/ Inativo"].astype(str).str.casefold().str.contains("inativ", na=False))
    if "IEG" in df.columns:
        criterios.append(pd.to_numeric(df["IEG"], errors="coerce") < 0.4)
    if "IPS" in df.columns:
        criterios.append(pd.to_numeric(df["IPS"], errors="coerce") < 0.4)
    if "Rec Psicologia" in df.columns:
        criterios.append(df["Rec Psicologia"].astype(str).str.strip().replace({"nan": ""}) != "")

    if criterios:
        acompanhamento = int(pd.concat(criterios, axis=1).any(axis=1).sum())
    else:
        acompanhamento = 0

    # 8.3 Reuniões de Pais: aluno alto risco e Indicado marcado
    if "Indicado" in df.columns:
        indicado_mask = df["Indicado"].apply(_truthy)
        reuniao_pais = int((alto_mask & indicado_mask).sum())
    else:
        reuniao_pais = 0

    acoes = {
        "plano_reforco": plano_reforco,
        "acompanhamento": acompanhamento,
        "reuniao_pais": reuniao_pais,
    }

    # Alertas prioritários (badge)
    alertas = int(alto_total)
    if "Indicado" in df.columns:
        alertas += int(df["Indicado"].apply(_truthy).sum())
    if "Rec Psicologia" in df.columns:
        alertas += int((df["Rec Psicologia"].astype(str).str.strip().replace({"nan": ""}) != "").sum())

    # Próximos prazos (config manual)
    proximos_prazos = [
        {"titulo": "Provas de Recuperação", "data": "A definir"},
        {"titulo": "Reuniões de Pais", "data": "A definir"},
        {"titulo": "Relatórios", "data": "A definir"},
    ]

    indicadores = {
        "alto_risco": {"total": alto_total, "percentual": pct(alto_total)},
        "medio_risco": {"total": medio_total, "percentual": pct(medio_total)},
        "regulares": {"total": reg_total, "percentual": pct(reg_total)},
        "total": total,
    }

    return {
        "indicadores": indicadores,
        "alunos_alto_risco": alunos_alto_risco_payload,
        "disciplinas": disciplinas,
        "tendencia": tendencia,
        "acoes": acoes,
        "proximos_prazos": proximos_prazos,
        "alertas_count": alertas,
        "media_geral": media_geral_display,
    }


# =========================
# ROUTES
# =========================
@app.get("/")
def root() -> Any:
    return redirect(url_for("dashboard"))


@app.get("/dashboard")
def dashboard() -> Any:
    sheet = request.args.get("sheet", DEFAULT_SHEET)
    if sheet not in AVAILABLE_SHEETS:
        sheet = DEFAULT_SHEET
    escola = request.args.get("escola")
    q = request.args.get("q")

    df_raw = _read_xlsx_sheet(DATA_XLSX_PATH, sheet)
    df_std = _standardize_columns(df_raw)
    df_std = df_std.loc[:, ~df_std.columns.duplicated()].copy()  # anti-duplicadas

    escolas = _select_escolas(df_std)
    # Se a escola do querystring não existir neste sheet, faz fallback
    if escola and escola != "Todas" and escola not in escolas:
        escola = "Todas"

    escola_nome = escola or (escolas[0] if escolas else "Todas")

    df_filtered = _apply_filters(df_std, escola if escola_nome != "Todas" else None, q)

    # Se mesmo assim não tiver dados, cai para "Todas" automaticamente
    if df_filtered.empty and escola_nome != "Todas":
        escola_nome = "Todas"
        df_filtered = _apply_filters(df_std, None, q)

    ctx = _build_dashboard_context(df_filtered)

    # User info (placeholder controlado por servidor)
    usuario_nome = "Prof. Ana"
    usuario_cargo = "Coordenadora Pedagógica"

    return render_template(
        "home.html",
        # base.html
        alertas_count=ctx["alertas_count"],
        escola_nome=escola_nome,
        usuario_nome=usuario_nome,
        usuario_cargo=usuario_cargo,
        # home.html
        indicadores=ctx["indicadores"],
        media_geral=ctx["media_geral"],
        alunos_alto_risco=ctx["alunos_alto_risco"],
        disciplinas=ctx["disciplinas"],
        tendencia=ctx["tendencia"],
        acoes=ctx["acoes"],
        proximos_prazos=ctx["proximos_prazos"],
        # extras
        sheet=sheet,
        available_sheets=AVAILABLE_SHEETS,
        ano_base="".join([c for c in sheet if c.isdigit()]) or "base",
        ano_previsto=_next_year_from_sheet(sheet),
        escolas=escolas,
        q=q or "",
    )

@app.post("/predict")
def predict() -> Response:
    """
    Endpoint oficial do Datathon.

    Input:
      - JSON dict (1 aluno) ou JSON list[dict] (batch)
      - Opcional: sheet (querystring) para enriquecer dados via Excel quando só vier RA

    Output:
      - predictions: lista com {ra, risk_score, risk_label}
      - model_version (se metadata existir)
      - used_model: bool
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "JSON inválido ou ausente."}), 400

    # Aceita 1 registro ou batch
    if isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
        rows = payload
    else:
        return jsonify({"error": "Formato inválido. Envie dict ou list[dict]."}), 400

    sheet = request.args.get("sheet", DEFAULT_SHEET)
    if sheet not in AVAILABLE_SHEETS:
        sheet = DEFAULT_SHEET

    df_in = pd.DataFrame(rows, dtype=object)
    df_in = _standardize_columns(df_in)
    df_in = df_in.loc[:, ~df_in.columns.duplicated()].copy()
    df_in = _coerce_numeric_all(df_in)

    # Se vier somente RA, tenta enriquecer buscando no XLSX (best-effort)
    if "RA" in df_in.columns:
        ra_series = df_in["RA"].astype(str).str.strip()
        ra_series = ra_series[ra_series != ""]
        if not ra_series.empty:
            try:
                df_base_raw = _read_xlsx_sheet(DATA_XLSX_PATH, sheet)
                df_base = _standardize_columns(df_base_raw)
                df_base = df_base.loc[:, ~df_base.columns.duplicated()].copy()
                df_base["RA"] = df_base["RA"].astype(str).str.strip()

                df_lookup = df_base[df_base["RA"].isin(ra_series.tolist())].copy()
                if not df_lookup.empty:
                    # Merge mantendo prioridade para valores do input quando existirem
                    df_lookup = df_lookup.set_index("RA")
                    df_in_idx = df_in.copy()
                    df_in_idx["RA"] = df_in_idx["RA"].astype(str).str.strip()
                    df_in_idx = df_in_idx.set_index("RA")

                    merged = df_lookup.join(df_in_idx, how="left", rsuffix="_in")
                    for c in list(df_in_idx.columns):
                        c_in = f"{c}_in"
                        if c_in in merged.columns:
                            merged[c] = merged[c_in].where(
                                merged[c_in].notna() & (merged[c_in].astype(str).str.strip() != ""),
                                merged[c],
                            )
                            merged = merged.drop(columns=[c_in])
                    df_in = merged.reset_index()
            except Exception:
                # best-effort: se falhar, segue com o que veio no request
                pass

    pre, model, metadata = _load_model_bundle()
    feature_cols = metadata.get("feature_columns") if isinstance(metadata, dict) else None

    used_model = bool(pre is not None and model is not None and isinstance(feature_cols, list) and len(feature_cols) > 0)

    if used_model:
        score, label = _predict_risk_with_model(df_in, pre, model, [str(c) for c in feature_cols])
    else:
        score, label = _predict_risk_fallback(df_in)

    preds = []
    for i in df_in.index:
        ra = ""
        if "RA" in df_in.columns:
            ra = str(df_in.loc[i, "RA"]).strip()
        preds.append(
            {
                "ra": ra,
                "risk_score": float(score.loc[i]) if i in score.index else 0.0,
                "risk_label": str(label.loc[i]) if i in label.index else "Regular",
            }
        )

    model_version = None
    if isinstance(metadata, dict):
        mv = metadata.get("model_version")
        if isinstance(mv, str) and mv.strip():
            model_version = mv.strip()

    return jsonify(
        {
            "model_version": model_version,
            "used_model": used_model,
            "predictions": preds,
        }
    ), 200


def _load_df_with_risk(sheet: str, escola: Optional[str], q: Optional[str]) -> pd.DataFrame:
    df_raw = _read_xlsx_sheet(DATA_XLSX_PATH, sheet)
    df_std = _standardize_columns(df_raw)
    df_std = df_std.loc[:, ~df_std.columns.duplicated()].copy()

    df_filtered = _apply_filters(df_std, escola if escola and escola != "Todas" else None, q)

    df = _standardize_columns(df_filtered)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = _coerce_numeric(df, ["Matem", "Portug", "Inglês", "INDE 22", "IEG", "IPS"])

    pre, model, metadata = _load_model_bundle()
    if pre is not None and model is not None:
        feature_cols = metadata.get("feature_columns")
        if isinstance(feature_cols, list) and feature_cols:
            score, risco = _predict_risk_with_model(df, pre, model, [str(c) for c in feature_cols])
        else:
            score, risco = _predict_risk_fallback(df)
    else:
        score, risco = _predict_risk_fallback(df)

    df = df.assign(_risk_score=score, _risk_label=risco)
    df["_media"] = df.apply(_calc_media_disciplinas, axis=1)
    df["_media_classe"] = df["_media"].apply(_media_class)
    return df

@app.route("/intervencoes/plano-reforco", methods=["GET", "POST"])
def intervencao_plano_reforco() -> Any:
    msg = None 


    sheet = request.args.get("sheet", DEFAULT_SHEET)
    if sheet not in AVAILABLE_SHEETS:
        sheet = DEFAULT_SHEET

    escola = request.args.get("escola")
    q = request.args.get("q")

    df = _load_df_with_risk(sheet, escola, q)

    # 1) Critério de risco: Alto OU Muito Alto
    alto_mask = df["_risk_label"].isin(["Alto", "Muito Alto"])

    # 2) Critério "sem recuperação": se não existir Rec Av*, considera todos como sem rec
    rec_cols = [c for c in ["Rec Av1", "Rec Av2", "Rec Av3", "Rec Av4"] if c in df.columns]
    if rec_cols:
        rec_df = df[rec_cols].astype("string")
        rec_df = rec_df.apply(lambda s: s.str.strip(), axis=0).fillna("")
        rec_df = rec_df.replace({"nan": "", "NaN": "", "None": ""})
        rec_present = rec_df.ne("").any(axis=1)
    else:
        rec_present = pd.Series(False, index=df.index, dtype="bool")

    # 3) Lê confirmados e remove da lista pendente
    store = _read_plano_reforco_store()
    confirmados_ra = set(store[store["sheet"].astype(str) == sheet]["ra"].astype(str).str.strip().tolist())

    # pendentes = alto/muito alto + sem rec + ainda não confirmado
    pendentes_df = df.loc[alto_mask & (~rec_present)].copy()
    pendentes_df["RA"] = pendentes_df.get("RA", "").astype(str).str.strip()
    pendentes_df = pendentes_df[~pendentes_df["RA"].isin(confirmados_ra)]
    pendentes_df = pendentes_df.sort_values(by="_risk_score", ascending=False, na_position="last")

    pendentes: List[Dict[str, Any]] = []
    for _, r in pendentes_df.iterrows():
        pendentes.append(
            {
                "ra": str(r.get("RA", "")).strip(),
                "nome": str(r.get("Nome", "")).strip(),
                "turma": str(r.get("Turma", "")).strip(),
                "media": (f"{float(r.get('_media')):.1f}".replace(".", ",") if pd.notna(r.get("_media")) else "-"),
                "media_classe": str(r.get("_media_classe", "ok")),
                "risco": str(r.get("_risk_label", "Regular")),
            }
        )

    # confirmados (para o mesmo sheet)
    confirmados_df = store[store["sheet"].astype(str) == sheet].copy()
    confirmados: List[Dict[str, Any]] = confirmados_df.to_dict(orient="records")

    if request.method == "POST":
        selecionados = request.form.getlist("ra")
        disciplina = (request.form.get("disciplina") or "").strip()
        observacao = (request.form.get("observacao") or "").strip()
        data = (request.form.get("data") or "").strip()

        # grava somente os selecionados
        selected_rows: List[Dict[str, str]] = []
        pend_map = {p["ra"]: p for p in pendentes}

        for ra in selecionados:
            ra_s = str(ra).strip()
            p = pend_map.get(ra_s)
            if not p:
                continue
            selected_rows.append(
                {
                    "sheet": sheet,
                    "ra": ra_s,
                    "nome": str(p["nome"]),
                    "turma": str(p["turma"]),
                    "data": data,
                    "disciplina": disciplina,
                    "observacao": observacao,
                }
            )

        msg: str
        if selected_rows:
            _append_plano_reforco_rows(selected_rows)
            msg = f"Plano de reforço confirmado para {len(selected_rows)} aluno(s)."
        else:
            msg = "Nenhum aluno selecionado para confirmação."

    return render_template(
        "intervencao_plano_reforco.html",
        pendentes=pendentes,
        confirmados=_read_plano_reforco_store().query("sheet == @sheet").to_dict(orient="records"),
        msg=msg,
        sheet=sheet,
        available_sheets=AVAILABLE_SHEETS,
        escola_nome=escola or "Todas",
        alertas_count=int(alto_mask.sum()),
        usuario_nome="Prof. Ana",
        usuario_cargo="Coordenadora Pedagógica",
    )

@app.route("/intervencoes/acompanhamento", methods=["GET", "POST"])
def intervencao_acompanhamento() -> Any:
    sheet = request.args.get("sheet", DEFAULT_SHEET)
    if sheet not in AVAILABLE_SHEETS:
        sheet = DEFAULT_SHEET

    escola = request.args.get("escola")
    q = request.args.get("q")

    df = _load_df_with_risk(sheet, escola, q)
    df = standardize_columns(df)   # ← mantém a normalização

    # NOVA REGRA: somente alunos em risco MUITO ALTO
    mask = df["_risk_label"] == "Muito Alto"
    alvo = df[mask].copy().sort_values(by="_risk_score", ascending=False)

    alunos = [
        {
            "ra": str(r.get("RA", "")).strip(),
            "nome": str(r.get("Nome", "")).strip(),
            "turma": str(r.get("Turma", "")).strip(),
            "media": (f"{float(r.get('_media')):.1f}".replace(".", ",") if pd.notna(r.get("_media")) else "-"),
            "media_classe": str(r.get("_media_classe", "ok")),
            "risco": str(r.get("_risk_label", "Regular")),
        }
        for _, r in alvo.iterrows()
    ]

    if request.method == "POST":
        flash(f"Alertas de acompanhamento enviados para {len(alunos)} alunos (simulação).")
        return redirect(url_for("intervencao_acompanhamento", sheet=sheet))

    return render_template(
        "intervencao_acompanhamento.html",
        alunos=alunos,
        sheet=sheet,
        available_sheets=AVAILABLE_SHEETS,
        escola_nome=escola or "Todas",
        alertas_count=int((df["_risk_label"] == "Muito Alto").sum()),  # ← CORRIGIDO
        usuario_nome="Prof. Ana",
        usuario_cargo="Coordenadora Pedagógica",
    )


@app.route("/intervencoes/reuniao-pais", methods=["GET", "POST"])
def intervencao_reuniao_pais() -> Any:
    sheet = request.args.get("sheet", DEFAULT_SHEET)
    if sheet not in AVAILABLE_SHEETS:
        sheet = DEFAULT_SHEET

    escola = request.args.get("escola")
    q = request.args.get("q")

    df = _load_df_with_risk(sheet, escola, q)

    indicado_mask = df["Indicado"].apply(_truthy) if "Indicado" in df.columns else pd.Series(False, index=df.index)
    alto_mask = df["_risk_label"].isin(["Alto", "Muito Alto"])

    alvo = df[alto_mask & indicado_mask].copy().sort_values(by="_risk_score", ascending=False)

    alunos = [
        {
            "ra": str(r.get("RA", "")).strip(),
            "nome": str(r.get("Nome", "")).strip(),
            "turma": str(r.get("Turma", "")).strip(),
            "media": (f"{float(r.get('_media')):.1f}".replace(".", ",") if pd.notna(r.get("_media")) else "-"),
            "media_classe": str(r.get("_media_classe", "ok")),
            "risco": str(r.get("_risk_label", "Regular")),
        }
        for _, r in alvo.iterrows()
    ]

    if request.method == "POST":
        data = (request.form.get("data") or "").strip()
        flash(f"Reuniões agendadas para {len(alunos)} alunos em '{data or 'data não informada'}' (simulação).")
        return redirect(url_for("intervencao_reuniao_pais", sheet=sheet))

    return render_template(
        "intervencao_reuniao_pais.html",
        alunos=alunos,
        sheet=sheet,
        available_sheets=AVAILABLE_SHEETS,
        escola_nome=escola or "Todas",
        alertas_count=int((df["_risk_label"].isin(["Alto", "Muito Alto"])).sum()),
        usuario_nome="Prof. Ana",
        usuario_cargo="Coordenadora Pedagógica",
    )

@app.get("/export")
def export_relatorio() -> Response:
    sheet = request.args.get("sheet", DEFAULT_SHEET)
    if sheet not in AVAILABLE_SHEETS:
        sheet = DEFAULT_SHEET

    escola = request.args.get("escola") or "Todas"
    q = request.args.get("q") or ""

    df_raw = _read_xlsx_sheet(DATA_XLSX_PATH, sheet)
    df_std = _standardize_columns(df_raw)
    df_std = df_std.loc[:, ~df_std.columns.duplicated()].copy()

    df_filtered = _apply_filters(df_std, None if escola == "Todas" else escola, q)

    df = _standardize_columns(df_filtered)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = _coerce_numeric(df, ["Matem", "Portug", "Inglês", "INDE 22"])

    pre, model, metadata = _load_model_bundle()
    if pre is not None and model is not None:
        feature_cols = metadata.get("feature_columns")
        if isinstance(feature_cols, list) and feature_cols:
            score, risco = _predict_risk_with_model(df, pre, model, [str(c) for c in feature_cols])
        else:
            score, risco = _predict_risk_fallback(df)
    else:
        score, risco = _predict_risk_fallback(df)

    df = df.assign(_risk_score=score, _risk_label=risco)
    alto_mask = df["_risk_label"].isin(["Alto", "Muito Alto"])

    df_out = df.loc[alto_mask].copy()
    df_out["_media"] = df_out.apply(_calc_media_disciplinas, axis=1)

    export_df = pd.DataFrame(
        {
            "RA": df_out.get("RA", pd.Series("", index=df_out.index)).astype(str).str.strip(),
            "Aluno": df_out.get("Nome", pd.Series("", index=df_out.index)).astype(str).str.strip(),
            "Turma": df_out.get("Turma", pd.Series("", index=df_out.index)).astype(str).str.strip(),
            "Média": pd.to_numeric(df_out["_media"], errors="coerce").round(1),
            "Risco": df_out["_risk_label"].astype(str),
        }
    )

    # ordena por maior risco (score) e depois por média menor
    export_df = export_df.sort_values(by=["Risco", "Média"], ascending=[True, True], na_position="last")

    csv_text = export_df.to_csv(index=False, sep=";", lineterminator="\n")
    csv_bytes = ("\ufeff" + csv_text).encode("utf-8")  # BOM p/ Excel

    filename = f"relatorio_alto_risco_{sheet}.csv"
    return Response(
        csv_bytes,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

@app.get("/aluno/<ra>")
def aluno_detalhe(ra: str) -> Any:
    sheet = request.args.get("sheet", DEFAULT_SHEET)

    df_raw = _read_xlsx_sheet(DATA_XLSX_PATH, sheet)
    df = _standardize_columns(df_raw)
    df = _coerce_numeric(
        df,
        [
            "Matem",
            "Portug",
            "Inglês",
            "INDE 22",
            "Pedra 20",
            "Pedra 21",
            "Pedra 22",
            "IEG",
            "IPS",
            "IAA",
            "IDA",
            "IPV",
            "IAN",
        ],
    )

    ra_norm = str(ra).strip()
    row_df = df[df["RA"].astype(str).str.strip() == ra_norm]
    if row_df.empty:
        return abort(404, description=f"Aluno não encontrado. RA={ra_norm}")

    # Garantia: pega exatamente 1 linha
    row = row_df.iloc[0]

    # Badge e escola (para o base.html)
    escola_nome = str(row.get("Instituição de ensino", "Todas")).strip() or "Todas"

    # Predição (modelo se existir, senão fallback)
    pre, model, metadata = _load_model_bundle()
    if pre is not None and model is not None:
        feature_cols = metadata.get("feature_columns")
        if isinstance(feature_cols, list) and feature_cols:
            score_s, label_s = _predict_risk_with_model(row_df, pre, model, [str(c) for c in feature_cols])
        else:
            score_s, label_s = _predict_risk_fallback(row_df)
    else:
        score_s, label_s = _predict_risk_fallback(row_df)

    risk_score = float(score_s.iloc[0]) if not score_s.empty else 0.0
    risco = str(label_s.iloc[0]) if not label_s.empty else "Regular"

    # Média disciplinas
    media = _calc_media_disciplinas(row)
    media_fmt = f"{media:.1f}".replace(".", ",") if pd.notna(media) else "-"

    # Campos exibidos
    aluno_payload = {
        "ra": ra_norm,
        "nome": str(row.get("Nome", "")).strip(),
        "turma": str(row.get("Turma", "")).strip(),
        "risk_score": f"{risk_score:.2f}".replace(".", ","),
        "risco": risco,
        "media": media_fmt,
        "inde_22": (f"{float(row.get('INDE 22')):.1f}".replace(".", ",") if pd.notna(row.get("INDE 22")) else "-"),
        "ieg": (f"{float(row.get('IEG')):.2f}".replace(".", ",") if pd.notna(row.get("IEG")) else "-"),
        "matem": (f"{float(row.get('Matem')):.1f}".replace(".", ",") if pd.notna(row.get("Matem")) else "-"),
        "portug": (f"{float(row.get('Portug')):.1f}".replace(".", ",") if pd.notna(row.get("Portug")) else "-"),
        "ingles": (f"{float(row.get('Inglês')):.1f}".replace(".", ",") if pd.notna(row.get("Inglês")) else "-"),
        "iaa": (f"{float(row.get('IAA')):.2f}".replace(".", ",") if pd.notna(row.get("IAA")) else "-"),
        "ips": (f"{float(row.get('IPS')):.2f}".replace(".", ",") if pd.notna(row.get("IPS")) else "-"),
        "ida": (f"{float(row.get('IDA')):.2f}".replace(".", ",") if pd.notna(row.get("IDA")) else "-"),
        "ipv": (f"{float(row.get('IPV')):.2f}".replace(".", ",") if pd.notna(row.get("IPV")) else "-"),
        "ian": (f"{float(row.get('IAN')):.2f}".replace(".", ",") if pd.notna(row.get("IAN")) else "-"),
        "pedra_20": (str(row.get("Pedra 20")) if pd.notna(row.get("Pedra 20")) else "-"),
        "pedra_21": (str(row.get("Pedra 21")) if pd.notna(row.get("Pedra 21")) else "-"),
        "pedra_22": (str(row.get("Pedra 22")) if pd.notna(row.get("Pedra 22")) else "-"),
        "rec_psicologia": (str(row.get("Rec Psicologia", "")).strip() or "-"),
        "indicado": ("Sim" if _truthy(row.get("Indicado")) else "Não"),
        "rec_av1": (str(row.get("Rec Av1", "")).strip() or "-"),
        "rec_av2": (str(row.get("Rec Av2", "")).strip() or "-"),
        "rec_av3": (str(row.get("Rec Av3", "")).strip() or "-"),
        "rec_av4": (str(row.get("Rec Av4", "")).strip() or "-"),
    }

    # Recomendações (texto idêntico à lógica do dashboard)
    # Plano de Reforço: sem recuperação preenchida
    rec_vals = [
        str(row.get("Rec Av1", "")).strip(),
        str(row.get("Rec Av2", "")).strip(),
        str(row.get("Rec Av3", "")).strip(),
        str(row.get("Rec Av4", "")).strip(),
    ]
    sem_recuperacao = not any(v and v.casefold() != "nan" for v in rec_vals)

    # Acompanhamento: critérios OR
    acomp_flags = []
    if pd.notna(row.get("IEG")):
        acomp_flags.append(float(row.get("IEG")) < 0.4)
    if pd.notna(row.get("IPS")):
        acomp_flags.append(float(row.get("IPS")) < 0.4)
    rec_psico = str(row.get("Rec Psicologia", "")).strip()
    acomp_flags.append(bool(rec_psico and rec_psico.casefold() != "nan"))

    # Reuniões de Pais: alto risco + indicado
    reuniao_pais = risco in {"Alto", "Muito Alto"} and _truthy(row.get("Indicado"))

    recomendacoes = {
        "plano_reforco": "Aluno sem recuperação" if sem_recuperacao else "Aluno com recuperação registrada",
        "acompanhamento": "Sinal para acompanhamento" if any(acomp_flags) else "Sem sinal crítico de acompanhamento",
        "reunioes_pais": "Aluno crítico indicado" if reuniao_pais else "Sem critério para reunião de pais",
    }

    # Badge: reaproveita cálculo do dashboard no DF inteiro (coerente com topo)
    ctx_badge = _build_dashboard_context(df)
    alertas_count = int(ctx_badge.get("alertas_count", 0))

    # User info (placeholder controlado por servidor)
    usuario_nome = "Prof. Ana"
    usuario_cargo = "Coordenadora Pedagógica"

    return render_template(
        "aluno_detalhe.html",
        alertas_count=alertas_count,
        escola_nome=escola_nome,
        usuario_nome=usuario_nome,
        usuario_cargo=usuario_cargo,
        aluno=aluno_payload,
        recomendacoes=recomendacoes,
    )


@app.get("/alunos-risco")
def alunos_risco() -> Any:
    sheet = request.args.get("sheet", DEFAULT_SHEET)
    if sheet not in AVAILABLE_SHEETS:
        sheet = DEFAULT_SHEET

    escola = request.args.get("escola")
    q = request.args.get("q")

    df_raw = _read_xlsx_sheet(DATA_XLSX_PATH, sheet)
    df_std = _standardize_columns(df_raw)
    df_std = df_std.loc[:, ~df_std.columns.duplicated()].copy()

    escolas = _select_escolas(df_std)
    if escola and escola != "Todas" and escola not in escolas:
        escola = "Todas"
    escola_nome = escola or (escolas[0] if escolas else "Todas")

    df_filtered = _apply_filters(df_std, escola if escola_nome != "Todas" else None, q)

    # Se ficar vazio por filtro, cai pra "Todas"
    if df_filtered.empty and escola_nome != "Todas":
        escola_nome = "Todas"
        df_filtered = _apply_filters(df_std, None, q)

    # gera risco e média por aluno
    df = _standardize_columns(df_filtered)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = _coerce_numeric(df, ["Matem", "Portug", "Inglês", "INDE 22"])

    pre, model, metadata = _load_model_bundle()
    if pre is not None and model is not None:
        feature_cols = metadata.get("feature_columns")
        if isinstance(feature_cols, list) and feature_cols:
            score, risco = _predict_risk_with_model(df, pre, model, [str(c) for c in feature_cols])
        else:
            score, risco = _predict_risk_fallback(df)
    else:
        score, risco = _predict_risk_fallback(df)

    df = df.assign(_risk_score=score, _risk_label=risco)
    df["_media"] = df.apply(_calc_media_disciplinas, axis=1)
    df["_media_classe"] = df["_media"].apply(_media_class)

    rows: List[Dict[str, Any]] = []
    for _, r in df.sort_values(by="_risk_score", ascending=False).iterrows():
        ra = str(r.get("RA", "")).strip()
        rows.append(
            {
                "ra": ra,
                "nome": str(r.get("Nome", "")).strip(),
                "turma": str(r.get("Turma", "")).strip(),
                "media": (f"{float(r.get('_media')):.1f}".replace(".", ",") if pd.notna(r.get("_media")) else "-"),
                "media_classe": str(r.get("_media_classe", "ok")),
                "risco": str(r.get("_risk_label", "Regular")),
            }
        )

    # badge / usuário
    ctx_badge = _build_dashboard_context(df_filtered)
    alertas_count = int(ctx_badge.get("alertas_count", 0))
    usuario_nome = "Prof. Ana"
    usuario_cargo = "Coordenadora Pedagógica"

    return render_template(
        "alunos_risco.html",
        alertas_count=alertas_count,
        escola_nome=escola_nome,
        usuario_nome=usuario_nome,
        usuario_cargo=usuario_cargo,
        alunos=rows,
        sheet=sheet,
        available_sheets=AVAILABLE_SHEETS,
        q=q or "",
        escolas=escolas,
    )

@app.route("/api/tendencia", methods=["GET"])
def api_tendencia() -> dict:
    sheet = request.args.get("sheet", DEFAULT_SHEET)
    if sheet not in AVAILABLE_SHEETS:
        sheet = DEFAULT_SHEET

    df_raw = _read_xlsx_sheet(DATA_XLSX_PATH, sheet)
    df_std = _standardize_columns(df_raw)
    df_std = df_std.loc[:, ~df_std.columns.duplicated()].copy()
    df = _coerce_numeric_all(df_std)

    tendencia_media, tendencia_risco = calcular_tendencia(df)

    return {
        "labels": ["Pedra 20", "Pedra 21", "Pedra 22"],
        "media": tendencia_media,
        "risco": tendencia_risco,
    }

def _read_plano_reforco_store() -> pd.DataFrame:
    if not PLANO_REFORCO_STORE.exists():
        return pd.DataFrame(columns=["sheet", "ra", "nome", "turma", "data", "disciplina", "observacao"])
    try:
        return pd.read_csv(PLANO_REFORCO_STORE, sep=";", dtype=str).fillna("")
    except Exception:
        return pd.DataFrame(columns=["sheet", "ra", "nome", "turma", "data", "disciplina", "observacao"])


def _append_plano_reforco_rows(rows: List[Dict[str, str]]) -> None:
    df_old = _read_plano_reforco_store()
    df_new = pd.DataFrame(rows, dtype=str).fillna("")
    out = pd.concat([df_old, df_new], ignore_index=True)

    # dedup por sheet+ra (mantém o primeiro registro)
    out["_k"] = out["sheet"].astype(str) + "::" + out["ra"].astype(str)
    out = out.drop_duplicates(subset=["_k"], keep="first").drop(columns=["_k"])

    out.to_csv(PLANO_REFORCO_STORE, index=False, sep=";", encoding="utf-8-sig")

@app.get("/alertas")
def alertas() -> Any:
    return redirect(url_for("dashboard"))


@app.get("/turmas")
def turmas() -> Any:
    return redirect(url_for("dashboard"))


@app.get("/disciplinas")
def disciplinas() -> Any:
    return redirect(url_for("dashboard"))


@app.get("/evolucao")
def evolucao() -> Any:
    return redirect(url_for("dashboard"))


@app.get("/comparar-turmas")
def comparar_turmas() -> Any:
    return redirect(url_for("dashboard"))


@app.get("/relatorios")
def relatorios() -> Any:
    return redirect(url_for("dashboard"))


@app.get("/intervencoes")
def intervencoes() -> Any:
    return redirect(url_for("dashboard"))


@app.get("/configuracoes")
def configuracoes() -> Any:
    return redirect(url_for("dashboard"))


@app.get("/suporte")
def suporte() -> Any:
    return redirect(url_for("dashboard"))


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Execução local:
    #   set PEDE_XLSX_PATH="caminho/para/BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
    #   python app.py
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)