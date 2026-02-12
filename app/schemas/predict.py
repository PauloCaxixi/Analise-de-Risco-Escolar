from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Dicionário de features (pode enviar subset; o servidor completa faltantes com null).",
        examples=[{"IDADE_ALUNO_2020": 12, "IAN_2022": 7.5, "IDA_2022": 6.0, "IEG_2022": 8.0}],
    )


class PredictResponse(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Probabilidade de risco (0 a 1).")
    risk_label: str = Field(..., description="LOW | MEDIUM | HIGH")
    model_version: str = Field(..., description="Versão do modelo em produção.")
    request_id: str = Field(..., description="ID único da requisição para auditoria/logs.")
