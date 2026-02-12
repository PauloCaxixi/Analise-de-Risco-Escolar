from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VersionResponse(BaseModel):
    api_version: str = Field(..., description="Versão da API.")
    model_version: str = Field(..., description="Versão do modelo.")
    test_year: Optional[str] = Field(None, description="Ano/aba usada como validação final.")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Métricas do modelo.")
    dropped_all_null_columns: List[str] = Field(default_factory=list, description="Colunas removidas por serem 100% nulas.")
