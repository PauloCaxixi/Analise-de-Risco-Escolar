from typing import Dict, Any


def validate_features(features: Dict[str, Any]) -> None:
    if not features:
        raise ValueError("Payload de features vazio")

    if not isinstance(features, dict):
        raise TypeError("Features devem ser um dicionário")
