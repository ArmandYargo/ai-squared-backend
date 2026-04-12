from pathlib import Path
from typing import Any

import yaml

from src.dem_ai_assistant.core.models import (
    GeometryConfig,
    MaterialConfig,
    ScenarioConfig,
)


def load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty: {path}")

    return data


def load_yaml_config(path: Path) -> MaterialConfig | ScenarioConfig | GeometryConfig:
    data = load_yaml_file(path)

    if "psd_bands" in data:
        return MaterialConfig(**data)

    if "duration_s" in data and "feed_profile" in data:
        return ScenarioConfig(**data)

    if "transfer_point_name" in data and "belt" in data:
        return GeometryConfig(**data)

    raise ValueError(f"Unrecognized config schema: {path}")