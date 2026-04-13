from pathlib import Path

from src.dem_ai_assistant.core.config_loader import load_yaml_file
from src.dem_ai_assistant.core.study_models import DesignStudyConfig


def load_design_study_config(path: Path) -> DesignStudyConfig:
    data = load_yaml_file(path)
    return DesignStudyConfig(**data)
