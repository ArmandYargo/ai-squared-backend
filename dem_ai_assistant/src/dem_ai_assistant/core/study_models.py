from pydantic import BaseModel, Field


class DesignStudyConfig(BaseModel):
    """
    Defines a bounded design study:
    one material, multiple scenarios, and multiple geometry options.
    """

    study_name: str
    material_file: str
    scenario_files: list[str] = Field(min_length=1)
    design_option_files: list[str] = Field(min_length=1)
    baseline_option_file: str
    # "placeholder" (default) or "pychrono" when implemented
    solver_name: str = "placeholder"
