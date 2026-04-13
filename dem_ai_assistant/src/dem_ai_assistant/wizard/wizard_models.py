"""
Structured inputs for the DEM input wizard.
NSC contact is executed today; SMC selection is stored for future use.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel


class PSDWizardBand(BaseModel):
    size_min_mm: float
    size_max_mm: float
    mass_fraction: float


class WizardParticleSection(BaseModel):
    psd_bands: List[PSDWizardBand]
    particle_shape: Literal["sphere"] = "sphere"
    particle_density_kg_m3: float = 2500.0
    use_particle_count: bool = True
    particle_count: int = 200
    total_particle_mass_kg: float = 50.0
    spawn_x_min_m: float = -0.4
    spawn_x_max_m: float = 0.4
    spawn_y_min_m: float = 0.15
    spawn_y_max_m: float = 0.55
    spawn_z_min_m: float = -0.15
    spawn_z_max_m: float = 0.15


class WizardMaterialSection(BaseModel):
    young_modulus_pa: float = 5.0e8
    poisson_ratio: float = 0.3
    friction_static: float = 0.45
    friction_rolling: float = 0.08
    restitution: float = 0.25
    cohesion_pa: float = 0.0


class WizardGeometrySection(BaseModel):
    cad_mesh_path: str
    collision_mesh_path: Optional[str] = None
    mesh_scale: float = 1.0
    mesh_offset_x_m: float = 0.0
    mesh_offset_y_m: float = 0.0
    mesh_offset_z_m: float = 0.0


class WizardGravitySection(BaseModel):
    gx: float = 0.0
    gy: float = -9.81
    gz: float = 0.0


class WizardSolverSection(BaseModel):
    timestep_s: float = 0.001
    end_time_s: float = 2.0
    output_every_n_steps: int = 50
    collision_envelope_m: float = 0.002
    collision_margin_m: float = 0.002
    contact_model: Literal["NSC", "SMC"] = "NSC"


class WizardEgressSection(BaseModel):
    """Egress when |horizontal_axis| exceeds limit (default: x-axis, belt cross-wise)."""

    axis: Literal["x", "y", "z"] = "x"
    limit_abs_m: float = 0.55


class WizardVisualizationSection(BaseModel):
    use_irrlicht: bool = True
    window_width: int = 1024
    window_height: int = 768


class WizardDEMConfig(BaseModel):
    """Full wizard payload saved next to run outputs."""

    particle: WizardParticleSection
    material: WizardMaterialSection
    geometry: WizardGeometrySection
    gravity: WizardGravitySection
    solver: WizardSolverSection
    egress: WizardEgressSection
    visualization: WizardVisualizationSection

    run_name: str = "wizard_run"
