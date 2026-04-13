"""
Console + tkinter file dialogs (works in Spyder on Windows).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import List, Optional

from src.dem_ai_assistant.wizard.wizard_models import (
    PSDWizardBand,
    WizardDEMConfig,
    WizardEgressSection,
    WizardGeometrySection,
    WizardGravitySection,
    WizardMaterialSection,
    WizardParticleSection,
    WizardSolverSection,
    WizardVisualizationSection,
)


def _pick_file(title: str, patterns: tuple) -> Optional[Path]:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        filetypes=list(patterns) + [("All files", "*.*")],
    )
    root.destroy()
    if not path:
        return None
    return Path(path)


def _ask(prompt: str, default: str) -> str:
    s = input(f"{prompt} [{default}]: ").strip()
    return s if s else default


def _ask_float(prompt: str, default: float) -> float:
    s = _ask(prompt, str(default))
    return float(s)


def _ask_int(prompt: str, default: int) -> int:
    s = _ask(prompt, str(default))
    return int(float(s))


def _ask_bool(prompt: str, default: bool) -> bool:
    d = "y" if default else "n"
    s = _ask(f"{prompt} (y/n)", d).lower()
    return s in ("y", "yes", "1", "true")


def collect_wizard_interactive() -> WizardDEMConfig:
    print("\n" + "=" * 70)
    print("DEM AI ASSISTANT — INPUT WIZARD (PyChrono)")
    print("Workflow: CAD mesh → (optional) simplified collision mesh → particles →")
    print("material → gravity → solver → run → report (+ optional Irrlicht).")
    print("=" * 70 + "\n")

    print("--- 1) GEOMETRY / CAD ---")
    print("Open file dialog for main chute / boundary mesh (.obj or .stl, metres).")
    cad = _pick_file(
        "Select CAD / boundary mesh",
        (("Mesh files", "*.obj *.stl"),),
    )
    if cad is None:
        raise SystemExit("No mesh selected. Cancelled.")
    simp = _ask_bool("Use a separate simplified collision mesh? (else same file)", False)
    collision_path: Optional[Path] = None
    if simp:
        print("Select simplified collision mesh…")
        collision_path = _pick_file(
            "Select simplified collision mesh",
            (("Mesh files", "*.obj *.stl"),),
        )
        if collision_path is None:
            print("No file chosen — using main CAD for collision.")
    mesh_scale = _ask_float("Uniform mesh scale (1.0 = file units are metres)", 1.0)
    ox = _ask_float("Mesh offset X (m)", 0.0)
    oy = _ask_float("Mesh offset Y (m)", 0.0)
    oz = _ask_float("Mesh offset Z (m)", 0.0)

    geometry = WizardGeometrySection(
        cad_mesh_path=str(cad.resolve()),
        collision_mesh_path=str(collision_path.resolve()) if collision_path else None,
        mesh_scale=mesh_scale,
        mesh_offset_x_m=ox,
        mesh_offset_y_m=oy,
        mesh_offset_z_m=oz,
    )

    print("\n--- 2) PARTICLES ---")
    print("PSD: enter number of size bands, then for each: d_min_mm, d_max_mm, mass_fraction (sum≈1).")
    n_bands = _ask_int("How many PSD bands?", 2)
    bands: List[PSDWizardBand] = []
    for i in range(n_bands):
        print(f"  Band {i + 1}:")
        smin = _ask_float("    min diameter (mm)", 5.0 if i == 0 else 20.0)
        smax = _ask_float("    max diameter (mm)", 20.0 if i == 0 else 50.0)
        frac = _ask_float("    mass fraction in this band", 0.5)
        bands.append(PSDWizardBand(size_min_mm=smin, size_max_mm=smax, mass_fraction=frac))
    shape = _ask("Particle shape (sphere only in this build)", "sphere")
    if shape.lower() != "sphere":
        print("Note: only 'sphere' is executed; storing shape for future use.")
    rho = _ask_float("Particle density (kg/m³)", 2500.0)
    use_count = _ask_bool("Specify particle count? (if no, uses total mass)", True)
    count = 200
    mass_kg = 50.0
    if use_count:
        count = _ask_int("Number of particles", 200)
    else:
        mass_kg = _ask_float("Total particle mass (kg)", 50.0)

    print("Spawn box (metres), relative to world origin:")
    sx0 = _ask_float("  X min", -0.4)
    sx1 = _ask_float("  X max", 0.4)
    sy0 = _ask_float("  Y min", 0.15)
    sy1 = _ask_float("  Y max", 0.55)
    sz0 = _ask_float("  Z min", -0.15)
    sz1 = _ask_float("  Z max", 0.15)

    particle = WizardParticleSection(
        psd_bands=bands,
        particle_shape="sphere",
        particle_density_kg_m3=rho,
        use_particle_count=use_count,
        particle_count=count,
        total_particle_mass_kg=mass_kg,
        spawn_x_min_m=min(sx0, sx1),
        spawn_x_max_m=max(sx0, sx1),
        spawn_y_min_m=min(sy0, sy1),
        spawn_y_max_m=max(sy0, sy1),
        spawn_z_min_m=min(sz0, sz1),
        spawn_z_max_m=max(sz0, sz1),
    )

    print("\n--- 3) MATERIAL / CONTACT ---")
    E = _ask_float("Young's modulus E (Pa)", 5.0e8)
    nu = _ask_float("Poisson's ratio", 0.3)
    mu_s = _ask_float("Static friction coefficient (particle–wall typical)", 0.45)
    mu_r = _ask_float("Rolling friction coefficient", 0.08)
    rest = _ask_float("Coefficient of restitution", 0.25)
    coh = _ask_float("Cohesion / adhesion scale (Pa, 0 = none)", 0.0)

    material = WizardMaterialSection(
        young_modulus_pa=E,
        poisson_ratio=nu,
        friction_static=mu_s,
        friction_rolling=mu_r,
        restitution=rest,
        cohesion_pa=coh,
    )

    print("\n--- 4) GRAVITY (m/s²) ---")
    gx = _ask_float("g_x", 0.0)
    gy = _ask_float("g_y", -9.81)
    gz = _ask_float("g_z", 0.0)
    gravity = WizardGravitySection(gx=gx, gy=gy, gz=gz)

    print("\n--- 5) SOLVER / NUMERICS ---")
    dt = _ask_float("Timestep (s)", 0.001)
    t_end = _ask_float("End simulation time (s)", 2.0)
    out_n = _ask_int("Output / diagnostics every N steps", 50)
    env_m = _ask_float("Collision envelope (m)", 0.002)
    margin_m = _ask_float("Collision margin (m)", 0.002)
    cm = _ask("Contact model NSC or SMC", "NSC").upper()
    if cm not in ("NSC", "SMC"):
        cm = "NSC"

    solver = WizardSolverSection(
        timestep_s=dt,
        end_time_s=t_end,
        output_every_n_steps=out_n,
        collision_envelope_m=env_m,
        collision_margin_m=margin_m,
        contact_model=cm,  # type: ignore[arg-type]
    )

    print("\n--- 6) EGRESS DETECTION ---")
    ax = _ask("Axis for lateral egress (x / y / z)", "x").lower()
    if ax not in ("x", "y", "z"):
        ax = "x"
    lim = _ask_float(f"Egress when |{ax}| exceeds (m)", 0.55)
    egress = WizardEgressSection(axis=ax, limit_abs_m=lim)  # type: ignore[arg-type]

    print("\n--- 7) VISUALISATION ---")
    use_vis = _ask_bool("Open Irrlicht viewer? (needs pychrono.irrlicht)", True)
    vis = WizardVisualizationSection(use_irrlicht=use_vis)

    name = _ask("Run name (for output folder)", "wizard_run")

    return WizardDEMConfig(
        particle=particle,
        material=material,
        geometry=geometry,
        gravity=gravity,
        solver=solver,
        egress=egress,
        visualization=vis,
        run_name=name.replace(" ", "_"),
    )
