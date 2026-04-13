"""
Execute WizardDEMConfig in PyChrono: mesh boundary, PSD spheres, optional Irrlicht, reporting.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.dem_ai_assistant.simulation import pychrono_bridge as pb
from src.dem_ai_assistant.wizard.wizard_models import WizardDEMConfig, PSDWizardBand


def _normalize_psd(bands: List[PSDWizardBand]) -> List[PSDWizardBand]:
    s = sum(b.mass_fraction for b in bands)
    if s <= 0:
        raise ValueError("PSD mass fractions must sum to > 0.")
    return [
        PSDWizardBand(
            size_min_mm=b.size_min_mm,
            size_max_mm=b.size_max_mm,
            mass_fraction=b.mass_fraction / s,
        )
        for b in bands
    ]


def _mean_diameter_mm(bands: List[PSDWizardBand]) -> float:
    bands = _normalize_psd(bands)
    return sum(
        0.5 * (b.size_min_mm + b.size_max_mm) * b.mass_fraction for b in bands
    )


def _sample_radius_m(bands: List[PSDWizardBand], rng: random.Random) -> float:
    bands = _normalize_psd(bands)
    r = rng.random()
    c = 0.0
    for b in bands:
        c += b.mass_fraction
        if r <= c:
            d_mm = rng.uniform(b.size_min_mm, b.size_max_mm)
            return max(1e-4, d_mm * 0.5 / 1000.0)
    d_mm = rng.uniform(bands[-1].size_min_mm, bands[-1].size_max_mm)
    return max(1e-4, d_mm * 0.5 / 1000.0)


def _particle_count_from_mass(mass_kg: float, rho: float, bands: List[PSDWizardBand]) -> int:
    d_mean_mm = _mean_diameter_mm(bands)
    r_m = max(1e-4, d_mean_mm * 0.5 / 1000.0)
    vol = (4.0 / 3.0) * math.pi * r_m**3
    m_one = rho * vol
    return max(1, int(mass_kg / max(m_one, 1e-9)))


def _compliance_from_elastic(E: float, nu: float) -> float:
    """Rough NSC normal compliance from E, nu (metres/N order)."""
    try:
        if nu >= 0.49 or E <= 0:
            return 1e-7
        kn = abs(2.0 * E * (1.0 - nu) / max((1.0 + nu) * (1.0 - 2.0 * nu), 1e-9))
        return float(1.0 / max(kn, 1e3))
    except Exception:
        return 1e-7


def _make_material_wizard(chrono: Any, cfg: WizardDEMConfig) -> Any:
    m = cfg.material
    if hasattr(chrono, "ChContactMaterialNSC"):
        mat = chrono.ChContactMaterialNSC()
    else:
        mat = chrono.ChMaterialSurfaceNSC()
    if hasattr(mat, "SetFriction"):
        mat.SetFriction(max(0.01, min(1.2, m.friction_static)))
    if hasattr(mat, "SetRestitution"):
        mat.SetRestitution(max(0.0, min(1.0, m.restitution)))
    comp = _compliance_from_elastic(m.young_modulus_pa, m.poisson_ratio)
    coh = min(m.cohesion_pa / 4000.0, 1.0)
    comp = comp + 5e-5 * coh
    if hasattr(mat, "SetCompliance"):
        mat.SetCompliance(comp)
    elif hasattr(mat, "SetComplianceNormal"):
        mat.SetComplianceNormal(comp)
    if m.friction_rolling > 0 and hasattr(mat, "SetRollingFriction"):
        try:
            mat.SetRollingFriction(m.friction_rolling)
        except Exception:
            pass
    return mat


def _configure_collision_wizard(chrono: Any, system: Any, env: float, margin: float) -> None:
    if hasattr(chrono, "ChCollisionSystem") and hasattr(
        chrono.ChCollisionSystem, "Type_BULLET"
    ):
        try:
            system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        except Exception:
            pass
    if hasattr(chrono, "ChCollisionModel"):
        try:
            chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(float(env))
            chrono.ChCollisionModel.SetDefaultSuggestedMargin(float(margin))
        except Exception:
            pass


def _axis_value(pos: Any, axis: str) -> float:
    if axis == "x":
        return float(pos.x)
    if axis == "y":
        return float(pos.y)
    return float(pos.z)


@dataclass
class WizardRunReport:
    egress_first_crossing_count: int = 0
    time_first_egress_s: Optional[float] = None
    egress_locations: List[Dict[str, float]] = field(default_factory=list)
    total_mass_particles_kg: float = 0.0
    simulation_time_s: float = 0.0
    impact_force_samples: List[float] = field(default_factory=list)
    freeboard_mean_m: float = 0.0
    freeboard_min_m: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        hist = _histogram(self.impact_force_samples, bins=[0, 50, 200, 500, 2000, 1e9])
        mass_t = max(self.total_mass_particles_kg / 1000.0, 1e-12)
        dur = max(self.simulation_time_s, 1e-12)
        rate = self.egress_first_crossing_count / mass_t / dur
        return {
            "egress": {
                "events_first_crossing_per_particle": self.egress_first_crossing_count,
                "time_first_egress_s": self.time_first_egress_s,
                "egress_rate_per_tonne_per_s": rate,
                "egress_location_map": self.egress_locations,
            },
            "impact": {
                "note": "Histogram uses contact force samples (N) as impact proxy; not Hertz energy yet.",
                "contact_force_samples_count": len(self.impact_force_samples),
                "max_contact_force_N": max(self.impact_force_samples)
                if self.impact_force_samples
                else 0.0,
                "force_histogram_N_bins": hist,
            },
            "freeboard_edge_distance_m": {
                "mean_m": self.freeboard_mean_m,
                "min_m": self.freeboard_min_m,
            },
            "mass_particles_kg": self.total_mass_particles_kg,
            "simulation_time_s": self.simulation_time_s,
            "notes": self.notes,
        }


def _histogram(values: List[float], bins: List[float]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        key = f"{lo:g}-{hi:g}"
        out[key] = sum(1 for v in values if lo <= v < hi)
    return out


def run_wizard_dem(
    cfg: WizardDEMConfig,
    project_root: Path,
) -> Tuple[Path, WizardRunReport]:
    chrono = pb.import_chrono()
    rng = random.Random(42)

    notes: List[str] = []
    if cfg.solver.contact_model == "SMC":
        notes.append(
            "SMC was requested; this build still uses NSC contact. SMC can be added later."
        )

    system = chrono.ChSystemNSC()
    g = cfg.gravity
    gvec = pb.vec3(chrono, g.gx, g.gy, g.gz)
    if hasattr(system, "Set_G_acc"):
        system.Set_G_acc(gvec)
    elif hasattr(system, "SetGravitationalAcceleration"):
        system.SetGravitationalAcceleration(gvec)
    elif hasattr(system, "SetGravity"):
        system.SetGravity(gvec)

    _configure_collision_wizard(
        chrono,
        system,
        cfg.solver.collision_envelope_m,
        cfg.solver.collision_margin_m,
    )
    material = _make_material_wizard(chrono, cfg)

    mesh_path = Path(cfg.geometry.collision_mesh_path or cfg.geometry.cad_mesh_path)
    mesh_path = mesh_path.resolve()
    off = (
        cfg.geometry.mesh_offset_x_m,
        cfg.geometry.mesh_offset_y_m,
        cfg.geometry.mesh_offset_z_m,
    )
    pb.add_fixed_mesh_body(
        chrono,
        system,
        mesh_path,
        material,
        7800.0,
        cfg.geometry.mesh_scale,
        off,
    )

    floor_half_x = 3.0
    floor_half_y = 0.02
    floor_half_z = 2.0
    floor = chrono.ChBodyEasyBox(
        floor_half_x * 2,
        floor_half_y * 2,
        floor_half_z * 2,
        1000,
        True,
        True,
        material,
    )
    floor.SetPos(pb.vec3(chrono, 0.0, -floor_half_y, 0.0))
    if hasattr(floor, "SetFixed"):
        floor.SetFixed(True)
    elif hasattr(floor, "SetBodyFixed"):
        floor.SetBodyFixed(True)
    system.Add(floor)

    bands = _normalize_psd(cfg.particle.psd_bands)
    pcfg = cfg.particle
    if pcfg.use_particle_count:
        n_spheres = max(1, pcfg.particle_count)
    else:
        n_spheres = _particle_count_from_mass(
            pcfg.total_particle_mass_kg, pcfg.particle_density_kg_m3, bands
        )

    sphere_bodies: List[Any] = []
    total_mass = 0.0
    for _ in range(n_spheres):
        r = _sample_radius_m(bands, rng)
        rho = pcfg.particle_density_kg_m3
        total_mass += rho * (4.0 / 3.0) * math.pi * r**3
        sph = chrono.ChBodyEasySphere(r, rho, True, True, material)
        for _try in range(80):
            x = rng.uniform(pcfg.spawn_x_min_m, pcfg.spawn_x_max_m)
            y = rng.uniform(pcfg.spawn_y_min_m, pcfg.spawn_y_max_m)
            z = rng.uniform(pcfg.spawn_z_min_m, pcfg.spawn_z_max_m)
            sph.SetPos(pb.vec3(chrono, x, y, z))
            break
        system.Add(sph)
        sphere_bodies.append(sph)

    dt = max(1e-6, cfg.solver.timestep_s)
    n_steps = int(math.ceil(cfg.solver.end_time_s / dt))
    n_steps = min(max(1, n_steps), 200_000)
    out_every = max(1, cfg.solver.output_every_n_steps)

    axis = cfg.egress.axis
    lim = cfg.egress.limit_abs_m
    already_egressed: set[int] = set()
    egress_locs: List[Dict[str, float]] = []
    t_first: Optional[float] = None
    force_samples: List[float] = []

    def report_contacts() -> None:
        if not hasattr(chrono, "ReportContactCallback"):
            return
        try:
            cc = system.GetContactContainer()

            class CB(chrono.ReportContactCallback):
                def __init__(self, store: List[float]) -> None:
                    chrono.ReportContactCallback.__init__(self)
                    self._store = store

                def OnReportContact(
                    self,
                    vA: Any,
                    vB: Any,
                    cA: Any,
                    dist: float,
                    rad: float,
                    force: Any,
                    torque: Any,
                    modA: Any,
                    modB: Any,
                ) -> bool:
                    try:
                        if hasattr(force, "Length"):
                            fn = float(force.Length())
                        else:
                            fn = math.sqrt(
                                float(force.x) ** 2
                                + float(force.y) ** 2
                                + float(force.z) ** 2
                            )
                        self._store.append(fn)
                    except Exception:
                        pass
                    return True

            cb = CB(force_samples)
            cc.ReportAllContacts(cb)
        except Exception:
            pass

    vis = None
    if cfg.visualization.use_irrlicht:
        try:
            import pychrono.irrlicht as chronoirr  # type: ignore[import-untyped]

            vis = chronoirr.ChVisualSystemIrrlicht()
            vis.AttachSystem(system)
            vis.SetWindowSize(cfg.visualization.window_width, cfg.visualization.window_height)
            vis.SetWindowTitle("DEM AI Assistant — wizard")
            vis.Initialize()
            try:
                if hasattr(chrono, "GetChronoDataFile"):
                    vis.AddLogo(chrono.GetChronoDataFile("logo_chrono_alpha.png"))
                else:
                    vis.AddLogo()
            except Exception:
                pass
            vis.AddSkyBox()
            vis.AddCamera(pb.vec3(chrono, 1.5, 0.8, 1.2))
            vis.AddTypicalLights()
        except Exception as exc:
            notes.append(f"Irrlicht not available ({exc!r}); running headless.")
            vis = None

    sim_time = 0.0

    step = 0
    while step < n_steps:
        if vis is not None:
            if not vis.Run():
                notes.append("Irrlicht window closed; stopping early.")
                break
            vis.BeginScene()
            vis.Render()
            vis.EndScene()

        system.DoStepDynamics(dt)
        if hasattr(system, "GetChTime"):
            sim_time = float(system.GetChTime())
        else:
            sim_time = float(step + 1) * dt
        if step % out_every == 0:
            report_contacts()

        for bi, body in enumerate(sphere_bodies):
            pos = body.GetPos()
            av = abs(_axis_value(pos, axis))
            if av > lim and bi not in already_egressed:
                already_egressed.add(bi)
                t_ev = float(sim_time)
                if t_first is None:
                    t_first = t_ev
                egress_locs.append(
                    {
                        "t_s": t_ev,
                        "x_m": float(pos.x),
                        "y_m": float(pos.y),
                        "z_m": float(pos.z),
                    }
                )

        step += 1
        if vis is None and step % out_every == 0:
            pass

    if vis is not None:
        try:
            vis.Close()
        except Exception:
            pass

    freeboard_samples: List[float] = []
    for body in sphere_bodies:
        pos = body.GetPos()
        av = abs(_axis_value(pos, axis))
        if av < lim:
            freeboard_samples.append(lim - av)
    fb_mean = sum(freeboard_samples) / len(freeboard_samples) if freeboard_samples else 0.0
    fb_min = min(freeboard_samples) if freeboard_samples else 0.0

    rep = WizardRunReport(
        egress_first_crossing_count=len(already_egressed),
        time_first_egress_s=t_first,
        egress_locations=egress_locs,
        total_mass_particles_kg=total_mass,
        simulation_time_s=float(sim_time),
        impact_force_samples=force_samples,
        freeboard_mean_m=fb_mean,
        freeboard_min_m=fb_min,
        notes=notes,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (project_root / "runs" / "wizard_runs" / f"{cfg.run_name}_{stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "wizard_config.json"
    try:
        cfg_path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    except Exception:
        cfg_path.write_text(
            json.dumps(cfg.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    report_path = run_dir / "wizard_report.json"
    report_path.write_text(
        json.dumps(rep.to_dict(), indent=2),
        encoding="utf-8",
    )
    summary = run_dir / "wizard_summary.txt"
    d = rep.to_dict()
    lines = [
        "DEM AI ASSISTANT — WIZARD RUN SUMMARY",
        "=" * 60,
        f"Simulation time (s): {rep.simulation_time_s:.4f}",
        f"Particle mass total (kg): {rep.total_mass_particles_kg:.3f}",
        f"Throughput (tph): not computed (no controlled feed mass-flow boundary in this wizard).",
        f"Egress events (first crossing per particle): {rep.egress_first_crossing_count}",
        f"Time of first egress (s): {rep.time_first_egress_s}",
        f"Egress rate (per tonne per s): {d['egress']['egress_rate_per_tonne_per_s']:.6f}",
        f"Max contact force proxy (N): {d['impact']['max_contact_force_N']:.2f}",
        f"Freeboard / edge distance mean & min (m): {rep.freeboard_mean_m:.4f} / {rep.freeboard_min_m:.4f}",
        "",
        "Notes:",
    ]
    lines.extend(f"  - {n}" for n in rep.notes)
    summary.write_text("\n".join(lines), encoding="utf-8")

    return run_dir, rep
