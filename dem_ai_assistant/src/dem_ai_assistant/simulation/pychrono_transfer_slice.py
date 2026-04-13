"""
PyChrono rigid-sphere transfer slice (CPU, NSC / Bullet collision).

Optional CAD import: set geometry `cad_file` to a watertight .obj or .stl (metres).
Export from CAD (STEP → mesh) in Blender, Meshmixer, or your CAD exporter.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, List

from src.dem_ai_assistant.simulation.input_builder import SimulationInput
from src.dem_ai_assistant.simulation.output_schema import (
    EdgeStats,
    FlowStats,
    ImpactStats,
    ParticleStats,
    SimulationRawOutput,
)
from src.dem_ai_assistant.simulation import pychrono_bridge as pb


def run_pychrono_transfer_slice(sim: SimulationInput) -> SimulationRawOutput:
    chrono = pb.import_chrono()
    random.seed(42)

    system = chrono.ChSystemNSC()
    gvec = pb.vec3(chrono, 0.0, -9.81, 0.0)
    if hasattr(system, "Set_G_acc"):
        system.Set_G_acc(gvec)
    elif hasattr(system, "SetGravitationalAcceleration"):
        system.SetGravitationalAcceleration(gvec)
    elif hasattr(system, "SetGravity"):
        system.SetGravity(gvec)
    pb.configure_collision_system(chrono, system)

    material = pb.make_contact_material(chrono, sim)

    half_w = 0.30 + (sim.skirt_spacing_mm / 1000.0) * 0.55
    half_w = max(0.22, min(half_w, 1.2))
    offset_m = sim.lateral_offset_mm / 1000.0

    floor_half_x = 2.0
    floor_half_z = 1.0
    floor_half_y = 0.02
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

    use_mesh = bool(sim.cad_mesh_path)
    mesh_body: Any = None
    if use_mesh:
        mesh_path = Path(sim.cad_mesh_path)
        rho_wall = 7800.0
        mesh_body = pb.add_fixed_mesh_body(
            chrono,
            system,
            mesh_path,
            material,
            rho_wall,
            sim.cad_mesh_scale,
            (0.0, 0.0, 0.0),
        )
    else:
        wall_thick = 0.04
        wall_h = 0.45
        wall_depth = floor_half_z * 2 + 0.2
        for sign in (-1.0, 1.0):
            wall = chrono.ChBodyEasyBox(
                wall_thick,
                wall_h,
                wall_depth,
                1000,
                True,
                True,
                material,
            )
            wx = sign * (half_w + wall_thick / 2)
            wall.SetPos(pb.vec3(chrono, wx, wall_h / 2 - floor_half_y, 0.0))
            if hasattr(wall, "SetFixed"):
                wall.SetFixed(True)
            elif hasattr(wall, "SetBodyFixed"):
                wall.SetBodyFixed(True)
            system.Add(wall)

    rho = 2500.0
    radius = 0.025
    n_spheres = int(min(420, max(96, sim.throughput_tph / 22.0)))
    spawn_y = floor_half_y + radius + 0.35
    grid_n = int(math.ceil(math.sqrt(n_spheres)))
    spacing = 2.1 * radius

    sphere_bodies: List[Any] = []
    for ix in range(grid_n):
        for iz in range(grid_n):
            if len(sphere_bodies) >= n_spheres:
                break
            sx = (ix - grid_n / 2) * spacing * 0.85 + 0.02 * random.uniform(-1, 1)
            sz = (iz - grid_n / 2) * spacing * 0.85 + 0.02 * random.uniform(-1, 1)
            if abs(sx) > half_w - 1.8 * radius:
                continue
            sph = chrono.ChBodyEasySphere(radius, rho, True, True, material)
            sph.SetPos(pb.vec3(chrono, sx, spawn_y, sz))
            idx = len(sphere_bodies)
            if sim.restart_enabled and idx % 4 == 0:
                if hasattr(sph, "SetLinVel"):
                    sph.SetLinVel(pb.vec3(chrono, sim.belt_speed_m_s * 0.25, -0.5, 0.0))
                elif hasattr(sph, "SetPos_dt"):
                    sph.SetPos_dt(pb.vec3(chrono, sim.belt_speed_m_s * 0.15, 0.0, 0.0))
            system.Add(sph)
            sphere_bodies.append(sph)
        if len(sphere_bodies) >= n_spheres:
            break

    dt = max(1e-5, float(sim.timestep_s))
    duration = max(0.05, float(sim.duration_s))
    n_steps = int(math.ceil(duration / dt))
    n_steps = min(n_steps, 15000)

    belt_top = floor_half_y
    belt_band = 0.09
    max_outside = 0
    max_ke = 0.0

    for _step in range(n_steps):
        system.DoStepDynamics(dt)
        outside = 0
        step_ke = 0.0
        for body in sphere_bodies:
            pos = body.GetPos()
            x = float(pos.x) + offset_m
            y = float(pos.y)
            v = body.GetLinVel() if hasattr(body, "GetLinVel") else body.GetPos_dt()
            vx = float(v.x)
            vy = float(v.y)
            vz = float(v.z)
            step_ke += 0.5 * (vx * vx + vy * vy + vz * vz)
            if abs(x) > half_w - 0.5 * radius:
                outside += 1
        max_outside = max(max_outside, outside)
        max_ke = max(max_ke, step_ke)

    margins_mm: list[float] = []
    left_bias_samples: list[float] = []
    right_bias_samples: list[float] = []
    for body in sphere_bodies:
        pos = body.GetPos()
        x = float(pos.x) + offset_m
        y = float(pos.y)
        if belt_top <= y <= belt_top + belt_band:
            m_mm = (half_w - abs(x)) * 1000.0
            margins_mm.append(m_mm)
            if x < 0:
                left_bias_samples.append(abs(x))
            else:
                right_bias_samples.append(abs(x))

    if margins_mm:
        mean_m = sum(margins_mm) / len(margins_mm)
        min_m = min(margins_mm)
    else:
        mean_m = (half_w - 0.5 * radius) * 1000.0
        min_m = max(0.0, (half_w - radius) * 1000.0)

    lb = sum(left_bias_samples) / len(left_bias_samples) if left_bias_samples else 0.05
    rb = sum(right_bias_samples) / len(right_bias_samples) if right_bias_samples else 0.05
    lb_n = lb / (lb + rb + 1e-9)
    rb_n = rb / (lb + rb + 1e-9)

    coarse = max(1, n_spheres // 6)
    if margins_mm:
        spread = max(margins_mm) - min(margins_mm)
        stability = max(0.15, min(0.95, 0.92 - 0.008 * spread))
    else:
        stability = 0.55

    mass_ref = n_spheres * (4.0 / 3.0 * math.pi * radius**3 * rho)
    impact_j = max(1.0, 0.5 * mass_ref / max(n_spheres, 1) * max_ke / max(n_steps, 1) * 0.02)

    mesh_note = ""
    if use_mesh and sim.cad_mesh_path:
        mesh_note = f" CAD mesh: {sim.cad_mesh_path} (scale={sim.cad_mesh_scale})."
    elif use_mesh:
        mesh_note = " CAD mesh: (path unset)."

    notes = (
        f"PyChrono NSC rigid spheres (n={len(sphere_bodies)}); steps={n_steps}, dt={dt:.5g}s; "
        f"channel half-width={half_w:.3f}m.{mesh_note}"
    )

    meta = {
        "solver": "pychrono",
        "pychrono_steps": n_steps,
        "pychrono_dt": dt,
        "channel_half_width_m": half_w,
        "sphere_radius_m": radius,
        "sphere_count": len(sphere_bodies),
        "cad_mesh_loaded": bool(use_mesh),
        "cad_mesh_path": sim.cad_mesh_path,
    }
    if mesh_body is not None:
        meta["cad_mesh_body"] = True

    return SimulationRawOutput(
        particle_stats=ParticleStats(
            total_particles=len(sphere_bodies),
            coarse_particles=coarse,
            fines_fraction=0.2,
        ),
        flow_stats=FlowStats(
            throughput_tph=sim.throughput_tph,
            estimated_spill_events=int(max_outside),
            estimated_recirculation_index=min(
                0.45, 0.12 + 0.25 * (max_outside / max(len(sphere_bodies), 1))
            ),
        ),
        edge_stats=EdgeStats(
            mean_edge_margin_mm=mean_m,
            min_edge_margin_mm=min_m,
            left_edge_bias=float(min(0.45, max(0.02, lb_n))),
            right_edge_bias=float(min(0.45, max(0.02, rb_n))),
        ),
        impact_stats=ImpactStats(
            max_impact_energy_j=min(500.0, impact_j * 8.0),
            p95_impact_energy_j=min(400.0, impact_j * 5.5),
        ),
        stability_score=float(stability),
        notes=notes,
        metadata=meta,
    )
