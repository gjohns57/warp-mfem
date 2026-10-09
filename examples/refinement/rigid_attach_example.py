"""A free rigid box attached to a soft cube by penalty springs.

The cube's top face is pinned; a rigid box (mass ``--box-mass``) is attached to
the cube's bottom-face nodes through ``mfem.refinement.rigid.RigidCoupling``.
The box pulls the cube down and the cube pulls it back, both solved together
(implicit, Schur-eliminated rigid DOFs) inside ``RefinementSolver``. Refinement
is off; this exercises only the soft<->rigid coupling. Headless: prints the
box height as the system settles.

Run with ``uv run -m examples.refinement.rigid_attach_example``.
"""

import argparse

import numpy as np
import warp as wp
import newton

from examples.refinement.cube_contact_refinement import build_grid_cube_mesh
from mfem.refinement.rigid import RigidCoupling
from mfem.refinement.solver import RefinementSolver


def build_scene(
    grid: int = 3,
    box_mass: float = 100.0,
    box_half: float = 0.25,
    stiffness: float = 1.0e6,
    pin_top: bool = True,
    gravity: float = -9.81,
    box_velocity=(0.0, 0.0, 0.0),
    iterations: int = 10,
    graph: bool = False,
    **solver_kwargs,
):
    verts, tets = build_grid_cube_mesh(grid)
    builder = newton.ModelBuilder(gravity=gravity)
    # Obstacle for Contact (it needs at least one shape); far below, never touched.
    builder.add_shape_plane(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, -50.0), wp.quat_identity()))
    inertia = box_mass * (2.0 * box_half) ** 2 / 6.0 * np.eye(3)
    box = builder.add_body(
        xform=wp.transform(wp.vec3(0.5, 0.5, -box_half), wp.quat_identity()),
        mass=box_mass,
        inertia=wp.mat33(inertia.astype(np.float32)),
    )
    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=wp.float32(1.0),
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=verts,
        indices=tets.flatten(),
        density=1000.0,
        k_mu=1.0e4,
        k_lambda=1.0e4,
        k_damp=0.0,
        add_surface_mesh_edges=False,
        validate_mesh=False,
    )
    model = builder.finalize()
    if pin_top:
        inv_mass = model.particle_inv_mass.numpy()
        inv_mass[verts[:, 2] > 1.0 - 1e-6] = 0.0
        model.particle_inv_mass.assign(inv_mass)

    state_0, state_1 = model.state(), model.state()
    state_0.body_qd.assign(np.array([[*box_velocity, 0, 0, 0]], dtype=np.float32))

    bottom = np.nonzero(verts[:, 2] < 1e-6)[0]
    rigid = RigidCoupling(model, state_0, [box], bottom, np.zeros(len(bottom), dtype=np.int32), verts[bottom], stiffness)

    solver = RefinementSolver(
        model,
        iterations,
        max_tets=len(tets),
        enable_refinement=False,
        rigid_coupling=rigid,
        cg_use_cuda_graph=graph,
        cg_check_every=0 if graph else 10,
        **{"line_search": False, "attachment_stiffness": 1.0e5, **solver_kwargs},
    )
    return model, solver, rigid, state_0, state_1, verts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--box-mass", type=float, default=100.0)
    parser.add_argument("--stiffness", type=float, default=1.0e6)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    with wp.ScopedDevice(args.device):
        model, solver, rigid, s0, s1, _ = build_scene(box_mass=args.box_mass, stiffness=args.stiffness)
        control = model.control()
        for step in range(args.steps):
            solver.step(s0, s1, control, None, args.dt)
            s0, s1 = s1, s0
            if step % 10 == 0:
                z = s0.body_q.numpy()[0, 2]
                print(f"step {step:4d}  box z = {z:+.4f}  vz = {s0.body_qd.numpy()[0, 2]:+.4f}")


if __name__ == "__main__":
    main()
