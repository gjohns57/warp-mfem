"""A rigid mesh (a scanned octopus by default) hanging off the free end of a cantilevered soft beam.

The beam's x = 0 end is clamped (zero inverse mass, pinned by the solver's
Dirichlet attachments). A free rigid body is penalty-attached to the nodes of the
beam's opposite end face through ``mfem.refinement.rigid.RigidCoupling``. Gravity
loads the body, the beam bends and pushes back on it, and the body both drags the
tip down and rotates with it -- soft and rigid DOFs are solved together.

The run prints the body's height against the Euler-Bernoulli static tip
deflection ``P L^3 / (3 E I)`` (the beam oscillates about it; implicit Euler
damps the motion) and saves a plot of body drop and tilt over time.

Run with ``uv run -m examples.refinement.rigid_beam_example`` (add ``--view`` for an
interactive polyscope window).
"""

import argparse
from pathlib import Path

import numpy as np
import warp as wp
import newton

from mfem.refinement.rigid import RigidCoupling
from mfem.refinement.solver import RefinementSolver

# Corner order matches cube_contact_refinement.build_grid_cube_mesh.
LOCAL_TETS = [(3, 2, 4, 0), (3, 1, 4, 0), (3, 5, 7, 4), (3, 6, 7, 4), (3, 6, 2, 4), (3, 5, 1, 4)]


def build_beam_mesh(nx: int, ny: int, nz: int, length: float, width: float, height: float):
    """Structured ``nx*ny*nz`` hex grid, each cell cut into 6 tets."""
    xs, ys, zs = np.linspace(0, length, nx + 1), np.linspace(0, width, ny + 1), np.linspace(0, height, nz + 1)

    def vidx(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    verts = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32)
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                corner = [vidx(i + dx, j + dy, k + dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
                tets += [tuple(corner[c] for c in t) for t in LOCAL_TETS]
    tets = np.array(tets, dtype=np.int32)
    for t in tets:  # newton drops non-positive-volume tets
        if np.dot(np.cross(verts[t[1]] - verts[t[0]], verts[t[2]] - verts[t[0]]), verts[t[3]] - verts[t[0]]) < 0.0:
            t[2], t[3] = t[3], t[2]
    return verts, tets


DEFAULT_MESH = Path(__file__).resolve().parents[2] / "models" / "octopus_initial_full.msh__sf.obj"


def load_obj(path):
    verts, faces = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(c) for c in parts[1:4]])
        elif parts[0] == "f":
            idx = [int(t.split("/")[0]) - 1 for t in parts[1:]]
            faces += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


def mass_properties(verts, faces):
    """Volume, centre of mass and inertia tensor about the COM (unit density) of a closed,
    outward-oriented triangle mesh, by summing signed tetrahedra against the origin."""
    a, b, c = (verts[faces[:, i]] for i in range(3))
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    total = vol.sum()
    com = (vol[:, None] * (a + b + c) / 4.0).sum(0) / total
    # second moments of a tet (vertices 0, a, b, c): V/20 * (sum_i v_i v_i^T + (sum v)(sum v)^T)
    cov = np.zeros((3, 3))
    for v, (p, q, r) in zip(vol, zip(a, b, c)):
        cov += v / 20.0 * (np.outer(p, p) + np.outer(q, q) + np.outer(r, r) + np.outer(p + q + r, p + q + r))
    cov -= total * np.outer(com, com)
    inertia = np.trace(cov) * np.eye(3) - cov
    return total, com, inertia


def load_rigid_mesh(path, size, up_axis="y"):
    """Rigid body mesh scaled so its longest extent is ``size`` metres and re-oriented to z-up.

    Returns ``(verts_com, faces, unit_inertia_per_mass, anchor_offset)``: vertices relative to the
    centre of mass (the body frame), the inertia tensor per unit mass, and the COM position
    relative to the (x-min, y-centre, z-centre) of the bounding box -- the point that is placed
    on the beam's end-face centre.
    """
    verts, faces = load_obj(path)
    if up_axis == "y":  # (x, y, z) -> (x, -z, y)
        verts = verts @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64).T
    verts *= size / np.ptp(verts, axis=0).max()
    vol, com, inertia = mass_properties(verts, faces)
    if vol < 0:  # inward-facing winding
        faces = faces[:, ::-1].copy()
        vol, com, inertia = mass_properties(verts, faces)
    lo, hi = verts.min(0), verts.max(0)
    anchor = np.array([lo[0], 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])])
    return (verts - com).astype(np.float32), faces, inertia / vol, com - anchor


def build_scene(
    cells=(16, 4, 4),
    length=1.0,
    thickness=0.2,
    youngs=3.0e6,
    poisson=0.3,
    rigid_mass=20.0,
    mesh_path=DEFAULT_MESH,
    mesh_size=0.5,
    stiffness=1.0e6,
    iterations=6,
    **solver_kwargs,
):
    verts, tets = build_beam_mesh(*cells, length, thickness, thickness)
    builder = newton.ModelBuilder(gravity=-9.81)
    # Contact needs at least one shape; this plane is far below and never touched.
    builder.add_shape_plane(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, -50.0), wp.quat_identity()))
    mesh_v, mesh_f, unit_inertia, com_offset = load_rigid_mesh(mesh_path, mesh_size)
    center = np.array([length, thickness / 2, thickness / 2]) + com_offset
    body = builder.add_body(
        xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
        mass=rigid_mass,
        inertia=wp.mat33((rigid_mass * unit_inertia).astype(np.float32)),
    )
    mu = youngs / (2 * (1 + poisson))
    lmbda = youngs * poisson / ((1 + poisson) * (1 - 2 * poisson))
    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=wp.float32(1.0), vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=verts, indices=tets.flatten(), density=1000.0, k_mu=mu, k_lambda=lmbda, k_damp=0.0,
        add_surface_mesh_edges=False, validate_mesh=False,
    )
    model = builder.finalize()

    inv_mass = model.particle_inv_mass.numpy()
    inv_mass[verts[:, 0] < 1e-6] = 0.0  # clamp x = 0
    model.particle_inv_mass.assign(inv_mass)

    s0, s1 = model.state(), model.state()
    tip = np.nonzero(verts[:, 0] > length - 1e-6)[0]
    rigid = RigidCoupling(model, s0, [body], tip, np.zeros(len(tip), dtype=np.int32), verts[tip], stiffness)
    solver = RefinementSolver(
        model, iterations, max_tets=len(tets), enable_refinement=False, rigid_coupling=rigid,
        **{"line_search": False, "attachment_stiffness": 1.0e7, "cg_use_cuda_graph": False,
           "cg_check_every": 10, **solver_kwargs},
    )
    I = thickness**4 / 12.0
    static_tip = (rigid_mass * 9.81) * length**3 / (3 * youngs * I)
    return model, solver, rigid, s0, s1, verts, tip, static_tip, (mesh_v, mesh_f)


class Stepper:
    """Advances the sim two steps per call (state_0 -> state_1 -> state_0), either eagerly or by
    replaying a CUDA graph of that pair. Two steps because the ping-pong of the state buffers is
    baked into a captured graph, so the capture must return to its starting buffer."""

    def __init__(self, scene_kwargs: dict, dt: float, graph: bool):
        self.dt, self.graph = dt, graph
        if graph:
            # Warm-up on a throwaway scene so Warp loads every module before capture
            # (module loads are not allowed inside a capture).
            model, solver, _, s0, s1, *_ = build_scene(**scene_kwargs)
            for a, b in ((s0, s1), (s1, s0)):
                solver.step(a, b, model.control(), None, dt)
            wp.synchronize()
        (self.model, self.solver, self.rigid, self.s0, self.s1, self.verts, self.tip, self.static_tip, self.rigid_mesh) = build_scene(
            **scene_kwargs
        )
        self.control = self.model.control()
        self._graph = None
        if graph:
            with wp.ScopedCapture() as capture:
                self._pair()
            self._graph = capture.graph

    def _pair(self):
        self.solver.step(self.s0, self.s1, self.control, None, self.dt)
        self.solver.step(self.s1, self.s0, self.control, None, self.dt)

    def advance(self):
        """Two steps; the current state is always ``self.s0`` afterwards."""
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._pair()


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply a Warp transform ``(px, py, pz, qx, qy, qz, qw)`` to ``points``."""
    q, p = pose[3:6], pose[:3]
    w = pose[6]
    t = 2.0 * np.cross(q, points)
    return (points + w * t + np.cross(q, t) + p).astype(np.float32)


def run_viewer(stepper, args):
    """Interactive polyscope window; the physics steps every frame (space toggles pause)."""
    import polyscope as ps
    import polyscope.imgui as psim

    model, verts, s0 = stepper.model, stepper.verts, stepper.s0
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps.set_window_size(1280, 720)
    tris = model.tri_indices.numpy()
    beam = ps.register_surface_mesh("beam", s0.particle_q.numpy(), tris, smooth_shade=False)
    beam.set_color((0.30, 0.55, 0.80))
    beam.set_edge_width(1.0)
    body_v, body_f = stepper.rigid_mesh
    body_vis = ps.register_surface_mesh("rigid body", transform_points(body_v, s0.body_q.numpy()[0]), body_f, smooth_shade=True)
    body_vis.set_color((0.85, 0.30, 0.25))
    clamp = ps.register_point_cloud("clamped", verts[verts[:, 0] < 1e-6], radius=0.006)
    clamp.set_color((0.1, 0.1, 0.1))
    ps.look_at((0.7, -2.6, 0.9), (0.75, 0.1, 0.0))

    state = {"paused": False, "t": 0.0, "pairs_per_frame": args.steps_per_frame}

    def callback():
        _, state["paused"] = psim.Checkbox("paused", state["paused"])
        _, state["pairs_per_frame"] = psim.SliderInt("step pairs / frame", state["pairs_per_frame"], 1, 4)
        if not state["paused"]:
            for _ in range(state["pairs_per_frame"]):
                stepper.advance()
                state["t"] += 2 * args.dt
        cur = stepper.s0
        beam.update_vertex_positions(cur.particle_q.numpy())
        pose = cur.body_q.numpy()[0]
        body_vis.update_vertex_positions(transform_points(body_v, pose))
        psim.Text(f"t = {state['t']:.3f} s   body z = {pose[2]:+.4f} m")

    ps.set_user_callback(callback)
    ps.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--mass", type=float, default=20.0, help="rigid body mass [kg]")
    parser.add_argument("--mesh", default=str(DEFAULT_MESH), help="closed OBJ surface for the rigid body (y-up)")
    parser.add_argument("--mesh-size", type=float, default=0.5, help="longest extent of the rigid mesh [m]")
    parser.add_argument("--youngs", type=float, default=3.0e6)
    parser.add_argument("--cells", type=int, nargs=3, default=(16, 4, 4), metavar=("NX", "NY", "NZ"),
                        help="beam hex cells along x, y, z (6 tets each)")
    parser.add_argument("--stiffness", type=float, default=1.0e6, help="penalty stiffness per attachment node")
    parser.add_argument("--out", default="results/figures/rigid_beam.png")
    parser.add_argument("--device", default=None)
    parser.add_argument("--view", action="store_true", help="open an interactive polyscope window instead of the headless run")
    parser.add_argument("--steps-per-frame", type=int, default=1, help="viewer: step *pairs* per frame")
    parser.add_argument("--graph", action="store_true", help="CUDA-graph capture the solver step")
    args = parser.parse_args()

    with wp.ScopedDevice(args.device):
        scene = dict(cells=tuple(args.cells), rigid_mass=args.mass, mesh_path=args.mesh, mesh_size=args.mesh_size, youngs=args.youngs, stiffness=args.stiffness)
        if args.graph:
            # check_every=0 only exits CG early inside a captured graph
            scene.update(cg_use_cuda_graph=True, cg_check_every=0)
        stepper = Stepper(scene, args.dt, args.graph)
        if args.view:
            run_viewer(stepper, args)
            return
        z0 = stepper.s0.body_q.numpy()[0, 2]
        times, body_z, tilt = [], [], []
        for pair in range(args.steps // 2):
            stepper.advance()
            pose = stepper.s0.body_q.numpy()[0]
            times.append(2 * (pair + 1) * args.dt)
            body_z.append(pose[2] - z0)
            tilt.append(2.0 * np.arcsin(np.clip(pose[4], -1, 1)))  # rotation about y (qy)
            if pair % 10 == 0:
                print(f"t={times[-1]:.3f}  body drop={-body_z[-1]:+.4f} m  tilt={np.degrees(tilt[-1]):+.2f} deg")
        static_tip = stepper.static_tip
        print(f"Euler-Bernoulli static tip deflection: {static_tip:.4f} m  (max drop reached: {-min(body_z):.4f} m)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    a.plot(times, -np.array(body_z), label="body drop")
    a.axhline(static_tip, color="k", ls="--", lw=1, label="beam theory (static)")
    a.set_ylabel("drop [m]"); a.legend()
    b.plot(times, np.degrees(tilt)); b.set_ylabel("body tilt [deg]"); b.set_xlabel("time [s]")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
