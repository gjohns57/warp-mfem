import numpy as np
import pytest
import warp as wp
import warp.sparse as ws
import newton

from mfem.refinement.rigid import RigidCoupling
from mfem.types import vec6

wp.config.quiet = True


def build(device, anchors_offset=0.0):
    builder = newton.ModelBuilder(gravity=-9.81)
    pos = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)]
    for p in pos:
        builder.add_particle(pos=p, vel=(0.0, 0.0, 0.0), mass=1.0)
    q = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 0.5)), 0.4)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.3, 0.2, -0.5), q), mass=2.0, inertia=wp.mat33(np.diag([0.5, 0.7, 0.9])))
    model = builder.finalize(device=device)
    state = model.state()
    nodes = [0, 1, 2, 3, 4, 2]
    world = np.array(pos, dtype=np.float32)[nodes] + anchors_offset
    rc = RigidCoupling(model, state, [body], nodes, [0] * len(nodes), world, 50.0 + np.arange(len(nodes)))
    return model, state, rc


def trial_energy(rc, state, x, twist, dt=0.01):
    xq = wp.array(x.astype(np.float32), dtype=wp.vec3)
    tw = wp.array(twist.reshape(1, 6).astype(np.float32), dtype=vec6)
    kin = wp.zeros(rc.num_bodies, dtype=wp.float32)
    pen = wp.zeros(rc.num_attachments, dtype=wp.float32)
    rc.energy_trial(xq, state.body_q, tw, dt, kin, pen)
    return pen.numpy().sum(), kin.numpy().sum()


@pytest.mark.parametrize("device", ["cpu"])
def test_penalty_gradient_matches_finite_difference(device):
    with wp.ScopedDevice(device):
        model, state, rc = build(device, anchors_offset=0.05)
        dt = 0.01
        rc.begin_step(model, state, dt)
        rc.evaluate(state, dt)
        x0 = state.particle_q.numpy().copy()
        g_node = rc.node_gradient.numpy()
        # penalty-only part of body gradient = total - kinetic
        kin_g = wp.zeros(1, dtype=vec6)
        h = 1e-3
        fd_node = np.zeros_like(x0)
        for i in range(x0.shape[0]):
            for c in range(3):
                xp, xm = x0.copy(), x0.copy()
                xp[i, c] += h
                xm[i, c] -= h
                ep = trial_energy(rc, state, xp, np.zeros(6))[0]
                em = trial_energy(rc, state, xm, np.zeros(6))[0]
                fd_node[i, c] = (ep - em) / (2 * h)
        np.testing.assert_allclose(g_node, fd_node, rtol=2e-2, atol=2e-2)

        fd_body = np.zeros(6)
        for c in range(6):
            tp, tm = np.zeros(6), np.zeros(6)
            tp[c], tm[c] = h, -h
            ep = sum(trial_energy(rc, state, x0, tp, dt))
            em = sum(trial_energy(rc, state, x0, tm, dt))
            fd_body[c] = (ep - em) / (2 * h)
        np.testing.assert_allclose(rc.body_gradient.numpy()[0], fd_body, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize("device", ["cpu"])
def test_schur_matvec_matches_dense(device):
    with wp.ScopedDevice(device):
        # e = 0 so the Gauss-Newton Hessian equals the exact penalty Hessian
        model, state, rc = build(device)
        dt = 0.01
        rc.begin_step(model, state, dt)
        rc.evaluate(state, dt)
        n = state.particle_q.shape[0]
        x0 = state.particle_q.numpy().copy()

        # Dense (3n+6) penalty Hessian by FD of the gradient from energy_trial
        h = 1e-3
        def pen_grad(x, tw):
            g = np.zeros(3 * n + 6)
            for i in range(3 * n + 6):
                dx, dtw = np.zeros(3 * n), np.zeros(6)
                if i < 3 * n:
                    dx[i] = h
                else:
                    dtw[i - 3 * n] = h
                ep = trial_energy(rc, state, x + dx.reshape(n, 3), tw + dtw)[0]
                dx, dtw = -dx, -dtw
                em = trial_energy(rc, state, x + dx.reshape(n, 3), tw + dtw)[0]
                g[i] = (ep - em) / (2 * h)
            return g

        K = np.zeros((3 * n + 6, 3 * n + 6))
        for i in range(3 * n + 6):
            ex = np.zeros(3 * n)
            et = np.zeros(6)
            if i < 3 * n:
                ex[i] = h
            else:
                et[i - 3 * n] = h
            gp = pen_grad(x0 + ex.reshape(n, 3), et)
            gm = pen_grad(x0 - ex.reshape(n, 3), -et)
            K[i] = (gp - gm) / (2 * h)
        K = 0.5 * (K + K.T)
        # kinetic body block at the predicted pose (zero error)
        Hbb_kin = rc.body_hessian.numpy()[0] - K[3 * n:, 3 * n:]
        # add soft-side mass-like diagonal so Hxx is SPD
        M = 3.0 * np.eye(3 * n)
        Hxx = K[: 3 * n, : 3 * n] + M
        Hxb = K[: 3 * n, 3 * n:]
        Hbb = K[3 * n:, 3 * n:] + Hbb_kin
        S = Hxx - Hxb @ np.linalg.solve(Hbb, Hxb.T)

        base = ws.bsr_diag(wp.full(n, 3.0 * np.eye(3, dtype=np.float32), dtype=wp.mat33), rows_of_blocks=n, cols_of_blocks=n)
        # fold the penalty x-block into the base matrix, as the solver's assembly does
        Pxx = K[: 3 * n, : 3 * n].astype(np.float32)
        dense_base = Pxx + M
        xv = np.random.default_rng(0).standard_normal((n, 3)).astype(np.float32)
        y = wp.zeros(n, dtype=wp.vec3)
        z = wp.zeros(n, dtype=wp.vec3)

        def base_matvec(x, y_, z_, alpha, beta):
            z_.assign((alpha * (dense_base @ x.numpy().reshape(-1)) ).reshape(n, 3).astype(np.float32))

        rc.schur_matvec(wp.array(xv, dtype=wp.vec3), y, z, 1.0, 0.0, base_matvec)
        np.testing.assert_allclose(z.numpy().reshape(-1), S @ xv.reshape(-1), rtol=5e-2, atol=5e-1)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="needs CUDA")
def test_two_way_coupling_and_graph_capture():
    """A heavier box drags the soft cube down further (soft feels the rigid body),
    and the box falls slower than free fall (rigid feels the soft body); a graph-captured
    run matches the eager one."""
    from examples.refinement.rigid_attach_example import build_scene

    dt = 1.0 / 120.0

    def run(mass, graph=False, steps=20):
        with wp.ScopedDevice("cuda:0"):
            model, solver, rigid, s0, s1, verts = build_scene(box_mass=mass, stiffness=1.0e6, iterations=6, graph=graph)
            c = model.control()
            if graph:
                solver.step(s0, s1, c, None, dt)
                solver.step(s1, s0, c, None, dt)
                model, solver, rigid, s0, s1, verts = build_scene(box_mass=mass, stiffness=1.0e6, iterations=6, graph=graph)
                with wp.ScopedCapture() as cap:
                    solver.step(s0, s1, c, None, dt)
                    solver.step(s1, s0, c, None, dt)
                for _ in range(steps // 2):
                    wp.capture_launch(cap.graph)
            else:
                for _ in range(steps // 2):
                    solver.step(s0, s1, c, None, dt)
                    solver.step(s1, s0, c, None, dt)
            bottom = verts[:, 2] < 1e-6
            return s0.body_q.numpy()[0, :3].copy(), s0.particle_q.numpy()[bottom, 2].mean()

    light, light_nodes = run(1.0)
    heavy, heavy_nodes = run(200.0)
    z0 = -0.25
    free_fall = z0 - 0.5 * 9.81 * (20 * dt) ** 2
    assert heavy[2] > free_fall + 1e-3  # held up by the soft body
    assert heavy_nodes < light_nodes - 5e-5  # soft body pulled down by the box
    assert abs(heavy[2] - heavy_nodes + 0.25) < 5e-2  # box stays attached to the nodes

    captured, _ = run(200.0, graph=True)
    np.testing.assert_allclose(captured, heavy, atol=1e-4)
