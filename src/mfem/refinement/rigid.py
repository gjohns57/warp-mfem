"""Penalty coupling between free rigid bodies and soft-body nodes.

Each attachment ties soft node ``i`` to a point fixed in rigid body ``b``::

    E_a = 1/2 k_a |x_i - (p_b + R_b r_a)|^2

so the soft body and the rigid body push on each other with equal and opposite
force/torque. The rigid bodies are extra Newton unknowns (a world-frame
twist ``xi = (dp, dtheta)`` per body) solved together with the soft nodes.

To keep the soft solver's CG on ``dx`` only, the 6-DOF rigid blocks are
eliminated by a Schur complement::

    [H_xx H_xb] [dx ]   [r_x]        (S = H_xx - H_xb H_bb^-1 H_bx) dx = r_x - H_xb H_bb^-1 r_b
    [H_bx H_bb] [dxi] = [r_b]        dxi = H_bb^-1 (r_b - H_bx dx)

with ``r = -gradient`` (the sign convention of ``RefinementSolver._assemble_system``).
Hessians are Gauss-Newton (SPD). Rigid inertia is ``1/(2 dt^2) (xi - xi~)^T M (xi - xi~)``
with the same predicted-position convention as ``integrate_particles``
(``x~ = x + dt v + dt^2 g / 2``) so soft and rigid parts feel the same gravity.

Everything is preallocated and kernel-only so a step can be CUDA-graph captured.
"""

import numpy as np
import warp as wp
import warp.optim.linear

from mfem.numerical.cholesky import cholesky6
from mfem.types import mat33, mat66, vec6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@wp.func
def skew(v: wp.vec3):
    return wp.matrix_from_rows(
        wp.vec3(0.0, -v[2], v[1]),
        wp.vec3(v[2], 0.0, -v[0]),
        wp.vec3(-v[1], v[0], 0.0),
    )


@wp.func
def rotvec_to_quat(theta: wp.vec3):
    angle = wp.length(theta)
    if angle < 1.0e-9:
        return wp.normalize(wp.quat(0.5 * theta[0], 0.5 * theta[1], 0.5 * theta[2], 1.0))
    return wp.quat_from_axis_angle(theta / angle, angle)


@wp.func
def quat_to_rotvec(q: wp.quat):
    """Rotation vector of a unit quaternion, shortest-arc."""
    if q[3] < 0.0:
        q = wp.quat(-q[0], -q[1], -q[2], -q[3])
    v = wp.vec3(q[0], q[1], q[2])
    s = wp.length(v)
    if s < 1.0e-9:
        return 2.0 * v
    return v * (2.0 * wp.atan2(s, q[3]) / s)


@wp.func
def apply_twist(pose: wp.transform, xi: vec6):
    p = wp.transform_get_translation(pose) + wp.vec3(xi[0], xi[1], xi[2])
    dq = rotvec_to_quat(wp.vec3(xi[3], xi[4], xi[5]))
    q = wp.normalize(dq * wp.transform_get_rotation(pose))
    return wp.transform(p, q)


@wp.func
def world_inertia(q: wp.quat, inertia_body: mat33):
    R = wp.quat_to_matrix(q)
    return R * inertia_body * wp.transpose(R)


@wp.func
def solve6(H: mat66, rhs: vec6):
    """H^-1 rhs for SPD H via Cholesky (Warp has no 6x6 inverse)."""
    L = cholesky6(H)
    y = vec6()
    for i in range(6):
        s = rhs[i]
        for k in range(i):
            s -= L[i, k] * y[k]
        y[i] = s / L[i, i]
    x = vec6()
    for ii in range(6):
        i = 5 - ii
        s = y[i]
        for k in range(i + 1, 6):
            s -= L[k, i] * x[k]
        x[i] = s / L[i, i]
    return x


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
@wp.kernel
def predict_bodies(
    body_ids: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    gravity: wp.array[wp.vec3],
    dt: wp.float32,
    pred_pose: wp.array[wp.transform],
):
    """Predicted pose ``(p~, q~)`` (no torque, no gyroscopic term). ``body_qd`` is
    Newton's spatial vector ``(linear velocity of the COM, angular velocity)``, world frame."""
    b = wp.tid()
    pose = body_q[body_ids[b]]
    qd = body_qd[body_ids[b]]
    v = wp.spatial_top(qd)
    w = wp.spatial_bottom(qd)
    p = wp.transform_get_translation(pose) + v * dt + gravity[0] * dt * dt / 2.0
    q = wp.normalize(rotvec_to_quat(w * dt) * wp.transform_get_rotation(pose))
    pred_pose[b] = wp.transform(p, q)


@wp.kernel
def rigid_kinetic_terms(
    body_ids: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    pred_pose: wp.array[wp.transform],
    mass: wp.array[wp.float32],
    inertia_body: wp.array[mat33],
    dt: wp.float32,
    # outputs
    gradient: wp.array[vec6],
    hessian: wp.array[mat66],
):
    """Gradient (+ , not negated) and Hessian of the rigid inertial term at the current pose."""
    b = wp.tid()
    pose = body_q[body_ids[b]]
    pred = pred_pose[b]
    inv_dt2 = 1.0 / (dt * dt)
    pred_q = wp.transform_get_rotation(pred)
    I_w = world_inertia(pred_q, inertia_body[b])

    dp = wp.transform_get_translation(pose) - wp.transform_get_translation(pred)
    th = quat_to_rotvec(wp.transform_get_rotation(pose) * wp.quat_inverse(pred_q))
    f = mass[b] * dp * inv_dt2
    tq = I_w * th * inv_dt2
    gradient[b] = vec6(f[0], f[1], f[2], tq[0], tq[1], tq[2])

    H = mat66()
    for i in range(3):
        H[i, i] = mass[b] * inv_dt2
        for j in range(3):
            H[3 + i, 3 + j] = I_w[i, j] * inv_dt2
    hessian[b] = H


@wp.kernel
def kinetic_energy_trial(
    body_ids: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    twist: wp.array[vec6],
    pred_pose: wp.array[wp.transform],
    mass: wp.array[wp.float32],
    inertia_body: wp.array[mat33],
    dt: wp.float32,
    energy: wp.array[wp.float32],
):
    b = wp.tid()
    pose = apply_twist(body_q[body_ids[b]], twist[b])
    pred = pred_pose[b]
    pred_q = wp.transform_get_rotation(pred)
    I_w = world_inertia(pred_q, inertia_body[b])
    dp = wp.transform_get_translation(pose) - wp.transform_get_translation(pred)
    th = quat_to_rotvec(wp.transform_get_rotation(pose) * wp.quat_inverse(pred_q))
    energy[b] = 0.5 * (mass[b] * wp.dot(dp, dp) + wp.dot(th, I_w * th)) / (dt * dt)


@wp.kernel
def penalty_derivatives(
    att_node: wp.array[wp.int32],
    att_body: wp.array[wp.int32],
    att_anchor: wp.array[wp.vec3],
    att_k: wp.array[wp.float32],
    body_ids: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    particle_q: wp.array[wp.vec3],
    # outputs (gradients accumulate with atomics; zero them first)
    node_gradient: wp.array[wp.vec3],
    body_gradient: wp.array[vec6],
    body_hessian: wp.array[mat66],
    node_hessian_blocks: wp.array[mat33],
    att_w: wp.array[wp.vec3],
):
    """Gradient (positive, +dE) and Gauss-Newton Hessian pieces. ``J = [I, -I, [w]x]``."""
    a = wp.tid()
    n = att_node[a]
    b = att_body[a]
    k = att_k[a]
    pose = body_q[body_ids[b]]
    w = wp.quat_rotate(wp.transform_get_rotation(pose), att_anchor[a])
    y = wp.transform_get_translation(pose) + w
    e = particle_q[n] - y
    ke = k * e

    wp.atomic_add(node_gradient, n, ke)
    kw = wp.cross(e, w) * k  # d/dtheta = -k w x e = k e x w
    wp.atomic_add(body_gradient, b, vec6(-ke[0], -ke[1], -ke[2], kw[0], kw[1], kw[2]))

    # Jb = [-I, [w]x];  k Jb^T Jb
    W = skew(w)
    Wt = wp.transpose(W)
    WtW = Wt * W
    H = mat66()
    for i in range(3):
        H[i, i] = k
        for j in range(3):
            H[i, 3 + j] = -k * W[i, j]
            H[3 + i, j] = -k * Wt[i, j]
            H[3 + i, 3 + j] = k * WtW[i, j]
    wp.atomic_add(body_hessian, b, H)

    node_hessian_blocks[a] = wp.identity(3, dtype=wp.float32) * k
    att_w[a] = w


@wp.kernel
def penalty_energy_trial(
    att_node: wp.array[wp.int32],
    att_body: wp.array[wp.int32],
    att_anchor: wp.array[wp.vec3],
    att_k: wp.array[wp.float32],
    body_ids: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    twist: wp.array[vec6],
    particle_q: wp.array[wp.vec3],
    energy: wp.array[wp.float32],
):
    a = wp.tid()
    b = att_body[a]
    pose = apply_twist(body_q[body_ids[b]], twist[b])
    y = wp.transform_point(pose, att_anchor[a])
    e = particle_q[att_node[a]] - y
    energy[a] = 0.5 * att_k[a] * wp.dot(e, e)


@wp.kernel
def schur_gather(
    att_node: wp.array[wp.int32],
    att_body: wp.array[wp.int32],
    att_k: wp.array[wp.float32],
    att_w: wp.array[wp.vec3],
    x: wp.array[wp.vec3],
    t: wp.array[vec6],
):
    """t[b] += H_bx x = k Jb^T x = k (-x, x cross w)."""
    a = wp.tid()
    xi = x[att_node[a]]
    k = att_k[a]
    c = wp.cross(xi, att_w[a]) * k
    wp.atomic_add(t, att_body[a], vec6(-k * xi[0], -k * xi[1], -k * xi[2], c[0], c[1], c[2]))


@wp.kernel
def solve_bodies(
    hessian: wp.array[mat66],
    rhs: wp.array[vec6],
    out: wp.array[vec6],
):
    b = wp.tid()
    out[b] = solve6(hessian[b], rhs[b])


@wp.kernel
def schur_scatter(
    att_node: wp.array[wp.int32],
    att_body: wp.array[wp.int32],
    att_k: wp.array[wp.float32],
    att_w: wp.array[wp.vec3],
    u: wp.array[vec6],
    scale: wp.float32,
    z: wp.array[wp.vec3],
):
    """z_i += scale * H_xb u = scale * k (-u_p + w cross u_theta)."""
    a = wp.tid()
    ub = u[att_body[a]]
    up = wp.vec3(ub[0], ub[1], ub[2])
    ut = wp.vec3(ub[3], ub[4], ub[5])
    d = att_k[a] * (-up + wp.cross(att_w[a], ut)) * scale
    wp.atomic_add(z, att_node[a], d)


@wp.kernel
def backsubstitute_bodies(
    hessian: wp.array[mat66],
    rhs: wp.array[vec6],
    hbx_dx: wp.array[vec6],
    out: wp.array[vec6],
):
    b = wp.tid()
    out[b] = solve6(hessian[b], rhs[b] - hbx_dx[b])


@wp.kernel
def commit_bodies(
    body_ids: wp.array[wp.int32],
    twist: wp.array[vec6],
    dt: wp.float32,
    body_q_in: wp.array[wp.transform],
    # in/out
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Applies the accepted twist to ``body_q`` and sets the velocity."""
    b = wp.tid()
    i = body_ids[b]
    new_pose = apply_twist(body_q[i], twist[b])
    body_q[i] = new_pose
    # velocities from the finite difference to the step-start pose
    p0 = wp.transform_get_translation(body_q_in[i])
    q0 = wp.transform_get_rotation(body_q_in[i])
    v = (wp.transform_get_translation(new_pose) - p0) / dt
    w = quat_to_rotvec(wp.transform_get_rotation(new_pose) * wp.quat_inverse(q0)) / dt
    body_qd[i] = wp.spatial_vector(v, w)


@wp.kernel
def zero_vec6(a: wp.array[vec6]):
    a[wp.tid()] = vec6()


@wp.kernel
def zero_mat66(a: wp.array[mat66]):
    a[wp.tid()] = mat66()


@wp.kernel
def zero_vec3(a: wp.array[wp.vec3]):
    a[wp.tid()] = wp.vec3()


@wp.kernel
def add_scaled_vec6(a: wp.array[vec6], b: wp.array[vec6], alpha: wp.float32, out: wp.array[vec6]):
    out[wp.tid()] = a[wp.tid()] + alpha * b[wp.tid()]


@wp.kernel
def copy_dynamic_bodies(
    body_ids: wp.array[wp.int32],
    src_q: wp.array[wp.transform],
    src_qd: wp.array[wp.spatial_vector],
    dst_q: wp.array[wp.transform],
    dst_qd: wp.array[wp.spatial_vector],
):
    b = wp.tid()
    i = body_ids[b]
    dst_q[i] = src_q[i]
    dst_qd[i] = src_qd[i]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
class RigidCoupling:
    """Free rigid bodies attached to soft nodes by penalty springs.

    Args:
        model: Newton model. The dynamic bodies must be in ``model.body_*`` (use
            ``builder.add_body(...)`` with mass/inertia, or pass them explicitly below).
        body_ids: indices into ``state.body_q`` of the dynamic bodies (others stay
            kinematic/static and are never touched).
        node_indices / body_slots / world_anchors / stiffness: one entry per attachment;
            ``body_slots`` index into ``body_ids``, ``world_anchors`` are the world positions
            (in the *initial* configuration) of the rigid-side attachment points. They are
            converted to body-frame anchors using the current ``state.body_q``.
        mass / inertia: optional overrides, else taken from ``model.body_mass``/``body_inertia``.
            Inertia is about the COM in the body frame; ``body_q`` is taken to be the COM frame.
    """

    def __init__(
        self,
        model,
        state,
        body_ids,
        node_indices,
        body_slots,
        world_anchors,
        stiffness,
        mass=None,
        inertia=None,
        max_particles: int | None = None,
    ):
        device = model.device
        self.device = device
        self.num_bodies = len(body_ids)
        self.num_attachments = len(node_indices)
        self.max_particles = max_particles or model.particle_count
        nb, na = self.num_bodies, self.num_attachments

        body_ids_np = np.asarray(body_ids, dtype=np.int32)
        self.body_ids = wp.array(body_ids_np, dtype=wp.int32, device=device)
        if mass is None:
            mass = model.body_mass.numpy()[body_ids_np]
        if inertia is None:
            inertia = model.body_inertia.numpy()[body_ids_np]
        self.mass = wp.array(np.asarray(mass, dtype=np.float32), dtype=wp.float32, device=device)
        self.inertia_body = wp.array(np.asarray(inertia, dtype=np.float32).reshape(nb, 3, 3), dtype=mat33, device=device)

        node_np = np.asarray(node_indices, dtype=np.int32)
        slot_np = np.asarray(body_slots, dtype=np.int32)
        k_np = np.broadcast_to(np.asarray(stiffness, dtype=np.float32), (na,)).copy()
        world_np = np.asarray(world_anchors, dtype=np.float32).reshape(na, 3)
        pose_np = state.body_q.numpy()[body_ids_np]
        anchors = np.zeros((na, 3), dtype=np.float32)
        for a in range(na):
            p = pose_np[slot_np[a], :3]
            qx, qy, qz, qw = pose_np[slot_np[a], 3:]
            # inverse rotate (world - p) into the body frame
            v = world_np[a] - p
            u = np.cross([qx, qy, qz], v)
            anchors[a] = v - 2.0 * qw * u + 2.0 * np.cross([qx, qy, qz], u)
        self.att_node = wp.array(node_np, dtype=wp.int32, device=device)
        self.att_body = wp.array(slot_np, dtype=wp.int32, device=device)
        self.att_anchor = wp.array(anchors, dtype=wp.vec3, device=device)
        self.att_k = wp.array(k_np, dtype=wp.float32, device=device)

        # scratch
        self.pred_pose = wp.zeros(nb, dtype=wp.transform, device=device)
        self.body_gradient = wp.zeros(nb, dtype=vec6, device=device)  # +grad (kinetic + penalty)
        self.body_hessian = wp.zeros(nb, dtype=mat66, device=device)
        self._kin_gradient = wp.zeros(nb, dtype=vec6, device=device)
        self._kin_hessian = wp.zeros(nb, dtype=mat66, device=device)
        self.node_gradient = wp.zeros(self.max_particles, dtype=wp.vec3, device=device)
        self.node_hessian_blocks = wp.zeros(na, dtype=mat33, device=device)
        self.att_w = wp.zeros(na, dtype=wp.vec3, device=device)
        self.twist = wp.zeros(nb, dtype=vec6, device=device)  # accepted-so-far twist inside a Newton iteration
        self.dtwist = wp.zeros(nb, dtype=vec6, device=device)  # Newton direction
        self._t = wp.zeros(nb, dtype=vec6, device=device)
        self._u = wp.zeros(nb, dtype=vec6, device=device)
        self._rhs_b = wp.zeros(nb, dtype=vec6, device=device)
        self.penalty_energy = wp.zeros(na, dtype=wp.float32, device=device)
        self.kinetic_energy = wp.zeros(nb, dtype=wp.float32, device=device)
        self.node_count = wp.array([na], dtype=wp.int32, device=device)

    # -- per-step ---------------------------------------------------------
    def begin_step(self, model, state, dt: float):
        wp.launch(
            predict_bodies,
            dim=self.num_bodies,
            inputs=[self.body_ids, state.body_q, state.body_qd, model.gravity, dt],
            outputs=[self.pred_pose],
            device=self.device,
        )

    # -- per-Newton-iteration ----------------------------------------------
    def evaluate(self, state, dt: float):
        """Fills node_gradient / node_hessian_blocks / body_gradient / body_hessian at the current pose."""
        wp.launch(zero_vec3, dim=self.max_particles, inputs=[self.node_gradient], device=self.device)
        wp.launch(
            rigid_kinetic_terms,
            dim=self.num_bodies,
            inputs=[self.body_ids, state.body_q, self.pred_pose, self.mass, self.inertia_body, dt],
            outputs=[self.body_gradient, self.body_hessian],
            device=self.device,
        )
        wp.launch(
            penalty_derivatives,
            dim=self.num_attachments,
            inputs=[
                self.att_node, self.att_body, self.att_anchor, self.att_k,
                self.body_ids, state.body_q, state.particle_q,
            ],
            outputs=[
                self.node_gradient, self.body_gradient, self.body_hessian,
                self.node_hessian_blocks, self.att_w,
            ],
            device=self.device,
        )
        # rhs_b = -gradient_b
        wp.launch(zero_vec6, dim=self.num_bodies, inputs=[self._rhs_b], device=self.device)
        wp.launch(add_scaled_vec6, dim=self.num_bodies, inputs=[self._rhs_b, self.body_gradient, -1.0], outputs=[self._rhs_b], device=self.device)
        wp.launch(zero_vec6, dim=self.num_bodies, inputs=[self.twist], device=self.device)

    def eliminate_rhs(self, rhs_x: wp.array):
        """rhs_x <- rhs_x - H_xb H_bb^-1 rhs_b (in place)."""
        self._schur_correction(self._rhs_b, rhs_x, scale=-1.0)

    def _schur_correction(self, body_vec: wp.array, z: wp.array, scale: float):
        wp.launch(solve_bodies, dim=self.num_bodies, inputs=[self.body_hessian, body_vec], outputs=[self._u], device=self.device)
        wp.launch(
            schur_scatter,
            dim=self.num_attachments,
            inputs=[self.att_node, self.att_body, self.att_k, self.att_w, self._u, scale],
            outputs=[z],
            device=self.device,
        )

    def schur_matvec(self, x, y, z, alpha, beta, base_matvec):
        """z = alpha * S x + beta * y with S = H_xx - H_xb H_bb^-1 H_bx; ``base_matvec``
        applies H_xx (the assembled BSR, kI already included)."""
        base_matvec(x, y, z, alpha, beta)
        wp.launch(zero_vec6, dim=self.num_bodies, inputs=[self._t], device=self.device)
        wp.launch(
            schur_gather,
            dim=self.num_attachments,
            inputs=[self.att_node, self.att_body, self.att_k, self.att_w, x],
            outputs=[self._t],
            device=self.device,
        )
        self._schur_correction(self._t, z, scale=-float(alpha))

    def recover_twist(self, dx: wp.array):
        """dtwist = H_bb^-1 (rhs_b - H_bx dx)."""
        wp.launch(zero_vec6, dim=self.num_bodies, inputs=[self._t], device=self.device)
        wp.launch(
            schur_gather,
            dim=self.num_attachments,
            inputs=[self.att_node, self.att_body, self.att_k, self.att_w, dx],
            outputs=[self._t],
            device=self.device,
        )
        wp.launch(
            backsubstitute_bodies,
            dim=self.num_bodies,
            inputs=[self.body_hessian, self._rhs_b, self._t],
            outputs=[self.dtwist],
            device=self.device,
        )
        return self.dtwist

    # -- merit function -------------------------------------------------
    def energy_trial(self, particle_q, state_body_q, twist, dt: float, kinetic_out: wp.array, penalty_out: wp.array):
        wp.launch(
            kinetic_energy_trial,
            dim=self.num_bodies,
            inputs=[self.body_ids, state_body_q, twist, self.pred_pose, self.mass, self.inertia_body, dt],
            outputs=[kinetic_out],
            device=self.device,
        )
        wp.launch(
            penalty_energy_trial,
            dim=self.num_attachments,
            inputs=[
                self.att_node, self.att_body, self.att_anchor, self.att_k,
                self.body_ids, state_body_q, twist, particle_q,
            ],
            outputs=[penalty_out],
            device=self.device,
        )

    def commit(self, state_in, state_out, dt: float):
        """Apply the accepted twist (``self.twist``) to ``state_out.body_q`` and set body_qd."""
        wp.launch(
            commit_bodies,
            dim=self.num_bodies,
            inputs=[self.body_ids, self.twist, dt, state_in.body_q],
            outputs=[state_out.body_q, state_out.body_qd],
            device=self.device,
        )

    def mirror_bodies(self, src_state, dst_state):
        """Copy only the dynamic bodies' pose + velocity (leaves kinematic bodies alone)."""
        wp.launch(
            copy_dynamic_bodies,
            dim=self.num_bodies,
            inputs=[self.body_ids, src_state.body_q, src_state.body_qd, dst_state.body_q, dst_state.body_qd],
            device=self.device,
        )
