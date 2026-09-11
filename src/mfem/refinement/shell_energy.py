"""ARAP membrane shell elements over the surface triangulation.

The surface tris tracked in AdditionalState (tri_indices / active_tri_count,
kept in lockstep with tet refinement by scatter_tris) are given a constant-
strain-triangle membrane energy:

    E_tri = A_rest * mu_s / 2 * ||F - R(F)||^2

with F the 3x2 in-plane deformation gradient and R the closest rotation
(polar factor of F). mu_s is a *surface* shear modulus in Pa*m -- i.e. the
bulk Lame mu of the shell material times its thickness.

This is deliberately the 2D analogue of the tet ARAP energy the solver
already uses, and it is integrated the same way Contact is: a direct
position-based energy whose gradient is added to the global rhs and whose
Hessian is added to the global lhs. The Hessian uses the Gauss-Newton
approximation (the dR/dF term is dropped), which makes it CONSTANT per
topology -- blocks (i, j) = mu_s * A * dot(w_i, w_j) * I3, where the w_i
are rows of the inverse rest-shape matrix -- so it is rebuilt only after a
refine pass (refresh()), not per Newton iteration. It is PSD by
construction. Because every surface tri is a face of some active tet, all
its (i, j) blocks already exist in the global Hessian's sparsity from the
tet constraint product, so the bsr_axpy in the solver never grows topology.

Rest state (2x2 inverse shape matrix + rest area) is derived on the fly
from AdditionalState.rest_particle_q, which claim_new_vertices keeps valid
across refinement, so split child tris automatically get correct rest data.
"""

import warp as wp
import warp.sparse as ws
from newton import State

from mfem.refinement.additional_state import AdditionalState


@wp.func
def tri_rest_frame(
    rest_particle_q: wp.array[wp.vec3],
    i0: wp.int32,
    i1: wp.int32,
    i2: wp.int32,
) -> tuple[wp.mat22, wp.float32]:
    """Inverse 2D rest-shape matrix W = Dm^-1 and rest area of one tri."""
    r0 = rest_particle_q[i0]
    e1 = rest_particle_q[i1] - r0
    e2 = rest_particle_q[i2] - r0

    n = wp.cross(e1, e2)
    twice_area = wp.length(n)
    if twice_area < 1.0e-12:
        # Degenerate rest tri: contributes nothing.
        return wp.mat22(), wp.float32(0.0)

    u = wp.normalize(e1)
    v = wp.cross(n / twice_area, u)
    D_m = wp.mat22(
        wp.dot(e1, u), wp.dot(e2, u),
        wp.dot(e1, v), wp.dot(e2, v),
    )
    return wp.inverse(D_m), 0.5 * twice_area


@wp.func
def membrane_F(
    particle_q: wp.array[wp.vec3],
    i0: wp.int32,
    i1: wp.int32,
    i2: wp.int32,
    W: wp.mat22,
) -> tuple[wp.vec3, wp.vec3]:
    """Columns f1, f2 of the 3x2 deformation gradient F = [x1-x0, x2-x0] @ W."""
    x0 = particle_q[i0]
    d1 = particle_q[i1] - x0
    d2 = particle_q[i2] - x0
    f1 = d1 * W[0, 0] + d2 * W[1, 0]
    f2 = d1 * W[0, 1] + d2 * W[1, 1]
    return f1, f2


@wp.func
def membrane_energy_density(f1: wp.vec3, f2: wp.vec3) -> wp.float32:
    """||F - R||^2 / 2 for the 3x2 F with columns f1, f2.

    Uses ||F||^2 - 2 (sigma1 + sigma2) + 2, with sigma1 + sigma2 =
    sqrt(tr(F^T F) + 2 sqrt(det(F^T F))) -- no explicit SVD needed."""
    c11 = wp.dot(f1, f1)
    c12 = wp.dot(f1, f2)
    c22 = wp.dot(f2, f2)
    det_c = wp.max(c11 * c22 - c12 * c12, 0.0)
    sigma_sum = wp.sqrt(wp.max(c11 + c22 + 2.0 * wp.sqrt(det_c), 1.0e-20))
    return 0.5 * (c11 + c22 - 2.0 * sigma_sum + 2.0)


@wp.func
def membrane_piola(f1: wp.vec3, f2: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    """Columns of F - R(F), R the polar rotation factor of the 3x2 F.

    R = F @ S^-1 with S = sqrt(F^T F), computed closed-form for the 2x2 SPD
    C = F^T F: sqrt(C) = (C + sqrt(det C) I) / sqrt(tr C + 2 sqrt(det C)),
    and det(sqrt(C)) = sqrt(det C)."""
    c11 = wp.dot(f1, f1)
    c12 = wp.dot(f1, f2)
    c22 = wp.dot(f2, f2)
    s = wp.sqrt(wp.max(c11 * c22 - c12 * c12, 0.0))
    denom = wp.sqrt(wp.max(c11 + c22 + 2.0 * s, 1.0e-20))
    if s < 1.0e-10:
        # (Near-)rank-deficient F: the polar factor is ill-defined; return
        # zero stress rather than a garbage direction.
        return wp.vec3(), wp.vec3()

    # S = (C + s I) / denom, S^-1 = adj(S) / det(S), det(S) = s.
    inv11 = (c22 + s) / (denom * s)
    inv12 = -c12 / (denom * s)
    inv22 = (c11 + s) / (denom * s)
    r1 = f1 * inv11 + f2 * inv12
    r2 = f1 * inv12 + f2 * inv22
    return f1 - r1, f2 - r2


@wp.kernel
def shell_refresh_kernel(
    active_tri_count: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    rest_particle_q: wp.array[wp.vec3],
    mu_s: wp.float32,
    tri_pose: wp.array[wp.mat22],
    tri_area: wp.array[wp.float32],
    hessian_rows: wp.array[wp.int32],
    hessian_cols: wp.array[wp.int32],
    hessian_values: wp.array[wp.mat33],
):
    tid = wp.tid()
    if tid >= active_tri_count[0]:
        return

    i0 = tri_indices[tid, 0]
    i1 = tri_indices[tid, 1]
    i2 = tri_indices[tid, 2]

    W, area = tri_rest_frame(rest_particle_q, i0, i1, i2)
    tri_pose[tid] = W
    tri_area[tid] = area

    # Per-vertex shape-gradient 2-vectors: F depends on x1 through row 0 of
    # W, on x2 through row 1, and on x0 through minus their sum.
    w1 = wp.vec2(W[0, 0], W[0, 1])
    w2 = wp.vec2(W[1, 0], W[1, 1])
    w0 = -(w1 + w2)

    # Gauss-Newton Hessian blocks: (i, j) -> mu_s * A * dot(w_i, w_j) * I3.
    for i in range(3):
        wi = w0
        if i == 1:
            wi = w1
        if i == 2:
            wi = w2
        for j in range(3):
            wj = w0
            if j == 1:
                wj = w1
            if j == 2:
                wj = w2
            slot = tid * 9 + i * 3 + j
            hessian_rows[slot] = tri_indices[tid, i]
            hessian_cols[slot] = tri_indices[tid, j]
            hessian_values[slot] = mu_s * area * wp.dot(wi, wj) * wp.identity(3, dtype=wp.float32)


@wp.kernel
def shell_energy_gradient_kernel(
    active_tri_count: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    tri_pose: wp.array[wp.mat22],
    tri_area: wp.array[wp.float32],
    mu_s: wp.float32,
    energy: wp.array[wp.float32],
    gradient: wp.array[wp.vec3],
):
    tid = wp.tid()
    if tid >= active_tri_count[0]:
        # Inactive slots must hold zero energy: the line-search accumulator
        # sums the whole array.
        energy[tid] = 0.0
        return

    i0 = tri_indices[tid, 0]
    i1 = tri_indices[tid, 1]
    i2 = tri_indices[tid, 2]
    W = tri_pose[tid]
    area = tri_area[tid]

    f1, f2 = membrane_F(particle_q, i0, i1, i2, W)
    energy[tid] = mu_s * area * membrane_energy_density(f1, f2)

    p1, p2 = membrane_piola(f1, f2)
    p1 *= mu_s * area
    p2 *= mu_s * area

    # dE/dD = P @ W^T; column k of D is x_{k+1} - x0.
    g1 = p1 * W[0, 0] + p2 * W[0, 1]
    g2 = p1 * W[1, 0] + p2 * W[1, 1]
    wp.atomic_add(gradient, i1, g1)
    wp.atomic_add(gradient, i2, g2)
    wp.atomic_add(gradient, i0, -(g1 + g2))


@wp.kernel
def shell_energy_only_kernel(
    active_tri_count: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    tri_pose: wp.array[wp.mat22],
    tri_area: wp.array[wp.float32],
    mu_s: wp.float32,
    energy: wp.array[wp.float32],
):
    tid = wp.tid()
    if tid >= active_tri_count[0]:
        energy[tid] = 0.0
        return

    f1, f2 = membrane_F(
        particle_q,
        tri_indices[tid, 0],
        tri_indices[tid, 1],
        tri_indices[tid, 2],
        tri_pose[tid],
    )
    energy[tid] = mu_s * tri_area[tid] * membrane_energy_density(f1, f2)


class Shell:
    """Membrane shell elements over AdditionalState's surface tris.

    Mirrors Contact's role in the solver: evaluate() fills a per-particle
    gradient (and per-tri energy) at the current positions; hessian is a
    preassembled BSR matrix the solver bsr_axpy's into the global lhs.
    refresh() must be called after every topology change (refine pass)."""

    def __init__(self, max_particles: int, max_tris: int, mu_s: float, max_incident_tris: int):
        self.mu_s = mu_s
        self.max_tris = max_tris

        self.tri_pose = wp.zeros(max_tris, dtype=wp.mat22)
        self.tri_area = wp.zeros(max_tris, dtype=wp.float32)

        self.energy = wp.zeros(max_tris, dtype=wp.float32)
        self.gradient = wp.zeros(max_particles, dtype=wp.vec3)

        self._hessian_rows = wp.zeros(max_tris * 9, dtype=wp.int32)
        self._hessian_cols = wp.zeros(max_tris * 9, dtype=wp.int32)
        self._hessian_values = wp.zeros(max_tris * 9, dtype=wp.mat33)
        self._hessian_block_count = wp.zeros(1, dtype=wp.int32)
        self.hessian = ws.bsr_zeros(
            max_particles,
            max_particles,
            wp.mat33,
            topology="padded",
            row_capacity=3 * max_incident_tris,
        )

    def refresh(self, additional_state: AdditionalState):
        """Rebuild rest state + the constant Gauss-Newton Hessian for the
        current triangulation. Call after every refine pass."""
        wp.launch(
            shell_refresh_kernel,
            dim=self.max_tris,
            inputs=[
                additional_state.active_tri_count,
                additional_state.tri_indices,
                additional_state.rest_particle_q,
                self.mu_s,
            ],
            outputs=[
                self.tri_pose,
                self.tri_area,
                self._hessian_rows,
                self._hessian_cols,
                self._hessian_values,
            ],
        )

        wp.copy(self._hessian_block_count, additional_state.active_tri_count)
        self._hessian_block_count *= 9

        ws.bsr_set_from_triplets(
            self.hessian,
            self._hessian_rows,
            self._hessian_cols,
            self._hessian_values,
            count=self._hessian_block_count,
            topology="padded",
        )

    def evaluate(self, state: State, additional_state: AdditionalState):
        self.gradient.zero_()
        wp.launch(
            shell_energy_gradient_kernel,
            dim=self.max_tris,
            inputs=[
                additional_state.active_tri_count,
                additional_state.tri_indices,
                state.particle_q,
                self.tri_pose,
                self.tri_area,
                self.mu_s,
            ],
            outputs=[
                self.energy,
                self.gradient,
            ],
        )

    def evaluate_energy_only(self, particle_q: wp.array, additional_state: AdditionalState):
        wp.launch(
            shell_energy_only_kernel,
            dim=self.max_tris,
            inputs=[
                additional_state.active_tri_count,
                additional_state.tri_indices,
                particle_q,
                self.tri_pose,
                self.tri_area,
                self.mu_s,
            ],
            outputs=[
                self.energy,
            ],
        )
