"""CUDA-graph-safe adaptive mesh refinement for RefinementSolver.

Ports the candidate-scoring / conflict-resolution / edge-bisection logic from
mesh_topology.py's FEMTopology.split_edges() (dynamic allocation, host
readbacks) into a fixed-buffer form: every scratch array is preallocated once
to a static ceiling and reused via `wp.copy`/atomic-append counters, so this
can run inside a captured CUDA graph.

Cadence: a real refine pass is only evaluated every `refine_every_n_steps`
calls, via a device-resident counter that gates candidate *scores* rather
than branching (see refine()'s docstring for why). additional_state_0 always
plays the "current mesh" role and additional_state_1 the "scratch, rebuilt
every step" role -- RefinementSolver.step() keeps these as fixed Python
object references and copies additional_state_1's final content back into
additional_state_0 at the end of every step (see copy_additional_state),
rather than swapping which object plays which role.

That copy-back, not a swap, is deliberate: unlike the newton.State
state_in/state_out swap (which happens in the *caller*, between separate,
uncaptured calls to solver.step()), additional_state_0/_1 are swapped
*inside* step() itself. A CUDA graph permanently bakes in the specific
device pointers each captured kernel reads/writes; reassigning a Python
attribute after the fact doesn't change what an already-captured graph does
on replay. So a plain `self._additional_state_0, self._additional_state_1 =
self._additional_state_1, self._additional_state_0` would only "take effect"
for calls still inside the same capture trace (irrelevant here, since
solver.step() is called once per trace) -- across separate capture_launch
replays, whichever object was captured as the read side would silently stay
frozen at whatever it held at capture time. Copying data into a fixed
additional_state_0 avoids this trap entirely.

Because a linear tet's deformation gradient F is constant over the whole
element, and an affine map commutes with taking a midpoint, splitting a tet
at the true rest-space midpoint of an edge (not just its current-space
midpoint) gives each child *exactly* the same F as the parent. That's why
AdditionalState carries a `rest_particle_q` buffer: it lets a new child's
tet_pose be recomputed exactly, and its tet_stretch/tet_lambda inherited
verbatim from the parent instead of reset (no discontinuity).
"""

import numpy as np
import warp as wp

from mfem.refinement.additional_state import AdditionalState
from mfem.refinement.elastic_energy import arap_energy_hessian_kernel
from mfem.refinement.geometry_hash import (
    canonical_edge,
    get_hashtable_size,
    hashmap_find_edge,
    hashmap_insert_edge,
    hashset_find_edge,
)
from mfem.refinement.mesh_topology import remove_worst_candidate, update_tet_candidates
from mfem.types import vec6
from newton import State

# Sentinel score for inactive candidate slots (padding beyond active_tet_count).
# Must sort below every real score (real scores are non-negative lengths).
_SENTINEL_SCORE = wp.constant(wp.float32(-1.0e30))


@wp.kernel
def emit_split_edge_candidates(
    active_tet_count: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_energy: wp.array[wp.float32],
    should_refine: wp.array[wp.int32],
    candidate_edges: wp.array[wp.vec2i],
    candidate_scores: wp.array[wp.float32],
):
    """Every active tet nominates its 6 edges, scored by the tet's elastic
    (ARAP) energy as of the *previous* step's SQP solve (solver._elastic_energy,
    written by _compute_elastic_derivatives every iteration and never
    reallocated -- refine() reads whatever it holds from the last iteration
    of the step that just finished, before this step's own solve has run).
    Inactive tet slots emit sentinel edges/scores so they always sort to the
    bottom and can never be chosen as a top-K winner. On steps where
    should_refine is 0, every real score is also forced to the sentinel, so
    the rest of the (always-run) pipeline naturally reduces to a verbatim
    carry-forward with zero accepted candidates -- this is what lets refine()
    avoid wp.capture_if, whose conditional-body graphs disallow the memory
    allocation that bsr_set_from_triplets/radix_sort_pairs perform
    internally (fine in the unconditional graph, illegal inside a
    conditional one)."""
    tid = wp.tid()

    if tid >= active_tet_count[0] or should_refine[0] == 0:
        sentinel = wp.vec2i(-1, -1)
        for k in range(6):
            candidate_edges[tid * 6 + k] = sentinel
            candidate_scores[tid * 6 + k] = _SENTINEL_SCORE
        return

    t0 = tet_indices[tid, 0]
    t1 = tet_indices[tid, 1]
    t2 = tet_indices[tid, 2]
    t3 = tet_indices[tid, 3]

    score = tet_energy[tid]

    candidate_edges[tid * 6 + 0] = canonical_edge(t0, t1)
    candidate_edges[tid * 6 + 1] = canonical_edge(t0, t2)
    candidate_edges[tid * 6 + 2] = canonical_edge(t0, t3)
    candidate_edges[tid * 6 + 3] = canonical_edge(t1, t2)
    candidate_edges[tid * 6 + 4] = canonical_edge(t1, t3)
    candidate_edges[tid * 6 + 5] = canonical_edge(t2, t3)
    for k in range(6):
        candidate_scores[tid * 6 + k] = score


@wp.kernel
def extract_top_k_candidates(
    candidate_edges: wp.array[wp.vec2i],
    candidate_order: wp.array[wp.int32],
    candidate_scores_sorted: wp.array[wp.float32],
    max_candidates: wp.int32,
    k: wp.int32,
    min_refine_score: wp.float32,
    split_hashmap_size: wp.int32,
    split_candidates: wp.array[wp.vec2i],
    split_scores: wp.array[wp.float32],
    split_hashmap_keys: wp.array[wp.uint64],
    split_hashmap_values: wp.array[wp.int32],
):
    """Compact the top-K highest-scoring candidates (radix_sort_pairs sorts
    ascending, so top-K is the last K slots) and seed the conflict-resolution
    hashmap with the ones actually worth splitting."""
    tid = wp.tid()
    rank = max_candidates - k + tid
    original_idx = candidate_order[rank]
    edge = candidate_edges[original_idx]
    score = candidate_scores_sorted[rank]

    split_candidates[tid] = edge
    split_scores[tid] = score

    if score > min_refine_score:
        hashmap_insert_edge(split_hashmap_keys, split_hashmap_values, split_hashmap_size, edge, tid)


@wp.kernel
def get_tet_split_edge_candidates_fixed(
    active_tet_count: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    split_hashmap_keys: wp.array[wp.uint64],
    split_hashmap_values: wp.array[wp.int32],
    split_hashmap_size: wp.int32,
    tet_candidates: wp.array2d[wp.int32],
    tet_candidate_ct: wp.array[wp.int32],
):
    """Same logic as mesh_topology.get_tet_split_edge_candidates, but guarded
    on active_tet_count: tids beyond it hold garbage tet_indices (never
    written this pass) and must not be allowed to probe the shared hashmap,
    or they could spuriously steal/pollute a conflict-resolution slot."""
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        tet_candidate_ct[tid] = 0
        return

    count = 0
    for j in range(4):
        for i in range(j):
            edge = canonical_edge(tet_indices[tid, i], tet_indices[tid, j])
            candidate_index = hashmap_find_edge(split_hashmap_keys, split_hashmap_values, split_hashmap_size, edge)
            if candidate_index != -1:
                tet_candidates[tid, count] = candidate_index
                count += 1
    tet_candidate_ct[tid] = count


@wp.kernel
def claim_new_vertices(
    split_candidates: wp.array[wp.vec2i],
    split_hashmap_keys: wp.array[wp.uint64],
    split_hashmap_size: wp.int32,
    old_vertex_count: wp.array[wp.int32],
    max_particles: wp.int32,
    rest_particle_q_0: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    new_vertex_count: wp.array[wp.int32],
    rest_particle_q_1: wp.array[wp.vec3],
    candidate_new_vertex_id: wp.array[wp.int32],
):
    """For every candidate that survived conflict resolution, atomically
    claim a new vertex slot and write its rest + current position/velocity
    as the midpoint of its edge's two endpoints (rest and current
    respectively). Vertex ids are append-only, so writing past
    old_vertex_count never disturbs an existing particle.

    max_particles bounds-checks the claim: it caps how many vertex slots
    actually exist, and is unrelated to K (max_new_vertices_per_refine) --
    without this, a claim past the buffer's end would write out of bounds
    and silently corrupt whatever GPU memory follows it."""
    tid = wp.tid()
    edge = split_candidates[tid]
    if edge[0] < 0:
        return
    if not hashset_find_edge(split_hashmap_keys, split_hashmap_size, edge):
        return

    claimed = wp.atomic_add(new_vertex_count, 0, 1)
    new_idx = old_vertex_count[0] + claimed
    if new_idx >= max_particles:
        return
    candidate_new_vertex_id[tid] = claimed

    a = edge[0]
    b = edge[1]
    particle_q[new_idx] = 0.5 * (particle_q[a] + particle_q[b])
    particle_qd[new_idx] = 0.5 * (particle_qd[a] + particle_qd[b])
    rest_particle_q_1[new_idx] = 0.5 * (rest_particle_q_0[a] + rest_particle_q_0[b])


@wp.func
def copy_unsplit_tet(
    tid: wp.int32,
    base: wp.int32,
    old_tet_indices: wp.array2d[wp.int32],
    old_tet_stretch: wp.array[vec6],
    old_tet_lambda: wp.array[vec6],
    old_tet_materials: wp.array2d[wp.float32],
    old_tet_poses: wp.array[wp.mat33],
    new_tet_indices: wp.array2d[wp.int32],
    new_tet_stretch: wp.array[vec6],
    new_tet_lambda: wp.array[vec6],
    new_tet_materials: wp.array2d[wp.float32],
    new_tet_poses: wp.array[wp.mat33],
):
    for i in range(4):
        new_tet_indices[base, i] = old_tet_indices[tid, i]
    new_tet_stretch[base] = old_tet_stretch[tid]
    new_tet_lambda[base] = old_tet_lambda[tid]
    for i in range(3):
        new_tet_materials[base, i] = old_tet_materials[tid, i]
    new_tet_poses[base] = old_tet_poses[tid]


@wp.kernel
def scatter_refined_tets(
    active_tet_count: wp.array[wp.int32],
    max_tets: wp.int32,
    old_tet_indices: wp.array2d[wp.int32],
    old_tet_stretch: wp.array[vec6],
    old_tet_lambda: wp.array[vec6],
    old_tet_materials: wp.array2d[wp.float32],
    old_tet_poses: wp.array[wp.mat33],
    tet_candidates: wp.array2d[wp.int32],
    tet_candidate_ct: wp.array[wp.int32],
    split_candidates: wp.array[wp.vec2i],
    candidate_new_vertex_id: wp.array[wp.int32],
    old_vertex_count: wp.array[wp.int32],
    rest_particle_q: wp.array[wp.vec3],
    new_tet_count: wp.array[wp.int32],
    new_tet_indices: wp.array2d[wp.int32],
    new_tet_stretch: wp.array[vec6],
    new_tet_lambda: wp.array[vec6],
    new_tet_materials: wp.array2d[wp.float32],
    new_tet_poses: wp.array[wp.mat33],
):
    """Atomic-append replacement for mesh_topology.scatter_tets. An unsplit
    tet copies through to one freshly-claimed slot; a split tet claims two
    consecutive slots and both children replace one endpoint of the winning
    edge with the new vertex. Children inherit tet_stretch/tet_lambda
    verbatim from the parent (exact given the rest-space-midpoint split, see
    module docstring) and get a freshly computed tet_pose from
    rest_particle_q. Output order is not deterministic across runs (which
    physical slot a tet lands in depends on thread scheduling), but the set
    of resulting tets/vertices is.

    max_tets bounds-checks every claim: one accepted candidate edge can be
    shared by many incident tets, all of which split when it does, so the
    number of new tets created per pass is NOT simply bounded by K (the
    number of accepted candidates) -- without this check, enough refine
    passes could claim a slot past the buffer's end and silently corrupt
    whatever GPU memory follows it. A tet whose winning candidate lost its
    vertex claim to this same overflow (candidate_new_vertex_id[cid] == -1)
    falls back to being copied through unsplit, rather than referencing a
    vertex that was never actually created."""
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return

    unsplit = tet_candidate_ct[tid] == 0
    cid = wp.int32(-1)
    new_vid = wp.int32(-1)
    if not unsplit:
        cid = tet_candidates[tid, 0]
        claimed = candidate_new_vertex_id[cid]
        if claimed < 0:
            unsplit = True
        else:
            new_vid = claimed + old_vertex_count[0]

    if unsplit:
        base = wp.atomic_add(new_tet_count, 0, 1)
        if base >= max_tets:
            return
        copy_unsplit_tet(
            tid, base,
            old_tet_indices, old_tet_stretch, old_tet_lambda, old_tet_materials, old_tet_poses,
            new_tet_indices, new_tet_stretch, new_tet_lambda, new_tet_materials, new_tet_poses,
        )
        return

    split_edge = split_candidates[cid]
    base = wp.atomic_add(new_tet_count, 0, 2)

    # Each of the 2 claimed slots is checked independently, not
    # all-or-nothing: atomic_add's claimed ranges gaplessly tile
    # [0, new_tet_count_final), so if a range straddles max_tets (its first
    # slot in bounds, second one not), the in-bounds slot is still owed a
    # write -- every valid index that gets claimed by *any* thread must end
    # up written, or a later kernel iterating up to the (clamped)
    # active_tet_count would read a slot nothing ever populated. The
    # straddling tet's un-writable sibling is simply dropped (loses that one
    # tet, not memory safety) -- the same "drop on overflow" philosophy this
    # codebase already uses for padded-topology BSR row capacity.
    for j in range(2):
        slot = base + j
        if slot >= max_tets:
            continue

        for i in range(4):
            if old_tet_indices[tid, i] == split_edge[j]:
                new_tet_indices[slot, i] = new_vid
            else:
                new_tet_indices[slot, i] = old_tet_indices[tid, i]

        new_tet_stretch[slot] = old_tet_stretch[tid]
        new_tet_lambda[slot] = old_tet_lambda[tid]
        for k in range(3):
            new_tet_materials[slot, k] = old_tet_materials[tid, k]

        r0 = rest_particle_q[new_tet_indices[slot, 0]]
        r1 = rest_particle_q[new_tet_indices[slot, 1]]
        r2 = rest_particle_q[new_tet_indices[slot, 2]]
        r3 = rest_particle_q[new_tet_indices[slot, 3]]
        Dm = wp.matrix_from_cols(r1 - r0, r2 - r0, r3 - r0)
        new_tet_poses[slot] = wp.inverse(Dm)


@wp.kernel
def finalize_active_counts(
    old_particle_count: wp.array[wp.int32],
    max_tets: wp.int32,
    max_particles: wp.int32,
    new_tet_count: wp.array[wp.int32],
    new_vertex_count: wp.array[wp.int32],
    active_tet_count_out: wp.array[wp.int32],
    active_particle_count_out: wp.array[wp.int32],
):
    # new_tet_count/new_vertex_count are raw atomic-add totals and can
    # overshoot the buffer ceiling (claims past it are refused by
    # scatter_refined_tets/claim_new_vertices, but the counters themselves
    # aren't bounds-aware) -- clamp here so active_tet_count/
    # active_particle_count never exceed what's actually allocated, which
    # every other kernel in this solver trusts as an invariant.
    active_tet_count_out[0] = wp.min(new_tet_count[0], max_tets)
    active_particle_count_out[0] = wp.min(old_particle_count[0] + new_vertex_count[0], max_particles)


@wp.kernel
def interpolate_new_particle_mass(
    split_candidates: wp.array[wp.vec2i],
    split_hashmap_keys: wp.array[wp.uint64],
    split_hashmap_size: wp.int32,
    candidate_new_vertex_id: wp.array[wp.int32],
    old_vertex_count: wp.array[wp.int32],
    particle_mass: wp.array[wp.float32],
    particle_inv_mass: wp.array[wp.float32],
):
    """New vertices start with no representation in model.particle_mass at
    all (it's built once, at model-finalize time, from the original mesh) --
    without this, the global system would be singular for their 3 DOFs. Not
    exactly mass-conserving (a small amount of mass is added per split); an
    accepted simplification for a research prototype."""
    tid = wp.tid()
    edge = split_candidates[tid]
    if edge[0] < 0:
        return
    if not hashset_find_edge(split_hashmap_keys, split_hashmap_size, edge):
        return
    claimed = candidate_new_vertex_id[tid]
    if claimed < 0:
        # Claim failed in claim_new_vertices (max_particles exhausted) --
        # skip, or this would write mass into old_vertex_count[0] - 1, i.e.
        # corrupt the last legitimately-existing particle.
        return

    new_idx = old_vertex_count[0] + claimed
    mass = 0.5 * (particle_mass[edge[0]] + particle_mass[edge[1]])
    particle_mass[new_idx] = mass
    if mass > 0.0:
        particle_inv_mass[new_idx] = 1.0 / mass
    else:
        particle_inv_mass[new_idx] = 0.0


@wp.kernel
def compute_should_refine(
    step_counter: wp.array[wp.int32],
    period: wp.int32,
    should_refine: wp.array[wp.int32],
):
    should_refine[0] = wp.where(step_counter[0] % period == 0, 1, 0)


@wp.kernel
def increment_step_counter(step_counter: wp.array[wp.int32]):
    step_counter[0] += 1


class RefinementBuffers:
    """Fixed-capacity scratch state for refine(), allocated once."""

    def __init__(self, max_tets: int, max_new_vertices_per_refine: int):
        self.max_tets = max_tets
        self.max_candidates = 6 * max_tets
        self.k = max_new_vertices_per_refine

        # radix_sort_pairs requires keys/values storage >= 2*count.
        self.candidate_edges = wp.empty(self.max_candidates, dtype=wp.vec2i)
        self.candidate_scores = wp.empty(2 * self.max_candidates, dtype=wp.float32)
        self.candidate_order = wp.empty(2 * self.max_candidates, dtype=wp.int32)
        self.candidate_order_iota = wp.array(np.arange(self.max_candidates, dtype=np.int32))

        self.split_candidates = wp.empty(self.k, dtype=wp.vec2i)
        self.split_scores = wp.empty(self.k, dtype=wp.float32)

        self.split_hashmap_size = get_hashtable_size(self.k, 0.5)
        self.split_hashmap_keys = wp.empty(self.split_hashmap_size, dtype=wp.uint64)
        self.split_hashmap_values = wp.empty(self.split_hashmap_size, dtype=wp.int32)

        self.tet_candidates = wp.empty((max_tets, 6), dtype=wp.int32)
        self.tet_candidate_ct = wp.zeros(max_tets, dtype=wp.int32)

        self.candidate_new_vertex_id = wp.empty(self.k, dtype=wp.int32)

        self.new_tet_count = wp.zeros(1, dtype=wp.int32)
        self.new_vertex_count = wp.zeros(1, dtype=wp.int32)

        self.step_counter = wp.zeros(1, dtype=wp.int32)
        self.should_refine = wp.zeros(1, dtype=wp.int32)


def copy_additional_state(dst: AdditionalState, src: AdditionalState) -> None:
    """Device-side (capture-safe) copy of every field, src -> dst. Called by
    RefinementSolver.step() at the end of every step to bring
    additional_state_0 up to date with additional_state_1's final content,
    in place of a Python reference swap -- see the module docstring for why
    a swap doesn't survive CUDA graph replay here."""
    wp.copy(dst.tet_indices, src.tet_indices)
    wp.copy(dst.tet_stretch, src.tet_stretch)
    wp.copy(dst.tet_lambda, src.tet_lambda)
    wp.copy(dst.tet_materials, src.tet_materials)
    wp.copy(dst.tet_poses, src.tet_poses)
    wp.copy(dst.active_tet_count, src.active_tet_count)
    wp.copy(dst.active_particle_count, src.active_particle_count)
    wp.copy(dst.rest_particle_q, src.rest_particle_q)


def refine(
    solver,
    additional_state_0: AdditionalState,
    additional_state_1: AdditionalState,
    state_out: State,
    min_refine_score: float,
) -> None:

    buffers: RefinementBuffers = solver._refine_buffers
    max_tets = buffers.max_tets

    # Vertex-indexed and append-only (never reassigned to a new slot the way
    # tet-indexed fields can be), so this is always carried forward in full
    # regardless of whether a real refine pass triggers this step.
    wp.copy(additional_state_1.rest_particle_q, additional_state_0.rest_particle_q)

    wp.launch(
        compute_should_refine,
        dim=1,
        inputs=[buffers.step_counter, solver.refine_every_n_steps],
        outputs=[buffers.should_refine],
    )
    wp.launch(increment_step_counter, dim=1, inputs=[], outputs=[buffers.step_counter])

    wp.launch(
        emit_split_edge_candidates,
        dim=max_tets,
        inputs=[
            additional_state_0.active_tet_count,
            additional_state_0.tet_indices,
            # additional_state_0's tet slots line up with solver._elastic_energy
            # because copy_additional_state() (end of the previous step) copies
            # additional_state_1's content into additional_state_0 slot-for-slot,
            # and solver._elastic_energy was last written against
            # additional_state_1 at that same slot during that step's SQP loop.
            solver._elastic_energy,
            buffers.should_refine,
        ],
        outputs=[buffers.candidate_edges, buffers.candidate_scores],
    )

    wp.copy(buffers.candidate_order[: buffers.max_candidates], buffers.candidate_order_iota)
    wp.utils.radix_sort_pairs(buffers.candidate_scores, buffers.candidate_order, buffers.max_candidates)

    buffers.split_hashmap_keys.fill_(wp.uint64(0))
    wp.launch(
        extract_top_k_candidates,
        dim=buffers.k,
        inputs=[
            buffers.candidate_edges,
            buffers.candidate_order,
            buffers.candidate_scores,
            buffers.max_candidates,
            buffers.k,
            min_refine_score,
            buffers.split_hashmap_size,
        ],
        outputs=[
            buffers.split_candidates,
            buffers.split_scores,
            buffers.split_hashmap_keys,
            buffers.split_hashmap_values,
        ],
    )

    wp.launch(
        get_tet_split_edge_candidates_fixed,
        dim=max_tets,
        inputs=[
            additional_state_0.active_tet_count,
            additional_state_0.tet_indices,
            buffers.split_hashmap_keys,
            buffers.split_hashmap_values,
            buffers.split_hashmap_size,
        ],
        outputs=[buffers.tet_candidates, buffers.tet_candidate_ct],
    )

    # A tet can only legally bisect on one edge; greedily strip the
    # worst-scored conflicting edge (globally, via the shared hashmap) until
    # every tet has at most one surviving candidate. 5 is a static constant
    # (mirrors mesh_topology.py), not data-dependent, so this loop needs no
    # wp.capture_while.
    for i in range(5):
        wp.launch(
            remove_worst_candidate,
            dim=max_tets,
            inputs=[
                buffers.tet_candidates,
                buffers.tet_candidate_ct,
                buffers.split_hashmap_keys,
                buffers.split_hashmap_values,
                buffers.split_hashmap_size,
                5 - i,
                buffers.split_candidates,
                buffers.split_scores,
            ],
        )
        wp.launch(
            update_tet_candidates,
            dim=max_tets,
            inputs=[
                buffers.split_hashmap_keys,
                buffers.split_hashmap_values,
                buffers.split_hashmap_size,
                buffers.split_candidates,
                buffers.split_scores,
                buffers.tet_candidates,
                buffers.tet_candidate_ct,
            ],
        )

    buffers.new_tet_count.zero_()
    buffers.new_vertex_count.zero_()
    # -1 sentinel: an accepted candidate's vertex claim can fail if
    # max_particles is exhausted (see claim_new_vertices), and
    # scatter_refined_tets needs to distinguish "claim succeeded at id 0"
    # from "claim never happened."
    buffers.candidate_new_vertex_id.fill_(-1)

    wp.launch(
        claim_new_vertices,
        dim=buffers.k,
        inputs=[
            buffers.split_candidates,
            buffers.split_hashmap_keys,
            buffers.split_hashmap_size,
            additional_state_0.active_particle_count,
            solver.max_particles,
            additional_state_0.rest_particle_q,
            state_out.particle_q,
            state_out.particle_qd,
        ],
        outputs=[
            buffers.new_vertex_count,
            additional_state_1.rest_particle_q,
            buffers.candidate_new_vertex_id,
        ],
    )

    wp.launch(
        scatter_refined_tets,
        dim=max_tets,
        inputs=[
            additional_state_0.active_tet_count,
            max_tets,
            additional_state_0.tet_indices,
            additional_state_0.tet_stretch,
            additional_state_0.tet_lambda,
            additional_state_0.tet_materials,
            additional_state_0.tet_poses,
            buffers.tet_candidates,
            buffers.tet_candidate_ct,
            buffers.split_candidates,
            buffers.candidate_new_vertex_id,
            additional_state_0.active_particle_count,
            additional_state_1.rest_particle_q,
        ],
        outputs=[
            buffers.new_tet_count,
            additional_state_1.tet_indices,
            additional_state_1.tet_stretch,
            additional_state_1.tet_lambda,
            additional_state_1.tet_materials,
            additional_state_1.tet_poses,
        ],
    )

    wp.launch(
        finalize_active_counts,
        dim=1,
        inputs=[additional_state_0.active_particle_count, max_tets, solver.max_particles, buffers.new_tet_count, buffers.new_vertex_count],
        outputs=[additional_state_1.active_tet_count, additional_state_1.active_particle_count],
    )

    wp.launch(
        interpolate_new_particle_mass,
        dim=buffers.k,
        inputs=[
            buffers.split_candidates,
            buffers.split_hashmap_keys,
            buffers.split_hashmap_size,
            buffers.candidate_new_vertex_id,
            additional_state_0.active_particle_count,
            solver.model.particle_mass,
            solver.model.particle_inv_mass,
        ],
    )

    # Topology-dependent scratch (sparse constraint-gradient topology, the
    # mass matrix, and the ARAP Hessian blocks) all go stale the moment
    # tet_indices/active_tet_count change -- and since scatter_refined_tets
    # can reshuffle tet slot order even on a no-op pass (see docstring
    # above), these are refreshed every step, not just on steps that
    # actually split something.
    solver._update_constraint_gradient_topology(additional_state_1)
    solver._refresh_mass_matrix(additional_state_1)
    wp.launch(
        arap_energy_hessian_kernel,
        dim=max_tets,
        inputs=[
            additional_state_1.active_tet_count,
            additional_state_1.tet_materials,
            additional_state_1.tet_poses,
        ],
        outputs=[solver._hessian_ds_blocks, solver._inv_hessian_ds_blocks],
    )
