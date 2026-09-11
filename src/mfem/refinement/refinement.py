import warp as wp
from newton import State, Model, GeoType
from mfem.refinement.additional_state import AdditionalState
from mfem.refinement.geometry_hash import get_hashtable_size, hashtable_find, hashtable_insert
from mfem.refinement.contact import query_min_signed_distance
from mfem.types import vec6

class RefinementBuffers:


    def __init__(
        self,
        max_tets: int,
        max_vertices: int,
        max_tris: int,
        threshold: float = 0.8,
        hashmap_load_factor: float = 0.5,
        hashmap_edges_per_tet: int = 6,
    ):
        self.max_tets = max_tets
        self.max_vertices = max_vertices
        self.max_tris = max_tris
        self.candidate_hashmap_size = get_hashtable_size(max_tets * hashmap_edges_per_tet, hashmap_load_factor)
        self.candidate_hashmap_keys = wp.empty(self.candidate_hashmap_size, dtype=wp.uint64)
        self.candidate_hashmap_scores = wp.empty(self.candidate_hashmap_size, dtype=wp.float32)
        self.candidate_hashmap_flag = wp.empty(self.candidate_hashmap_size, dtype=wp.uint8)
        self.tet_candidate = wp.full(max_tets, wp.vec2i(-1, -1), dtype=wp.vec2i)
        self.tmp_rest_particle_q = wp.empty(max_vertices, dtype=wp.vec3)
        self.tet_split_counts = wp.zeros(max_tets + 1, dtype=wp.int32)
        self.new_vertex_index = wp.zeros(max_tets + 1, dtype=wp.int32)
        self.threshold = wp.array([threshold], dtype=wp.float32)
        # Set by check_split_capacity when a pass would overflow max_tets /
        # max_vertices; the pass is then dropped (see refine()). Read it back
        # from the host to detect a starved refinement.
        self.split_overflow = wp.zeros(1, dtype=wp.int32)
        # Counts refine() calls so refine_every can gate passes on the device
        # (a host-side check would be frozen into a captured CUDA graph).
        self.pass_counter = wp.zeros(1, dtype=wp.int32)
        # [sum of tet strain-energy density / mu, active tet count] for the
        # geometric score's relative elastic term (see elastic_density_stats).
        self.elastic_density_stats = wp.zeros(2, dtype=wp.float32)

        # Surface-triangle counterparts of tet_candidate/tet_split_counts,
        # updated by scatter_tris the same way tet_candidate/tet_split_counts
        # are updated by scatter_tets (see refine() below).
        self.tri_candidate = wp.full(max_tris, wp.vec2i(-1, -1), dtype=wp.vec2i)
        self.tri_split_counts = wp.zeros(max_tris + 1, dtype=wp.int32)

@wp.func
def edge_to_key(edge: wp.vec2i) -> wp.uint64:
    return wp.cast(wp.vec2i(wp.min(*edge), wp.max(*edge)), dtype=wp.uint64)
    

CANDIDATE_AVAILABLE = wp.constant(wp.uint8(0))
CANDIDATE_PENDING = wp.constant(wp.uint8(1))
CANDIDATE_SELECTED = wp.constant(wp.uint8(2))
CANDIDATE_UNAVAILABLE = wp.constant(wp.uint8(3))

# Default score assigned to edges whose midpoint has penetrated a rigid shape, so
# they always win over the threshold. Overridable per-call via refine(..., penetrating_edge_score=).
DEFAULT_PENETRATING_EDGE_SCORE = 1.0e6

# Default weights for the edge refinement score (see populate_candidates):
#   score += (tet_score * tet_score_weight + min(vertex_score) * vertex_score_weight) * edge_length
DEFAULT_TET_SCORE_WEIGHT = 0.0
DEFAULT_VERTEX_SCORE_WEIGHT = 0.1
# Default conflict-resolution sweeps for parallel edge-split selection.
DEFAULT_CONFLICT_ITERATIONS = 5
# Default parametric position of the new vertex along a split edge (0.5 = midpoint).
DEFAULT_SPLIT_POSITION = 0.5

# We need to change this to add more candidates

@wp.kernel
def populate_candidates(
    active_tet_count: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_scores: wp.array[wp.float32], # We wan't to split elements with higher elastic energy. I think this is already volume weighted
    vertex_scores: wp.array[wp.float32],
    particle_inv_mass: wp.array[wp.float32], # We wan't to avoid splitting edges between kinetic vertices
    particle_q: wp.array[wp.vec3], # We want to split longer edges so we need the position data
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_count: wp.int32,
    body_q: wp.array[wp.transform],
    tet_score_weight: wp.float32,
    vertex_score_weight: wp.float32,
    penetrating_edge_score: wp.float32,
    candidate_hashmap_size: wp.int32,
    candidate_hashmap_keys: wp.array[wp.uint64],
    candidate_hashmap_scores: wp.array[wp.float32],
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return


    for i in range(4):
        for j in range(i):
            edge = wp.vec2i(tet_indices[tid, i], tet_indices[tid, j])
            edge_key = edge_to_key(edge)
            if not (particle_inv_mass[edge[0]] == 0.0 and particle_inv_mass[edge[1]] == 0.0):
                index, _is_new_key = hashtable_insert(
                    candidate_hashmap_keys,
                    candidate_hashmap_size,
                    edge_key,
                    wp.uint64(0)
                )

                midpoint = 0.5 * (particle_q[edge[0]] + particle_q[edge[1]])
                midpoint_distance = query_min_signed_distance(
                    midpoint, shape_transform, shape_type, shape_scale, shape_body, shape_count, body_q
                )

                if midpoint_distance < 0.0:
                    wp.atomic_max(candidate_hashmap_scores, index, penetrating_edge_score)
                else:
                    wp.atomic_add(candidate_hashmap_scores, index, (tet_scores[tid] * tet_score_weight + (wp.min(vertex_scores[edge[0]], vertex_scores[edge[1]])) * vertex_score_weight)  * wp.length(particle_q[edge[1]] - particle_q[edge[0]]))

# ---------------------------------------------------------------------------
# "geometric" scoring (refine(..., scoring="geometric"))
#
# The legacy score above sums (energy * length) over every tet incident to an
# edge, so interior edges (valence 5-7) outscore surface edges (valence 2-4)
# for the same physics, the weights carry units (J*m), and contact enters
# only through the *min* endpoint barrier energy plus an SDF query at every
# edge midpoint. The geometric score is dimensionless and per-edge:
#
#   score(e) = max over incident tets / surface tris of
#              (L / h_min) * (L / L_max) * (w_e * s_elastic + w_c * s_contact)
#
#   L / h_min    edge length in units of the minimum edge length h_min; edges
#                shorter than 2 h_min are never split (so no child is < h_min)
#   L / L_max    length relative to the longest edge of that tet / tri:
#                longest-edge bisection, which keeps element quality bounded
#   s_elastic    how far the tet's strain-energy density / mu sits *above the
#                mesh-wide mean*: max(0, rho / mean(rho) - 1). Relative, so a
#                body that is strained everywhere (gravity sag, a rest shape
#                that differs from the start pose) does not split everywhere;
#                only the hot spots do. (Absolute rho / mu when the stats are
#                zero, e.g. in isolation tests.)
#   s_contact    contact proximity (d1 - d) / d1: 0 outside the barrier range,
#                1 at the surface, > 1 when penetrating -- so penetration is
#                still prioritised without a separate override
#
# Contact is taken from the tri-contact data the solver already computes: a
# surface tri within d1 of the tool scores its edges by proximity times
# (1 - bary of the opposite vertex), i.e. the edge nearest the closest point
# wins and the new vertex lands where the tool actually touches. A per-vertex
# term (capsule shapes only, so resting on the table never drives splits)
# covers vertex-only contact. Contributions combine with atomic_max, so an
# edge's score does not grow with its valence.
# ---------------------------------------------------------------------------

@wp.func
def contact_proximity(d: wp.float32, d1: wp.float32) -> wp.float32:
    """0 outside the barrier range, 1 at the surface, > 1 when penetrating."""
    return wp.max((d1 - d) / d1, wp.float32(0.0))


@wp.func
def tet_elastic_density(
    tet_energy: wp.array[wp.float32],
    tet_materials: wp.array2d[wp.float32],
    tet_poses: wp.array[wp.mat33],
    tid: wp.int32,
) -> wp.float32:
    """Strain-energy density over mu of one tet (0 for degenerate data).
    tet_poses holds DmInv, so rest volume = 1 / (6 det(DmInv))."""
    vol = wp.abs(1.0 / (6.0 * wp.determinant(tet_poses[tid])))
    mu = tet_materials[tid, 0]
    if mu > 0.0 and vol > 0.0:
        return wp.max(tet_energy[tid], wp.float32(0.0)) / (vol * mu)
    return wp.float32(0.0)


@wp.kernel
def elastic_density_stats(
    active_tet_count: wp.array[wp.int32],
    tet_energy: wp.array[wp.float32],
    tet_materials: wp.array2d[wp.float32],
    tet_poses: wp.array[wp.mat33],
    stats: wp.array[wp.float32],
):
    """stats[0] += sum of tet strain-energy density / mu, stats[1] += count."""
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return
    wp.atomic_add(stats, 0, tet_elastic_density(tet_energy, tet_materials, tet_poses, tid))
    wp.atomic_add(stats, 1, wp.float32(1.0))


@wp.kernel
def populate_candidates_geometric(
    active_tet_count: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_energy: wp.array[wp.float32],
    tet_materials: wp.array2d[wp.float32],
    tet_poses: wp.array[wp.mat33],
    particle_inv_mass: wp.array[wp.float32],
    particle_q: wp.array[wp.vec3],
    particle_distance: wp.array[wp.float32],
    particle_shape_id: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    contact_d1: wp.float32,
    min_edge_length: wp.float32,
    elastic_weight: wp.float32,
    vertex_contact_weight: wp.float32,
    elastic_stats: wp.array[wp.float32],
    candidate_hashmap_size: wp.int32,
    candidate_hashmap_keys: wp.array[wp.uint64],
    candidate_hashmap_scores: wp.array[wp.float32],
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return

    s_elastic = tet_elastic_density(tet_energy, tet_materials, tet_poses, tid)
    if elastic_stats[1] > 0.0 and elastic_stats[0] > 0.0:
        # Excess over the mesh-wide mean density: only hot spots score.
        mean_density = elastic_stats[0] / elastic_stats[1]
        s_elastic = wp.max(s_elastic / mean_density - 1.0, wp.float32(0.0))

    l_max = wp.float32(0.0)
    for i in range(4):
        for j in range(i):
            l_max = wp.max(l_max, wp.length(particle_q[tet_indices[tid, i]] - particle_q[tet_indices[tid, j]]))
    l_max = wp.max(l_max, wp.float32(1.0e-20))

    for i in range(4):
        for j in range(i):
            edge = wp.vec2i(tet_indices[tid, i], tet_indices[tid, j])
            if not (particle_inv_mass[edge[0]] == 0.0 and particle_inv_mass[edge[1]] == 0.0):
                # Every splittable edge is registered (score 0 if gated) so the
                # later tri pass and the selection kernels can find its slot.
                index, _is_new_key = hashtable_insert(
                    candidate_hashmap_keys,
                    candidate_hashmap_size,
                    edge_to_key(edge),
                    wp.uint64(0),
                )
                L = wp.length(particle_q[edge[1]] - particle_q[edge[0]])
                if L >= 2.0 * min_edge_length:
                    s_contact = wp.float32(0.0)
                    for k in range(2):
                        v = edge[k]
                        sid = particle_shape_id[v]
                        if sid >= 0:
                            if shape_type[sid] == GeoType.CAPSULE:
                                s_contact = wp.max(s_contact, contact_proximity(particle_distance[v], contact_d1))
                    score = (L / min_edge_length) * (L / l_max) * (
                        elastic_weight * s_elastic + vertex_contact_weight * s_contact
                    )
                    wp.atomic_max(candidate_hashmap_scores, index, score)


@wp.kernel
def populate_tri_candidates_geometric(
    active_tri_count: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    particle_inv_mass: wp.array[wp.float32],
    particle_q: wp.array[wp.vec3],
    tri_distance: wp.array[wp.float32],
    tri_bary: wp.array[wp.vec3],
    contact_d1: wp.float32,
    min_edge_length: wp.float32,
    tri_contact_weight: wp.float32,
    candidate_hashmap_size: wp.int32,
    candidate_hashmap_keys: wp.array[wp.uint64],
    candidate_hashmap_scores: wp.array[wp.float32],
):
    """Run after populate_candidates_geometric: raises the score of the edges
    of every surface tri inside the contact barrier range, favouring the edge
    nearest the tri's closest point to the tool."""
    tid = wp.tid()
    if tid >= active_tri_count[0]:
        return
    d = tri_distance[tid]
    if d >= contact_d1:
        return
    prox = contact_proximity(d, contact_d1)
    bary = tri_bary[tid]

    l_max = wp.float32(1.0e-20)
    for k in range(3):
        i = k + 1
        if i > 2:
            i = i - 3
        l_max = wp.max(l_max, wp.length(particle_q[tri_indices[tid, i]] - particle_q[tri_indices[tid, k]]))

    for k in range(3):
        # Edge opposite vertex k.
        i = k + 1
        if i > 2:
            i = i - 3
        j = k + 2
        if j > 2:
            j = j - 3
        edge = wp.vec2i(tri_indices[tid, i], tri_indices[tid, j])
        if not (particle_inv_mass[edge[0]] == 0.0 and particle_inv_mass[edge[1]] == 0.0):
            L = wp.length(particle_q[edge[1]] - particle_q[edge[0]])
            if L >= 2.0 * min_edge_length:
                key = edge_to_key(edge)
                index = hashtable_find(
                    candidate_hashmap_keys,
                    candidate_hashmap_size,
                    key,
                    wp.uint64(0),
                )
                if candidate_hashmap_keys[index] == key:
                    # 1 when the closest point lies on this edge, 2/3 at the centroid.
                    w = 1.0 - bary[k]
                    score = (L / min_edge_length) * (L / l_max) * tri_contact_weight * prox * w
                    wp.atomic_max(candidate_hashmap_scores, index, score)


@wp.kernel
def get_tet_candidate(
    active_tet_count: wp.array[wp.int32],
    threshold: wp.array[wp.float32],
    tet_indices: wp.array2d[wp.int32],
    candidate_hashmap_size: wp.int32,
    candidate_hashmap_keys: wp.array[wp.uint64],
    candidate_hashmap_scores: wp.array[wp.float32],
    candidate_hashmap_flag: wp.array[wp.uint8],
    tet_candidate: wp.array[wp.vec2i],
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return
    
    high_score = threshold[0]
    high_score_edge = wp.vec2i(-1, -1)
    high_score_index = wp.int32(0)


    for i in range(4):
        for j in range(i):
            edge = wp.vec2i(tet_indices[tid, i], tet_indices[tid, j])
            key = edge_to_key(edge)

            index =  hashtable_find(
                candidate_hashmap_keys,
                candidate_hashmap_size,
                key,
                wp.uint64(0)
            )


            if candidate_hashmap_keys[index] == key and candidate_hashmap_scores[index] > high_score:
                high_score = candidate_hashmap_scores[index]
                high_score_edge = edge
                high_score_index = index

    if high_score > threshold[0]:
        candidate_hashmap_flag[high_score_index] = CANDIDATE_PENDING
    tet_candidate[tid] = high_score_edge

@wp.kernel
def remove_conflicting_candidates(
    active_tet_count: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_candidate: wp.array[wp.vec2i],
    candidate_hashmap_size: wp.int32,
    candidate_hashmap_keys: wp.array[wp.uint64],
    candidate_hashmap_scores: wp.array[wp.float32],
    candidate_hashmap_flag: wp.array[wp.uint8],
):
    tid = wp.tid()
    if tid >= active_tet_count[0] or tet_candidate[tid][0] == -1:
        return
    
    tet_candidate_key = edge_to_key(tet_candidate[tid])
    candidate_index = hashtable_find(
        candidate_hashmap_keys,
        candidate_hashmap_size,
        tet_candidate_key,
        wp.uint64(0),
    )


    for i in range(4):
        for j in range(i):

            edge = wp.vec2i(tet_indices[tid, i], tet_indices[tid, j])
            edge_key = edge_to_key(edge)

            if edge_key != tet_candidate_key:
                index = hashtable_find(
                    candidate_hashmap_keys,
                    candidate_hashmap_size,
                    edge_key,
                    wp.uint64(0),
                )

                # Send this candidate back to being available
                if candidate_hashmap_flag[index] == CANDIDATE_PENDING:
                    candidate_hashmap_flag[index] = CANDIDATE_AVAILABLE



@wp.kernel
def finalize_nonconflicting_candidates(
    active_tet_count: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_candidate: wp.array[wp.vec2i],
    candidate_hashmap_size: wp.int32,
    candidate_hashmap_keys: wp.array[wp.uint64],
    candidate_hashmap_scores: wp.array[wp.float32],
    candidate_hashmap_flag: wp.array[wp.uint8],
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return
    
    candidate = tet_candidate[tid]
    # We have already established there are no viable edge splits for this tetrahedron
    if candidate[0] == -1:
        return
    candidate_key = edge_to_key(candidate)

    candidate_index = hashtable_find(
        candidate_hashmap_keys,
        candidate_hashmap_size,
        candidate_key,
        wp.uint64(0),
    )
    

    if candidate_hashmap_flag[candidate_index] != CANDIDATE_PENDING and candidate_hashmap_flag[candidate_index] != CANDIDATE_SELECTED:
        return
    
    candidate_hashmap_flag[candidate_index] = CANDIDATE_SELECTED

    # We have found a candidate that can be finalized so lets set all the other edges on the tet so that they cannot be claimed by any other tetrahedra
    for i in range(4):
        for j in range(i):
            edge = wp.vec2i(tet_indices[tid, i], tet_indices[tid, j])
            edge_key = edge_to_key(edge)

            if edge_key != candidate_key:

                index = hashtable_find(
                    candidate_hashmap_keys,
                    candidate_hashmap_size,
                    edge_key,
                    wp.uint64(0),
                )

                candidate_hashmap_flag[index] = CANDIDATE_UNAVAILABLE




@wp.kernel
def udpate_tet_candidate(
    active_tet_count: wp.array[wp.int32],
    threshold: wp.array[wp.float32],
    tet_indices: wp.array2d[wp.int32],
    tet_candidate: wp.array[wp.vec2i],
    candidate_hashmap_size: wp.int32,
    candidate_hashmap_keys: wp.array[wp.uint64],
    candidate_hashmap_scores: wp.array[wp.float32],
    candidate_hashmap_flag: wp.array[wp.uint8],
):
    tid = wp.tid()
    if tid >= active_tet_count[0] and tet_candidate[tid][0] != -1:
        return
    
    high_score = threshold[0]
    high_score_edge = wp.vec2i(-1, -1)
    high_score_index = wp.int32(0)

    for i in range(4):
        for j in range(i):
            edge = wp.vec2i(tet_indices[tid, i], tet_indices[tid, j])
            edge_key = edge_to_key(edge)

            index = hashtable_find(
                candidate_hashmap_keys,
                candidate_hashmap_size,
                edge_key,
                wp.uint64(0)
            )

            # Select the highest scored available candidate or if there is a selected candidate incident select that one
            if candidate_hashmap_keys[index] == edge_key and ((candidate_hashmap_scores[index] > high_score and candidate_hashmap_flag[index] != CANDIDATE_UNAVAILABLE) or candidate_hashmap_flag[index] == CANDIDATE_SELECTED):
                high_score_edge = edge
                high_score = candidate_hashmap_scores[index]
                high_score_index = index

    if high_score > threshold[0] and candidate_hashmap_flag[high_score_index] != CANDIDATE_SELECTED:
        candidate_hashmap_flag[high_score_index] = CANDIDATE_PENDING
    tet_candidate[tid] = high_score_edge


@wp.kernel
def invalidate_failed_candidates(
    active_tet_count: wp.array[wp.int32],
    tet_candidate: wp.array[wp.vec2i],
    candidate_hashtable_size: wp.int32,
    candidate_hashtable_keys: wp.array[wp.uint64],
    candidate_hashtable_scores: wp.array[wp.float32],
    candidate_hashtable_flag: wp.array[wp.uint8],
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return

    candidate = tet_candidate[tid]

    if candidate[0] == -1:
        return

    candidate_key = edge_to_key(candidate)

    index = hashtable_find(
        candidate_hashtable_keys,
        candidate_hashtable_size,
        candidate_key,
        wp.uint64(0),
    )

    if candidate_hashtable_flag[index] != CANDIDATE_SELECTED:
        tet_candidate[tid] = wp.vec2i(-1, -1)
    

# @wp.kernel
# def threshold_candidates(
#     active_tet_count: wp.array[wp.int32],
#     tet_candidate: wp.array[wp.vec2i],
#     candidate_hashmap_size: wp.int32,
#     candidate_hashmap_keys: wp.array[wp.uint64],
#     candidate_hashmap_scores: wp.array[wp.float32],
#     threshold: wp.array[wp.float32],
# ):
#     tid = wp.tid()
#     if tid > active_tet_count[0]:
#         return
    
#     candidate = tet_candidate[tid]
#     candidate_key = edge_to_key(candidate)

#     index = hashtable_find(
#         candidate_hashmap_keys,
#         candidate_hashmap_size,
#         edge_to_key(candidate),
#         wp.uint64(0),
#     )

#     if candidate_hashmap_scores[index] < threshold[0]:
#         tet_candidate[tid] = wp.vec2i(-1, -1)

#     pass

    

@wp.kernel
def gate_refine_pass(
    pass_counter: wp.array[wp.int32],
    refine_every: wp.int32,
    tet_candidate: wp.array[wp.vec2i],
):
    """Clear every candidate unless this is a refining pass (counter % every == 0)."""
    tid = wp.tid()
    if refine_every > 1:
        if pass_counter[0] % refine_every != 0:
            tet_candidate[tid] = wp.vec2i(-1, -1)


@wp.kernel
def advance_pass_counter(pass_counter: wp.array[wp.int32]):
    pass_counter[0] = pass_counter[0] + 1


@wp.kernel
def check_split_capacity(
    tet_index_map: wp.array[wp.int32],
    new_vertex_index_map: wp.array[wp.int32],
    old_active_particle_count: wp.array[wp.int32],
    max_tets: wp.int32,
    max_vertices: wp.int32,
    overflow: wp.array[wp.int32],
):
    """After the exclusive scans the last entries hold the pass's new tet
    total and new-vertex count; flag the pass if either exceeds capacity."""
    n_tets = tet_index_map[tet_index_map.shape[0] - 1]
    n_new_vertices = new_vertex_index_map[new_vertex_index_map.shape[0] - 1]
    if n_tets > max_tets or old_active_particle_count[0] + n_new_vertices > max_vertices:
        overflow[0] = 1
    else:
        overflow[0] = 0


@wp.kernel
def drop_candidates_on_overflow(
    overflow: wp.array[wp.int32],
    tet_candidate: wp.array[wp.vec2i],
):
    tid = wp.tid()
    if overflow[0] != 0:
        tet_candidate[tid] = wp.vec2i(-1, -1)


@wp.kernel
def populate_chosen_candidates(
    active_tet_count: wp.array[wp.int32],
    tet_candidate: wp.array[wp.vec2i],
    candidate_hashtable_size: wp.int32,
    candidate_hashtable_keys: wp.array[wp.uint64],
    tet_split_counts: wp.array[wp.int32],
    new_vertex_predicate: wp.array[wp.int32],
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return

    candidate = tet_candidate[tid]

    if candidate[0] == -1:
        tet_split_counts[tid] = 1
        new_vertex_predicate[tid] = 0
        return

    tet_split_counts[tid] = 2

    _index, is_new = hashtable_insert(
        candidate_hashtable_keys,
        candidate_hashtable_size,
        edge_to_key(candidate),
        wp.uint64(0)
    )

    if is_new:
        new_vertex_predicate[tid] = 1
    else:
        new_vertex_predicate[tid] = 0

@wp.kernel
def validate_candidate_selection(
    active_tet_count: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    candidate_hashtable_size: wp.int32,
    candidate_hashtable_keys: wp.array[wp.uint64],
    is_valid: wp.array[wp.int32],
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return
    count = wp.int32(0)

    for i in range(4):
        for j in range(i):
            edge = wp.vec2i(tet_indices[tid, i], tet_indices[tid, j])
            key = edge_to_key(edge)

            index = hashtable_find(
                candidate_hashtable_keys,
                candidate_hashtable_size,
                key,
                wp.uint64(0),
            )

            if candidate_hashtable_keys[index] == key:
                count += 1

    is_valid[tid] = wp.int32(count <= 1)

@wp.kernel
def make_candidate_new_vertex_mapping(
    active_tet_count: wp.array[wp.int32],
    active_particle_count: wp.array[wp.int32],
    tet_candidate: wp.array[wp.vec2i],
    new_vertex_index_map: wp.array[wp.int32],
    candidate_hashmap_size: wp.int32,
    candidate_hashmap_keys: wp.array[wp.uint64],
    canidate_hashmap_new_vertex: wp.array[wp.int32],  
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return

    if new_vertex_index_map[tid] < new_vertex_index_map[tid + 1]:
        candidate_key = edge_to_key(tet_candidate[tid])

        index = hashtable_find(
            candidate_hashmap_keys,
            candidate_hashmap_size,
            candidate_key,
            wp.uint64(0),
        )

        canidate_hashmap_new_vertex[index] = new_vertex_index_map[tid] + active_particle_count[0]

@wp.kernel
def claim_new_vertices(
    candidate_hashmap_size: wp.int32,
    candidate_hashmap_keys: wp.array[wp.uint64],
    candidate_hashmap_new_vertex: wp.array[wp.int32],
    old_active_tet_count: wp.array[wp.int32],
    tet_candidate: wp.array[wp.vec2i],
    new_vertex_index_map: wp.array[wp.int32],
    split_t: wp.float32,
    new_particle_q: wp.array[wp.vec3],
    new_particle_qd: wp.array[wp.vec3],
    new_rest_particle_q: wp.array[wp.vec3],
):
    tid = wp.tid()
    if tid >= old_active_tet_count[0]:
        return

    # Since this comes from an exclusive prefix scan if the next index is higher that means we should insert a vertex from this tet.
    if new_vertex_index_map[tid] >= new_vertex_index_map[tid + 1]:
        return

    candidate = tet_candidate[tid]
    candidate_key = edge_to_key(candidate)
    index = hashtable_find(
        candidate_hashmap_keys,
        candidate_hashmap_size,
        candidate_key,
        wp.uint64(0)
    )
    new_vertex_index = candidate_hashmap_new_vertex[index]

    new_particle_q[new_vertex_index] = new_particle_q[candidate[0]] + (new_particle_q[candidate[1]] - new_particle_q[candidate[0]]) * split_t
    new_particle_qd[new_vertex_index] = new_particle_qd[candidate[0]] + (new_particle_qd[candidate[1]] - new_particle_qd[candidate[0]]) * split_t
    new_rest_particle_q[new_vertex_index] = new_rest_particle_q[candidate[0]] + (new_rest_particle_q[candidate[1]] - new_rest_particle_q[candidate[0]]) * split_t


@wp.kernel
def scatter_tets(
    candidate_hashmap_size: wp.int32,
    candidate_hashmap_keys: wp.array[wp.uint64],
    candidate_hashmap_new_vertex: wp.array[wp.int32],
    old_active_tet_count: wp.array[wp.int32],
    old_tet_indices: wp.array2d[wp.int32],
    old_tet_stretch: wp.array[vec6],
    old_tet_poses: wp.array[wp.mat33],
    old_tet_lambda: wp.array[vec6],
    old_tet_materials: wp.array2d[wp.float32],
    tet_candidate: wp.array[wp.vec2i],
    tet_index_map: wp.array[wp.int32],
    density: wp.float32,
    new_rest_particle_q: wp.array[wp.vec3],
    new_particle_mass: wp.array[wp.float32],
    new_tet_indices: wp.array2d[wp.int32],
    new_tet_stretch: wp.array[vec6],
    new_tet_poses: wp.array[wp.mat33],
    new_tet_lambda: wp.array[vec6],
    new_tet_materials: wp.array2d[wp.float32],
):
    tid = wp.tid()
    if tid >= old_active_tet_count[0]:
        return

    if tet_index_map[tid] + 1 >= tet_index_map[tid + 1]:
        new_tet_indices[tet_index_map[tid], 0] = old_tet_indices[tid, 0]
        new_tet_indices[tet_index_map[tid], 1] = old_tet_indices[tid, 1]
        new_tet_indices[tet_index_map[tid], 2] = old_tet_indices[tid, 2]
        new_tet_indices[tet_index_map[tid], 3] = old_tet_indices[tid, 3]

        new_tet_stretch[tet_index_map[tid]] = old_tet_stretch[tid]
        new_tet_poses[tet_index_map[tid]] = old_tet_poses[tid]
        new_tet_lambda[tet_index_map[tid]] = old_tet_lambda[tid]
        new_tet_materials[tet_index_map[tid], 0] = old_tet_materials[tid, 0]
        new_tet_materials[tet_index_map[tid], 1] = old_tet_materials[tid, 1]
        new_tet_materials[tet_index_map[tid], 2] = old_tet_materials[tid, 2]

        tet_mass = density / (6.0 * wp.determinant(old_tet_poses[tid]))

        for i in range(4):
            new_particle_mass[old_tet_indices[tid, i]] += tet_mass / 4.0

        return



    candidate = tet_candidate[tid]
    candidate_key = edge_to_key(candidate)
    index = hashtable_find(
        candidate_hashmap_keys,
        candidate_hashmap_size,
        candidate_key,
        wp.uint64(0)
    )
    new_vertex_index = candidate_hashmap_new_vertex[index]

    split_edge_index = wp.vec2i(0, 0)
    for i in range(4):
        for j in range(i):
            edge_key = edge_to_key(wp.vec2i(old_tet_indices[tid, i], old_tet_indices[tid, j]))

            if edge_key == candidate_key:
                split_edge_index = wp.vec2i(i, j)
    
    for i in range(2):
        new_tet_indices[tet_index_map[tid] + i, 0] = old_tet_indices[tid, 0]
        new_tet_indices[tet_index_map[tid] + i, 1] = old_tet_indices[tid, 1]
        new_tet_indices[tet_index_map[tid] + i, 2] = old_tet_indices[tid, 2]
        new_tet_indices[tet_index_map[tid] + i, 3] = old_tet_indices[tid, 3]
        new_tet_indices[tet_index_map[tid] + i, split_edge_index[i]] = new_vertex_index

        t0 = new_rest_particle_q[new_tet_indices[tet_index_map[tid] + i,  0]]
        t1 = new_rest_particle_q[new_tet_indices[tet_index_map[tid] + i,  1]]
        t2 = new_rest_particle_q[new_tet_indices[tet_index_map[tid] + i,  2]]
        t3 = new_rest_particle_q[new_tet_indices[tet_index_map[tid] + i,  3]]
        D_m = wp.matrix_from_cols(t1 - t0, t2 - t0, t3 - t0)
        volume = wp.determinant(D_m) / 6.0
        tet_mass = volume * density

        for j in range(4):
            new_particle_mass[new_tet_indices[tet_index_map[tid] + i, j]] += tet_mass / 4.0

        new_tet_poses[tet_index_map[tid] + i] = wp.inverse(D_m)

        new_tet_stretch[tet_index_map[tid] + i] = old_tet_stretch[tid]
        new_tet_lambda[tet_index_map[tid] + i] = old_tet_lambda[tid]
        new_tet_materials[tet_index_map[tid] + i, 0] = old_tet_materials[tid, 0]
        new_tet_materials[tet_index_map[tid] + i, 1] = old_tet_materials[tid, 1]
        new_tet_materials[tet_index_map[tid] + i, 2] = old_tet_materials[tid, 2]



@wp.kernel
def get_tri_candidate(
    active_tri_count: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    candidate_hashmap_size: wp.int32,
    candidate_hashmap_keys: wp.array[wp.uint64],
    tri_candidate: wp.array[wp.vec2i],
    tri_split_counts: wp.array[wp.int32],
):
    """Per active surface tri, find whether one of its 3 edges was finalized
    as a split edge this refine pass (candidate_hashmap_keys has already been
    rebuilt by populate_chosen_candidates to contain only those edges), and
    set the 1/2 split count the same way populate_chosen_candidates does for
    tets."""
    tid = wp.tid()
    if tid >= active_tri_count[0]:
        return

    found_edge = wp.vec2i(-1, -1)
    for i in range(3):
        for j in range(i):
            edge = wp.vec2i(tri_indices[tid, i], tri_indices[tid, j])
            key = edge_to_key(edge)

            index = hashtable_find(
                candidate_hashmap_keys,
                candidate_hashmap_size,
                key,
                wp.uint64(0),
            )

            if candidate_hashmap_keys[index] == key:
                found_edge = edge

    tri_candidate[tid] = found_edge
    if found_edge[0] == -1:
        tri_split_counts[tid] = 1
    else:
        tri_split_counts[tid] = 2


@wp.kernel
def scatter_tris(
    candidate_hashmap_size: wp.int32,
    candidate_hashmap_keys: wp.array[wp.uint64],
    candidate_hashmap_new_vertex: wp.array[wp.int32],
    old_active_tri_count: wp.array[wp.int32],
    old_tri_indices: wp.array2d[wp.int32],
    tri_candidate: wp.array[wp.vec2i],
    tri_index_map: wp.array[wp.int32],
    new_tri_indices: wp.array2d[wp.int32],
):
    tid = wp.tid()
    if tid >= old_active_tri_count[0]:
        return

    if tri_index_map[tid] + 1 >= tri_index_map[tid + 1]:
        new_tri_indices[tri_index_map[tid], 0] = old_tri_indices[tid, 0]
        new_tri_indices[tri_index_map[tid], 1] = old_tri_indices[tid, 1]
        new_tri_indices[tri_index_map[tid], 2] = old_tri_indices[tid, 2]
        return

    candidate = tri_candidate[tid]
    candidate_key = edge_to_key(candidate)
    index = hashtable_find(
        candidate_hashmap_keys,
        candidate_hashmap_size,
        candidate_key,
        wp.uint64(0)
    )
    new_vertex_index = candidate_hashmap_new_vertex[index]

    split_edge_index = wp.vec2i(0, 0)
    for i in range(3):
        for j in range(i):
            edge_key = edge_to_key(wp.vec2i(old_tri_indices[tid, i], old_tri_indices[tid, j]))

            if edge_key == candidate_key:
                split_edge_index = wp.vec2i(i, j)

    for i in range(2):
        new_tri_indices[tri_index_map[tid] + i, 0] = old_tri_indices[tid, 0]
        new_tri_indices[tri_index_map[tid] + i, 1] = old_tri_indices[tid, 1]
        new_tri_indices[tri_index_map[tid] + i, 2] = old_tri_indices[tid, 2]
        new_tri_indices[tri_index_map[tid] + i, split_edge_index[i]] = new_vertex_index


def refine(
    model: Model,
    density: float,
    state_in: State,
    state_out: State,
    additional_state_in: AdditionalState,
    additional_state_out: AdditionalState,
    refinement_buffers: RefinementBuffers,
    tet_scores: wp.array[wp.float32],
    vertex_scores: wp.array[wp.float32],
    tet_score_weight: float = DEFAULT_TET_SCORE_WEIGHT,
    vertex_score_weight: float = DEFAULT_VERTEX_SCORE_WEIGHT,
    penetrating_edge_score: float = DEFAULT_PENETRATING_EDGE_SCORE,
    conflict_iterations: int = DEFAULT_CONFLICT_ITERATIONS,
    split_position: float = DEFAULT_SPLIT_POSITION,
    scoring: str = "legacy",
    particle_distance: wp.array | None = None,
    particle_shape_id: wp.array | None = None,
    tri_distance: wp.array | None = None,
    tri_bary: wp.array | None = None,
    contact_d1: float = 0.0,
    min_edge_length: float = 0.0,
    elastic_weight: float = 1.0,
    vertex_contact_weight: float = 1.0,
    tri_contact_weight: float = 1.0,
    refine_every: int = 1,
):
    """Split edges of ``additional_state_in`` into ``additional_state_out``.

    ``refine_every``: only every N-th call actually splits (the others still
    forward the topology unchanged); counted on the device so it survives
    CUDA-graph capture.

    ``scoring`` selects how candidate edges are scored:

    * ``"legacy"`` -- energy-weighted sum over incident tets plus the
      midpoint-penetration override (``tet_score_weight`` /
      ``vertex_score_weight`` / ``penetrating_edge_score``).
    * ``"geometric"`` -- dimensionless per-edge score from edge length,
      longest-edge ratio, elastic energy density and contact proximity; see
      populate_candidates_geometric. Needs ``particle_distance`` /
      ``particle_shape_id`` (Contact.distance / Contact.shape_id) and, for the
      tri term, ``tri_distance`` / ``tri_bary``, all evaluated at
      ``state_in``; ``contact_d1`` is the barrier range and
      ``min_edge_length`` the shortest edge the refinement may create.
    """

    refinement_buffers.candidate_hashmap_keys.zero_()
    refinement_buffers.candidate_hashmap_scores.zero_()
    refinement_buffers.candidate_hashmap_flag.zero_()
    # Add candidates to hashmap
    

    if scoring == "legacy":
        wp.launch(
            populate_candidates,
            dim=refinement_buffers.max_tets,
            inputs=[
                additional_state_in.active_tet_count,
                additional_state_in.tet_indices,
                tet_scores,
                vertex_scores,
                model.particle_inv_mass,
                state_in.particle_q,
                model.shape_transform,
                model.shape_type,
                model.shape_scale,
                model.shape_body,
                model.shape_count,
                state_in.body_q,
                tet_score_weight,
                vertex_score_weight,
                penetrating_edge_score,
                refinement_buffers.candidate_hashmap_size,
            ],
            outputs=[
                refinement_buffers.candidate_hashmap_keys,
                refinement_buffers.candidate_hashmap_scores,
            ]
        )
    elif scoring == "geometric":
        if particle_distance is None or particle_shape_id is None:
            raise ValueError("scoring='geometric' needs particle_distance and particle_shape_id")
        if min_edge_length <= 0.0 or contact_d1 <= 0.0:
            raise ValueError("scoring='geometric' needs min_edge_length > 0 and contact_d1 > 0")
        refinement_buffers.elastic_density_stats.zero_()
        wp.launch(
            elastic_density_stats,
            dim=refinement_buffers.max_tets,
            inputs=[
                additional_state_in.active_tet_count,
                tet_scores,
                additional_state_in.tet_materials,
                additional_state_in.tet_poses,
            ],
            outputs=[refinement_buffers.elastic_density_stats],
        )
        wp.launch(
            populate_candidates_geometric,
            dim=refinement_buffers.max_tets,
            inputs=[
                additional_state_in.active_tet_count,
                additional_state_in.tet_indices,
                tet_scores,
                additional_state_in.tet_materials,
                additional_state_in.tet_poses,
                model.particle_inv_mass,
                state_in.particle_q,
                particle_distance,
                particle_shape_id,
                model.shape_type,
                float(contact_d1),
                float(min_edge_length),
                float(elastic_weight),
                float(vertex_contact_weight),
                refinement_buffers.elastic_density_stats,
                refinement_buffers.candidate_hashmap_size,
            ],
            outputs=[
                refinement_buffers.candidate_hashmap_keys,
                refinement_buffers.candidate_hashmap_scores,
            ],
        )
        if tri_distance is not None and tri_bary is not None:
            wp.launch(
                populate_tri_candidates_geometric,
                dim=refinement_buffers.max_tris,
                inputs=[
                    additional_state_in.active_tri_count,
                    additional_state_in.tri_indices,
                    model.particle_inv_mass,
                    state_in.particle_q,
                    tri_distance,
                    tri_bary,
                    float(contact_d1),
                    float(min_edge_length),
                    float(tri_contact_weight),
                    refinement_buffers.candidate_hashmap_size,
                ],
                outputs=[
                    refinement_buffers.candidate_hashmap_keys,
                    refinement_buffers.candidate_hashmap_scores,
                ],
            )
    else:
        raise ValueError(f"Unknown refinement scoring {scoring!r} (expected 'legacy' or 'geometric')")

    wp.launch(
        get_tet_candidate,
        dim=refinement_buffers.max_tets,
        inputs=[
            additional_state_in.active_tet_count,
            refinement_buffers.threshold,
            additional_state_in.tet_indices,
            refinement_buffers.candidate_hashmap_size,
            refinement_buffers.candidate_hashmap_keys,
            refinement_buffers.candidate_hashmap_scores,
            refinement_buffers.candidate_hashmap_flag,
        ],
        outputs=[
            refinement_buffers.tet_candidate,
        ]
    )


    for _ in range(conflict_iterations):
        wp.launch(
            remove_conflicting_candidates,
            dim=refinement_buffers.max_tets,
            inputs=[
                additional_state_in.active_tet_count,
                additional_state_in.tet_indices,
                refinement_buffers.tet_candidate,
                refinement_buffers.candidate_hashmap_size,
                refinement_buffers.candidate_hashmap_keys,
                refinement_buffers.candidate_hashmap_scores,
                refinement_buffers.candidate_hashmap_flag,
            ]
        )


        wp.launch(
            finalize_nonconflicting_candidates,
            dim=refinement_buffers.max_tets,
            inputs=[
                additional_state_in.active_tet_count,
                additional_state_in.tet_indices,
                refinement_buffers.tet_candidate,
                refinement_buffers.candidate_hashmap_size,
                refinement_buffers.candidate_hashmap_keys,
                refinement_buffers.candidate_hashmap_scores,
                refinement_buffers.candidate_hashmap_flag,
            ]
        )

        wp.launch(
            udpate_tet_candidate,
            dim=refinement_buffers.max_tets,
            inputs=[
                additional_state_in.active_tet_count,
                refinement_buffers.threshold,
                additional_state_in.tet_indices,
                refinement_buffers.tet_candidate,
                refinement_buffers.candidate_hashmap_size,
                refinement_buffers.candidate_hashmap_keys,
                refinement_buffers.candidate_hashmap_scores,
                refinement_buffers.candidate_hashmap_flag,
            ]
        )




    wp.launch(
        remove_conflicting_candidates,
        dim=refinement_buffers.max_tets,
        inputs=[
            additional_state_in.active_tet_count,
            additional_state_in.tet_indices,
            refinement_buffers.tet_candidate,
            refinement_buffers.candidate_hashmap_size,
            refinement_buffers.candidate_hashmap_keys,
            refinement_buffers.candidate_hashmap_scores,
            refinement_buffers.candidate_hashmap_flag,
        ]
    )
    

    wp.launch(
        invalidate_failed_candidates,
        dim=refinement_buffers.max_tets,
        inputs=[
            additional_state_in.active_tet_count,
            refinement_buffers.tet_candidate,
            refinement_buffers.candidate_hashmap_size,
            refinement_buffers.candidate_hashmap_keys,
            refinement_buffers.candidate_hashmap_scores,
            refinement_buffers.candidate_hashmap_flag,
        ]
    )

    wp.launch(
        gate_refine_pass,
        dim=refinement_buffers.max_tets,
        inputs=[refinement_buffers.pass_counter, int(refine_every)],
        outputs=[refinement_buffers.tet_candidate],
    )
    wp.launch(advance_pass_counter, dim=1, inputs=[refinement_buffers.pass_counter])

    tet_index_map = refinement_buffers.tet_split_counts
    new_vertex_index_map = refinement_buffers.new_vertex_index

    def count_and_scan_splits():
        # Rebuild the hashmap from the finalized candidates only, count the
        # tets / new vertices each tet contributes, and exclusive-scan both so
        # the last entries hold the totals.
        refinement_buffers.candidate_hashmap_keys.zero_()
        refinement_buffers.tet_split_counts.zero_()
        refinement_buffers.new_vertex_index.zero_()
        wp.launch(
            populate_chosen_candidates,
            dim=refinement_buffers.max_tets,
            inputs=[
                additional_state_in.active_tet_count,
                refinement_buffers.tet_candidate,
                refinement_buffers.candidate_hashmap_size,
                refinement_buffers.candidate_hashmap_keys,
                refinement_buffers.tet_split_counts,
                refinement_buffers.new_vertex_index,
            ]
        )
        wp.utils.array_scan(refinement_buffers.tet_split_counts, tet_index_map, inclusive=False)
        wp.utils.array_scan(new_vertex_index_map, refinement_buffers.new_vertex_index, inclusive=False)

    count_and_scan_splits()

    # Capacity guard: a pass that would overflow max_tets / max_vertices is
    # dropped wholesale (every candidate cleared, counts rebuilt) rather than
    # writing past the buffers. Done on the device so it is CUDA-graph safe.
    wp.launch(
        check_split_capacity,
        dim=1,
        inputs=[
            tet_index_map,
            new_vertex_index_map,
            additional_state_in.active_particle_count,
            refinement_buffers.max_tets,
            refinement_buffers.max_vertices,
        ],
        outputs=[refinement_buffers.split_overflow],
    )
    wp.launch(
        drop_candidates_on_overflow,
        dim=refinement_buffers.max_tets,
        inputs=[refinement_buffers.split_overflow],
        outputs=[refinement_buffers.tet_candidate],
    )
    count_and_scan_splits()

    wp.copy(additional_state_out.active_tet_count, tet_index_map[-1:])
    additional_state_out.active_particle_count += new_vertex_index_map[-1:]

    candidate_new_vertex_hashmap_indices = refinement_buffers.candidate_hashmap_scores.view(dtype=wp.int32)

    wp.launch(
        make_candidate_new_vertex_mapping,
        dim=refinement_buffers.max_tets,
        inputs=[
            additional_state_in.active_tet_count,
            additional_state_in.active_particle_count,
            refinement_buffers.tet_candidate,
            new_vertex_index_map,
            refinement_buffers.candidate_hashmap_size,
            refinement_buffers.candidate_hashmap_keys,
        ],
        outputs=[
            candidate_new_vertex_hashmap_indices,
        ]
    )

    wp.launch(
        claim_new_vertices,
        dim=refinement_buffers.max_tets,
        inputs=[
            refinement_buffers.candidate_hashmap_size,
            refinement_buffers.candidate_hashmap_keys,
            candidate_new_vertex_hashmap_indices,
            additional_state_in.active_tet_count,
            refinement_buffers.tet_candidate,
            new_vertex_index_map,
            split_position,
        ],
        outputs=[
            state_out.particle_q,
            state_out.particle_qd,
            additional_state_out.rest_particle_q,
        ]
    )

    model.particle_mass.zero_()
    wp.launch(
        scatter_tets,
        dim=refinement_buffers.max_tets,
        inputs=[
            refinement_buffers.candidate_hashmap_size,
            refinement_buffers.candidate_hashmap_keys,
            candidate_new_vertex_hashmap_indices,
            additional_state_in.active_tet_count,
            additional_state_in.tet_indices,
            additional_state_in.tet_stretch,
            additional_state_in.tet_poses,
            additional_state_in.tet_lambda,
            additional_state_in.tet_materials,
            refinement_buffers.tet_candidate,
            tet_index_map,
            density,
        ],
        outputs=[
            additional_state_out.rest_particle_q,
            model.particle_mass,
            additional_state_out.tet_indices,
            additional_state_out.tet_stretch,
            additional_state_out.tet_poses,
            additional_state_out.tet_lambda,
            additional_state_out.tet_materials,
        ]
    )

    # Surface tris are updated the same way the tets are: any tri edge that
    # matches one of this pass's finalized split edges (candidate_hashmap_keys
    # was just rebuilt above, by populate_chosen_candidates, to contain only
    # those) gets its tri split in two around the same new vertex; everything
    # else is copied through unchanged.
    refinement_buffers.tri_split_counts.zero_()
    wp.launch(
        get_tri_candidate,
        dim=refinement_buffers.max_tris,
        inputs=[
            additional_state_in.active_tri_count,
            additional_state_in.tri_indices,
            refinement_buffers.candidate_hashmap_size,
            refinement_buffers.candidate_hashmap_keys,
        ],
        outputs=[
            refinement_buffers.tri_candidate,
            refinement_buffers.tri_split_counts,
        ]
    )

    tri_index_map = refinement_buffers.tri_split_counts
    wp.utils.array_scan(refinement_buffers.tri_split_counts, tri_index_map, inclusive=False)
    wp.copy(additional_state_out.active_tri_count, tri_index_map[-1:])

    wp.launch(
        scatter_tris,
        dim=refinement_buffers.max_tris,
        inputs=[
            refinement_buffers.candidate_hashmap_size,
            refinement_buffers.candidate_hashmap_keys,
            candidate_new_vertex_hashmap_indices,
            additional_state_in.active_tri_count,
            additional_state_in.tri_indices,
            refinement_buffers.tri_candidate,
            tri_index_map,
        ],
        outputs=[
            additional_state_out.tri_indices,
        ]
    )
