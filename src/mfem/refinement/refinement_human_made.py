import warp as wp
from newton import State, Model
from mfem.refinement.additional_state import AdditionalState
from mfem.refinement.geometry_hash import get_hashtable_size, hashtable_find, hashtable_insert
from mfem.types import vec6

class RefinementBuffers:


    def __init__(self, max_tets: int, max_vertices: int, threshold: float = 0.8):
        self.max_tets = max_tets
        self.max_vertices = max_vertices
        self.candidate_hashmap_size = get_hashtable_size(max_tets * 6, 0.5)
        self.candidate_hashmap_keys = wp.empty(self.candidate_hashmap_size, dtype=wp.uint64)
        self.candidate_hashmap_scores = wp.empty(self.candidate_hashmap_size, dtype=wp.float32)
        self.candidate_hashmap_flag = wp.empty(self.candidate_hashmap_size, dtype=wp.uint8)
        self.tet_candidate = wp.full(max_tets, wp.vec2i(-1, -1), dtype=wp.vec2i)
        self.tmp_rest_particle_q = wp.empty(max_vertices, dtype=wp.vec3)
        self.tet_split_counts = wp.zeros(max_tets + 1, dtype=wp.int32)
        self.new_vertex_index = wp.zeros(max_tets + 1, dtype=wp.int32)
        self.threshold = wp.array([threshold], dtype=wp.float32)

@wp.func
def edge_to_key(edge: wp.vec2i) -> wp.uint64:
    return wp.cast(wp.vec2i(wp.min(*edge), wp.max(*edge)), dtype=wp.uint64)
    

CANDIDATE_AVAILABLE = wp.constant(wp.uint8(0))
CANDIDATE_PENDING = wp.constant(wp.uint8(1))
CANDIDATE_SELECTED = wp.constant(wp.uint8(2))
CANDIDATE_UNAVAILABLE = wp.constant(wp.uint8(3))

# We need to change this to add more candidates

@wp.kernel
def populate_candidates(
    active_tet_count: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_scores: wp.array[wp.float32], # We wan't to split elements with higher elastic energy. I think this is already volume weighted
    vertex_scores: wp.array[wp.float32],
    particle_inv_mass: wp.array[wp.float32], # We wan't to avoid splitting edges between kinetic vertices
    particle_q: wp.array[wp.vec3], # We want to split longer edges so we need the position data
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

            
                wp.atomic_add(candidate_hashmap_scores, index, (tet_scores[tid] * 0.0 + (wp.max(vertex_scores[edge[0]], vertex_scores[edge[1]])) * 1.0)  * wp.length(particle_q[edge[1]] - particle_q[edge[0]]))
            
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
    

    if candidate_hashmap_flag[candidate_index] != CANDIDATE_PENDING:
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

    new_particle_q[new_vertex_index] = new_particle_q[candidate[0]] + (new_particle_q[candidate[1]] - new_particle_q[candidate[0]]) / 2.0
    new_particle_qd[new_vertex_index] = new_particle_qd[candidate[0]] + (new_particle_qd[candidate[1]] - new_particle_qd[candidate[0]]) / 2.0
    new_rest_particle_q[new_vertex_index] = new_rest_particle_q[candidate[0]] + (new_rest_particle_q[candidate[1]] - new_rest_particle_q[candidate[0]]) / 2.0


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



def refine(model: Model, density: float, state_in: State, state_out: State, additional_state_in: AdditionalState, additional_state_out: AdditionalState, refinement_buffers: RefinementBuffers, tet_scores: wp.array[wp.float32], vertex_scores: wp.array[wp.float32]):

    refinement_buffers.candidate_hashmap_keys.zero_()
    refinement_buffers.candidate_hashmap_scores.zero_()
    refinement_buffers.candidate_hashmap_flag.zero_()
    # Add candidates to hashmap
    

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
            refinement_buffers.candidate_hashmap_size,
        ],
        outputs=[
            refinement_buffers.candidate_hashmap_keys,
            refinement_buffers.candidate_hashmap_scores,
        ]
    )

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


    for _ in range(5):
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


    tet_index_map = refinement_buffers.tet_split_counts
    new_vertex_index_map = refinement_buffers.new_vertex_index
    wp.utils.array_scan(refinement_buffers.tet_split_counts, tet_index_map, inclusive=False)
    wp.utils.array_scan(new_vertex_index_map, refinement_buffers.new_vertex_index, inclusive=False)
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
