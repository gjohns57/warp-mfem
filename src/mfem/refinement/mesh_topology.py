import warp as wp
from newton import Model

from .geometry_hash import (
    get_hashtable_size,
    hashbag_add_tri,
    hashbag_find_tri,
    hashmap_find_edge,
    hashmap_insert_edge,
    hashmap_remove_edge,
    hashset_find_edge,
    hashset_insert_edge,
)


@wp.func
def sort4(
    i0: wp.int32, i1: wp.int32, i2: wp.int32, i3: wp.int32
) -> tuple[wp.int32, wp.int32, wp.int32, wp.int32]:

    c0 = wp.int32(i0 >= i1) + wp.int32(i0 >= i2) + wp.int32(i0 >= i3)
    c1 = wp.int32(i1 >= i0) + wp.int32(i1 >= i2) + wp.int32(i1 >= i3)
    c2 = wp.int32(i2 >= i0) + wp.int32(i2 >= i1) + wp.int32(i2 >= i3)
    c3 = wp.int32(i3 >= i0) + wp.int32(i3 >= i1) + wp.int32(i3 >= i2)

    r = wp.vec4i()
    r[c0] = i0
    r[c1] = i1
    r[c2] = i2
    r[c3] = i3

    return (r[0], r[1], r[2], r[3])


@wp.kernel
def sort_tet_indices(
    tets: wp.array2d[wp.int32],
):
    tid = wp.tid()
    i0, i1, i2, i3 = sort4(*tets[tid][:4])
    tets[tid, 0] = i0
    tets[tid, 1] = i1
    tets[tid, 2] = i2
    tets[tid, 3] = i3


@wp.kernel
def build_split_edge_lookup(
    split_edges: wp.array[wp.vec2i],
    hashmap_keys: wp.array[wp.uint64],
    hashmap_values: wp.array[wp.int32],
    hashmap_size: wp.int32,
):
    tid = wp.tid()

    hashmap_insert_edge(
        hashmap_keys, hashmap_values, hashmap_size, split_edges[tid], tid
    )


@wp.kernel
def fill_tri_bag(
    tets: wp.array2d[wp.int32],
    tri_bag_keys: wp.array[wp.uint64],
    tri_bag_counts: wp.array[wp.int32],
    tri_bag_size: wp.int32,
):
    tid = wp.tid()
    t0 = tets[tid, 0]
    t1 = tets[tid, 1]
    t2 = tets[tid, 2]
    t3 = tets[tid, 3]
    tri0 = wp.vec3i(t0, t1, t2)
    tri1 = wp.vec3i(t1, t2, t3)
    tri2 = wp.vec3i(t0, t2, t3)
    tri3 = wp.vec3i(t0, t1, t3)

    hashbag_add_tri(
        tri_bag_keys,
        tri_bag_counts,
        tri_bag_size,
        tri0,
    )
    hashbag_add_tri(
        tri_bag_keys,
        tri_bag_counts,
        tri_bag_size,
        tri1,
    )
    hashbag_add_tri(
        tri_bag_keys,
        tri_bag_counts,
        tri_bag_size,
        tri2,
    )
    hashbag_add_tri(
        tri_bag_keys,
        tri_bag_counts,
        tri_bag_size,
        tri3,
    )


# We would like a way to remove edge split candidates where one of the incident tetrahedra is incident to another candidate
# For each tet we could query all the incident edge split candidates and choose the one with the highest score and remove the others?
# What happens if two tets have the same incident edges will it be an issue when they try to remove the candadate?
# Probably not but we might have to do a kind of hashset / bloom filter to determine if the edge is already included as a candadate.
# But inlcuding is different than exlcuding


@wp.kernel
def get_surface_tri_counts(
    tets: wp.array2d[wp.int32],
    tri_bag_keys: wp.array[wp.uint64],
    tri_bag_counts: wp.array[wp.int32],
    tri_bag_size: wp.int32,
    tet_surface_tri_count: wp.array[wp.int32],
):
    tid = wp.tid()
    t0 = tets[tid, 0]
    t1 = tets[tid, 1]
    t2 = tets[tid, 2]
    t3 = tets[tid, 3]
    tri0 = wp.vec3i(t0, t1, t2)
    tri1 = wp.vec3i(t1, t2, t3)
    tri2 = wp.vec3i(t0, t2, t3)
    tri3 = wp.vec3i(t0, t1, t3)

    tet_surface_tri_count[tid] = (
        wp.int32(
            hashbag_find_tri(tri_bag_keys, tri_bag_counts, tri_bag_size, tri0) == 1
        )
        + wp.int32(
            hashbag_find_tri(tri_bag_keys, tri_bag_counts, tri_bag_size, tri1) == 1
        )
        + wp.int32(
            hashbag_find_tri(tri_bag_keys, tri_bag_counts, tri_bag_size, tri2) == 1
        )
        + wp.int32(
            hashbag_find_tri(tri_bag_keys, tri_bag_counts, tri_bag_size, tri3) == 1
        )
    )


@wp.func
def fix_tri_direction(
    vertices: wp.array[wp.vec3],
    tri: wp.vec3i,
    internal_vertex: wp.int32,
):
    t0 = vertices[tri[0]]
    t1 = vertices[tri[1]]
    t2 = vertices[tri[2]]
    p = vertices[internal_vertex]

    if wp.dot(p - t0, wp.cross(t1 - t0, t2 - t0)) > 0:
        tmp = tri[1]
        tri[1] = tri[0]
        tri[0] = tmp

    return tri


@wp.kernel
def get_surface_tris(
    tets: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    tri_bag_keys: wp.array[wp.uint64],
    tri_bag_counts: wp.array[wp.int32],
    tri_bag_size: wp.int32,
    index_map: wp.array[wp.int32],
    surface_tris: wp.array[wp.vec3i],
):
    tid = wp.tid()
    t0 = tets[tid, 0]
    t1 = tets[tid, 1]
    t2 = tets[tid, 2]
    t3 = tets[tid, 3]
    tri0 = wp.vec3i(t0, t1, t2)
    tri1 = wp.vec3i(t1, t2, t3)
    tri2 = wp.vec3i(t0, t2, t3)
    tri3 = wp.vec3i(t0, t1, t3)

    i = wp.int32(0)
    if hashbag_find_tri(tri_bag_keys, tri_bag_counts, tri_bag_size, tri0) == 1:
        surface_tris[index_map[tid] + i] = fix_tri_direction(vertices, tri0, t3)
        i += 1
    if hashbag_find_tri(tri_bag_keys, tri_bag_counts, tri_bag_size, tri1) == 1:
        surface_tris[index_map[tid] + i] = fix_tri_direction(vertices, tri1, t0)
        i += 1
    if hashbag_find_tri(tri_bag_keys, tri_bag_counts, tri_bag_size, tri2) == 1:
        surface_tris[index_map[tid] + i] = fix_tri_direction(vertices, tri2, t1)
        i += 1
    if hashbag_find_tri(tri_bag_keys, tri_bag_counts, tri_bag_size, tri3) == 1:
        surface_tris[index_map[tid] + i] = fix_tri_direction(vertices, tri3, t2)


@wp.kernel
def make_tet_graph(
    tets: wp.array2d[wp.int32],
    tmp_tris: wp.indexedarray2d(dtype=wp.int32),
    tmp_tri_tet: wp.indexedarray(dtype=wp.int32),
    tet_graph: wp.array2d[wp.int32],
):
    tid = wp.tid()

    if tid == 0:
        return

    if (
        tmp_tris[tid, 0] == tmp_tris[tid - 1, 0]
        and tmp_tris[tid, 1] == tmp_tris[tid - 1, 1]
        and tmp_tris[tid, 2] == tmp_tris[tid - 1, 2]
    ):
        # So we want to add an edge between these two tets but which triangle on each tet is this?
        # We need the order of the adjacencies to correspond with the edges so that we know which edge (trianle) to traverse
        for i in range(2):
            graph_edge_index = -1
            # Try and figure out what's going on here
            for j in range(3):
                if tmp_tris[tid - i, j] == tets[tmp_tri_tet[tid - i], j + 1]:
                    graph_edge_index = j + 1
            if graph_edge_index == -1:
                graph_edge_index = 0

            tet_graph[tmp_tri_tet[tid - i], graph_edge_index] = tmp_tri_tet[tid - 1 + i]


@wp.kernel
def get_tet_split_edge_candidates(
    tets: wp.array2d[wp.int32],
    split_hashmap_keys: wp.array[wp.uint64],
    split_hashmap_values: wp.array[wp.int32],
    split_hashmap_size: wp.int32,
    candidates: wp.array2d[wp.int32],
    candidate_ct: wp.array[wp.int32],
):
    tid = wp.tid()
    count = 0
    for j in range(4):
        for i in range(j):
            edge = wp.vec2i(tets[tid, i], tets[tid, j])

            candidate_index = hashmap_find_edge(
                split_hashmap_keys, split_hashmap_values, split_hashmap_size, edge
            )

            if candidate_index != -1:
                candidates[tid, count] = candidate_index
                count += 1

    candidate_ct[tid] = count


@wp.kernel
def remove_worst_candidate(
    tet_candidates: wp.array2d[wp.int32],
    tet_candidate_ct: wp.array[wp.int32],
    split_hashmap_keys: wp.array[wp.uint64],
    split_hashmap_values: wp.array[wp.int32],
    split_hashmap_size: wp.int32,
    target_candidate_ct: wp.int32,
    split_candidates: wp.array[wp.vec2i],
    split_scores: wp.array[wp.float32],
):
    tid = wp.tid()
    min_score = wp.float32(1e100)
    worst_index = wp.int32(-1)

    # If there is only one candidate then we are good
    if tet_candidate_ct[tid] <= 1 or tet_candidate_ct[tid] <= target_candidate_ct:
        return

    for i in range(tet_candidate_ct[tid]):
        score = split_scores[tet_candidates[tid, i]]

        if score < min_score:
            min_score = score
            worst_index = i

    edge = split_candidates[tet_candidates[tid, worst_index]]

    hashmap_remove_edge(
        split_hashmap_keys, split_hashmap_values, split_hashmap_size, edge
    )

    tet_candidates[tid, worst_index] = tet_candidates[tid, tet_candidate_ct[tid] - 1]
    tet_candidate_ct[tid] -= 1


@wp.kernel
def update_tet_candidates(
    split_hashmap_keys: wp.array[wp.uint64],
    split_hashmap_values: wp.array[wp.int32],
    split_hashmap_size: wp.int32,
    split_edges: wp.array[wp.vec2i],
    split_scores: wp.array[wp.float32],
    tet_candidates: wp.array2d[wp.int32],
    tet_candidate_ct: wp.array[wp.int32],
):
    tid = wp.tid()
    if tet_candidate_ct[tid] == 0:
        return

    best_score = wp.float32(0.0)
    best_index = wp.int32(-1)

    for i in range(tet_candidate_ct[tid]):
        score = split_scores[tet_candidates[tid, i]]

        if score > best_score or (
            score == best_score
            and tet_candidates[tid, i] < tet_candidates[tid, best_index]
        ):
            best_score = score
            best_index = i

    best_edge = split_edges[tet_candidates[tid, best_index]]

    if (
        hashmap_find_edge(
            split_hashmap_keys, split_hashmap_values, split_hashmap_size, best_edge
        )
        == -1
    ):
        tet_candidates[tid, best_index] = tet_candidates[tid, tet_candidate_ct[tid] - 1]
        tet_candidate_ct[tid] -= 1


@wp.kernel
def count_insertions(
    tets: wp.array2d[wp.int32],
    split_hashmap_keys: wp.array[wp.uint64],
    split_hashmap_values: wp.array[wp.int32],
    split_hashmap_size: wp.int32,
    old_vertex_ct: wp.int32,
    tet_insertions: wp.array[wp.int32],
    tet_insertion_details: wp.array[wp.vec3i],
):
    tid = wp.tid()

    for i in range(4):
        for j in range(i):
            edge = wp.vec2i(tets[tid, j], tets[tid, i])

            new_vertex = hashmap_find_edge(
                split_hashmap_keys, split_hashmap_values, split_hashmap_size, edge
            )

            if new_vertex != -1:
                tet_insertions[tid] = 2
                tet_insertion_details[tid] = wp.vec3i(new_vertex + old_vertex_ct, j, i)
                return

    tet_insertions[tid] = 1
    tet_insertion_details[tid] = wp.vec3i(-1)


@wp.kernel
def build_candidate_set(
    tets: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    candidate_set: wp.array[wp.uint64],
    candidate_set_table_size: wp.int32,
):
    tid = wp.tid()
    max_edge_len = wp.float32(0.0)
    longest_edge = wp.vec2i(-1, -1)

    for j in range(4):
        for i in range(j):
            edge = wp.vec2i(i, j)
            edge_len = wp.length(vertices[tets[tid, i]] - vertices[tets[tid, i]])

            if edge_len > max_edge_len:
                max_edge_len = edge_len
                longest_edge = edge

    hashset_insert_edge(candidate_set, candidate_set_table_size, longest_edge)


@wp.kernel
def tet_emit_edges(
    tets: wp.array2d[wp.int32],
    edges: wp.array2d[wp.vec2i],
    edge_index: wp.array2d[wp.int32],
):
    tid = wp.tid()
    t0 = tets[tid, 0]
    t1 = tets[tid, 1]
    t2 = tets[tid, 2]
    t3 = tets[tid, 3]
    edges[tid, 0] = wp.vec2i(t0, t1)
    edges[tid, 1] = wp.vec2i(t0, t2)
    edges[tid, 2] = wp.vec2i(t0, t3)
    edges[tid, 3] = wp.vec2i(t1, t2)
    edges[tid, 4] = wp.vec2i(t1, t3)
    edges[tid, 5] = wp.vec2i(t2, t3)
    edge_index[tid, 0] = 0 + tid * 6
    edge_index[tid, 1] = 1 + tid * 6
    edge_index[tid, 2] = 2 + tid * 6
    edge_index[tid, 3] = 3 + tid * 6
    edge_index[tid, 4] = 4 + tid * 6
    edge_index[tid, 5] = 5 + tid * 6


@wp.kernel
def extract_valid_candidate_predicate(
    split_candidates: wp.array[wp.vec2i],
    edge_candidate_keys: wp.array[wp.uint64],
    edge_candidate_hashtable_size: wp.int32,
    valid_candidate_predicate: wp.array[wp.int32],
):
    tid = wp.tid()

    edge = split_candidates[tid]

    valid_candidate_predicate[tid] = wp.int32(
        hashset_find_edge(edge_candidate_keys, edge_candidate_hashtable_size, edge)
    )


@wp.kernel
def extract_valid_candidate_indices(
    valid_candidate_index_map: wp.array[wp.int32],
    valid_candidate_indices: wp.array[wp.int32],
):
    tid = wp.tid()

    if valid_candidate_index_map[tid] != valid_candidate_index_map[tid + 1]:
        valid_candidate_indices[valid_candidate_index_map[tid]] = tid


@wp.kernel
def scatter_tets(
    old_tets: wp.array2d[wp.int32],
    old_vertex_count: wp.int32,
    tet_index: wp.array[wp.int32],
    tet_candidates: wp.array2d[wp.int32],
    split_edges: wp.array[wp.vec2i],
    valid_candidate_indices: wp.array[wp.int32],
    new_tets: wp.array2d[wp.int32],
):
    tid = wp.tid()

    if tet_index[tid] + 1 == tet_index[tid + 1]:
        for i in range(4):
            new_tets[tet_index[tid], i] = old_tets[tid, i]
    else:
        split_edge = split_edges[tet_candidates[tid, 0]]

        for j in range(2):
            for i in range(4):
                if old_tets[tid, i] == split_edge[j]:
                    new_tets[tet_index[tid] + j, i] = (
                        valid_candidate_indices[tet_candidates[tid, 0]]
                        + old_vertex_count
                    )
                else:
                    new_tets[tet_index[tid] + j, i] = old_tets[tid, i]


@wp.kernel
def edge_run_ends(
    redundant_edges: wp.array[wp.vec2i],
    edge_run_ends: wp.array[wp.int32],
):
    tid = wp.tid()

    if tid != 0 and (
        redundant_edges[tid][0] != redundant_edges[tid - 1][0]
        or redundant_edges[tid][1] != redundant_edges[tid - 1][1]
    ):
        edge_run_ends[tid] = 1


@wp.kernel
def build_unique_edges(
    redundant_edges: wp.array[wp.vec2i],
    dedup_index: wp.array[wp.int32],
    unique_edges: wp.array[wp.vec2i],
    edge_tet_list_indices: wp.array[wp.int32],
):
    tid = wp.tid()

    if tid == 0 or dedup_index[tid] != dedup_index[tid - 1]:
        unique_edges[dedup_index[tid]] = redundant_edges[tid]
        edge_tet_list_indices[dedup_index[tid]] = tid


@wp.kernel
def build_tet_unique_edge_map(
    edge_tet_list_indices: wp.array[wp.int32],
    edge_tet_lists: wp.array[wp.int32],
    tet_unique_edge_map: wp.array2d[wp.int32],
):
    tid = wp.tid()

    for i in range(edge_tet_list_indices[tid], edge_tet_list_indices[tid + 1]):
        tet_index = edge_tet_lists[i] // 6
        in_tet_index = edge_tet_lists[i] % 6
        tet_unique_edge_map[tet_index, in_tet_index] = tid


@wp.kernel
def volume_score_edges(
    tets: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    tet_edge_map: wp.array2d[wp.int32],
    scores: wp.array[wp.float32],
):
    tid = wp.tid()

    e01 = vertices[tets[tid, 1]] - vertices[tets[tid, 0]]
    e02 = vertices[tets[tid, 2]] - vertices[tets[tid, 0]]
    e03 = vertices[tets[tid, 3]] - vertices[tets[tid, 0]]
    edge_martix = wp.matrix_from_rows(e01, e02, e03)
    volume = wp.abs(wp.determinant(edge_martix) / wp.float32(6.0))

    for i in range(6):
        edge_index = tet_edge_map[tid, i]
        wp.atomic_add(scores, edge_index, volume)


class FEMTopology:
    def __init__(
        self,
        model: Model,
        max_vertices: int,
        max_edges: int,
    ):
        vertices = model.particle_q
        tets = model.tet_indices

        wp.launch(
            sort_tet_indices,
            dim=tets.shape[0],
            inputs=[tets],
        )

        self.frame_no = 0
        self.tets = tets
        self.tet_ct = tets.shape[0]
        self.vertices = wp.empty(max_vertices, dtype=wp.vec3)
        wp.copy(self.vertices, vertices)
        self.vertex_ct = vertices.shape[0]
        self._split_edge_hashmap_size = get_hashtable_size(max_edges, 0.5)
        self._split_edge_hashmap_keys = wp.empty(
            self._split_edge_hashmap_size, dtype=wp.uint64
        )
        self._split_edge_hashmap_values = wp.empty(
            self._split_edge_hashmap_size, dtype=wp.int32
        )
        self._is_surface_fresh = False

    def split_edges(
        self,
        split_candadates: wp.array[wp.vec2i],
        split_scores: wp.array[wp.float32],
    ):
        self._is_surface_fresh = False

        self._split_edge_hashmap_keys.fill_(wp.uint64(0))
        tet_candidates = wp.empty((self.tet_ct, 6), dtype=wp.int32)
        tet_candidate_ct = wp.zeros(self.tet_ct + 1, dtype=wp.int32)

        wp.launch(
            build_split_edge_lookup,
            dim=split_candadates.shape[0],
            inputs=[
                split_candadates,
            ],
            outputs=[
                self._split_edge_hashmap_keys,
                self._split_edge_hashmap_values,
                self._split_edge_hashmap_keys.shape[0],
            ],
        )
        wp.launch(
            get_tet_split_edge_candidates,
            dim=self.tet_ct,
            inputs=[
                self.tets,
                self._split_edge_hashmap_keys,
                self._split_edge_hashmap_values,
                self._split_edge_hashmap_keys.shape[0],
                tet_candidates,
                tet_candidate_ct,
            ],
        )
        # There are a maximum of 6 candidates per tet so we can just iterate a few times to remove the worst candidates until we are left with the best candidate for each tet
        # How is this different from removing the worst candidates all at once?
        # I think maybe this gives other tets the oportunity to choose a candidate that is not their best when there best would be removed by another tet. This way we can get a better distribution of candidates across the tets
        for i in range(5):
            wp.launch(
                remove_worst_candidate,
                dim=self.tet_ct,
                inputs=[
                    tet_candidates,
                    tet_candidate_ct,
                    self._split_edge_hashmap_keys,
                    self._split_edge_hashmap_values,
                    self._split_edge_hashmap_keys.shape[0],
                    5 - i,
                    split_candadates,
                    split_scores,
                ],
            )

            wp.launch(
                update_tet_candidates,
                dim=self.tet_ct,
                inputs=[
                    self._split_edge_hashmap_keys,
                    self._split_edge_hashmap_values,
                    self._split_edge_hashmap_keys.shape[0],
                    split_candadates,
                    split_scores,
                    tet_candidates,
                    tet_candidate_ct,
                ],
            )

        candidate_predicate = wp.zeros(split_candadates.shape[0] + 1, dtype=wp.int32)

        wp.launch(
            extract_valid_candidate_predicate,
            dim=split_candadates.shape[0],
            inputs=[
                split_candadates,
                self._split_edge_hashmap_keys,
                self._split_edge_hashmap_keys.shape[0],
                candidate_predicate,
            ],
        )

        valid_candidate_index_map = candidate_predicate
        wp.utils.array_scan(
            candidate_predicate, valid_candidate_index_map, inclusive=False
        )
        valid_candidate_ct = int(valid_candidate_index_map[-1:].numpy()[0])
        valid_candidate_indices = wp.empty(valid_candidate_ct, dtype=wp.int32)
        wp.launch(
            extract_valid_candidate_indices,
            dim=split_candadates.shape[0],
            inputs=[
                valid_candidate_index_map,
                valid_candidate_indices,
            ],
        )

        tet_index = tet_candidate_ct
        tet_index += 1
        wp.utils.array_scan(tet_index, tet_index, inclusive=False)
        new_tets_size = int(tet_index[-1:].numpy()[0])
        new_tets = wp.empty((new_tets_size, 4), dtype=wp.int32)

        wp.launch(
            scatter_tets,
            dim=self.tet_ct,
            inputs=[
                self.tets,
                self.vertex_ct,
                tet_index,
                tet_candidates,
                split_candadates,
                valid_candidate_index_map,
                new_tets,
            ],
        )

        wp.launch(
            sort_tet_indices,
            dim=new_tets_size,
            inputs=[
                new_tets,
            ],
        )

        self.tets = new_tets
        self.vertex_ct += valid_candidate_ct
        self.tet_ct = new_tets_size

        return valid_candidate_indices

    def get_surface_indices(self):
        # We could make a hashmap mapping triangles to tets: if we go to insert a triangle and it is already in the map then we know that that is not a surface triangle
        # Then we could keep a count of how many surface triangles there are per tet and build up the triangles list that way. The only issue is that I need to write new
        # hash functions but that is okay

        tri_bag_table_size = get_hashtable_size(4 * self.tet_ct, 0.5)
        tri_bag_keys = wp.zeros(tri_bag_table_size, dtype=wp.uint64)
        tri_bag_counts = wp.zeros(tri_bag_table_size, dtype=wp.int32)

        wp.launch(
            fill_tri_bag,
            dim=self.tet_ct,
            inputs=[
                self.tets,
                tri_bag_keys,
                tri_bag_counts,
                tri_bag_table_size,
            ],
        )

        surface_tri_counts = wp.zeros(self.tet_ct, dtype=wp.int32)
        wp.launch(
            get_surface_tri_counts,
            dim=self.tet_ct,
            inputs=[
                self.tets,
                tri_bag_keys,
                tri_bag_counts,
                tri_bag_table_size,
                surface_tri_counts,
            ],
        )

        tet_tri_index_map = surface_tri_counts
        surface_tri_ct = surface_tri_counts[-1:].numpy()[0]
        wp.utils.array_scan(surface_tri_counts, tet_tri_index_map, inclusive=False)
        surface_tri_ct += tet_tri_index_map[-1:].numpy()[0]

        tris = wp.empty((surface_tri_ct, 3), dtype=wp.int32)
        wp.launch(
            get_surface_tris,
            dim=self.tet_ct,
            inputs=[
                self.tets,
                self.vertices,
                tri_bag_keys,
                tri_bag_counts,
                tri_bag_table_size,
                tet_tri_index_map,
                tris.view(wp.vec3i),
            ],
        )

        self._is_surface_fresh = True
        self.tris = tris
        self.tri_ct = int(surface_tri_ct)

    def get_edges(self):
        edges = wp.empty((self.tet_ct * 2, 6), dtype=wp.vec2i)
        edge_index = wp.empty((self.tet_ct * 2, 6), dtype=wp.int32)

        wp.launch(
            tet_emit_edges,
            dim=self.tet_ct,
            inputs=[
                self.tets,
                edges,
                edge_index,
            ],
        )
        edges = edges.flatten()
        edge_index = edge_index.flatten()

        wp.utils.radix_sort_pairs(edges.view(wp.uint64), edge_index, self.tet_ct * 6)

        edge_dedup_index = wp.zeros(edges.shape[0] // 2, dtype=wp.int32)
        wp.launch(
            edge_run_ends,
            dim=edge_dedup_index.shape[0],
            inputs=[
                edges,
                edge_dedup_index,
            ],
        )
        wp.utils.array_scan(edge_dedup_index, edge_dedup_index)
        unique_edge_ct = int(edge_dedup_index[-1:].numpy()[0]) + 1

        unique_edges = wp.empty(unique_edge_ct * 2, dtype=wp.vec2i)
        edge_tet_list_indices = wp.full(
            unique_edge_ct + 1, edge_dedup_index.shape[0], dtype=wp.int32
        )
        wp.launch(
            build_unique_edges,
            dim=edge_dedup_index.shape[0],
            inputs=[
                edges,
                edge_dedup_index,
                unique_edges,
                edge_tet_list_indices,
            ],
        )
        edge_tet_lists = edge_index[: edge_dedup_index.shape[0]]
        tet_unique_edge_map = wp.empty((self.tet_ct, 6), dtype=wp.int32)
        wp.launch(
            build_tet_unique_edge_map,
            dim=unique_edge_ct,
            inputs=[
                edge_tet_list_indices,
                edge_tet_lists,
                tet_unique_edge_map,
            ],
        )

        self.edges = unique_edges
        self.edge_ct = unique_edge_ct
        self.tet_edge_map = tet_unique_edge_map
        self.edge_tet_indices = edge_tet_list_indices
        self.edge_tet_lists = edge_tet_lists


@wp.kernel
def get_midpoints(
    edges: wp.indexedarray1d(dtype=wp.vec2i),
    vertices: wp.array[wp.vec3],
    midpoints: wp.array[wp.vec3],
):
    tid = wp.tid()
    midpoints[tid] = (
        vertices[edges[tid][0]]
        + (vertices[edges[tid][1]] - vertices[edges[tid][0]]) / 2.0
    )


@wp.kernel
def get_edge_scores(
    edges: wp.array[wp.vec2i],
    vertices: wp.array[wp.vec3],
    scores: wp.array[wp.float32],
):
    tid = wp.tid()
    scores[tid] = wp.length(vertices[edges[tid][0]] - vertices[edges[tid][1]]) / (
        wp.max(
            wp.length(vertices[edges[tid][0]]),
            wp.length(vertices[edges[tid][1]]) + 0.01,
        )
    )
