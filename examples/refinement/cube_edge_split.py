"""Minimal worked example of a single edge split.

Builds the smallest mesh that can show off ``mfem.refinement.refinement.refine``
end to end: a unit cube cut into 6 tets (``scipy.spatial.Delaunay`` on the 8
corners -- the standard fan around the cube's internal space diagonal, see
``tests/utils.py::create_cube_example``). One ``refine()`` pass is forced to
split exactly that internal diagonal (it is the shared edge of every tet, and
the longest edge in the mesh) by handing the legacy scorer vertex scores that
are zero everywhere except at its two endpoints -- every other edge's
``min(vertex_scores)`` is then 0 and never clears the threshold.

Run with ``python -m examples.refinement.cube_edge_split``; prints the
tet/vertex counts and tet connectivity before and after, and saves a
before/after wireframe figure.
"""

from examples.config import apply_config
import argparse

import numpy as np
import warp as wp
import newton

from mfem.refinement.additional_state import AdditionalState
from mfem.refinement.refinement import RefinementBuffers, refine


def build_cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    """8 cube corners and their Delaunay tetrahedralization (6 tets sharing
    one internal space diagonal)."""
    import scipy.spatial

    corners = np.array(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)],
        dtype=np.float32,
    )
    tets = scipy.spatial.Delaunay(corners).simplices.astype(np.int32)

    # newton's add_tetrahedron silently drops any tet whose signed volume
    # (det([v1-v0, v2-v0, v3-v0]) / 6) isn't positive, and a valid
    # tessellation necessarily alternates handedness between adjacent tets --
    # so half of Delaunay's simplices need their last two vertices swapped.
    for t in tets:
        d = corners[t[1]] - corners[t[0]]
        e = corners[t[2]] - corners[t[0]]
        f = corners[t[3]] - corners[t[0]]
        if np.dot(np.cross(d, e), f) < 0.0:
            t[2], t[3] = t[3], t[2]
    return corners, tets


def find_shared_diagonal(tets: np.ndarray) -> tuple[int, int]:
    """The one edge common to every tet -- the cube's internal space
    diagonal in the fan decomposition ``build_cube_mesh`` produces."""
    from collections import Counter

    counts = Counter()
    for tet in tets:
        for i in range(4):
            for j in range(i):
                counts[tuple(sorted((int(tet[i]), int(tet[j]))))] += 1
    edge, count = counts.most_common(1)[0]
    assert count == len(tets), "expected an edge shared by every tet"
    return edge


def print_mesh(label: str, particle_q: np.ndarray, tets: np.ndarray) -> None:
    print(f"-- {label}: {len(tets)} tets, {len(particle_q)} vertices --")
    for i, tet in enumerate(tets):
        print(f"  tet {i}: {tuple(int(v) for v in tet)}")
    print()


def plot_before_after(
    q_before: np.ndarray, tets_before: np.ndarray,
    q_after: np.ndarray, tets_after: np.ndarray,
    split_edge: tuple[int, int], new_vertex: int, out_path: str,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401 (registers 3d projection)

    def tet_edges(tets: np.ndarray) -> set[tuple[int, int]]:
        edges = set()
        for tet in tets:
            for i in range(4):
                for j in range(i):
                    edges.add(tuple(sorted((int(tet[i]), int(tet[j])))))
        return edges

    fig = plt.figure(figsize=(9, 4.5))
    for col, (title, q, tets, highlight) in enumerate([
        ("before: 6 tets, 8 vertices", q_before, tets_before, split_edge),
        (f"after: {len(tets_after)} tets, {len(q_after)} vertices", q_after, tets_after, None),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        segs = [(q[a], q[b]) for a, b in tet_edges(tets)]
        ax.add_collection3d(Line3DCollection(segs, colors="0.5", linewidths=1.0))
        ax.scatter(*q.T, color="black", s=15)
        if highlight is not None:
            a, b = highlight
            ax.plot(*zip(q[a], q[b]), color="crimson", linewidth=2.5, label="edge to split")
            ax.legend(loc="upper left")
        else:
            ax.scatter(*q[new_vertex], color="crimson", s=40, label="new vertex")
            ax.legend(loc="upper left")
        ax.set_title(title)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1); ax.set_zlim(-0.1, 1.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="cube_edge_split.png", help="output figure path")
    parser.add_argument("--device", default=None, help="warp device (default: warp's default)")
    apply_config(parser, "cube_edge_split")
    args = parser.parse_args()

    wp.init()

    corners, tets = build_cube_mesh()
    split_edge = find_shared_diagonal(tets)
    n_particles, n_tets = len(corners), len(tets)

    # Headroom for exactly the one split this example performs: one new tet
    # per tet incident to the split edge minus one (a single edge split always
    # adds exactly 1 vertex and turns each tet touching that edge into 2), so
    # here (every tet touches it) max_tets must fit 2 * n_tets.
    max_particles = n_particles + 1
    max_tets = 2 * n_tets
    max_tris = 64  # generous headroom for the cube's 12 surface tris

    with wp.ScopedDevice(args.device):
        padded_particles = np.zeros((max_particles, 3), dtype=np.float32)
        padded_particles[:n_particles] = corners

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=wp.float32(1.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=padded_particles,
            indices=tets.flatten(),
            density=1000.0,
            k_mu=1.0e4,
            k_lambda=1.0e4,
            k_damp=0.0,
            add_surface_mesh_edges=False,
            validate_mesh=False,
        )
        model = builder.finalize()

        state_in = model.state()
        state_out = model.state()
        state_out.assign(state_in)

        additional_state_in = AdditionalState.from_model(model, max_tets=max_tets, max_tris=max_tris)
        additional_state_out = additional_state_in.clone()

        refinement_buffers = RefinementBuffers(
            max_tets=max_tets, max_vertices=max_particles, max_tris=max_tris, threshold=1.0e-6,
        )

        # Force exactly one candidate: give only the two endpoints of
        # split_edge a nonzero legacy vertex score, so min(vertex_scores) at
        # every other edge is 0 and stays under the threshold.
        vertex_scores_np = np.zeros(max_particles, dtype=np.float32)
        vertex_scores_np[list(split_edge)] = 1.0
        vertex_scores = wp.array(vertex_scores_np, dtype=wp.float32)
        tet_scores = wp.zeros(max_tets, dtype=wp.float32)

        print_mesh("before", corners, tets)
        print(f"forcing a split of edge {split_edge} "
              f"({corners[split_edge[0]]} -> {corners[split_edge[1]]}, "
              f"length {np.linalg.norm(corners[split_edge[0]] - corners[split_edge[1]]):.3f})\n")

        refine(
            model=model,
            density=1000.0,
            state_in=state_in,
            state_out=state_out,
            additional_state_in=additional_state_in,
            additional_state_out=additional_state_out,
            refinement_buffers=refinement_buffers,
            tet_scores=tet_scores,
            vertex_scores=vertex_scores,
            scoring="legacy",
        )

        n_tets_after = int(additional_state_out.active_tet_count.numpy()[0])
        n_particles_after = int(additional_state_out.active_particle_count.numpy()[0])
        q_after = state_out.particle_q.numpy()[:n_particles_after]
        tets_after = additional_state_out.tet_indices.numpy()[:n_tets_after]

        print_mesh("after", q_after, tets_after)
        new_vertex = n_particles_after - 1
        print(f"new vertex {new_vertex} at {q_after[new_vertex]} "
              f"(midpoint of {split_edge})")

        plot_before_after(corners, tets, q_after, tets_after, split_edge, new_vertex, args.out)


if __name__ == "__main__":
    main()
