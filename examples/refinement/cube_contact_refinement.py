"""Contact-driven refinement on a cube, rendered with polyscope.

Extends ``cube_edge_split`` from a single forced split to the real
contact-driven path used in production (``scoring="geometric"``,
``mfem.refinement.contact.Contact``): a translucent capsule ("poker") is held
close above a coarsely tetrahedralized cube -- close enough that some
vertices/tris fall inside the barrier range ``d1``, but never pushed into the
mesh -- and ``refine()`` is called once per phase against that fixed pose.
Edges near the tool score highest (see ``populate_candidates_geometric`` /
``populate_tri_candidates_geometric`` in ``refinement.py``), so the mesh
refines locally under the tool; each pass's conflict resolution only lets one
split per tet through, so full convergence of the contact patch takes several
passes even though the tool itself never moves.

No elasticity or dynamics are solved -- the mesh vertices never move, only
the topology does -- so this isolates the refinement scheduler from the
solver, exactly like ``cube_edge_split``.

Run with ``python -m examples.refinement.cube_contact_refinement``.
"""

from examples.config import apply_config
from pathlib import Path
import argparse
import math

import numpy as np
import warp as wp
import newton

from mfem.refinement.additional_state import AdditionalState
from mfem.refinement.contact import Contact
from mfem.refinement.refinement import RefinementBuffers, refine

CUBE_SIZE = 1.0


def build_grid_cube_mesh(n: int, size: float = CUBE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """``n`` subdivisions per axis: an ``(n+1)^3`` vertex grid, each of the
    ``n^3`` cells cut into 6 tets by the same fan-around-the-internal-diagonal
    pattern as ``cube_edge_split.build_cube_mesh`` (there obtained
    from ``scipy.spatial.Delaunay`` on one cube's corners; reused directly
    here per-cell since every axis-aligned cell tessellates identically)."""
    n1 = n + 1
    xs = np.linspace(0.0, size, n1, dtype=np.float32)

    def vidx(i: int, j: int, k: int) -> int:
        return (i * n1 + j) * n1 + k

    verts = np.array(
        [[xs[i], xs[j], xs[k]] for i in range(n1) for j in range(n1) for k in range(n1)],
        dtype=np.float32,
    )

    # Local corner order matches [dx, dy, dz for dx in (0,1) for dy in (0,1)
    # for dz in (0,1)], i.e. corner c has (dx,dy,dz) = (c>>2, (c>>1)&1, c&1).
    local_tets = [(3, 2, 4, 0), (3, 1, 4, 0), (3, 5, 7, 4), (3, 6, 7, 4), (3, 6, 2, 4), (3, 5, 1, 4)]

    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                corner_global = [
                    vidx(i + dx, j + dy, k + dz)
                    for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)
                ]
                for tpl in local_tets:
                    tets.append(tuple(corner_global[c] for c in tpl))
    tets = np.array(tets, dtype=np.int32)

    # newton's add_tetrahedron drops any tet with non-positive signed volume
    # (see cube_edge_split); fix orientation per tet.
    for t in tets:
        d = verts[t[1]] - verts[t[0]]
        e = verts[t[2]] - verts[t[0]]
        f = verts[t[3]] - verts[t[0]]
        if np.dot(np.cross(d, e), f) < 0.0:
            t[2], t[3] = t[3], t[2]
    return verts, tets


def capsule_cap_centers(bottom_point_z: float, radius: float, half_height: float) -> np.ndarray:
    """The two hemisphere-cap centers of a capsule whose lowest point (local
    -Z cap, see ``mfem.ipc.distance.capsule_sdf``) sits at ``bottom_point_z``,
    centered over the cube's top face. Polyscope draws a curve-network edge
    between them, radius ``radius``, as a capsule-shaped tube -- the same
    shape ``Contact``/``capsule_sdf`` actually score against."""
    body_z = bottom_point_z + half_height + radius
    return np.array(
        [[0.5 * CUBE_SIZE, 0.5 * CUBE_SIZE, body_z - half_height],
         [0.5 * CUBE_SIZE, 0.5 * CUBE_SIZE, body_z + half_height]],
        dtype=np.float32,
    )


def mesh_snapshot(state, additional_state, phase: int) -> dict:
    n_tets = int(additional_state.active_tet_count.numpy()[0])
    n_verts = int(additional_state.active_particle_count.numpy()[0])
    n_tris = int(additional_state.active_tri_count.numpy()[0])
    return dict(
        phase=phase,
        particle_q=state.particle_q.numpy()[:n_verts].copy(),
        tets=additional_state.tet_indices.numpy()[:n_tets].copy(),
        # additional_state.tri_indices, not re-derived from tets: refine()
        # only fully propagates a split to every tet touching an edge over
        # several passes (see module docstring), so mid-convergence the
        # volume mesh has non-conforming ("hanging node") faces between a
        # split and an unsplit neighbor tet. Re-deriving the boundary by
        # matching tet face triplets (surface_triangles_from_tets) silently
        # misses exactly the faces refinement just touched, since a big
        # triangle on one side no longer has a matching triplet on the
        # other; the incrementally-maintained tri list does not have this
        # problem since scatter_tris splits it in lockstep with tet_indices.
        tris=additional_state.tri_indices.numpy()[:n_tris].copy(),
    )


def render_polyscope(
    snapshots: list[dict], capsule_caps: np.ndarray, capsule_radius: float, out_path: str,
    pixels: int = 900, ssaa: int = 2,
) -> None:
    """One fixed camera for all phases (the mesh's bounding box never
    changes, only its topology does); real GPU depth testing means the tool
    and the cube's far faces occlude correctly, unlike the matplotlib
    version this replaces."""
    import polyscope as ps

    pts = np.concatenate([snapshots[0]["particle_q"], capsule_caps])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = 0.5 * (lo + hi)
    scene_radius = 0.5 * float(np.linalg.norm(hi - lo))
    # Mostly top-down: the tool only ever refines the top face, so a mostly
    # side-on view (as for the octopus poke) would hide the entire point of
    # the figure behind an unrefined side wall. eye = center - view_dir*dist
    # puts the camera opposite view_dir looking back along +view_dir, so a
    # top-down camera needs a *negative* z here -- with +1.0 the camera
    # looked from below, up through the (always-unrefined) bottom face,
    # which is why every phase rendered identically until this was caught.
    view_dir = np.array([0.22, -0.22, -1.0])
    view_dir /= np.linalg.norm(view_dir)

    ps.set_program_name("cube contact refinement")
    ps.init()
    ps.set_window_size(pixels, pixels)
    ps.set_build_gui(False)
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps.set_background_color((1.0, 1.0, 1.0))
    ps.set_transparency_mode("pretty")
    try:
        ps.set_SSAA_factor(ssaa)
    except Exception:
        pass
    fov = ps.get_view_camera_parameters().get_fov_vertical_deg()
    dist = scene_radius / math.sin(math.radians(fov) * 0.5) / 1.3
    ps.look_at(tuple(center - view_dir * dist), tuple(center), fly_to=False)

    # Rendered as two separate passes, composited by hand, instead of
    # registering both structures together: polyscope's "pretty" transparency
    # mode (needed for a see-through capsule) simply fails to draw a
    # transparent curve network at all once an opaque surface mesh also
    # shares the scene (verified in isolation -- a polyscope limitation, not
    # a camera or ordering mistake on our end), while its "simple" mode
    # avoids that but makes the "opaque" cube incorrectly translucent too.
    # Compositing capsule-over-cube by hand is safe here because the capsule
    # is always held above the top face, so it is always the near object.
    #
    # The composite can't use a real alpha channel either: polyscope's
    # screenshot ignores transparent_bg and always returns alpha=255, with
    # the true blending already baked into the RGB against its opaque white
    # background (see refinement_frames.content_mask's docstring for
    # the same issue). But since that background is known and the capsule's
    # transparency is a value *we* chose, the blend is invertible: a render
    # of just the capsule over white is `tool = c*a + 255*(1-a)`, so
    # `c*a = tool - 255*(1-a)`, and compositing the capsule over the cube
    # render `body` instead of white gives `tool + (1-a)*(body-255)`.
    tool_alpha = 0.25
    tool = ps.register_curve_network("tool", capsule_caps, np.array([[0, 1]], dtype=np.int32))
    tool.set_radius(capsule_radius, relative=False)
    tool.set_color((0.82, 0.18, 0.22))
    tool.set_material("clay")
    tool.set_transparency(tool_alpha)
    tool_buf = np.asarray(ps.screenshot_to_buffer(transparent_bg=True), dtype=np.uint8)[..., :3].astype(np.float32)
    ps.remove_all_structures()

    images = []
    for snap in snapshots:
        body = ps.register_surface_mesh("body", snap["particle_q"], snap["tris"], smooth_shade=False)
        body.set_color((0.30, 0.55, 0.80))
        body.set_edge_width(1.0)
        body.set_edge_color((0.05, 0.05, 0.05))
        body.set_material("clay")

        body_buf = np.asarray(ps.screenshot_to_buffer(transparent_bg=True), dtype=np.uint8)[..., :3].astype(np.float32)
        ps.remove_all_structures()
        composited = np.clip(tool_buf + (1.0 - tool_alpha) * (body_buf - 255.0), 0.0, 255.0)
        images.append(composited.astype(np.uint8))

    ps.shutdown()
    compose(snapshots, images, out_path)


def compose(snapshots: list[dict], images: list[np.ndarray], out_path: str) -> None:
    """Crop every panel to the union content box (one shared camera, so one
    box is enough) and lay the phases out in a single row with matplotlib --
    used here only to tile/label the polyscope renders, not to draw 3D."""
    import matplotlib.pyplot as plt

    def content_mask(im):
        # No usable alpha channel (see render_polyscope): background pixels
        # are just opaque white, so "not white" is the only signal there is.
        return (im[..., :3].astype(np.int16) < 250).any(axis=-1)

    h, w = images[0].shape[:2]
    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    for im in images:
        ys, xs = np.nonzero(content_mask(im))
        if ys.size:
            lo = np.minimum(lo, [ys.min(), xs.min()])
            hi = np.maximum(hi, [ys.max(), xs.max()])
    pad = 0.03 * max(hi - lo)
    y0, x0 = np.maximum(lo - pad, 0).astype(int)
    y1, x1 = np.minimum(hi + pad, [h - 1, w - 1]).astype(int) + 1

    fig, axes = plt.subplots(1, len(images), figsize=(4.3 * len(images), 4.9))
    for ax, snap, im in zip(axes, snapshots, images):
        ax.imshow(im[y0:y1, x0:x1], interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title(f"{snap['phase']} refinement phase{'s' if snap['phase'] != 1 else ''}"
                     f"\n{len(snap['tets'])} tets, {len(snap['particle_q'])} vertices",
                     pad=10)
    fig.tight_layout()
    # bbox_inches="tight" (not just fig.tight_layout(), which only arranges
    # axes within a fixed canvas) re-measures the actual rendered title text
    # and expands the saved canvas to fit it, so a two-line title stays
    # uncropped regardless of how tight_layout guessed its extent.
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/figures/cube_contact_refinement.png")
    parser.add_argument("--grid", type=int, default=4, help="cube subdivisions per axis")
    parser.add_argument("--phases", type=int, default=10, help="total refine() passes")
    parser.add_argument("--mid-phase", type=int, default=None,
                        help="middle snapshot's phase index (default: phases // 2)")
    parser.add_argument("--device", default=None)
    apply_config(parser, "cube_contact_refinement")
    args = parser.parse_args()
    mid_phase = args.mid_phase if args.mid_phase is not None else args.phases // 2

    wp.init()

    verts, tets = build_grid_cube_mesh(args.grid)
    n_particles, n_tets = len(verts), len(tets)

    # Headroom for a few levels of local bisection under the tool; refine()
    # silently drops a pass that would overflow either buffer rather than
    # crash, so generous headroom just avoids a starved-looking result.
    max_particles = n_particles + 400
    max_tets = 4 * n_tets
    max_tris = 4 * 6 * args.grid * args.grid  # 6 faces * grid^2 quads * 2 tris, x4 headroom

    capsule_radius = 0.10
    capsule_half_height = 0.22
    d1 = 0.16
    d0 = 0.5 * d1
    min_edge_length = CUBE_SIZE / args.grid * 0.12
    geometric_threshold = 1.0
    # Held fixed just above the top face for every phase -- close enough
    # that it falls inside the d1 barrier range (some proximity, no
    # penetration), never pushed down into the mesh.
    bottom_point_z = CUBE_SIZE + 0.35 * d1

    with wp.ScopedDevice(args.device):
        padded = np.zeros((max_particles, 3), dtype=np.float32)
        padded[:n_particles] = verts

        builder = newton.ModelBuilder(gravity=0.0)
        capsule_body = builder.add_body(xform=wp.transform_identity(), is_kinematic=True)
        builder.add_shape_capsule(capsule_body, radius=capsule_radius, half_height=capsule_half_height)
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=wp.float32(1.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=padded,
            indices=tets.flatten(),
            density=1000.0,
            k_mu=1.0e4,
            k_lambda=1.0e4,
            k_damp=0.0,
            add_surface_mesh_edges=False,
            validate_mesh=False,
        )
        model = builder.finalize()
        print(f"built {n_tets} tets / {n_particles} vertices / {model.tri_indices.shape[0]} surface tris")

        state_a, state_b = model.state(), model.state()
        add_a = AdditionalState.from_model(model, max_tets=max_tets, max_tris=max_tris)
        add_b = add_a.clone()

        contact = Contact(model, max_particles=max_particles, d0=d0, d1=d1, stiffness=1.0,
                           max_tris=max_tris, max_incident_tris=8, tri_contact=True)

        refinement_buffers = RefinementBuffers(
            max_tets=max_tets, max_vertices=max_particles, max_tris=max_tris, threshold=geometric_threshold,
        )
        tet_scores = wp.zeros(max_tets, dtype=wp.float32)   # elastic_weight=0 below: contact-only scoring
        vertex_scores = wp.zeros(max_particles, dtype=wp.float32)  # unused by geometric scoring

        body_z = bottom_point_z + capsule_half_height + capsule_radius
        xform = wp.transform(wp.vec3(0.5, 0.5, body_z), wp.quat_identity())
        wp.copy(state_a.body_q, wp.array([xform], dtype=wp.transform))

        snapshots = [mesh_snapshot(state_a, add_a, phase=0)]

        # Fixed roles throughout (matching RefinementSolver.step's pattern):
        # state_a/add_a is always the authoritative "current" mesh, state_b/
        # add_b the transient refine() output, folded back into a after every
        # phase. additional_state_out.active_particle_count is *incremented*
        # (not overwritten) by refine(), so add_b must already agree with
        # add_a's count going in -- ping-ponging the buffer identities instead
        # would leave the recycled "out" buffer's count stale by one phase.
        for phase in range(1, args.phases + 1):
            state_b.assign(state_a)
            contact.compute_distance(model, state_a, add_a)

            refine(
                model=model,
                density=1000.0,
                state_in=state_a,
                state_out=state_b,
                additional_state_in=add_a,
                additional_state_out=add_b,
                refinement_buffers=refinement_buffers,
                tet_scores=tet_scores,
                vertex_scores=vertex_scores,
                scoring="geometric",
                particle_distance=contact.distance,
                particle_shape_id=contact.shape_id,
                particle_distance_hessian=contact.distance_hessian,
                tri_distance=contact.tri_distance,
                tri_bary=contact.tri_bary,
                contact_d1=d1,
                min_edge_length=min_edge_length,
                elastic_weight=0.0,
                vertex_contact_weight=1.0,
                tri_contact_weight=1.0,
                curvature_weight=0.0,
            )

            n_tets_now = int(add_b.active_tet_count.numpy()[0])
            n_verts_now = int(add_b.active_particle_count.numpy()[0])
            print(f"phase {phase:2d}: tets={n_tets_now:4d}  vertices={n_verts_now:4d}")

            if phase in (mid_phase, args.phases):
                snapshots.append(mesh_snapshot(state_b, add_b, phase=phase))

            state_a.assign(state_b)
            add_b.asign(add_a)

        capsule_caps = capsule_cap_centers(bottom_point_z, capsule_radius, capsule_half_height)
        render_polyscope(snapshots, capsule_caps, capsule_radius, args.out)


if __name__ == "__main__":
    main()
