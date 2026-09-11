import numpy as np
import newton
import pytest
import warp as wp

from mfem.refinement.solver import RefinementSolver


def _rest_volume_total(solver: RefinementSolver):
    s0 = solver._additional_state_0
    active_tets = int(s0.active_tet_count.numpy()[0])
    active_verts = int(s0.active_particle_count.numpy()[0])
    tet_idx = s0.tet_indices.numpy()[:active_tets]
    rest_q = s0.rest_particle_q.numpy()

    assert tet_idx.max() < active_verts, "orphaned reference beyond active_particle_count"
    assert tet_idx.min() >= 0, "negative vertex index"

    vols = []
    for t in tet_idx:
        x0, x1, x2, x3 = rest_q[t[0]], rest_q[t[1]], rest_q[t[2]], rest_q[t[3]]
        Dm = np.stack([x1 - x0, x2 - x0, x3 - x0], axis=1)
        vols.append(np.linalg.det(Dm) / 6.0)
    vols = np.array(vols)
    assert (vols > -1e-9).all(), f"negative-volume (inverted) tet found: min={vols.min()}"
    return active_tets, active_verts, float(vols.sum())


def _build_grid_solver(refine_every_n_steps: int, max_new_vertices_per_refine: int, min_refine_score: float, **solver_kwargs):
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(wp.float32),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=3,
        dim_y=2,
        dim_z=2,
        cell_x=0.2,
        cell_y=0.2,
        cell_z=0.2,
        k_mu=1.0e1,
        k_lambda=1.0,
        k_damp=0.0,
        density=1.0,
        fix_left=True,
    )
    base_model = builder.finalize()

    from mfem.refinement.models import MFEMRefinementModel

    refinement_model = MFEMRefinementModel.from_model(
        base_model, max_particles=base_model.particle_count * 4, max_tets=base_model.tet_count * 4
    )

    builder2 = newton.ModelBuilder(gravity=-9.81)
    model = refinement_model.load_sim_model(builder2, mu=1.0e1)

    inv_mass = model.particle_inv_mass.numpy()
    inv_mass[0] = 0.0
    wp.copy(model.particle_inv_mass, wp.array(inv_mass, dtype=wp.float32, device=model.device))

    solver = RefinementSolver(
        model=model,
        iterations=8,
        max_tets=refinement_model.max_tets,
        refine_every_n_steps=refine_every_n_steps,
        max_new_vertices_per_refine=max_new_vertices_per_refine,
        min_refine_score=min_refine_score,
        **solver_kwargs,
    )
    return model, solver


# Per-scoring-mode solver kwargs: the geometric score is dimensionless and has
# its own threshold; the grid has no rigid shapes, so only its elastic term
# fires there. Edges must be >= 2 * refine_min_edge_length to be split, so
# keep that below the 0.2 grid cell.
SCORING_MODES = {
    # The solver's legacy tet weight defaults to 0 and the vertex term is
    # contact only, so give the legacy score an elastic term here.
    "legacy": dict(refine_tet_score_weight=1.0),
    "geometric": dict(refine_scoring="geometric", refine_geometric_threshold=1.0e-6, refine_min_edge_length=0.05),
}


@pytest.mark.parametrize("scoring", sorted(SCORING_MODES))
def test_refinement_conserves_volume_and_grows_mesh(scoring):
    """Splitting a tet must never orphan a vertex or change the total rest
    volume of the mesh, and the mesh should actually grow under gravity-
    induced deformation over enough steps."""
    model, solver = _build_grid_solver(
        refine_every_n_steps=3, max_new_vertices_per_refine=4, min_refine_score=1.0e-6, **SCORING_MODES[scoring]
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    dt = 1.0 / 240.0

    at0, av0, vol0 = _rest_volume_total(solver)

    # Generous margin: the exact step at which stretch first crosses
    # min_refine_score is sensitive to GPU floating-point rounding (kernel
    # scheduling/occupancy can vary with what else has run on the device this
    # session), so a tight step budget can be borderline flaky.
    for _ in range(60):
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

        q = state_0.particle_q.numpy()
        qd = state_0.particle_qd.numpy()
        assert np.isfinite(q).all()
        assert np.isfinite(qd).all()

        at, av, vol = _rest_volume_total(solver)
        # float32 positions -> volume accumulates a little rounding noise
        # over many splits; this is about catching a *systematic* volume
        # leak/gain, not chasing float32 ULPs.
        assert vol == pytest.approx(vol0, abs=1e-6)

    assert at > at0, "expected the mesh to have grown (gained tets) under gravity-induced deformation"
    assert av > av0, "expected the mesh to have grown (gained particles) under gravity-induced deformation"


@pytest.mark.parametrize("scoring", sorted(SCORING_MODES))
def test_refinement_under_cuda_graph_capture(scoring):
    """Capturing solver.step() into a CUDA graph and replaying it repeatedly
    must produce the same correctness invariants as eager stepping: no
    orphaned vertices, conserved volume, finite state, and a mesh that
    actually grows under gravity-induced deformation over enough replays.

    This does NOT assert bit-for-bit agreement with a separately-run eager
    solver over the exact same steps: near-simultaneous, near-tied candidate
    scores (e.g. a uniform grid early in the simulation) are resolved via an
    atomic-CAS hashtable insertion race, so *which* edge wins a close tie can
    legitimately vary run to run without indicating a bug -- what must not
    vary is correctness of whatever gets built."""
    if not wp.get_device().is_cuda:
        pytest.skip("CUDA graph capture requires a CUDA device")

    model, solver = _build_grid_solver(
        refine_every_n_steps=3, max_new_vertices_per_refine=4, min_refine_score=1.0e-6, **SCORING_MODES[scoring]
    )

    control = model.control()
    contacts = model.contacts()
    dt = 1.0 / 240.0
    # Generous margin: the exact step at which stretch first crosses
    # min_refine_score is sensitive to GPU floating-point rounding (kernel
    # scheduling/occupancy can vary with what else has run on the device this
    # session), so a tight step budget can be borderline flaky.
    n_steps = 60

    at0, av0, vol0 = _rest_volume_total(solver)

    states = [model.state(), model.state()]

    def simulate():
        solver.step(states[0], states[1], control, contacts, dt)
        states[0], states[1] = states[1], states[0]

    with wp.ScopedCapture() as capture:
        simulate()
        # Odd substep count (1 per capture): the graph permanently reads from
        # the original states[0] object and writes into the original
        # states[1] object. Copy the result back into states[1] so the
        # captured "state_in" reference holds the latest state before the
        # next replay.
        wp.copy(states[1].particle_q, states[0].particle_q)
        wp.copy(states[1].particle_qd, states[0].particle_qd)
        states[0], states[1] = states[1], states[0]
    graph = capture.graph

    for _ in range(n_steps):
        wp.capture_launch(graph)
        states[0], states[1] = states[1], states[0]

        q = states[0].particle_q.numpy()
        qd = states[0].particle_qd.numpy()
        assert np.isfinite(q).all()
        assert np.isfinite(qd).all()

        at, av, vol = _rest_volume_total(solver)
        assert vol == pytest.approx(vol0, abs=1e-6)

    assert at > at0, "expected the mesh to have grown (gained tets) under gravity-induced deformation"
    assert av > av0, "expected the mesh to have grown (gained particles) under gravity-induced deformation"


def test_refine_every_gates_passes_on_the_device():
    """With refine_every_n_steps=3 the mesh may only grow on every third
    solver step (steps 0, 3, 6, ...), counted on the device."""
    model, solver = _build_grid_solver(
        refine_every_n_steps=3, max_new_vertices_per_refine=4, min_refine_score=1.0e-6,
        **SCORING_MODES["geometric"]
    )
    s0, s1 = model.state(), model.state()
    control, contacts = model.control(), model.contacts()
    grew_on = []
    prev = _rest_volume_total(solver)[1]
    for i in range(12):
        solver.step(s0, s1, control, contacts, 1.0 / 240.0)
        s0, s1 = s1, s0
        cur = _rest_volume_total(solver)[1]
        if cur > prev:
            grew_on.append(i)
        prev = cur
    assert grew_on, "expected some refinement within 12 steps"
    assert all(i % 3 == 0 for i in grew_on), f"mesh grew on non-refining steps: {grew_on}"
