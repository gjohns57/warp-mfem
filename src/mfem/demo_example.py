import newton
import newton.examples
import newton.solvers
import numpy as np
import nvtx
import warp as wp
from newton._src.viewer.gl.opengl import RendererGL
from newton.viewer import ViewerGL, ViewerUSD
from pxr import UsdGeom
from pyglet.window.key import _0

from mfem import MFEMSolver
from mfem.boundary_condition import DirichletBoundaryCondition
from mfem.energies import ARAPEnergy, NeoHookeanEnergy
from mfem.refinement import FEMTopology
from mfem.refinement.mesh_topology import tet_emit_edges
from mfem.types import float_type, vec3


@wp.kernel
def distribute_tet_energies(
    tet_energies: wp.array[wp.float32],
    tet_indices: wp.array2d[wp.int32],
    normalization: wp.array[wp.float32],
    vertex_values: wp.array[wp.float32],
):
    tid = wp.tid()

    i0 = tet_indices[tid, 0]
    i1 = tet_indices[tid, 1]
    i2 = tet_indices[tid, 2]
    i3 = tet_indices[tid, 3]

    wp.atomic_add(vertex_values, i0, tet_energies[tid])
    wp.atomic_add(vertex_values, i1, tet_energies[tid])
    wp.atomic_add(vertex_values, i2, tet_energies[tid])
    wp.atomic_add(vertex_values, i3, tet_energies[tid])
    wp.atomic_add(normalization, i0, 1.0)
    wp.atomic_add(normalization, i1, 1.0)
    wp.atomic_add(normalization, i2, 1.0)
    wp.atomic_add(normalization, i3, 1.0)


@wp.kernel
def add_proximity_score(
    tet_energies: wp.array[wp.float32],
    tet_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
):
    tid = wp.tid()
    body_position = wp.transform_get_translation(body_q[0])
    x0 = particle_q[tet_indices[tid, 0]]
    x1 = particle_q[tet_indices[tid, 1]]
    x2 = particle_q[tet_indices[tid, 2]]
    x3 = particle_q[tet_indices[tid, 3]]

    radius = 0.5
    distance = (
        wp.min(
            wp.min(wp.length(body_position - x0), wp.length(body_position - x1)),
            wp.min(wp.length(body_position - x2), wp.length(body_position - x3)),
        )
        - radius
    )

    tet_energies[tid] += 0.025 / distance


# @wp.kernel
# def distribute_tet_energies_to_edges(
#     tet_energies: wp.array[wp.float32],
#     tet_indices: wp.array[wp.float32],
#     edges: wp.array[wp.float32],
# )


@wp.kernel
def distribute_tet_energies_to_edges(
    tet_energies: wp.array[wp.float32],
    tet_edges: wp.array2d[wp.int32],
    edge_energies: wp.array[wp.float32],
):
    tid = wp.tid()

    for i in range(6):
        edge_index = tet_edges[tid, i]
        wp.atomic_add(edge_energies, edge_index, tet_energies[tid])


def get_candidates(
    state: newton.State,
    topology: FEMTopology,
    tet_energies: wp.array[wp.float32],
    selection_percentile: wp.float32 = 0.9,
):
    edge_scores = wp.zeros(topology.edge_ct * 2, dtype=wp.float32)

    wp.launch(
        distribute_tet_energies_to_edges,
        dim=topology.tet_ct,
        inputs=[
            tet_energies,
            topology.tet_edges,
        ],
        outputs=[edge_scores],
    )

    candidates = topology.edges
    wp.utils.radix_sort_pairs(edge_scores, candidates)

    candidates = candidates[int(float(candidates.shape[0] * selection_percentile)) :]
    candidate_scores = edge_scores[
        int(float(edge_scores.shape[0] * selection_percentile)) :
    ]
    return candidates, candidate_scores


def refine_mesh(
    solver: MFEMSolver,
    model: newton.Model,
    state_0: newton.State,
    topology: FEMTopology,
):
    tet_energies = wp.zeros(model.tet_count, dtype=wp.float32)
    solver._material_model.energy(
        solver._s,
        model.tet_materials,
        solver._tet_rest_volumes,
        tet_energies,
    )

    candidates, candidate_scores = get_candidates(
        state_0,
        topology,
        tet_energies=tet_energies,
        selection_percentile=0.9,
    )

    topology.split_edges(candidates, candidate_scores)

    model.tet_indices = topology.tets
    model.tet_count = topology.tet_ct
    model.particle_count = topology.vertex_ct
    model.particle_q = topology.vertices


def make_lut(n=256):
    """Blue -> white -> red diverging LUT as a (1, n, 3) uint8 image."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    green = np.array([[0.10, 0.85, 0.2]])
    yellow = np.array([[0.85, 0.85, 0.01]])
    red = np.array([[0.85, 0.15, 0.15]])

    a = np.clip(t * 2.0, 0.0, 1.0)
    b = np.clip(t * 2.0 - 1.0, 0.0, 1.0)
    rgb = green * (1.0 - a) + yellow * a
    rgb = rgb * (1.0 - b) + red * b
    return (rgb * 255.0).astype(np.uint8)[None, :, :]  # (1, n, 3)


@wp.kernel
def color_vertices(
    value: wp.array[wp.float32],  # one scalar per particle
    lo: float,
    hi: float,
    uvs: wp.array[wp.vec2],
):
    i = wp.tid()
    uvs[i] = wp.vec2(wp.clamp((value[i] - lo) / (hi - lo), 0.0, 1.0), 0.5)


@wp.kernel
def rotate_top(
    top_positions: wp.array(dtype=vec3),
    center: wp.vec3,
    angle: wp.float32,
):
    tid = wp.tid()
    p = top_positions[tid]
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    cos_a = wp.cos(angle)
    sin_a = wp.sin(angle)
    top_positions[tid] = vec3(
        center[0] + cos_a * dx - sin_a * dy,
        center[1] + sin_a * dx + cos_a * dy,
        p[2],
    )


class MFEMExample:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.iterations = args.iterations
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.do_capture = args.graph_capture

        # self.sim_duration = args.sim_duration

        self.record_energy = args.record_energy

        builder = newton.ModelBuilder(gravity=-9.81)
        # builder.add_ground_plane()

        mu = args.mu
        lmbda = args.lmbda

        # Grid dimensions
        dim_x = args.dimx
        dim_y = args.dimy
        dim_z = args.dimz
        cell_size = 0.2
        energy = None
        if args.energy == "arap":
            energy = ARAPEnergy
        elif args.energy == "neohookean":
            energy = NeoHookeanEnergy

        builder.add_soft_grid(
            pos=wp.vec3(-cell_size * dim_x / 2.0, -cell_size * dim_y / 2.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=1.0,
            k_mu=mu,
            k_lambda=lmbda,
            k_damp=0.0,
        )

        # builder.add_sha

        # Color the mesh for VBD solver

        # body = builder.add_body(
        #     xform=(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
        # )

        # builder.add_shape_mesh(
        #     body, mesh=newton.Mesh.create_sphere(num_latitudes=10, num_longitudes=10)
        # )

        self.model = builder.finalize()
        self._color_gradient = make_lut()
        control_indices = np.argwhere(
            np.linalg.norm(
                self.model.particle_q.numpy().reshape((-1, 3))
                - np.array(
                    [
                        [
                            1.0,
                            0.0,
                            cell_size * dim_z,
                        ]
                    ]
                ),
                axis=1,
            )
            <= 3 * cell_size
        ).flatten()

        fixed_indices = np.argwhere(
            (
                self.model.particle_q.numpy()[:, 0] <= -cell_size * dim_x / 2.0
                # & (self.model.particle_q.numpy()[:, 0] <= 0.5)
                # & (self.model.particle_q.numpy()[:, 1] <= 0.5)
            )
        ).flatten()

        pin_idx = np.concat([fixed_indices])

        pin_positions = wp.array(self.model.particle_q.numpy()[pin_idx], dtype=vec3)
        self._attatchment = pin_positions[: len(fixed_indices)]
        # self._control = pin_positions[len(fixed_indices) :]
        # self._bottom_attatchment = pin_positions[: len(bottom_indices)]
        # self._top_attatchment = pin_positions[len(bottom_indices) :]
        self._bc = DirichletBoundaryCondition(
            pin_positions=pin_positions,
            constraint_indices=wp.array(pin_idx, wp.int32),
            constraint_stiffness=wp.array(
                np.concat(
                    [
                        np.full_like(fixed_indices, fill_value=1.0e3, dtype=wp.float32),
                        # np.full_like(control_indices, fill_value=0.0, dtype=wp.float32),
                    ]
                ),
                dtype=float_type,
            ),
            n_particles=self.model.particle_count,
        )

        self.solver = MFEMSolver(
            model=self.model,
            iterations=self.iterations,
            material_model=energy,
            record_energy=self.record_energy,
            line_search=args.line_search,
            attatchements=self._bc,
            # debug_logs=True,
        )

        # self._topology = FEMTopology(self.model, 1 << 20, 1 << 20)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        if args.graph_capture:
            self.capture()
        else:
            self.graph = None

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
                # With an odd number of substeps the graph reads from buffer A and writes
                # to buffer B, but A is never updated — every replay steps from the same
                # state.  Copy the output (state_0 after simulate's final swap) back to
                # the input buffer (state_1) so A always holds the latest state before the
                # next replay.
                if self.sim_substeps % 2 == 1:
                    wp.copy(self.state_1.particle_q, self.state_0.particle_q)
                    wp.copy(self.state_1.particle_qd, self.state_0.particle_qd)
                    self.state_0, self.state_1 = self.state_1, self.state_0
            self.graph = capture.graph
        else:
            self.graph = None

    def solver_log(self):
        return self.solver.get_log()

    def simulate(self):
        for _ in range(self.sim_substeps):
            # self.state_0.clear_forces()

            # apply forces to the model
            # self.viewer.apply_forces(self.state_0)

            # self.model.collide(self.state_0, self.contacts)
            with nvtx.annotate("solver_step", color="blue"):
                self.solver.step(
                    self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
                )

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    @nvtx.annotate("Solver Frame", color="green")
    def step(self):
        # wp.launch(
        #     rotate_top,
        #     dim=len(self._top_attatchment),
        #     inputs=[self._top_attatchment, self._rotation_center, self._rotation_angle],
        # )
        # self._control += wp.vec3(
        #     np.sin(1.0e-1 * self.sim_time) * 0.01,
        #     np.cos(1.0e-1 * self.sim_time) * 0.01,
        #     np.cos(1.0e-1 * self.sim_time) * 0.02,
        # )
        # self._bc.update_b(self.solver._accumulator)
        #


        with wp.ScopedTimer("Step"):
            if self.graph:
                wp.capture_launch(self.graph)
                self.state_0, self.state_1 = self.state_1, self.state_0
            else:
                self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        # Test that particles are in a reasonable range (soft body may settle or deform)
        # We check that they haven't exploded or collapsed completely
        # 4 grids, each roughly 1.2 x 0.4 x 0.4 in size, positioned along Y-axis
        # Initial positions: Y from 1.0 to ~3.2, X from 0 to 1.2, Z around 1.0 to 1.4
        # With fix_left=True, grids hang and sag significantly towards the ground
        p_lower = wp.vec3(-1.0, -0.5, 0.0)
        p_upper = wp.vec3(3.0, 4.0, 3.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        # ViewerUSD().log_mesh
        self.viewer.log_state(self.state_0)
        # particle_elastic_energies = wp.zeros(
        #     self.model.particle_count, dtype=wp.float32
        # )
        # normalization = wp.zeros(self.model.particle_count, dtype=wp.float32)
        # particle_colors = wp.empty(self.model.particle_q.shape[0], dtype=wp.vec2)

        # # self.solver._objective(self.state_0.particle_q, self.solver._s, wp.zeros(self.model.tet_count, dtype=wp.types.vector(length=6, dtype=wp.float32)))
        # elastic_energy = wp.zeros(self.model.tet_count, dtype=wp.float32)
        # self.solver._material_model.energy(
        #     self.solver._s,
        #     self.model.tet_materials,
        #     self.solver._tet_rest_volumes,
        #     elastic_energy,
        # )

        # wp.launch(
        #     add_proximity_score,
        #     dim=self.model.tet_count,
        #     inputs=[
        #         elastic_energy,
        #         self.model.tet_indices,
        #         self.state_0.particle_q,
        #         self.model.body_q,
        #     ],
        # )
        # # print(elastic_energy)
        # wp.launch(
        #     distribute_tet_energies,
        #     dim=self.model.tet_count,
        #     inputs=[
        #         elastic_energy,
        #         self.model.tet_indices,
        #     ],
        #     outputs=[
        #         normalization,
        #         particle_elastic_energies,
        #     ],
        # )

        # particle_elastic_energies /= normalization
        # wp.launch(
        #     color_vertices,
        #     dim=self.model.particle_q.shape[0],
        #     inputs=[
        #         particle_elastic_energies,
        #         0.0,
        #         0.1,
        #     ],
        #     outputs=[particle_colors],
        # )

        # self.viewer.log_mesh(
        #     "FEM_Beam",
        #     self.state_0.particle_q,
        #     self.model.tri_indices.flatten(),
        #     texture=self._color_gradient,
        #     uvs=particle_colors,
        #     color=wp.vec3(1.0, 1.0, 1.0),
        #     backface_culling=False,
        # )

        # scales = wp.full(self.model.body_q.shape[0], wp.vec3(0.5, 0.5, 0.0))
        # colors = wp.full(self.model.body_q.shape[0], wp.vec3(0.5, 0.5, 0.0))
        # material = wp.full(self.model.body_q.shape[0], wp.vec4(1.0, 1.0, 1.0, 0.0))
        # self.viewer.log_capsules(
        #     "Rigid Object",
        #     "unused?",
        #     self.model.body_q,
        #     scales,
        #     colors,
        #     material,
        # )

        # if isinstance(self.viewer, ViewerGL):
        #     # log_mesh()'s public API has no way to flip the shader's
        #     # texture_enable flag (material.w), so the albedo_map sample in
        #     # shape_fragment_shader is always skipped and the mesh renders
        #     # as flat ObjectColor. Reach into the backend object and set it
        #     # directly so the uploaded texture actually gets sampled.
        #     mesh_obj = self.viewer.objects[self.viewer._qualify("FEM_Beam")]
        #     r, m, c, _t = mesh_obj.material
        #     mesh_obj.material = (r, m, c, 1.0)

        #     # _upload_texture_from_file() (re-run every frame by update_texture())
        #     # always binds GL_REPEAT + mipmapped trilinear filtering. Our LUT is a
        #     # non-tiling blue->white->red ramp, so filtering across the wrap seam
        #     # or across mip levels blends the blue and red ends together and shows
        #     # up as stray purple. Clamp to the edge and drop mipmapping so only
        #     # the ramp's own colors are ever sampled.
        #     if mesh_obj.texture_id is not None:
        #         gl = RendererGL.gl
        #         gl.glBindTexture(gl.GL_TEXTURE_2D, mesh_obj.texture_id)
        #         gl.glTexParameteri(
        #             gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE
        #         )
        #         gl.glTexParameteri(
        #             gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE
        #         )
        #         gl.glTexParameteri(
        #             gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR
        #         )
        #         gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # self.viewer.log_state(self.state_0)

        # if isinstance(self.viewer, ViewerUSD):
        #     self.solver.log_aabbs(self.viewer)
        # self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

        # if self.sim_time >= self.sim_duration:
        #     self.viewer.close()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        # parser.add_argument(
        #     "--sim-duration",
        #     help="Duration of the simulation in seconds",
        #     type=float,
        #     default=10.0,
        # )

        parser.add_argument(
            "--record-energy",
            help="Record energy during simulation",
            type=bool,
            default=False,
        )

        parser.add_argument(
            "--dimx",
            help="Dimension X",
            type=int,
            default=15,
        )

        parser.add_argument(
            "--dimy",
            help="Dimension Y",
            type=int,
            default=10,
        )

        parser.add_argument(
            "--dimz",
            help="Dimension Z",
            type=int,
            default=10,
        )

        parser.add_argument(
            "--energy",
            help="Which energy model to use (arap, neohookean)",
            type=str,
            default="arap",
            choices=["arap", "neohookean"],
        )

        parser.add_argument(
            "--mu",
            help="First Lame's parameter for elasticity",
            type=float,
            default=1.0e1,
        )

        parser.add_argument(
            "--lmbda",
            help="Second Lame's parameter for elasticity",
            type=float,
            default=4.0e2,
        )
        parser.add_argument(
            "-l",
            "--line-search",
            help="Whether to use line search",
            action="store_true",
        )

        parser.add_argument(
            "-g",
            "--graph-capture",
            help="Whether to capture a Cuda graph",
            action="store_true",
        )

        parser.add_argument(
            "--iterations",
            help="SQP iterations per time step",
            type=int,
            default=8,
        )

        parser.add_argument(
            "--substeps",
            help="Time steps per frame",
            type=int,
            default=1,
        )

        parser.add_argument(
            "--fps",
            help="Frames per second",
            type=int,
            default=24,
        )

        return parser
