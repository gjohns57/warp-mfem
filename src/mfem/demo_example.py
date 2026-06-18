import newton
import newton.examples
import newton.solvers
import numpy as np
import nvtx
import warp as wp

from mfem import MFEMSolver
from mfem.boundary_condition import DirichletBoundaryCondition
from mfem.energies import ARAPEnergy, NeoHookeanEnergy
from mfem.types import float_type, vec3


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
        cell_size = 0.25
        energy = None
        if args.energy == "arap":
            energy = ARAPEnergy
        elif args.energy == "neohookean":
            energy = NeoHookeanEnergy

        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=0.5,
            k_mu=mu,
            k_lambda=lmbda,
            k_damp=0.0,
        )

        # Color the mesh for VBD solver

        self.model = builder.finalize()
        print(np.linalg.norm(self.model.particle_q.numpy().reshape((-1, 3))))
        control_indices = np.argwhere(
            np.linalg.norm(
                self.model.particle_q.numpy().reshape((-1, 3))
                - np.array(
                    [
                        [
                            cell_size * dim_x / 2.0,
                            cell_size * dim_y / 2.0,
                            cell_size * dim_z,
                        ]
                    ]
                ),
                axis=1,
            )
            <= 2 * cell_size
        ).flatten()

        fixed_indices = np.argwhere(
            (
                self.model.particle_q.numpy()[:, 2] <= 0.1
                # & (self.model.particle_q.numpy()[:, 0] <= 0.5)
                # & (self.model.particle_q.numpy()[:, 1] <= 0.5)
            )
        ).flatten()

        pin_idx = np.concat([fixed_indices, control_indices])

        pin_positions = wp.array(self.model.particle_q.numpy()[pin_idx], dtype=vec3)
        self._attatchment = pin_positions[: len(fixed_indices)]
        self._control = pin_positions[len(fixed_indices) :]
        # self._bottom_attatchment = pin_positions[: len(bottom_indices)]
        # self._top_attatchment = pin_positions[len(bottom_indices) :]

        self._bc = DirichletBoundaryCondition(
            pin_positions=pin_positions,
            constraint_indices=wp.array(pin_idx, wp.int32),
            constraint_stiffness=wp.array(
                np.concat(
                    [
                        np.full_like(fixed_indices, fill_value=1.0e3, dtype=wp.float32),
                        np.full_like(
                            control_indices, fill_value=1.0e-1, dtype=wp.float32
                        ),
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
        self._control += wp.vec3(
            np.sin(1.0 * self.sim_time) * 0.05,
            np.cos(1.0 * self.sim_time) * 0.05,
            np.cos(1.0 * self.sim_time) * 0.02,
        )
        self._bc.update_b(self.solver._accumulator)
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
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
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
            default=1.0e2,
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
