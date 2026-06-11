import newton
import newton.examples
import newton.solvers
import numpy as np
import warp as wp

from mfem import MFEMSolver, line_search
from mfem.energies import ARAPEnergy, NeoHookeanEnergy


class MFEMExample:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.iterations = 3
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.do_capture = args.graph_capture

        # self.sim_duration = args.sim_duration

        self.record_energy = args.record_energy

        builder = newton.ModelBuilder()
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
            pos=wp.vec3(0.0, 0.0, 1.0),
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

        fix_indices = np.argwhere(
            (
                (self.model.particle_q.numpy()[:, 0] <= 0.1)
                | (
                    (self.model.particle_q.numpy()[:, 2] >= 3.4)
                    & (self.model.particle_q.numpy()[:, 0] <= 1.0)
                )
            )
        ).flatten()

        print(args)
        self.solver = MFEMSolver(
            model=self.model,
            iterations=self.iterations,
            material_model=energy,
            record_energy=self.record_energy,
            fix_indices=fix_indices,  # Fix the leftmost particles of each grid
            line_search=args.line_search,
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
            self.graph = capture.graph
            # self.graph = None
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
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
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
            "--line-search",
            help="Whether to use line search",
            type=bool,
            default=False,
        )

        parser.add_argument(
            "--graph-capture",
            help="Whether to capture a Cuda graph",
            type=bool,
            default=False,
        )

        return parser
