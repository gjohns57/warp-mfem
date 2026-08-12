from mfem.refinement.solver import RefinementSolver
import newton
import newton.examples
import numpy as np
import nvtx
import warp as wp
from attr import dataclass
from mpmath.libmp.gammazeta import MAX_BERNOULLI_CACHE
from newton._src.viewer.gl.opengl import RendererGL
from newton._src.viewer.viewer_gl import ViewerGL
from sympy.printing.pretty.pretty_symbology import d

from mfem import MFEMSolver
from mfem.boundary_condition import DirichletBoundaryCondition
from mfem.energies import ARAPEnergy, NeoHookeanEnergy
from mfem.refinement import FEMTopology
from mfem.types import float_type, vec3, vec6

PARAM_MAX_PARTICLES: int = 0
PARAM_MAX_TETS: int = 1
PARAM_DENSITY: float = 2


@dataclass
class MFEMRefinementModel:
    particles: np.ndarray
    tet_indices: np.ndarray
    tet_materials: np.ndarray
    parameters: np.ndarray

    @classmethod
    def from_model(cls, model: newton.Model, max_particles: int, max_tets: int):
        particles = model.particle_q.numpy()
        tet_indices = model.tet_indices.numpy()
        tet_materials = model.tet_materials.numpy()
        params = np.array([max_particles, max_tets, 1.0])
        return cls(
            particles=particles,
            tet_indices=tet_indices,
            tet_materials=tet_materials,
            parameters=params,
        )

    def save(self, file: str):
        np.savez(
            file,
            particles=self.particles,
            tet_indices=self.tet_indices,
            tet_materials=self.tet_materials,
            parameters=self.parameters,
        )

    @staticmethod
    def load(file: str):
        model_store = np.load(file, allow_pickle=True)
        particles = model_store["particles"]
        tet_indices = model_store["tet_indices"]
        tet_materials = model_store["tet_materials"]
        parameters = model_store["parameters"]

        return MFEMRefinementModel(
            particles=particles,
            tet_indices=tet_indices,
            tet_materials=tet_materials,
            parameters=parameters,
        )

    @property
    def max_particles(self):
        return int(self.parameters[PARAM_MAX_PARTICLES])

    @property
    def max_tets(self):
        return int(self.parameters[PARAM_MAX_TETS])

    @property
    def density(self):
        return float(self.parameters[PARAM_DENSITY])


def load_sim_model(refinement_model: MFEMRefinementModel, builder: newton.ModelBuilder):

    particles = np.zeros(
        (refinement_model.max_particles, 3),
        dtype=np.float32,
    )

    particles[: refinement_model.particles.shape[0], :] = refinement_model.particles
    density = refinement_model.density

    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        vel=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=wp.float32(1.0),
        vertices=particles,
        indices=refinement_model.tet_indices.flatten(),
        density=density,
        k_mu=refinement_model.tet_materials[:, 0],
        k_lambda=refinement_model.tet_materials[:, 1],
        k_damp=refinement_model.tet_materials[:, 2],
        add_surface_mesh_edges=False,
        validate_mesh=False,
    )

    model = builder.finalize()


    wp.copy(model.particle_inv_mass, wp.array(np.where(model.particle_q.numpy().reshape(-1, 3)[:, 0] < 0.1, 0.0, model.particle_inv_mass.numpy()), device="cpu"))



    return model


class MFEMRefinementSim:
    def __init__(self, viewer, refinement_model: MFEMRefinementModel, args):
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
        self.model = load_sim_model(refinement_model, builder)


        self.solver = RefinementSolver(
            model=self.model,
            iterations=self.iterations,
            max_tets=refinement_model.max_tets
        )

        self._topology = FEMTopology(self.model, 1 << 20, 1 << 20)
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


    def simulate(self):
        for _ in range(self.sim_substeps):
            with nvtx.annotate("solver_step", color="blue"):
                self.solver.step(
                    self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
                )

            self.state_0, self.state_1 = self.state_1, self.state_0

    @nvtx.annotate("Solver Frame", color="green")
    def step(self):
        self._topology.get_edges()

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

        self.viewer.log_state(
            self.state_0,
        )
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


if __name__ == "__main__":
    parser = MFEMRefinementSim.create_parser()
    viewer, args = newton.examples.init(parser)

    builder = newton.ModelBuilder(gravity=-9.81)

    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(wp.float32),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=20,
        dim_y=10,
        dim_z=10,
        cell_x=0.2,
        cell_y=0.2,
        cell_z=0.2,
        k_mu=1.0e1,
        k_lambda=1.0,
        k_damp=0.0,
        density=1.0,
    )

    model = builder.finalize()

    refinement_model = MFEMRefinementModel.from_model(model, max_particles=model.particle_count, max_tets=int(model.tet_count * 4))




    sim = MFEMRefinementSim(viewer, refinement_model, args)

    with wp.ScopedDevice("cuda:0"):
        newton.examples.run(sim, args)
    sim.solver.write_timings("timings.json")
