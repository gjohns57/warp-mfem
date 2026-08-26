from mfem.refinement.solver import RefinementSolver
import newton
import newton.examples
import numpy as np
import nvtx
import warp as wp
import time
from typing import Callable
import math
import ast
from mfem.refinement.models import MFEMRefinementModel
from newton import Axis

import polyscope as ps
from scipy.spatial.transform import Rotation


@wp.kernel
def _translate_bodies_kernel(body_q: wp.array(dtype=wp.transform), offset: wp.vec3):
    tid = wp.tid()
    tf = body_q[tid]
    pos = wp.transform_get_translation(tf) + offset
    rot = wp.transform_get_rotation(tf)
    body_q[tid] = wp.transform(pos, rot)


class MFEMRefinementSim:
    def __init__(self,refinement_model: MFEMRefinementModel, args):
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

        self.capsule_radius = 1.0
        self.capsule_half_height = 0.5
        capsule_body = builder.add_body(xform=wp.transform_set_translation(wp.transform_identity(), wp.vec3(1.0, 0.0, 0.0)))
        builder.add_shape_capsule(capsule_body, radius=self.capsule_radius, half_height=self.capsule_half_height)

        self.model = refinement_model.load_sim_model(builder)


        self.solver = RefinementSolver(
            model=self.model,
            iterations=self.iterations,
            max_tets=refinement_model.max_tets,
            refine_every_n_steps=args.refine_every,
            max_new_vertices_per_refine=args.max_new_vertices,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()

        particle_ct = self.solver._additional_state_0.active_particle_count.numpy()[0]
        self.ps_volume = ps.register_volume_mesh("Soft body", self.model.particle_q[:particle_ct].numpy(), self.model.tet_indices.numpy())
        self.ps_volume.add_scalar_quantity("elastic energy", self.solver._elastic_energy[:self.solver._additional_state_0.active_tet_count.numpy()[0]].numpy(), defined_on='cells', vminmax=(0.0, 10.0), cmap='blues', enabled=True)
        self._old_vertex_count = particle_ct

        capsule_mesh = newton.Mesh.create_capsule(self.capsule_radius, self.capsule_half_height, up_axis=Axis.Z)
        self.ps_capsule = ps.register_surface_mesh("Capsule", capsule_mesh.vertices, capsule_mesh.indices.reshape(-1, 3))

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
        # self._topology.get_edges()
        wp.launch(
            _translate_bodies_kernel,
            dim=self.model.body_count,
            inputs=[self.state_0.body_q, wp.vec3(0.01, 0.0, 0.0)],
        )

        if self.graph:
            wp.capture_launch(self.graph)
            self.state_0, self.state_1 = self.state_1, self.state_0
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        # Lots of device to host copy but it is just in the rendering so I can ignore this part when presenting timings
        particle_count = self.solver._additional_state_0.active_particle_count.numpy()[0]
        tet_count = self.solver._additional_state_0.active_tet_count.numpy()[0]

        if self._old_vertex_count != particle_count:
            print(f"new tet count {tet_count}, new vert count {particle_count}")
        ps.remove_volume_mesh("Soft body")
        self.ps_volume = ps.register_volume_mesh("Soft body", self.state_0.particle_q[:particle_count].numpy(), self.solver._additional_state_0.tet_indices[:tet_count].numpy())
        self.ps_volume.set_edge_width(1.0)
        self.ps_volume.set_edge_color([0.0, 0.0, 0.0])
        self.ps_volume.add_scalar_quantity("elastic energy", self.solver._elastic_energy[:tet_count].numpy(), defined_on='cells', vminmax=(0.0, 0.01), cmap='blues', enabled=True)
        self._old_vertex_count = particle_count

        capsule_tf = self.state_0.body_q.numpy()[0]
        capsule_mat = np.eye(4)
        capsule_mat[:3, :3] = Rotation.from_quat(capsule_tf[3:7]).as_matrix()
        capsule_mat[:3, 3] = capsule_tf[:3]
        self.ps_capsule.set_transform(capsule_mat)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

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

        parser.add_argument(
            "--refine-every",
            help="Evaluate refinement candidates every N solver steps",
            type=int,
            default=10,
        )

        parser.add_argument(
            "--max-new-vertices",
            help="Maximum number of new vertices created per refinement pass",
            type=int,
            default=32,
        )

        return parser


def _apply_warp_config(parser, args):
    """Apply ``--warp-config`` overrides to :obj:`warp.config`.

    Each entry in ``args.warp_config`` must have the form ``KEY=VALUE``.  The
    key is validated to be an existing attribute of :obj:`warp.config`.  The
    value is parsed with :func:`ast.literal_eval`; if that fails the raw
    string is kept.

    Args:
        parser: The argument parser, used for error reporting.
        args: Parsed argument namespace containing ``warp_config``.
    """
    if not args.warp_config:
        return

    for entry in args.warp_config:
        if "=" not in entry:
            parser.error(f"invalid --warp-config format '{entry}': expected KEY=VALUE")

        key, value_str = entry.split("=", 1)

        if not hasattr(wp.config, key):
            parser.error(f"invalid --warp-config key '{key}': not a recognized warp.config setting")

        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str

        setattr(wp.config, key, value)

def init(parser):
    """Initialize Newton example components from parsed arguments.

    Args:
        parser: An argparse.ArgumentParser instance (should include arguments from
              create_parser()). If None, a default parser is created.

    Returns:
        tuple: (viewer, args) where viewer is configured based on args.viewer

    Raises:
        ValueError: If invalid viewer type or missing required arguments
    """
    import warp as wp  # noqa: PLC0415

    ps.init()
    ps.set_frame_tick_limit_fps_mode("block_to_hit_target")
    # parse args

    args = parser.parse_args()
    ps.set_max_fps(args.fps)


    # Apply --warp-config overrides before any Warp API calls
    _apply_warp_config(parser, args)

    # Suppress Warp compilation messages if requested
    if args.quiet:
        wp.config.log_level = max(wp.config.log_level, wp.LOG_WARNING)

    # Set device if specified
    if args.device:
        wp.set_device(args.device)


    return args


def run(sim: MFEMRefinementSim, args):
    # Edge width/color are reapplied every frame in render() itself, since
    # render() re-registers a fresh volume mesh object each frame.
    ps.set_up_dir('z_up')

    while not ps.window_requests_close():
        frame_start_time = time.perf_counter()

        with wp.ScopedTimer("Step and readback"):
            sim.step()

            sim.render()

        ps.frame_tick()


        # _throttle_render_fps(frame_start_time, sim.fps)

if __name__ == "__main__":
    parser = MFEMRefinementSim.create_parser()
    args = init(parser)

    
    refinement_model = MFEMRefinementModel.load("fem_beam.npz")

    sim = MFEMRefinementSim(refinement_model, args)

    run(sim, args)
    sim.solver.write_timings("timings.json")


