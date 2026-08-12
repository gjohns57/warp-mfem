"""Stand-alone Newton example: per-triangle coloring of a soft body's surface.

Newton's viewer has no per-vertex / per-face color attribute (the GL vertex
format is position + normal + uv only, and log_mesh() takes a single uniform
`color`).  This example works around that by:

  1. unwelding the surface triangles once, so every face owns its 3 corners,
  2. baking a 1-D colormap into a tiny (1 x 256) RGB texture,
  3. writing a per-corner UV each frame, u = scalar in [0, 1], v = 0.5,
  4. calling log_mesh(..., uvs=..., texture=lut, color=(1, 1, 1)).

The shader computes albedo = ObjectColor * texture(albedo_map, TexCoord), so a
white base color lets the LUT drive the result completely.

The scalar being visualized here is per-triangle area strain (current area /
rest area) of a soft grid dropped onto the ground.

The scene is a tetrahedral soft grid fixed along its left face, sagging under
gravity (VBD solver).  Tested against newton 1.4.0 / warp 1.15.0.

Run:
    python example_colored_soft_body.py                # OpenGL viewer
    python example_colored_soft_body.py --viewer null --device cpu   # headless
"""

import argparse

import newton
import numpy as np
import warp as wp


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
@wp.kernel
def compute_rest_areas(
    x: wp.array(dtype=wp.vec3),
    tri: wp.array2d(dtype=wp.int32),
    areas: wp.array(dtype=float),
):
    t = wp.tid()
    p0 = x[tri[t, 0]]
    p1 = x[tri[t, 1]]
    p2 = x[tri[t, 2]]
    areas[t] = 0.5 * wp.length(wp.cross(p1 - p0, p2 - p0))


@wp.kernel
def color_vertices(
    val: wp.array(dtype=float),  # one scalar per particle
    lo: float,
    hi: float,
    uvs: wp.array(dtype=wp.vec2),
):
    i = wp.tid()
    uvs[i] = wp.vec2(wp.clamp((val[i] - lo) / (hi - lo), 0.0, 1.0), 0.5)


@wp.kernel
def unweld_and_color(
    x: wp.array(dtype=wp.vec3),
    tri: wp.array2d(dtype=wp.int32),
    rest_area: wp.array(dtype=float),
    lo: float,
    hi: float,
    # outputs
    pts: wp.array(dtype=wp.vec3),
    uvs: wp.array(dtype=wp.vec2),
):
    """One thread per unwelded corner (3 * num_tris)."""
    i = wp.tid()
    t = i / 3
    c = i - 3 * t

    pts[i] = x[tri[t, c]]

    # per-triangle scalar: area strain, evaluated once per face and shared by
    # all 3 corners -> flat (faceted) color
    p0 = x[tri[t, 0]]
    p1 = x[tri[t, 1]]
    p2 = x[tri[t, 2]]
    area = 0.5 * wp.length(wp.cross(p1 - p0, p2 - p0))

    s = area / wp.max(rest_area[t], 1.0e-12)
    u = wp.clamp((s - lo) / (hi - lo), 0.0, 1.0)
    uvs[i] = wp.vec2(u, 0.5)


# --------------------------------------------------------------------------
# colormap
# --------------------------------------------------------------------------
def make_lut(n=256):
    """Blue -> white -> red diverging LUT as a (1, n, 3) uint8 image."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    blue = np.array([[0.15, 0.30, 0.85]])
    white = np.array([[0.95, 0.95, 0.95]])
    red = np.array([[0.85, 0.15, 0.15]])

    a = np.clip(t * 2.0, 0.0, 1.0)
    b = np.clip(t * 2.0 - 1.0, 0.0, 1.0)
    rgb = blue * (1.0 - a) + white * a
    rgb = rgb * (1.0 - b) + red * b
    return (rgb * 255.0).astype(np.uint8)[None, :, :]  # (1, n, 3)


# --------------------------------------------------------------------------
# example
# --------------------------------------------------------------------------
class Example:
    def __init__(self, viewer):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self._tex_enabled = False

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=12,
            dim_y=4,
            dim_z=4,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=1.0e3,
            k_mu=1.0e5,
            k_lambda=1.0e5,
            k_damp=1.0e2,
            fix_left=True,
        )
        builder.color()  # graph coloring required by the VBD solver

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=10,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # ---- one-time setup for the colored surface mesh -------------------
        device = self.model.device
        self.num_tris = self.model.tri_count
        self.tri = self.model.tri_indices  # (T, 3) int32

        self.rest_area = wp.zeros(self.num_tris, dtype=float, device=device)
        wp.launch(
            compute_rest_areas,
            dim=self.num_tris,
            inputs=[self.state_0.particle_q, self.tri],
            outputs=[self.rest_area],
            device=device,
        )

        n_corners = 3 * self.num_tris
        self.pts = wp.zeros(n_corners, dtype=wp.vec3, device=device)
        self.uvs = wp.zeros(n_corners, dtype=wp.vec2, device=device)
        # unwelded topology: corner i simply refers to vertex i
        self.idx = wp.array(
            np.arange(n_corners, dtype=np.int32), dtype=wp.int32, device=device
        )
        self.lut = make_lut()

        self.viewer.set_model(self.model)
        # suppress the built-in flat-grey /model/triangles mesh
        self.viewer.show_triangles = False

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def _enable_texture_sampling(self, name="/tissue/surface"):
        obj = getattr(self.viewer, "objects", {}).get(name)
        if obj is not None:
            r, m, c, _ = obj.material
            obj.material = (r, m, c, 1.0)

    def render(self):
        wp.launch(
            unweld_and_color,
            dim=3 * self.num_tris,
            inputs=[self.state_0.particle_q, self.tri, self.rest_area, 0.9, 1.1],
            outputs=[self.pts, self.uvs],
            device=self.model.device,
        )

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_mesh(
            "/tissue/surface",
            self.pts,
            self.idx,
            uvs=self.uvs,
            texture=self.lut,  # must be passed EVERY frame, else it is deleted
            color=(1.0, 1.0, 1.0),
            backface_culling=False,
        )

        if not self._tex_enabled:
            self._enable_texture_sampling()
            self._tex_enabled = True

        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", default="gl", choices=["gl", "null"])
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with wp.ScopedDevice(args.device):
        if args.viewer == "gl":
            viewer = newton.viewer.ViewerGL()
        else:
            viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)

        example = Example(viewer)

        while viewer.is_running():
            example.step()
            example.render()

        viewer.close()


if __name__ == "__main__":
    main()
