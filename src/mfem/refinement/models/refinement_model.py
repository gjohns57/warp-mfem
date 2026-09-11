import math

import warp as wp
import newton
import numpy as np
from attr import dataclass

PARAM_MAX_PARTICLES: int = 0
PARAM_MAX_TETS: int = 1
PARAM_DENSITY: float = 2
PARAM_MAX_TRIS: int = 3

@dataclass
class MFEMRefinementModel:
    particles: np.ndarray
    tet_indices: np.ndarray
    tet_materials: np.ndarray
    parameters: np.ndarray

    @classmethod
    def from_model(cls, model: newton.Model, max_particles: int, max_tets: int,
                   max_tris: int = 0):
        particles = model.particle_q.numpy()
        tet_indices = model.tet_indices.numpy()
        tet_materials = model.tet_materials.numpy()
        params = np.array([max_particles, max_tets, 1.0, max_tris])
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
    def load(file: str, scale: wp.float32, translation: wp.float32):
        model_store = np.load(file, allow_pickle=True)
        particles = model_store["particles"]
        tet_indices = model_store["tet_indices"]
        tet_materials = model_store["tet_materials"]
        parameters = model_store["parameters"]

        return MFEMRefinementModel(
            particles=particles * scale + translation,
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
    def max_tris(self):
        """Baked surface-tri ceiling; 0 (missing on older .npz) lets the solver
        fall back to deriving it from the max_tets headroom factor."""
        if len(self.parameters) <= PARAM_MAX_TRIS:
            return 0
        return int(self.parameters[PARAM_MAX_TRIS])

    @property
    def density(self):
        return float(self.parameters[PARAM_DENSITY])

    def resolve_max_particles(self, mult: float | None) -> int:
        """``mult`` * the mesh's actual particle count, rounded up; ``None``
        keeps the ceiling baked into the .npz. Expressing the override as a
        multiple of the mesh's own size means the caller (e.g. the sim's
        ``--max-particles-mult`` CLI flag) doesn't need to know that count."""
        if mult is None:
            return self.max_particles
        return math.ceil(mult * self.particles.shape[0])

    def resolve_max_tets(self, mult: float | None) -> int:
        """Like :meth:`resolve_max_particles`, relative to the mesh's actual
        tet count."""
        if mult is None:
            return self.max_tets
        return math.ceil(mult * self.tet_indices.shape[0])

    def resolve_max_tris(self, mult: float | None, model: newton.Model) -> int:
        """Like :meth:`resolve_max_particles`, relative to ``model``'s actual
        surface-tri count. Unlike particle/tet counts, the surface
        triangulation isn't stored on the .npz -- it's derived by
        ``add_soft_mesh`` -- so this takes the already-built sim ``model``
        (from :meth:`load_sim_model`) rather than deriving it from ``self``."""
        if mult is None:
            return self.max_tris
        return math.ceil(mult * model.tri_indices.shape[0])

    def load_sim_model(self, builder: newton.ModelBuilder, mu: float, max_particles: int | None = None):
        """``max_particles`` pads the particle buffer for refinement growth; it
        defaults to the ceiling baked into the .npz (``self.max_particles``) but
        callers (e.g. the sim's ``--max-particles-mult`` CLI flag, resolved via
        :meth:`resolve_max_particles`) may override it. Whatever value is used
        here must also be passed to ``RefinementSolver`` as ``max_particles=``
        -- the solver validates the two agree."""
        if max_particles is None:
            max_particles = self.max_particles

        particles = np.zeros(
            (max_particles, 3),
            dtype=np.float32,
        )

        particles[: self.particles.shape[0], :] = self.particles
        density = self.density


        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_euler(wp.vec3(wp.PI / 2.0, 0.0, 0.0), 0, 1, 2),
            scale=wp.float32(1.0),
            vertices=particles,
            indices=self.tet_indices.flatten(),
            density=density,
            k_mu=mu,
            k_lambda=self.tet_materials[:, 1],
            k_damp=self.tet_materials[:, 2],
            add_surface_mesh_edges=False,
            validate_mesh=False,
        )

        model = builder.finalize()


        # wp.copy(model.particle_inv_mass, wp.array(np.where(model.particle_q.numpy().reshape(-1, 3)[:, 0] < 0.1, 0.0, model.particle_inv_mass.numpy()), device="cpu"))

        return model