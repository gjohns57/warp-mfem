import warp as wp
import newton
import numpy as np
from attr import dataclass

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


    def load_sim_model(self, builder: newton.ModelBuilder):

        particles = np.zeros(
            (self.max_particles, 3),
            dtype=np.float32,
        )

        particles[: self.particles.shape[0], :] = self.particles
        density = self.density

        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 2.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=wp.float32(1.0),
            vertices=particles,
            indices=self.tet_indices.flatten(),
            density=density,
            k_mu=self.tet_materials[:, 0],
            k_lambda=self.tet_materials[:, 1],
            k_damp=self.tet_materials[:, 2],
            add_surface_mesh_edges=False,
            validate_mesh=False,
        )

        model = builder.finalize()


        wp.copy(model.particle_inv_mass, wp.array(np.where(model.particle_q.numpy().reshape(-1, 3)[:, 0] < 0.1, 0.0, model.particle_inv_mass.numpy()), device="cpu"))

        return model