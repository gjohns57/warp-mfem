import warp as wp
from mfem.types import vec6
from newton import Model
from mfem.refinement.kernels import precompute_tet_stretch

@wp.kernel
def get_active_particles(
    tet_indices: wp.array2d[wp.int32],
    active_particles: wp.array[wp.int32],
):
    tid = wp.tid()
    for i in range(4):
        count = tet_indices[tid, i] + 1
        if count > active_particles[0]:
            wp.atomic_max(active_particles, 0, count)

class AdditionalState:
    def __init__(
        self,
        tet_indices: wp.array2d[wp.int32],
        tet_stretch: wp.array[vec6],
        tet_lambda: wp.array[vec6],
        tet_poses: wp.array[wp.mat33],
        tet_materials: wp.array2d[wp.float32],
        active_tet_count: wp.array[wp.int32],
        active_particle_count: wp.array[wp.int32],
        rest_particle_q: wp.array[wp.vec3],
        tri_indices: wp.array2d[wp.int32],
        active_tri_count: wp.array[wp.int32],
    ):
        self.tet_indices = tet_indices
        self.tet_stretch = tet_stretch
        self.tet_lambda = tet_lambda
        self.tet_poses = tet_poses
        self.tet_materials = tet_materials
        self.active_tet_count = active_tet_count
        self.active_particle_count = active_particle_count

        self.rest_particle_q = rest_particle_q

        # Surface triangulation (indices into the same particle array as
        # tet_indices), kept up to date across refinement passes the same way
        # tet_indices is -- see refine() / scatter_tris in refinement.py.
        self.tri_indices = tri_indices
        self.active_tri_count = active_tri_count

    @classmethod
    def from_model(
        cls,
        model: Model,
        max_tets: int,
        max_tris: int,
    ):
        tet_indices = wp.empty((max_tets, 4), dtype=wp.int32)
        tet_stretch = wp.empty(max_tets, dtype=vec6)
        tet_lambda = wp.zeros(max_tets, dtype=vec6)
        tet_poses = wp.empty(max_tets, dtype=wp.mat33)
        tet_materials = wp.empty((max_tets, 3), dtype=wp.float32)
        active_tet_count = wp.array([model.tet_count], dtype=wp.int32)
        active_particle_count = wp.zeros(1, dtype=wp.int32)
        # model.particle_q holds each particle's build-time (rest) position;
        # if the caller padded particle_count for refinement headroom, the
        # padding slots come along too, which is fine since they're inactive.
        rest_particle_q = wp.clone(model.particle_q)

        wp.copy(tet_indices, model.tet_indices)
        wp.copy(tet_poses, model.tet_poses)
        wp.copy(tet_materials, model.tet_materials)

        # model.tri_indices is the surface triangulation newton computes for
        # the soft mesh at build time (add_soft_mesh), indices into the same
        # particle array as tet_indices. It's already (num_tris, 3), not flat.
        tri_count = model.tri_indices.shape[0]
        tri_indices = wp.empty((max_tris, 3), dtype=wp.int32)
        active_tri_count = wp.array([tri_count], dtype=wp.int32)
        wp.copy(tri_indices, model.tri_indices)

        wp.launch(
            get_active_particles,
            dim=model.tet_count,
            inputs=[
                model.tet_indices,
                active_particle_count,
            ]
        )


        wp.launch(
            precompute_tet_stretch,
            dim=model.tet_count,
            inputs=[
                model.particle_q,
                model.tet_indices,
                model.tet_poses,
                tet_stretch,
            ],
        )

        return cls(
            tet_indices,
            tet_stretch,
            tet_lambda,
            tet_poses,
            tet_materials,
            active_tet_count,
            active_particle_count,
            rest_particle_q,
            tri_indices,
            active_tri_count,
        )

    def clone(self):
        return AdditionalState(
            wp.clone(self.tet_indices),
            wp.clone(self.tet_stretch),
            wp.clone(self.tet_lambda),
            wp.clone(self.tet_poses),
            wp.clone(self.tet_materials),
            wp.clone(self.active_tet_count),
            wp.clone(self.active_particle_count),
            wp.clone(self.rest_particle_q),
            wp.clone(self.tri_indices),
            wp.clone(self.active_tri_count),
        )

    def asign(self, other: AdditionalState):
        wp.copy(other.active_particle_count, self.active_particle_count)
        wp.copy(other.active_tet_count, self.active_tet_count)
        wp.copy(other.rest_particle_q, self.rest_particle_q)
        wp.copy(other.tet_indices, self.tet_indices)
        wp.copy(other.tet_lambda, self.tet_lambda)
        wp.copy(other.tet_poses, self.tet_poses)
        wp.copy(other.tet_stretch, self.tet_stretch)
        wp.copy(other.tet_materials, self.tet_materials)
        wp.copy(other.active_tri_count, self.active_tri_count)
        wp.copy(other.tri_indices, self.tri_indices)
