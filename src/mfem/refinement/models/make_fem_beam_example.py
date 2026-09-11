import newton
import warp as wp
from mfem.refinement.models import MFEMRefinementModel

if __name__ == "__main__":
    builder = newton.ModelBuilder(gravity=-9.81)

    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(wp.float32),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=8,
        dim_y=4,
        dim_z=4,
        cell_x=1.0,
        cell_y=1.0,
        cell_z=1.0,
        k_mu=6.0,
        k_lambda=1.0,
        k_damp=0.0,
        density=1.0,
    )

    model = builder.finalize()

    # Give particles the same 4x headroom convention already used for tets,
    # so refinement (which creates new vertices) has somewhere to grow into.
    refinement_model = MFEMRefinementModel.from_model(model, max_particles=int(model.particle_count * 16), max_tets=int(model.tet_count * 32))
    refinement_model.save("fem_beam.npz")