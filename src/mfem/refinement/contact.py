from newton import State, Model, GeoType
from mfem.refinement.additional_state import AdditionalState
from mfem.ipc.distance import capsule_sdf
import warp as wp
import warp.sparse as ws



# We can see if it is cost effective to build a array that stores indices for surface points instead of just checking
# all points. If we end up building a surface tri mesh at each step we could at the same time build a surface points
# indices array


@wp.func
def barrier(d: wp.float32, d0: wp.float32, d1: wp.float32, b_d0: wp.float32, db_d0 : wp.float32, d2b_d0: wp.float32) -> tuple[wp.float32, wp.float32, wp.float32]:
    energy = wp.float32(0.0)
    d_energy = wp.float32(0.0)
    d2_energy = wp.float32(0.0)
    if d < d1 and d > d0:
        log_d_d_1 = wp.log(d / d1)
        energy = -(d - d1) * (d - d1) * log_d_d_1
        d_energy = -(2.0 * (d - d1) * log_d_d_1 + (d - d1) * (d - d1) / d)
        d2_energy = -(2.0 * log_d_d_1 + 4.0 * (d - d1) / d - (d - d1) * (d - d1) / (d * d))
    elif d <= d0:
        energy = b_d0 + db_d0 * (d - d0) + 0.5 *  d2b_d0 * (d - d0) * (d - d0)
        d_energy = db_d0 + d2b_d0 * (d - d0)
        d2_energy = d2b_d0

    return energy, d_energy, d2_energy

@wp.kernel
def evaluate_barrier_energy(
    active_particle_count: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_count: wp.int32,
    body_q: wp.array[wp.transform],
    stiffness: wp.array[wp.float32],
    d0: wp.float32, # The distance closer to which (or if the signed distance is negative) we use a quadratic extrapolation of the log barrier
    d1: wp.float32, # The distance after which the barrier energy is 0
    quadratic_barrier_coefficients: wp.vec3,
    barrier_energy: wp.array[wp.float32],
    barrier_gradient: wp.array[wp.vec3],
    barrier_hessian: wp.array[wp.mat33],
):
    tid = wp.tid()
    if tid >= active_particle_count[0]:
        return
    x = particle_q[tid]
    k = stiffness[0]
    energy = wp.float32(0.0)
    gradient = wp.vec3(0.0)
    hessian = wp.mat33(0.0)


    for shape in range(shape_count):
        d = wp.float32(0.0)
        grad_d = wp.vec3(0.0)
        hess_d = wp.mat33(0.0)

        # Transform x into shape local frame
        body_transform = body_q[shape_body[shape]]
        shape_world_transform = wp.transform_multiply(body_transform, shape_transform[shape])
        x_local = wp.transform_point(wp.transform_inverse(shape_world_transform), x)

        if shape_type[shape] == GeoType.CAPSULE:
            scale = shape_scale[shape]
            d, n_local, hess_local = capsule_sdf(x_local, scale[0], scale[1])

            # Gradient direction transforms as a normal (rotation only, no translation)
            grad_d = wp.transform_vector(shape_world_transform, n_local)

            # Hessian transforms as a bilinear form under the shape's rotation: R H R^T
            rotation = wp.quat_to_matrix(wp.transform_get_rotation(shape_world_transform))
            hess_d = rotation * hess_local * wp.transpose(rotation)
        else:
            continue


        if d > d1:
            continue
        e, de, d2e = barrier(d, d0, d1, *quadratic_barrier_coefficients)
        energy += e
        gradient += de * grad_d
        hessian += d2e * wp.outer(grad_d, grad_d) + de * hess_d

    barrier_energy[tid] = k * energy
    barrier_gradient[tid] = k * gradient
    barrier_hessian[tid] = k * hessian

class Contact:
    def __init__(self, model: Model, max_particles: int, d0: float, d1: float, stiffness: float):
        self.barrier_energy = wp.zeros(max_particles, dtype=wp.float32)
        self.barrier_gradient = wp.zeros(max_particles, dtype=wp.vec3)
        self.barrier_hessian_blocks = wp.zeros(max_particles, dtype=wp.mat33)
        self._quadratic_barrier_coefficients = wp.vec3(*barrier(d0, 0.0, d1, 0.0, 0.0, 0.0))
        self._stiffness = wp.array([stiffness], dtype=wp.float32)
        self.d0 = d0
        self.d1 = d1

    def evaluate(self, model: Model, state: State, additional_state: AdditionalState) -> tuple[wp.array[wp.float32], ws.array[wp.vec3], wp.array[wp.mat33]]:

        wp.launch(
            evaluate_barrier_energy,
            dim=model.particle_count,
            inputs=[
                additional_state.active_particle_count,
                state.particle_q,
                model.shape_transform,
                model.shape_type,
                model.shape_scale,
                model.shape_body,
                model.shape_count,
                state.body_q,
                self._stiffness,
                self.d0,
                self.d1,
                self._quadratic_barrier_coefficients,
            ],
            outputs=[
                self.barrier_energy,
                self.barrier_gradient,
                self.barrier_hessian_blocks,
            ]
        )