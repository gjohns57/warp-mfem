from decimal import DecimalTuple
import warp as wp
import warp.fem as fem
import numpy as np

import warp.render

def gen_tetmesh(res, bounds_lo: wp.vec3 | None = None, bounds_hi: wp.vec3 | None = None):
    """Construct a tetrahedral mesh by diving each cell of a dense 3D grid into five tetrahedrons.

    Args:
        res: Resolution of the grid along each dimension
        bounds_lo: Position of the lower bound of the axis-aligned grid
        bounds_hi: Position of the upper bound of the axis-aligned grid

    Returns:
        Tuple of ndarrays: (Vertex positions, Tetrahedron vertex indices)
    """

    if bounds_lo is None:
        bounds_lo = wp.vec3(0.0)

    if bounds_hi is None:
        bounds_hi = wp.vec3(1.0)

    Nx = res[0]
    Ny = res[1]
    Nz = res[2]

    x = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
    y = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)
    z = np.linspace(bounds_lo[2], bounds_hi[2], Nz + 1)

    positions = np.transpose(np.meshgrid(x, y, z, indexing="ij"), axes=(1, 2, 3, 0)).reshape(-1, 3)

    vidx = fem.utils.grid_to_tets(Nx, Ny, Nz)

    return wp.array(positions, dtype=wp.vec3), wp.array(vidx, dtype=int)

# @fem.integrand
# def elastic_gradient_dx(
#     s: fem.Sample,
#     domain: fem.Domain,
#     displacement: fem.Field,
#     stretch: fem.Field,
# ):
    

@fem.integrand
def constraint_integrand(
    s: fem.Sample,
    domain: fem.Domain,
    displacement: fem.Field,
    stretch: fem.Field,
    lagrange_multiplier: fem.Field,
):
    sym_deformation = wp.identity(n=3, dtype=wp.float32) + fem.D(displacement, s)

    return wp.ddot(lagrange_multiplier(s), sym_deformation - stretch(s))    


@fem.integrand
def constraint_grad_x(
    s: fem.Sample,
    domain: fem.Domain,
    displacement: fem.Field,
    stretch: fem.Field,
    lagrange_multiplier: fem.Field,
    ):

    return wp.array([1, 2, 3, 4], dtype=wp.float32)


class Example:
    def __init__(self) -> None:

        bounds_lo = wp.vec3(0.0, 0.0, 0.0)
        bounds_hi = wp.vec3(1.0, 0.5, 2.0)
        positions, tetrahedra = gen_tetmesh((10, 10, 10), bounds_lo=bounds_lo, bounds_hi=bounds_hi)
        self._geo = fem.Tetmesh(tet_vertex_indices=tetrahedra, positions=positions)
        self.tetrahedra = self._geo.tet_vertex_indices

        self._displacement_space = fem.make_polynomial_space(self._geo, degree=1, dtype=wp.vec3, element_basis=fem.ElementBasis.LAGRANGE)
        self._stretch_space = fem.make_polynomial_space(self._geo, degree=0, dof_mapper=fem.SymmetricTensorMapper(wp.mat33), element_basis=fem.ElementBasis.LAGRANGE)
        self._lagrange_multiplier_space = fem.make_polynomial_space(self._geo, degree=0, dof_mapper=fem.SymmetricTensorMapper(wp.mat33), element_basis=fem.ElementBasis.LAGRANGE)

        self._cell_constraint_space = fem.make_polynomial_space(self._geo, degree=0, element_basis=fem.ElementBasis.LAGRANGE)


        self._displacement_field: fem.DiscreteField = self._displacement_space.make_field()
        self._stretch_field: fem.DiscreteField = self._stretch_space.make_field()
        self._lagrange_multiplier_field = self._lagrange_multiplier_space.make_field()
        self._constraint_field
        # print(type(self._stretch_field.dof_values.numpy()[0][0]))

        identity_sym = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, 0.0, 0.0], dtype=np.float32)
        self._stretch_field.dof_values.fill_(identity_sym)
        self._lagrange_multiplier_field.dof_values.fill_(identity_sym)


    def step(self):

        domain = fem.Cells(geometry=self._geo)

        constraint_values = wp.zeros(len(self.tetrahedra), dtype=float)
        fem.interpolate(
            integrand=constraint_integrand,
            dest=constraint_values,
            at=domain,
            fields={
                "displacement": self._displacement_field,
                "stretch": self._stretch_field,
                "lagrange_multiplier": self._lagrange_multiplier_field
            },
        )
        

        displacement_trial = fem.make_trial(space= self._displacement_space, domain=domain)
        stretch_test = fem.make_test(space=self._stretch_space, domain=domain);
        grad_x = fem.integrate(
            integrand=constraint_integrand,
            domain=domain,
            fields={
                "displacement": displacement_trial,
                "stretch": self._stretch_field,
                "lagrange_multiplier": self._lagrange_multiplier_field
            },
            assembly="nodal"
        )

        print(grad_x)

        # print(per_tet_output.numpy())

        # fem.integrate()
        


if __name__ == "__main__":
    example = Example()

    for i in range(1):
        example.step()

