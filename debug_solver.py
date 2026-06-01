"""Debug: energy convergence test over multiple time steps."""
import sys
sys.path.insert(0, "/home/user/warp-mfem/src")

import numpy as np
import warp as wp
import newton

from mfem import ARAPEnergy, MFEMSolver

wp.init()

builder = newton.ModelBuilder()
builder.add_soft_grid(
    pos=wp.vec3(0.0, 1.0, 1.0),
    rot=wp.quat_identity(),
    vel=wp.vec3(0.0, -1.0, 0.0),
    dim_x=1, dim_y=1, dim_z=1,
    cell_x=1.0, cell_y=1.0, cell_z=1.0,
    density=1.0e-3, k_mu=1.0e1, k_lambda=1.0e1, k_damp=0.0,
    fix_left=True,
)
builder.color()
model = builder.finalize()

solver = MFEMSolver(model=model, iterations=8, material_model=ARAPEnergy)
state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.contacts()
dt = 1.0 / 30.0

print("\n=== Energy convergence across Newton iterations (first time step) ===")
solver.step(state_0, state_1, control, contacts, dt)
print("\n=== Done ===")
