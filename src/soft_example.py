# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody Hanging
#
# This simulation demonstrates volumetric soft bodies (tetrahedral grids) hanging
# from fixed particles on the left side. Four grids with different damping values
# (1e-1 to 1e-4) showcase the effect of damping on Neo-Hookean elastic behavior.
#
# Command: uv run -m newton.examples softbody.example_softbody_hanging
#
###########################################################################


from mfem import MFEMExample
from newton.examples import init, run

if __name__ == "__main__":
    parser = MFEMExample.create_parser()
    # with wp.ScopedDevice("cuda:0"):
    viewer, args = init(parser)
    example = MFEMExample(viewer, args)
    run(example, args)

    if example.record_energy:
        solver_log = example.solver_log()
        example.solver.write_log("solver_log.parquet")

    example.solver.write_timings("timings_old.json")