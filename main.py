import matplotlib.pyplot as plt
import newton
import warp as wp
from newton.examples import init, run

from mfem import MFEMExample

if __name__ == "__main__":
    parser = MFEMExample.create_parser()
    # with wp.ScopedDevice("cuda:0"):
    viewer, args = init(parser)
    example = MFEMExample(viewer, args)
    fig = plt.figure()
    run(example, args)

    plt.title("Energy per iteration")
    plt.savefig("energy.png")
