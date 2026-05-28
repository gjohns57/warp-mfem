import warp as wp
from warp.types import matrix, vector


class vec6(vector(length=6, dtype=wp.float32)):
    """"""


class vec9(vector(length=9, dtype=wp.float32)):
    """"""


class mat99(matrix(shape=(9, 9), dtype=wp.float32)):
    """9x9 Matrix for partial^2 I_1 / partial f^2"""


class mat912(matrix(shape=(9, 12), dtype=wp.float32)):
    """dF/dx: Jacobian of the 9 deformation gradient components w.r.t. 12 vertex DOFs."""


class mat63(matrix(shape=(6, 3), dtype=wp.float32)):
    """6 x 3 Matrix"""


class mat612(matrix(shape=(6, 12), dtype=wp.float32)):
    """6 x 12 Matrix"""


class mat69(matrix(shape=(6, 9), dtype=wp.float32)):
    """6 x 9 Matrix"""


class mat66(matrix(shape=(6, 6), dtype=wp.float32)):
    """6 x 6 Matrix"""
