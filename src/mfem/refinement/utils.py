from mfem.types import mat69, mat912, mat99, vec6, vec9
import warp as wp

SYM = wp.constant(wp.diag(vec6(1.0, 1.0, 1.0, 2.0, 2.0, 2.0)))
INV_SYM = wp.constant(wp.diag(vec6(1.0, 1.0, 1.0, 0.5, 0.5, 0.5)))

@wp.func
def sym6_to_mat33(s: vec6) -> wp.mat33:
    return wp.mat33(
        s[0], s[3], s[4],
        s[3], s[1], s[5],
        s[4], s[5], s[2],
    )

@wp.func
def mat33_to_sym6(mat: wp.mat33) -> vec6:
    return vec6(
        mat[0, 0], mat[1, 1], mat[2, 2],
        mat[0, 1], mat[0, 2], mat[1, 2],
    )

@wp.func
def flatten(mat: wp.mat33) -> vec9:
    c0 = mat[:, 0]
    c1 = mat[:, 1]
    c2 = mat[:, 2]

    return vec9(*c0, *c1, *c2)

@wp.func
def unflatten(vec: vec9) -> wp.mat33:
    return wp.matrix_from_cols(vec[0:3], vec[3:6], vec[6:9])


@wp.func
def grad_det_sym_vec6(s: vec6) -> vec6:
    return vec6(
        s[1] * s[2] - s[5] * s[5],
        s[0] * s[2] - s[4] * s[4],
        s[0] * s[1] - s[3] * s[3],
        2.0 * (s[4] * s[5] - s[2] * s[3]),
        2.0 * (s[3] * s[5] - s[1] * s[4]),
        2.0 * (s[3] * s[4] - s[0] * s[5]),
    )

@wp.func
def rotational_svd(A: wp.mat33):
    U = wp.mat33()
    sigma = wp.vec3()
    V = wp.mat33()
    wp.svd3(A, U, sigma, V)

    det_U = wp.determinant(U)
    det_V = wp.determinant(V)

    flipped = det_U * det_V
    sigma[2] *= flipped
    if det_U < 0 and det_V >= 0:
        U[0, 2] *= flipped
        U[1, 2] *= flipped
        U[2, 2] *= flipped
    if det_V < 0 and det_U >= 0:
        V[0, 2] *= flipped
        V[1, 2] *= flipped
        V[2, 2] *= flipped

    return U, sigma, V


@wp.func
def stretch_component(_U: wp.mat33, sigma: wp.vec3, V: wp.mat33) -> wp.mat33:
    return V * wp.diag(sigma) * wp.transpose(V)


@wp.func
def stretch_gradient(F: wp.mat33, U: wp.mat33, sigma: wp.vec3, V: wp.mat33) -> mat69:
    dR_dF = rotation_gradient(U, sigma, V)
    R = U * wp.transpose(V)
    grad = mat69()
    FT = wp.transpose(F)

    for i in range(3):
        for j in range(3):
            dS_dF_ij = FT * unflatten(dR_dF[:, j * 3 + i])
            dS_dF_ij[j, 0] += R[i, 0]
            dS_dF_ij[j, 1] += R[i, 1]
            dS_dF_ij[j, 2] += R[i, 2]
            grad[:, i * 3 + j] = mat33_to_sym6(dS_dF_ij)

    return grad


@wp.func
def rotation_gradient(U: wp.mat33, sigma: wp.vec3, V: wp.mat33) -> mat99:

    T0 = wp.mat33(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0) # Twist about the z-axis
    T1 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0) # Twist about the x-axis
    T2 = wp.mat33(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0) # Twist about the y-axis

    T0 = (1.0 / wp.sqrt(2.0)) * U * T0 * wp.transpose(V)
    T1 = (1.0 / wp.sqrt(2.0)) * U * T1 * wp.transpose(V)
    T2 = (1.0 / wp.sqrt(2.0)) * U * T2 * wp.transpose(V)

    t0 = flatten(T0)
    t1 = flatten(T1)
    t2 = flatten(T2)

    s0 = sigma[0]
    s1 = sigma[1]
    s2 = sigma[2]

    s01 = wp.max(s0 + s1, 1.0e-8)
    s12 = wp.max(s1 + s2, 1.0e-8)
    s02 = wp.max(s0 + s2, 1.0e-8)

    dR_dF = (
        (2.0 / s01) * wp.outer(t0, t0)
        + (2.0 / s12) * wp.outer(t1, t1)
        + (2.0 / s02) * wp.outer(t2, t2)
    )
    return dR_dF

@wp.func
def deformation_gradient_dF_dx(DmInv: wp.mat33) -> mat912:
    m = DmInv[0, 0]
    n = DmInv[0, 1]
    o = DmInv[0, 2]
    p = DmInv[1, 0]
    q = DmInv[1, 1]
    r = DmInv[1, 2]
    s = DmInv[2, 0]
    t = DmInv[2, 1]
    u = DmInv[2, 2]

    t1 = -m - p - s
    t2 = -n - q - t
    t3 = -o - r - u

    # Row-major: row i*3+j corresponds to F[i,j]; nonzero cols are 3*k+i for vertices k=0..3
    dFdx = mat912()
    dFdx[0, 0] = t1
    dFdx[0, 3] = m
    dFdx[0, 6] = p
    dFdx[0, 9] = s
    dFdx[1, 0] = t2
    dFdx[1, 3] = n
    dFdx[1, 6] = q
    dFdx[1, 9] = t
    dFdx[2, 0] = t3
    dFdx[2, 3] = o
    dFdx[2, 6] = r
    dFdx[2, 9] = u
    dFdx[3, 1] = t1
    dFdx[3, 4] = m
    dFdx[3, 7] = p
    dFdx[3, 10] = s
    dFdx[4, 1] = t2
    dFdx[4, 4] = n
    dFdx[4, 7] = q
    dFdx[4, 10] = t
    dFdx[5, 1] = t3
    dFdx[5, 4] = o
    dFdx[5, 7] = r
    dFdx[5, 10] = u
    dFdx[6, 2] = t1
    dFdx[6, 5] = m
    dFdx[6, 8] = p
    dFdx[6, 11] = s
    dFdx[7, 2] = t2
    dFdx[7, 5] = n
    dFdx[7, 8] = q
    dFdx[7, 11] = t
    dFdx[8, 2] = t3
    dFdx[8, 5] = o
    dFdx[8, 8] = r
    dFdx[8, 11] = u
    return dFdx


if __name__ == "__main__":
    print("Okay")
