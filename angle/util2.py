import numpy as np


def to_ZAMO_up(vec, nu, mu1, mu2, omega, psi):
    basis = np.matrix([[nu, 0, 0, 0],
                       [0, mu1, 0, 0],
                       [0, 0, mu2, 0],
                       [-omega * nu, 0, 0, psi]])

    return np.asarray(np.matmul(basis, vec)).reshape(-1)


def to_ZAMO_down(vec, nu, mu1, mu2, omega, psi):
    basis = np.matrix([[1 / nu, 0, 0, omega / nu],
                       [0, 1 / mu1, 0, 0],
                       [0, 0, 1 / mu2, 0],
                       [0, 0, 0, 1 / psi]])

    return np.asarray(np.matmul(basis, vec)).reshape(-1)


def a_up_b_down(vec, v, gv):
    basis = np.matrix([[gv, 0, 0, gv * v],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [gv * v, 0, 0, gv]])

    return np.asarray(np.matmul(basis, vec)).reshape(-1)


def a_down_b_up(vec, v, gv):
    basis = np.matrix([[gv, 0, 0, -gv * v],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [-gv * v, 0, 0, gv]])

    return np.asarray(np.matmul(basis, vec)).reshape(-1)


def b_up_d_down(vec, u1, u3, gu):
    basis = np.matrix([[gu, gu * u1, 0, gu * u3],
                       [gu * u1, 1 + gu ** 2 * u1 ** 2 / (1 + gu), 0, gu ** 2 * u1 * u3 / (1 + gu)],
                       [0, 0, 1, 0],
                       [gu * u3, gu ** 2 * u1 * u3 / (1 + gu), 0, 1 + gu ** 2 * u3 ** 2 / (1 + gu)]])

    return np.asarray(np.matmul(basis, vec)).reshape(-1)


def b_down_d_up(vec, u1, u3, gu):
    basis = np.matrix([[gu, -gu * u1, 0, -gu * u3],
                       [-gu * u1, 1 + gu ** 2 * u1 ** 2 / (1 + gu), 0, gu ** 2 * u1 * u3 / (1 + gu)],
                       [0, 0, 1, 0],
                       [-gu * u3, gu ** 2 * u1 * u3 / (1 + gu), 0, 1 + gu ** 2 * u3 ** 2 / (1 + gu)]])

    return np.asarray(np.matmul(basis, vec)).reshape(-1)


def main3(dx, u1, u3, v, nu, mu1, mu2, psi, omega):
    gu = 1 / np.sqrt(1 - u1 ** 2 - u3 ** 2)
    gv = 1 / np.sqrt(1 - v ** 2)

    #print('Start debugging ...')
    # redefine k as the momentum, i.e.:
    dx[0] = (- nu ** 2 + omega ** 2 * psi ** 2) * dx[0] - psi ** 2 * omega * dx[3]
    dx[1] *= mu1 ** 2
    dx[2] *= mu2 ** 2
    dx[3] = - psi ** 2 * omega * dx[0] + psi ** 2 * dx[3]

    p_a = to_ZAMO_down(dx, nu, mu1, mu2, omega, psi)
    p_b = a_up_b_down(p_a, v, gv)
    p_d = b_up_d_down(p_b, u1, u3, gu)

    alpha = -(gv * gu + gv * v * gu * u3) ** 2 + (gu * u1) ** 2 + (p_d[0] / p_d[2]) ** 2 + (gv * v * gu + gv * gu * u3) ** 2
    alpha = 1 / np.sqrt(alpha)

    fd0 = - alpha
    fd1 = 0
    fd2 = alpha * p_d[0] / p_d[2]
    fd3 = 0

    f_d = np.array([fd0, fd1, fd2, fd3])
    f_b = b_up_d_down(f_d, u1, u3, gu)
    f_a = a_up_b_down(f_b, v, gv)

    ft = 1 / nu * f_a[0]
    fr = 1 / mu1 * f_a[1]
    fth = 1 / mu2 * f_a[2]
    fphi = 1 / psi * (omega * f_a[0] + f_a[3])

    return ft, fr, fth, fphi

