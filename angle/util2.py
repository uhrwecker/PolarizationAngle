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


def matrix_g00(gv, v, gu, u1, u3):
    g_ab = np.diag([-1, 1, 1, 1])
    lorentz_ac = np.matrix([[gv, 0, 0, gv * v],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [gv * v, 0, 0, gv]])
    lorentz_cd = np.matrix([[gu, gu * u1, 0, gu * u3],
                            [gu * u1, 1 + gu ** 2 * u1 ** 2 / (1 + gu), 0, gu ** 2 * u1 * u3 / (1 + gu)],
                            [0, 0, 1, 0],
                            [gu * u3, gu ** 2 * u1 * u3 / (1 + gu), 0, 1 + gu ** 2 * u3 ** 2 / (1 + gu)]])

    lac = np.matmul(lorentz_ac, lorentz_ac)
    lcd = np.matmul(lorentz_cd, lorentz_cd)
    g_ac = np.matmul(lac, g_ab)

    return np.matmul(lcd, g_ac)[0][0]


def dd_metric(vec, gv, v, gu, u1, u3):
    g_ab = np.diag([-1, 1, 1, 1])
    lorentz_ac = np.matrix([[gv, 0, 0, gv * v],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [gv * v, 0, 0, gv]])
    lorentz_cd = np.matrix([[gu, gu * u1, 0, gu * u3],
                            [gu * u1, 1 + gu ** 2 * u1 ** 2 / (1 + gu), 0, gu ** 2 * u1 * u3 / (1 + gu)],
                            [0, 0, 1, 0],
                            [gu * u3, gu ** 2 * u1 * u3 / (1 + gu), 0, 1 + gu ** 2 * u3 ** 2 / (1 + gu)]])

    lac = np.matmul(lorentz_ac, lorentz_ac)
    lcd = np.matmul(lorentz_cd, lorentz_cd)
    g_ac = np.matmul(lac, g_ab)

    g_dd = np.matmul(lcd, g_ac)

    return np.asarray(np.matmul(g_dd, vec)).reshape(-1)


def main(dx, u1, u3, v, nu, mu1, mu2, psi, omega):
    gu = 1 / np.sqrt(1 - u1 ** 2 - u3 ** 2)
    gv = 1 / np.sqrt(1 - v ** 2)

    beta = matrix_g00(gv, v, gu, u1, u3)[0, 0]
    delta = -matrix_g00(gv, v, gu, u1, u3)[0, -1]

    # redefine k as the momentum, i.e.:
    dx[0] = (- nu ** 2 + omega ** 2 * psi ** 2) * dx[0] - psi ** 2 * omega * dx[3]
    dx[1] *= mu1 ** 2
    dx[2] *= mu2 ** 2
    dx[3] = - psi ** 2 * omega * dx[0] + psi ** 2 * dx[3]

    k_a = to_ZAMO_up(dx, nu, mu1, mu2, omega, psi)
    k_b = a_down_b_up(k_a, v, gv)
    k_d = b_down_d_up(k_b, u1, u3, gu)

    #kf = dd_metric(k_d, gv, v, gu, u1, u3)

    fd0 = -np.sqrt(k_d[2] ** 2 / ((beta * k_d[0] + delta * k_d[3]) ** 2 + beta * k_d[2] ** 2))
    fd1 = 0
    fd2 = np.sqrt((beta * k_d[0] + delta * k_d[3]) ** 2 / ((beta * k_d[0] + delta * k_d[3]) ** 2 + beta * k_d[2] ** 2))
    fd3 = 0

    f_d = np.array([fd0, fd1, fd2, fd3])
    ff = dd_metric(f_d, gv, v, gu, u1, u3)

    print(beta * f_d[0], f_d[2], ff)
    print(beta * f_d[0] ** 2 + f_d[2] ** 2, np.dot(f_d, ff))
    print(beta * f_d[0] * k_d[0] + f_d[2] * k_d[2] + delta * f_d[0] * k_d[3], np.dot(ff, k_d))

    print('Start debugging ...')

    f_b = b_up_d_down(f_d, u1, u3, gu)
    f_a = a_up_b_down(f_d, v, gv)

    ff_a = a_down_b_up(ff, v, gv)

    print(f_d)
    print(f_a)
    print(- f_a[0] ** 2 + f_a[1] ** 2 + f_a[2] ** 2 + f_a[3] ** 2)
    print(np.dot(ff_a, f_a), np.dot(ff_a, k_a))
    #print(- f_a[0] * k_a[0] + f_a[1] * k_a[1] + f_a[2] * k_a[2] + f_a[3] * k_a[3])
    #print(- f_a[0] * ffb[0] + f_a[1] * ffb[1] + f_a[2] * ffb[2] + f_a[3] * ffb[3])

    ft = 1 / nu * f_a[0]
    fr = 1 / mu1 * f_a[1]
    fth = 1 / mu2 * f_a[2]
    fphi = 1 / psi * (omega * f_a[0] + f_a[3])

    st_prod = (- nu ** 2 + omega ** 2 * psi ** 2) * ft * dx[0] + \
              mu1 ** 2 * dx[1] * fr + mu2 ** 2 * dx[2] * fth - \
              omega * psi ** 2 * (dx[0] * fphi + dx[3] * ft) + psi ** 2 * fphi * dx[3]

    #st_prod = - nu ** 2 * ft ** 2 + mu1 ** 2 * fr ** 2 + mu2 ** 2 * fth ** 2 + psi ** 2 * fphi ** 2

    print('Checking if definition went right: fmu * kmu = {} (should be nearly 0)'.format(st_prod))

    #st_prod = - (1 - 2 / 35) * ft * dx[0] + 1 / (1 - 2 / 35) * fr * dx[1] + 35 ** 2 * fth * dx[2] + 35 ** 2 * np.sin(1.) ** 2 * fphi * dx[3]

    #print('Checking if definition went right: fmu * kmu = {} (should be nearly 0)'.format(st_prod))

    return ft, fr, fth, fphi


def main2(dx, u1, u3, v, nu, mu1, mu2, psi, omega):
    gu = 1 / np.sqrt(1 - u1 ** 2 - u3 ** 2)
    gv = 1 / np.sqrt(1 - v ** 2)

    beta = matrix_g00(gv, v, gu, u1, u3)[0, 0]
    delta = -matrix_g00(gv, v, gu, u1, u3)[0, -1]

    print('Start debugging ...')
    # redefine k as the momentum, i.e.:
    dx[0] = (- nu ** 2 + omega ** 2 * psi ** 2) * dx[0] - psi ** 2 * omega * dx[3]
    dx[1] *= mu1 ** 2
    dx[2] *= mu2 ** 2
    dx[3] = - psi ** 2 * omega * dx[0] + psi ** 2 * dx[3]

    p_a = to_ZAMO_down(dx, nu, mu1, mu2, omega, psi)
    p_b = a_up_b_down(p_a, v, gv)
    p_d = b_up_d_down(p_b, u1, u3, gu)

    # kf = dd_metric(k_d, gv, v, gu, u1, u3)

    fd0 = np.sqrt(p_d[2] ** 2 / np.abs(p_d[0] ** 2 + beta * p_d[2] ** 2))
    fd1 = 0
    fd2 = -np.sqrt(p_d[0] ** 2 / np.abs(p_d[0] ** 2 + beta * p_d[2] ** 2))
    fd3 = 0

    f_d = np.array([fd0, fd1, fd2, fd3])
    ff = dd_metric(f_d, gv, v, gu, u1, u3)

    #print(beta * f_d[0], f_d[2], ff)
    #print(beta * f_d[0] ** 2 + f_d[2] ** 2, np.dot(f_d, ff))
    #print(f_d[0] * p_d[0] + f_d[2] * p_d[2], np.dot(f_d, p_d))

    f_b = b_up_d_down(f_d, u1, u3, gu)
    f_a = a_up_b_down(f_b, v, gv)

    ff_b = b_down_d_up(ff, u1, u3, gu)
    ff_a = a_down_b_up(ff_b, v, gv)

    #print(f_d)
    #print(f_a)
    #print(- f_a[0] ** 2 + f_a[1] ** 2 + f_a[2] ** 2 + f_a[3] ** 2)
    #print(np.dot(ff_a, f_a), np.dot(f_a, p_a))
    # print(- f_a[0] * k_a[0] + f_a[1] * k_a[1] + f_a[2] * k_a[2] + f_a[3] * k_a[3])
    # print(- f_a[0] * ffb[0] + f_a[1] * ffb[1] + f_a[2] * ffb[2] + f_a[3] * ffb[3])

    ft = 1 / nu * f_a[0]
    fr = 1 / mu1 * f_a[1]
    fth = 1 / mu2 * f_a[2]
    fphi = 1 / psi * (omega * f_a[0] + f_a[3])

    st_prod = (- nu ** 2 + omega ** 2 * psi ** 2) * ft ** 2 + \
              mu1 ** 2 * fr ** 2 + mu2 ** 2 * fth ** 2 - \
              omega * psi ** 2 * (ft * fphi + fphi * ft) + psi ** 2 * fphi ** 2

    # st_prod = - nu ** 2 * ft ** 2 + mu1 ** 2 * fr ** 2 + mu2 ** 2 * fth ** 2 + psi ** 2 * fphi ** 2
    #print('Checking if definition went right: fmu * fmu = {} (should be nearly 1)'.format(st_prod))#

    #print('Checking if definition went right: fmu * kmu = {} (should be nearly 0)'.format(np.dot(dx, np.array([ft, fr, fth, fphi]))))

    # st_prod = - (1 - 2 / 35) * ft * dx[0] + 1 / (1 - 2 / 35) * fr * dx[1] + 35 ** 2 * fth * dx[2] + 35 ** 2 * np.sin(1.) ** 2 * fphi * dx[3]

    # print('Checking if definition went right: fmu * kmu = {} (should be nearly 0)'.format(st_prod))

    return ft, fr, fth, fphi


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

    #print(f_a, alpha)

    ft = 1 / nu * f_a[0]
    fr = 1 / mu1 * f_a[1]
    fth = 1 / mu2 * f_a[2]
    fphi = 1 / psi * (omega * f_a[0] + f_a[3])

    st_prod = (- nu ** 2 + omega ** 2 * psi ** 2) * ft ** 2 + \
              mu1 ** 2 * fr ** 2 + mu2 ** 2 * fth ** 2 - \
              omega * psi ** 2 * (ft * fphi + fphi * ft) + psi ** 2 * fphi ** 2

    # st_prod = - nu ** 2 * ft ** 2 + mu1 ** 2 * fr ** 2 + mu2 ** 2 * fth ** 2 + psi ** 2 * fphi ** 2
    #print('Checking if definition went right: fmu * fmu = {} (should be nearly 1)'.format(st_prod))#

    #print('Checking if definition went right: fmu * kmu = {} (should be nearly 0)'.format(np.dot(dx, np.array([ft, fr, fth, fphi]))))

    return ft, fr, fth, fphi

