import numpy as np


def transform_f(dt, dr, dth, dphi, u1, u3, v, nu, mu1, mu2, psi, omega):
    #print('Getting that f: ')
    # 0. get gammas

    gu = 1 / np.sqrt(1 - u1 ** 2 - u3 ** 2)
    gv = 1 / np.sqrt(1 - v ** 2)

    # 1. transform velocities to SURF reference frame

    ka0 = nu * dt
    ka1 = mu1 * dr
    ka2 = mu2 * dth
    ka3 = - omega * nu * dt + psi * dphi

    kc0 = gv * ka0 + gv * v * ka3
    kc1 = ka1
    kc2 = ka2
    kc3 = gv * v * ka0 + gv * ka0

    kd0 = gu * kc0 + gu * u1 * kc1 + gu * u3 * kc3
    kd1 =  gu * u1 * kc0 + (1 + gu ** 2 * u1 ** 2 / (1 + gu)) * kc1 + gu ** 2 * u1 * u3 / (1 + gu) * kc3
    kd2 = kc2
    kd3 =  gu * u3 * kc0 + gu ** 2 * u1 * u3 / (1 + gu) * kc1 + (1 + gu ** 2 * u3 ** 2 / (1 + gu)) * kc3

    # 2. define f in the SURF frame
    #print(-kc0**2 + kc1 ** 2 + kc3 ** 2 + kc2 ** 2)

    fd0 = np.sqrt(kd2 ** 2 / (kd0 ** 2 - kd2 ** 2))
    fd1 = 0
    fd2 = np.sqrt(kd0 ** 2 / (kd0 ** 2 - kd2 ** 2))
    fd3 = 0

    #print('Checking if definition went right: fd * kd = {} (should be nearly 0)'.format(- fd0 * kd0 + fd2 * kd2))

    # 3. transform f into spacetime coords

    fc0 = gu * fd0 + gu * u1 * fd1 + gu * u3 * fd3
    fc1 = gu * u1 * fd0 + (1 + gu ** 2 * u1 ** 2 / (1 + gu)) * fd1 + gu ** 2 * u1 * u3 / (1 + gu) * fd3
    fc2 = fd2
    fc3 = gu * u3 * fd0 + gu ** 2 * u1 * u3 / (1 + gu) * fd1 + (1 + gu ** 2 * u3 ** 2 / (1 + gu)) * fd3

    #print('Checking if definition went right: fc * kc = {} (should be nearly 0)'.format(- fc0 * kc0 + fc1 * kc1 +
    #                                                                                    fc2 * kc2 + fc3 * kc3))

    fa0 = gv * fc0 + gv * v * fc3
    fa1 = fc1
    fa2 = fc2
    fa3 = gv * v * fc0 + gv * fc3

    #print('Checking if definition went right: fa * ka = {} (should be nearly 0)'.format(- fa0 * ka0 + fa1 * ka1 +
    #                                                                                    fa2 * ka2 + fa3 * ka3))

    ft = 1 / nu * fa0
    fr = 1 / mu1 * fa1
    fth = 1 / mu2 * fa2
    fphi = 1 / psi * (omega * fa0 + fa3)

    st_prod = (- nu ** 2 + omega ** 2 * psi ** 2) * ft * dt + \
        mu1 ** 2 * dr * fr + mu2 ** 2 * dth * fth - \
        omega * psi ** 2 * (dt * fphi + dphi * ft) + psi ** 2 * fphi * dphi

    #print('Checking if definition went right: fmu * kmu = {} (should be nearly 0)'.format(st_prod))

    return ft, fr, fth, fphi
