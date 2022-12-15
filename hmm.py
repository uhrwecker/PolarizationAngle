import numpy as np
from sympy import *
import json
from one_lightray import solver
from angle import equ, util


def main():
    fp = 'Z:/Data/v/0.0/data/-0.0_-0.002097136923322517_-5.305874463909178.json'
    fp = 'Z:/Data/v/3.141592653589793/data/-0.0_-0.002595070280410998_7.901996434418482.json'

    with open(fp, 'r') as f:
        data = json.load(f)

    robs = data['INITIAL_DATA']['r0']
    tobs = data['INITIAL_DATA']['theta0']
    pobs = data['INITIAL_DATA']['phi0']
    l = data['CONSTANTS_OF_MOTION']['lambda']
    q = data['CONSTANTS_OF_MOTION']['q']
    dt = data['INITIAL_DATA']['dt']
    dr = data['INITIAL_DATA']['dr']
    dtheta = data['INITIAL_DATA']['dtheta']
    dphi = data['INITIAL_DATA']['dphi']

    bha = data['EMITTER']['bh_a']

    u1 = data['VELOCITIES']['surf_u1']
    u3 = data['VELOCITIES']['surf_u3']
    v = data['VELOCITIES']['orbit']

    sigma = robs ** 2 + bha ** 2 * np.cos(tobs) ** 2
    delta = robs ** 2 - 2 * robs + bha ** 2
    A = (robs ** 2 + bha ** 2) ** 2 - bha ** 2 * delta * np.sin(tobs) ** 2

    nu = np.sqrt(sigma * delta / A)
    mu1 = np.sqrt(sigma / delta)
    mu2 = np.sqrt(sigma)
    psi = np.sqrt(A / sigma) * np.sin(tobs)
    omega = 2 * robs * bha / A

    alpha = data['OBSERVER']['alpha']
    beta = data['OBSERVER']['beta']

    ft, fr, fth, fphi = util.transform_f(dt, dr, dtheta, dphi, u1, u3, v, nu, mu1, mu2, psi, omega)

    K1 = robs * (dt * fr - dr * ft)
    K2 = - robs ** 3 * (dphi * fth - dtheta * fphi)

    Ex = (beta * K2 + alpha * K1) / np.sqrt((K1 ** 2 + K2 ** 2) * (alpha ** 2 + beta ** 2))
    Ey = (beta * K1 - alpha * K2) / np.sqrt((K1 ** 2 + K2 ** 2) * (alpha ** 2 + beta ** 2))

    print('Thats the angle everyone is gettin')
    print(np.arctan(- Ex / Ey))

    salvation = solver.ODESolverPolAngle(robs, tobs, pobs, dt, dr, dtheta, dphi, bha=bha)

    print('Solving the geod eq...')
    s, res = salvation.solve()
    r = res[:, 2]
    th = res[:, 4]
    phi = res[:, 6]

    x = r * np.cos(phi) * np.sin(th)
    y = r * np.sin(phi) * np.sin(th)
    z = r * np.cos(th)

    x0 = 35 * np.cos(0) * np.sin(1)
    y0 = 35 * np.sin(0) * np.sin(1)
    z0 = 35 * np.cos(1)

    dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    idx = np.where(dist == np.amin(dist))[0][0]

    print('Solving analytically ...')
    res_t, res_r, res_th, res_p, keys = equ.my_fav_fun()#ft, fr, fth, fphi, r, k1, k2, kt, kr, kth, kphi, alpha, beta)

    valus = [35., K1, K2, res[:, 1][idx], res[:, 3][idx], res[:, 5][idx], res[:, 7][idx], -(1 - 2 / 35.), np.sin(1.) ** 2]
    #valus = [r[0], K1, K2, dt, dr, dtheta, dphi, 1 - 2 / r[0], np.sin(th[0]) ** 2]

    ft = res_t.subs({keys[i]: valus[i] for i in range(len(valus))})
    fr = res_r.subs({keys[i]: valus[i] for i in range(len(valus))})
    fth = res_th.subs({keys[i]: valus[i] for i in range(len(valus))})
    fp = res_p.subs({keys[i]: valus[i] for i in range(len(valus))})

    print('Werte für raumzeit f: ')
    print(ft, fr, fth, fp)

    robs = 35.
    tobs = 1.

    sigma = robs ** 2 + bha ** 2 * np.cos(tobs) ** 2
    delta = robs ** 2 - 2 * robs + bha ** 2
    A = (robs ** 2 + bha ** 2) ** 2 - bha ** 2 * delta * np.sin(tobs) ** 2

    nu = np.sqrt(sigma * delta / A)
    mu1 = np.sqrt(sigma / delta)
    mu2 = np.sqrt(sigma)
    psi = np.sqrt(A / sigma) * np.sin(tobs)
    omega = 2 * robs * bha / A

    f0 = nu * N(ft)
    f1 = mu1* N(fr)
    f2 = mu2* N(fth)
    f3 = - omega * nu * N(ft) + psi * N(fp)

    print('Components in the 3-space of the observer')
    print(f1 / f0, f2 / f0, f3 / f0)

    f1 /= f0
    f2 /= f0
    f3 /= f0

    angle = f2 / sqrt(f2 ** 2 + f3 ** 2)

    a = data['OBSERVER']['alpha']
    b = data['OBSERVER']['beta']

    angle = float(angle) #/ np.sqrt(K1 ** 2 + K2 ** 2)
    print('Hopefully the angle: ')
    print( np.arccos(angle))
    #0.508828978649526

    #7.74840479036009 -7.30262502142743 0.0243623536653093 -0.0161501282893279

if __name__ == '__main__':
    main()