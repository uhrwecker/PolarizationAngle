import numpy as np


def calculate_pol_angle(ft, fr, fth, fphi, robs, tobs, bha):
    sigma = robs ** 2 + bha ** 2 * np.cos(tobs) ** 2
    delta = robs ** 2 - 2 * robs + bha ** 2
    A = (robs ** 2 + bha ** 2) ** 2 - bha ** 2 * delta * np.sin(tobs) ** 2

    nu = np.sqrt(sigma * delta / A)
    mu1 = np.sqrt(sigma / delta)
    mu2 = np.sqrt(sigma)
    psi = np.sqrt(A / sigma) * np.sin(tobs)
    omega = 2 * robs * bha / A

    f0 = nu * ft
    f1 = mu1 * fr
    f2 = mu2 * fth
    f3 = - omega * nu * ft + psi * fphi

    #print('Components in the 3-space of the observer')
    #print(f1 / f0, f2 / f0, f3 / f0)

    f1 /= f0
    f2 /= f0
    f3 /= f0

    angle = f2 / np.sqrt(f2 ** 2 + f3 ** 2)

    return angle
