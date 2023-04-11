import numpy as np
from angle import util, util2


def compute_f_initially(rem, them, bha, dt, dr, dtheta, dphi, u1, u3, v):
    sigma = rem ** 2 + bha ** 2 * np.cos(them) ** 2
    delta = rem ** 2 - 2 * rem + bha ** 2
    A = (rem ** 2 + bha ** 2) ** 2 - bha ** 2 * delta * np.sin(them) ** 2

    nu = np.sqrt(sigma * delta / A)
    mu1 = np.sqrt(sigma / delta)
    mu2 = np.sqrt(sigma)
    psi = np.sqrt(A / sigma) * np.sin(them)
    omega = 2 * rem * bha / A

    ft, fr, fth, fphi = util2.main3(np.array([dt, dr, dtheta, dphi]), u1, u3, v, nu, mu1, mu2, psi, omega)

    Al = dt * fr - dr * ft + bha * np.sin(them) ** 2 * (dr * fphi - dphi * fr)
    Be = np.sin(them) * ((rem ** 2 + bha ** 2) * (dphi * fth - dtheta * fphi) - bha * (dt * fth - dtheta * ft))

    K1 = rem * Al - bha * np.cos(them) * Be
    K2 = - rem * Be - bha * np.cos(them) * Al

    return ft, fr, fth, fphi, K1, K2