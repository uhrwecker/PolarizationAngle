import numpy as np

def new_angles(robs, dr, dth, dph, ft, fr, fth, fphi, mu1, mu2, psi, epsilon):
    zeta = - 1 / np.sqrt(1 - 2 / robs)
    cos_theta = - mu1 * (dr + epsilon * fr) / zeta
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    cos_phi = mu2 * (dth + epsilon * fth) / np.sqrt(zeta ** 2 - (dr + epsilon * fr) ** 2)
    sin_phi = - psi * (dph + epsilon * fphi) / np.sqrt(zeta ** 2 - (dr + epsilon * fr) ** 2)

    xp = - 2 * (1 - cos_theta) / sin_theta * sin_phi
    yp = 2 * (1 - cos_theta) / sin_theta * cos_phi

    return xp, yp

def calculate_pol_angle(ft, fr, fth, fphi, robs, tobs, bha, alpha, beta, dt, dr, dth, dph):
    sigma = robs ** 2 + bha ** 2 * np.cos(tobs) ** 2
    delta = robs ** 2 - 2 * robs + bha ** 2
    A = (robs ** 2 + bha ** 2) ** 2 - bha ** 2 * delta * np.sin(tobs) ** 2

    nu = np.sqrt(sigma * delta / A)
    mu1 = np.sqrt(sigma / delta)
    mu2 = np.sqrt(sigma)
    psi = np.sqrt(A / sigma) * np.sin(tobs)
    omega = 2 * robs * bha / A

    x0, y0 = new_angles(robs, dr, dth, dph, ft, fr, fth, fphi, mu1, mu2, psi, 0)
    xp, yp = new_angles(robs, dr, dth, dph, ft, fr, fth, fphi, mu1, mu2, psi, 0.01)

    x = np.abs(x0 - xp)
    if xp < x0:
        x *= -1
    y = np.abs(y0 - yp)
    if yp < y0:
        y *= -1
    angle = -np.arctan2(y, x)
    return angle
