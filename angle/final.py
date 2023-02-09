import numpy as np


def calculate_pol_angle(ft, fr, fth, fphi, robs, tobs, bha, alpha, beta):
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
    f3 = -omega * nu * ft + psi * fphi

    #print('Components in the 3-space of the observer')
    #print(f1 / f0, f2 / f0, f3 / f0)

    f1 /= f0
    f2 /= f0
    f3 /= f0

    factor_a = 1
    factor_b = 1

    #if beta > 0:
    #    factor_b *= -1
    alpha2 = alpha + factor_a * f3 / f1
    beta2 = beta + factor_b * f2 / f1

    #angle = np.arctan2(1, 0) - np.arctan2(f2 / np.sqrt((f2 ** 2 + f3 ** 2) * (alpha ** 2 + beta ** 2)),
    #                                      f3 / np.sqrt((f2 ** 2 + f3 ** 2) * (alpha ** 2 + beta ** 2)))
    #angle = np.arctan2(factor * f3 / f1 - 0 * factor * f2 / f1,
    #                   0 * factor * f3 / f1 + factor * f2 / f1)
    m = (beta - beta2) / (alpha - alpha2)
    x = np.abs(alpha - alpha2)
    if alpha2 < alpha:
        x *= -1
    y = np.abs(beta - beta2)
    if beta2 < beta:
        y *= -1
    angle = -np.arctan2(y, x)
    #if alpha2 < alpha:
    #    angle = np.arctan2(-m, -1)
    #print(alpha, alpha2, beta, beta2, m, angle+2*np.pi, angle / np.pi * 180)
    #small_angle = np.arccos(m / (np.sqrt(1 + m ** 2)))

    #if m >= 0:
    #    big_angle = np.pi - small_angle
    #    if big_angle < 0:
    #        big_angle += 2 * np.pi
    #    if alpha2 > alpha:
    #        angle = big_angle + np.pi
    #    else:
    #        angle = big_angle
    #else:
    #    if alpha2 > alpha:
    #        angle = small_angle + np.pi
    #    else:
    #        angle = small_angle

    if angle < 0:
        angle += 2 * np.pi

    return angle
