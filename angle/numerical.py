import numpy as np


def kerr(f, r, theta, a, kt, kr, kth, kp, kappa1, kappa2):
    ft, fr, fth, fp = f

    a = 0
    sigma = r ** 2 + a ** 2 * np.cos(theta) ** 2
    delta = r ** 2 - 2 * r + a ** 2
    big_a = (r ** 2 + a ** 2) ** 2 - delta * a ** 2 * np.sin(theta) ** 2
    omega = 2 * a * r / big_a

    gtt = sigma * delta / big_a - omega ** 2 * np.sin(theta) ** 2 * big_a / sigma
    grr = sigma / delta
    gthth = sigma
    gpp = np.sin(theta) ** 2 * big_a / sigma
    gtp = 0#- omega * np.sin(theta) ** 2 * big_a / sigma

    A = kt * fr - kr * ft + a * np.sin(theta) ** 2 * (kr * fp - kp * fr)
    B = np.sin(theta) * ((r**2 + a**2) * (kp * fth - kth * fp) - a * (kt * fth - kth * ft))

    term1 = 0 - gtt * kt * ft + gtp * (ft * kp + kt * fp) + grr * fr * kr + gthth * fth * kth + gpp * fp * kp
    term2 = -1 - gtt * ft ** 2 + 2 * gtp * ft * fp + grr * fr ** 2 + gthth * fth ** 2 + gpp * fp ** 2
    term3 = -kappa1 + r * A - a * np.cos(theta) * B
    term4 = -kappa2 - r * B - a * np.cos(theta) * B

    return term1, term2, term3, term4