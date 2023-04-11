import numpy as np


def geod(w, t, m=1, a=0):
    """
        Defines the differential equations for lightlike geodesics in Schwarzschild background.
        :param w: iterable; vector of the state variables
                  [t, t', r, r', theta, theta', phi, phi']
        :param t: float; time parameter, deprecated here.
        :return: iterable; vector of differntiated variables
                  [t', t'', r', r'', theta', theta'', phi', phi'']
    """

    t, td, r, rd, th, thd, phi, phid, ft, fr, fth, fph = w

    # for convenience:
    alpha = 1 - 2 * m / r

    f = [td,
         - 2 * m * td * rd / (r ** 2 * alpha),
         rd,
         -m * td ** 2 / r ** 2 * alpha + m / r ** 2 * rd ** 2 / alpha + r * alpha * thd ** 2 + alpha * r * np.sin(
             th) ** 2 * phid**2,
         thd,
         -2 / r * rd * thd + np.cos(th) * np.sin(th) * phid ** 2,
         phid,
         - 2 / r * rd * phid - 2 / np.tan(th) * thd * phid,
         - 1 / (r ** 2 - 2 * r) * (td * fr + rd * ft),
         - (r - 2) / r ** 3 * td * ft + 1 / (r ** 2 - 2 * r) * rd * fr + (r - 2) * thd * fth + (r - 2) * np.sin(th) ** 2 * phid * fph,
         - 1 / r * (rd * fth + thd * fr) + 1 * np.sin(th) * np.cos(th) * phid * fph,
         - 1 / r * (rd * fph + phid * fr) - 1 / np.tan(th) * (thd * fph + phid * fth)
         ]

    return f
