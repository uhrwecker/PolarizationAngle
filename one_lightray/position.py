import numpy as np


def get_values_at_obs(r, th, ph, dt, dr, dth, dph, ft, fr, fth, fph):
    # robs = 35, tobs = 1., pobs = 0.
    x = r * np.cos(ph) * np.sin(th)
    y = r * np.sin(ph) * np.sin(th)
    z = r * np.cos(th)

    x0 = 35 * np.cos(0) * np.sin(1)
    y0 = 35 * np.sin(0) * np.sin(1)
    z0 = 35 * np.cos(1)

    dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    idx = np.where(dist == np.amin(dist))[0][0]

    return dt[idx], dr[idx], dth[idx], dph[idx], ft[idx], fr[idx], fth[idx], fph[idx]