import numpy as np


def f_phi(r, theta, kt, kr, kth, kp, kappa1, kappa2):
    alpha = 1 - 2 / r
    beta = np.sin(theta) ** 2

    th_nom = - kth * kappa2 * (- alpha ** 2 * kt ** 2 + alpha * beta * kp ** 2 * r ** 2 + kr ** 2)

    root = - alpha ** 4 * beta * kp ** 2 * kt ** 4 * r ** 4 + \
           alpha ** 4 * beta * kt ** 4 * kappa2 ** 2 + \
           alpha ** 4 * kt ** 4 * kth ** 2 * r ** 4 - \
           alpha ** 3 * beta ** 2 * kp ** 4 * kt ** 2 * r ** 6 + \
        alpha ** 3 * beta ** 2 * kp ** 2 * kt ** 2 * kappa2 ** 2 * r ** 2 - \
        2 * alpha ** 3 * beta * kp ** 2 * kt ** 2 * kth ** 2 * r ** 6 + \
        alpha ** 3 * beta * kp ** 2
