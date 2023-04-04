import numpy as np


def schwarzschild(r, theta, a, kt, kr, kth, kp, kappa1, kappa2):
    a = 0
    gtt = - (1 - 2 / r)
    grr = 1 / (1 - 2 / r)
    gthth = r ** 2
    gpp = r ** 2 * np.sin(theta) ** 2

    C1 = r * kt
    C2 = r * kr
    C3 = r ** 3 * kth
    C4 = r ** 3 * kp

    C5 = gtt * kt
    C6 = grr * kr
    C7 = gthth * kth
    C8 = gpp * kp

    C9 = C6 / C1 * kappa1 + C8 / C3 * kappa2
    C10 = C7 + C8 * C4 / C3
    C11 = - (C5 + C6 * C2 / C1)

    C12 = kappa1 + C2 * C9 / C11
    C13 = C2 * C10 / C11

    C14 = gtt * C9 ** 2 / C11 ** 2
    C15 = 2 * gtt * C9 * C10 / C11 ** 2
    C16 = gtt * C10 ** 2 / C11 ** 2
    C17 = grr * C12 ** 2 / C1 ** 2
    C18 = 2 * grr * C12 * C13 / C1 ** 2
    C19 = grr * C13 ** 2 / C1 ** 2
    C20 = gpp * kappa2 ** 2 / C3 ** 2
    C21 = 2 * gpp * kappa2 * C4 / C3 ** 2
    C22 = gpp * C4 ** 2 / C3 ** 2

    C23 = (C14 + C17 + C20) / (C16 + C19 + gthth + C22) - 1
    C24 = (C15 + C18 + C21) / (C16 + C19 + gthth + C22)

    fth1 = - C24 / 2 + np.sqrt(C24 ** 2 / 4 - C23)
    fth2 = - C24 / 2 - np.sqrt(C24 ** 2 / 4 - C23)

    ft1 = C9 / C11 + C10 / C11 * fth1
    ft2 = C9 / C11 + C10 / C11 * fth2

    fr1 = C12 / C1 + C13 / C1 * fth1
    fr2 = C12 / C1 + C13 / C1 * fth2

    fp1 = kappa2 / C3 + C4 / C3 * fth1
    fp2 = kappa2 / C3 + C4 / C3 * fth2

    return (ft1, fr1, fth1, fp1), (ft2, fr2, fth2, fp2)

def constants(r, theta, a, kt, kr, kth, kp, kappa1, kappa2):
    a = 0
    sigma = r ** 2 + a ** 2 * np.cos(theta) ** 2
    delta = r ** 2 - 2 * r + a ** 2
    big_a = (r ** 2 + a ** 2) ** 2 - delta * a ** 2 * np.sin(theta) ** 2
    omega = 2 * a * r / big_a

    #gtt = - sigma * delta / big_a + omega ** 2 * np.sin(theta) ** 2 * big_a / sigma
    #grr = sigma / delta
    #gthth = sigma
    #gpp = np.sin(theta) ** 2 * big_a / sigma
    #gtp = - 2 * omega * np.sin(theta) ** 2 * big_a / sigma

    gtt = -(1 - 2 / r)
    grr = -1 / gtt
    gthth = r ** 2
    gpp = r ** 2 * np.sin(theta) ** 2
    gtp = 0

    C1 = gtt * kt + gtp * kp
    C2 = grr * kr
    C3 = gthth * kth
    C4 = gpp * kp + gtp * kt

    C5 = gtt
    C6 = 2 * gtp
    C7 = grr
    C8 = gthth
    C9 = gpp

    C10 = - r * kr - a ** 2 * np.cos(theta) * np.sin(theta) * kth
    C11 = r * kt - r * a * np.sin(theta) ** 2 * kp
    C12 = - a * np.cos(theta) * np.sin(theta) * ((r ** 2 + a ** 2) * kp - a * kt)
    C13 = r * a * np.sin(theta) ** 2 * kr + (r ** 2 + a ** 2) * a * np.cos(theta) * np.sin(theta) * kth
    C14 = a * np.cos(theta) * kr - r * a * np.sin(theta) * kth
    C15 = - a * np.cos(theta) * kt + a ** 2 * np.cos(theta) * np.sin(theta) ** 2 * kp
    C16 = - (r ** 2 + a ** 2) * r * np.sin(theta) * kp + r * a * np.sin(theta) * kt
    C17 = - a ** 2 * np.cos(theta) * np.sin(theta) ** 2 * kr + (r ** 2 + a ** 2) * r * np.sin(theta) * kth

    C18 = - C2 / C1
    C19 = - C3 / C1
    C20 = - C4 / C1

    C21 = kappa1 / (C10 * C18 + C11)
    C22 = - (C10 * C19 + C12) / (C10 * C18 + C11)
    C23 = - (C10 * C20 + C13) / (C10 * C18 + C11)

    C24 = C18 * C21
    C25 = C18 * C22 + C19
    C26 = C18 * C23 + C20

    C27 = (kappa2 - C14 * C24 - C15 * C21) / (C14 * C26 + C15 * C23 + C17)
    C28 = - (C14 * C25 + C15 * C22 + C16) / (C14 * C26 + C15 * C23 + C17)

    C29 = C21 + C23 * C27
    C30 = C22 + C23 * C28
    C31 = C24 + C26 * C27
    C32 = C25 + C26 * C28

    C33 = C5 * C32 ** 2 + C6 * C30 * C32 + C7 * C30 ** 2 + C8 + C9 * C28 ** 2
    C34 = 2 * C5 * C31 * C32 + C6 * (C27 * C32 + C31 * C28) + 2 * C7 * C29 * C30 + 2 * C9 * C27 * C28
    C35 = C5 * C31 ** 2 + C6 * C31 * C27 + C7 * C29 + C9 * C27 ** 2 - 1
    C36 = - C34 ** 2 / (2 * C33)
    C37 = C36 ** 2 - C35 / C33
    C38 = C32 * C36 + C31
    C39 = C32 ** 2 * C37
    C40 = C30 * C36 + C29
    C41 = C30 ** 2 * C37
    C42 = C28 * C36 + C27
    C43 = C28 ** 2 * C37

    return (C38 + np.sqrt(C39), C40 + np.sqrt(C41), C36 + np.sqrt(C37), C42 + np.sqrt(C43)), \
           (C38 - np.sqrt(C39), C40 - np.sqrt(C41), C36 - np.sqrt(C37), C42 - np.sqrt(C43))
