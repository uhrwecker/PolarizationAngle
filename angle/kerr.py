from sympy import Symbol, solve, print_latex, simplify, sin, cos


def my_fav_fun():#ft, fr, fth, fphi, r, k1, k2, kt, kr, kth, kphi, alpha, beta):
    ft = Symbol('f^t')
    fr = Symbol('f^r')
    fth = Symbol('f^theta')
    fphi = Symbol('f^phi')
    r = Symbol('r')

    k1 = Symbol('kappa_1')
    k2 = Symbol('kappa_2')

    kt = Symbol('k^t')
    kr = Symbol('k^r')
    kth = Symbol('k^theta')
    kphi = Symbol('k^phi')

    a = Symbol('a')
    delta = Symbol('delta')
    theta = Symbol('theta')
    alpha = Symbol('alpha')
    beta = sin(theta) ** 2
    rho = r ** 2 + a ** 2 * (1 - beta)

    gtt = 1 - 2 * r / rho ** 2
    gtph = - 2 * a * r * beta / rho ** 2     # only one g_tphi = g_phit
    grr = rho ** 2 / delta
    gthth = rho ** 2
    gpp = beta / rho ** 2 * ((r ** 2 + a ** 2) ** 2 - a ** 2 * delta * beta)

    t1_t = - r * kr - a ** 2 * cos(theta) * sin(theta) * kth
    t1_r = r * kt - r * a * sin(theta) ** 2 * kphi
    t1_th = - a * cos(theta) * sin(theta) * (r ** 2 + a ** 2) * kphi + a ** 2 * cos(theta) * sin(theta) * kt
    t1_ph = r * a * sin(theta) ** 2 * kr + a * cos(theta) * sin(theta) * (r ** 2 + a ** 2) * kth

    term1 = k1 - t1_t * ft - t1_r * fr - t1_th * fth - t1_ph * kphi

    t2_t = a * cos(theta) * kr - r * a * sin(theta) * kth
    t2_r = - a * cos(theta) * kt + a ** 2 * beta * cos(theta) * kphi
    t2_th = - r * sin(theta) * (r ** 2 + a ** 2) * kphi + a * r * sin(theta) * kt
    t2_ph = - a ** 2 * beta * cos(theta) * kr + r * sin(theta) * (r ** 2 + a ** 2) * kth

    term2 = k2 - t2_t * ft - t2_r * fr - t2_th * fth - t2_ph * fphi

    term3 = 1 + gtt * ft ** 2 - 2 * gtph * ft * fphi - grr * fr ** 2 - gthth * fth ** 2 - gpp * fphi ** 2
    term4 = - gtt * ft * kt + gtph * ft * kphi + gtph * fphi * kt + grr * fr * kr + gthth * fth * kth + gpp * fphi * kphi

    print('Solving the set of equations ...')
    res = solve([term1, term2, term3, term4], ft, fr, fth, fphi, dict=True)



    print('Simplifying ...')
    #res_phi = simplify(res[0][fphi])#.subs(gthth, r**2))#gtt, alpha))
    #res_phi = simplify(res_phi.subs(grr, 1 / alpha))
    #res_phi = simplify(res_phi.subs(gthth, r ** 2))

    res_t = simplify(res[0][ft])
    res_r = simplify(res[0][fr])
    res_th = simplify(res[0][fth])
    res_p = simplify(res[0][fphi])

    #print('t component:')
    #print_latex(res_t)
    #print('r component')
    #print_latex(res_r)
    #print('theta component')
    #print_latex(res_th)
    #print('phi component')
    #print_latex(res_p)

    return res_t, res_r, res_th, res_p, [r, k1, k2, kt, kr, kth, kphi, alpha, beta]


if __name__ == '__main__':
    res_t, res_r, res_th, res_p, keys = my_fav_fun()

    print('t component:')
    print_latex(res_t)
    print('r component')
    print_latex(res_r)
    print('theta component')
    print_latex(res_th)
    print('phi component')
    print_latex(res_p)