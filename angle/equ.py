from sympy import Symbol, solve, print_latex, simplify


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

    heta = Symbol('theta')
    alpha = Symbol('alpha')
    beta = Symbol('beta')

    gtt = Symbol('g_tt')
    grr = Symbol('g_rr')
    gthth = Symbol('g_thetatheta')
    gphiphi = Symbol('g_phiphi')

    gtt = alpha
    grr = 1 / alpha
    gthth = r ** 2
    gpp = r ** 2 * beta


    #term1 = fr + 1 / (r * kt) * k1 + kr / kt * ft
    #term2 = fth - 1 / (r ** 3 * kphi) * k2 + kth / kphi * fphi
    term1 = k1 - r * kt * fr + r * kr * ft
    term2 = k2 + r ** 3 * kphi * fth - r ** 3 * kth * fphi

    term3 = 1 + gtt * ft ** 2 - grr * fr ** 2 - gthth * fth ** 2 - gpp * fphi ** 2
    term4 = - gtt * ft * kt + grr * fr * kr + gthth * fth * kth + gpp * fphi * kphi

    #print('Solving the set of equations ...')
    res = solve([term1, term2, term3, term4], ft, fr, fth, fphi, dict=True)



    #print('Simplifying ...')
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
#print(print_latex(simplify(res[1][fphi].subs({gtt: alpha, grr: 1 / alpha, gthth: r ** 2, gpp: r ** 2 * sin(theta)**2}))))
# g_{phiphi} g_{thetatheta} \left(k^{\phi}\right)^{2} \left(k^{\theta}\right)^{2} r^{3} + g_{phiphi} g_{thetatheta} k^{\phi} k^{\theta} \kappa_{2}    g_{phiphi} k^{\phi} r^{3} \left(g_{phiphi} \left(k^{\phi}\right)^{2} - g_{rr} \left(k^{r}\right)^{2} + g_{tt} \left(k^{t}\right)^{2}\right)
# g_{phiphi} g_{thetatheta} k^{\phi} k^{\theta} \left(k^{\phi} k^{\theta} r^{3} + \kappa_{2}\right)                                                   g_{phiphi} k^{\phi} r^{3} \left(g_{phiphi} \left(k^{\phi}\right)^{2} - g_{rr} \left(k^{r}\right)^{2} + g_{tt} \left(k^{t}\right)^{2}\right)
