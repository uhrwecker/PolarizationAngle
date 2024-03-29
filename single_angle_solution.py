from load import prepare
from angle import equ, initial, final, analytical, numerical
from one_lightray import solver, position

import numpy as np
from scipy.optimize import fsolve


def main(data, u1, u3, v, ft=None, fr=None, fth=None, fph=None, keys=None):
    # Step 1: Prepare the data
    data = prepare.prepare_data(data)

    # Step 2:
    #if type(None) == type(ft):
    #    ft3, fr3, fth3, fph3, keys = equ.my_fav_fun()

    # Step 3: Get the initial values of the Penrose-Walker constant:
    ft, fr, fth, fph, K1, K2 = initial.compute_f_initially(data['r_em'], data['th_em'], data['bha'],
                                                     data['dt_em'], data['dr_em'], data['dth_em'], data['dph_em'],
                                                     u1, u3, v)
    #print(ft, fr, fth, fph)
    #reh = fsolve(numerical.kerr, [ft, fr, fth, fph], args=(data['r_em'], data['th_em'], data['bha'],
    #                                                  data['dt_em'], data['dr_em'], data['dth_em'], data['dph_em'], K1,
    #                                                  K2))
    #print(reh)
    #print(numerical.kerr([ft, fr, fth, fph], data['r_em'], data['th_em'], data['bha'],
    #                                                  data['dt_em'], data['dr_em'], data['dth_em'], data['dph_em'], K1,
    #                                                  K2))
    #print(data['alpha'], data['beta'])
    # Step 4: Calculate the components of the light ray
    salvation = solver.ODESolverPolAngle(data['r_em'], data['th_em'], data['ph_em'],
                                         data['dt_em'], data['dr_em'], data['dth_em'], data['dph_em'],
                                         bha=data['bha'])

    s, res = salvation.solve(ft, fr, fth, fph)

    #eval_1 = 1000
    #print(f'at {res[:, 2][eval_1]}')
    #print(- (1 - 2 / res[:, 2][eval_1]) * res[:, 8][eval_1] ** 2
    #      + 1 / (1 - 2 / res[:, 2][eval_1]) * res[:, 9][eval_1] ** 2
    #      + res[:, 2][eval_1] ** 2 * res[:, 10][eval_1] ** 2 +
    #      res[:, 2][eval_1] ** 2 * np.sin(res[:, 4][eval_1]) ** 2 * res[:, 11][eval_1] ** 2)
    #print(- (1 - 2 / res[:, 2][eval_1]) * res[:, 8][eval_1] * res[:, 1][eval_1]
    #      + 1 / (1 - 2 / res[:, 2][eval_1]) * res[:, 9][eval_1] * res[:, 3][eval_1]
    #      + res[:, 2][eval_1] ** 2 * res[:, 10][eval_1] * res[:, 5][eval_1] +
    #      res[:, 2][eval_1] ** 2 * np.sin(res[:, 4][eval_1]) ** 2 * res[:, 11][eval_1] * res[:, 7][eval_1])


    # Step 5: Get the light ray components at the observer:
    dt, dr, dth, dph, ft, fr, fth, fph = position.get_values_at_obs(res[:, 2], res[:, 4], res[:, 6],
                                                  res[:, 1], res[:, 3], res[:, 5], res[:, 7],
                                                  res[:, 8], res[:, 9], res[:, 10], res[:, 11])
    #print(- (1 - 2 / 35) * ft ** 2 + 1 / (1 - 2 / 35) * fr ** 2 + 35 ** 2 * fth ** 2 + 35 ** 2 * np.sin(1) ** 2 * fph ** 2)
    #print(- (1 - 2 / 35) * ft * dt + 1 / (1 - 2 / 35) * fr * dr + 35 ** 2 * fth * dth + 35 ** 2 * np.sin(1) ** 2 * fph * dph)
    #print(ft, fr, fth, fph)
    #print('HMMMM')
    K1 = 35 * (dt * fr - dr * ft)
    K2 = - 35 ** 3 * np.sin(1) * (dph * fth - dth * fph)
    himwich_x = (data['beta'] * K2 + data['alpha'] * K1) / (np.sqrt((K1 ** 2 + K2 ** 2) * (data['beta'] ** 2 + data['alpha']**2)))
    himwich_y = -(data['beta'] * K1 - data['alpha'] * K2) / (np.sqrt((K1 ** 2 + K2 ** 2) * (data['beta'] ** 2 + data['alpha']**2)))

    himwich = np.arccos(himwich_y)
    if himwich_x < 0:
        himwich = 2 * np.pi - himwich

   # print(numerical.kerr([ft, fr, fth, fph], 35., 1., 0., dt, dr, dth, dph, K1, K2))
    #print(final.calculate_pol_angle(ft, fr, fth, fph, 35., 1., data['bha'], data['alpha'], data['beta'], dt, dr, dth, dph))
    #re = fsolve(numerical.kerr, [ft, fr, fth, fph], args=(35., 1., 0.0, dt, dr, dth, dph, K1, K2))
    #print(re)
    #print(numerical.kerr(re, 35., 1., 0., dt, dr, dth, dph, K1, K2))
        #np.arctan(-(data['beta'] * K2 - data['alpha'] * K1) / (data['beta'] * K1 + data['alpha'] * K2) )
    return final.calculate_pol_angle(ft, fr, fth, fph, 35., 1., data['bha'], data['alpha'], data['beta'],dt, dr, dth, dph), himwich, ft, fr, fth, fph
    #print(final.calculate_pol_angle(ft, fr, fth, fph, 35., 1., data['bha'], data['alpha'], data['beta']))


    # Step 6: Evaluate f at observer:
    values = [35., K1, K2, dt, dr, dth, dph, (1 - 2 / 35.), np.sin(1.) ** 2]

    #try:
    #    ft = float(ft3.subs({keys[i]: values[i] for i in range(len(values))}))
    #    fr = float(fr3.subs({keys[i]: values[i] for i in range(len(values))}))
    #    fth = float(fth3.subs({keys[i]: values[i] for i in range(len(values))}))
    #    fph = float(fph3.subs({keys[i]: values[i] for i in range(len(values))}))
    #except:
    #    ft = float(ft3.subs({keys[i]: values[i] for i in range(len(values))}))
    #    fr = float(fr3.subs({keys[i]: values[i] for i in range(len(values))}))
    #    fth = float(fth3.subs({keys[i]: values[i] for i in range(len(values))}))
    #    fph = float(fph3.subs({keys[i]: values[i] for i in range(len(values))}))
    #    return 0.
    #print(ft, fr, fth, fph)

    #re = fsolve(numerical.kerr, [-1, 1, 1, -1], args=(35., 1., 0.0, dt, dr, dth, dph, K1, K2))
    #print(re)
    #print(numerical.kerr(re, 35., 1., 0., dt, dr, dth, dph, K1, K2))
    #print(numerical.kerr([ft, fr, fth, fph], 35., 1., 0., dt, dr, dth, dph, K1, K2))
    #f1, f2 = analytical.constants(35., 1., 0., dt, dr, dth, dph, K1, K2)
    #f1, f2 = analytical.schwarzschild(35., 1., 0., dt, dr, dth, dph, K1, K2)

    #print('analytical 1: ')
    #print(final.calculate_pol_angle(f1[0], f1[1], f1[2], f1[3], 35., 1., data['bha'], data['alpha'], data['beta']))
    #print(final.calculate_pol_angle(f2[0], f2[1], f2[2], f2[3], 35., 1., data['bha'], data['alpha'], data['beta']))
    #print(numerical.kerr(f1, 35., 1., 0., dt, dr, dth, dph, K1, K2))
    #print(numerical.kerr(f2, 35., 1., 0., dt, dr, dth, dph, K1, K2))
    #pol_angle1 = final.calculate_pol_angle(f1[0], f1[1], f1[2], f1[3], 35., 1., data['bha'], data['alpha'], data['beta'])
    #pol_angle2 = final.calculate_pol_angle(f2[0], f2[1], f2[2], f2[3], 35., 1., data['bha'], data['alpha'], data['beta'])

    # Step 7: Calculate polarization angle at observer:
    #print('semi-analytical: ')
    #pol_angle = final.calculate_pol_angle(ft, fr, fth, fph, 35., 1., data['bha'], data['alpha'], data['beta'])
    #print(re[0], re[1], re[2], re[3])
    #print(ft, fr, fth, fph)
    #print(pol_angle)
    pol_angle = final.calculate_pol_angle(re[0], re[1], re[2], re[3], 35., 1., data['bha'], data['alpha'], data['beta'])
    #print('Numerically: ')
    print(pol_angle)
    #a = data['alpha']
    #b = data['beta']
    #pol_angle = np.arctan(-(b * K2 - a * K1)/(b * K1 + a * K2))
    #print(pol_angle)

    return pol_angle, himwich, re[0], re[1], re[2], re[3]#pol_angle1, pol_angle2, (ft, fr, fth, fph)


if __name__ == '__main__':
    import json
    #fp = ["E:/Schwarzschild/higher_resolution/redshift_dist_3pi-2_sphere/s0/4.700020505370556/data/-0.0_-9.274632265003175_0.06328533788104738.json",
    #      "E:/Schwarzschild/higher_resolution/redshift_dist_3pi-2_sphere/s0/4.700020505370556/data/-0.0_-9.274632265003175_0.06320562215004391.json",
    #      "E:/Schwarzschild/higher_resolution/redshift_dist_3pi-2_sphere/s0/4.700020505370556/data/-0.0_-9.274632265003175_0.06324548001554564.json"]
    fp = ["E:/Schwarzschild/higher_resolution/redshift_dist_pi_sphere/s0/3.141592653589793/data/0.0_0.003216327538174798_7.899462050150876.json",
          "E:/Schwarzschild/higher_resolution/redshift_dist_pi_sphere/s0/3.141592653589793/data/0.0_0.003216327538174798_7.899542532379293.json",
          "E:/Schwarzschild/higher_resolution/redshift_dist_pi_sphere/s0/3.141592653589793/data/0.0_0.003296809766591781_7.899462050150876.json",
          "E:/Schwarzschild/higher_resolution/redshift_dist_pi_sphere/s0/3.141592653589793/data/0.0_0.003296809766591781_7.899542532379293.json",
          "E:/Schwarzschild/higher_resolution/redshift_dist_pi_sphere/s0/3.141592653589793/data/0.0_-0.0033027329636008598_7.902117963688637.json"]
    #0.0_0.003216327538174798_7.899462050150876
    #0.0_0.003216327538174798_7.899542532379293
    #0.0_0.003135845309757815_7.899542532379293
    #0.0_0.003135845309757815_7.899462050150876
    s = 0.00175

    for f in fp:
        with open(f, 'r') as ff:
            config = json.load(ff)
        try:
            rho = config['EMITTER']['rho']
        except:
            rho = config['EMITTER']['a']
        T = config['EMITTER']['Theta']
        P = config['EMITTER']['Phi']

        u1 = 5 * s / (2 * rho) * np.sin(P) * np.sin(T)
        u3 = 5 * s / (2 * rho) * np.cos(P) * np.sin(T)

        v = config['VELOCITIES']['orbit']
        print('----------------------------')
        print('Starting run for ')
        print('{}\n'.format(f))
        main(config, u1, u3, v)
        print('----------------------------')
