from load import prepare
from angle import equ, initial, final, analytical, numerical
from one_lightray import solver, position

import numpy as np
from scipy.optimize import fsolve


def main(data, u1, u3, v, ft=None, fr=None, fth=None, fph=None, keys=None):
    # Step 1: Prepare the data
    data = prepare.prepare_data(data)

    # Step 3: Get the initial values of the Penrose-Walker constant:
    ft, fr, fth, fph, K1, K2 = initial.compute_f_initially(data['r_em'], data['th_em'], data['bha'],
                                                     data['dt_em'], data['dr_em'], data['dth_em'], data['dph_em'],
                                                     u1, u3, v)

    # Step 4: Calculate the components of the light ray
    salvation = solver.ODESolverPolAngle(data['r_em'], data['th_em'], data['ph_em'],
                                         data['dt_em'], data['dr_em'], data['dth_em'], data['dph_em'],
                                         bha=data['bha'])

    s, res = salvation.solve(ft, fr, fth, fph)

    # Step 5: Get the light ray components at the observer:
    dt, dr, dth, dph, ft, fr, fth, fph = position.get_values_at_obs(res[:, 2], res[:, 4], res[:, 6],
                                                  res[:, 1], res[:, 3], res[:, 5], res[:, 7],
                                                  res[:, 8], res[:, 9], res[:, 10], res[:, 11])

    K1 = 35 * (dt * fr - dr * ft)
    K2 = - 35 ** 3 * np.sin(1) * (dph * fth - dth * fph)
    himwich_x = (data['beta'] * K2 + data['alpha'] * K1) / (np.sqrt((K1 ** 2 + K2 ** 2) * (data['beta'] ** 2 + data['alpha']**2)))
    himwich_y = -(data['beta'] * K1 - data['alpha'] * K2) / (np.sqrt((K1 ** 2 + K2 ** 2) * (data['beta'] ** 2 + data['alpha']**2)))

    himwich = np.arccos(himwich_y)
    if himwich_x < 0:
        himwich = 2 * np.pi - himwich

    return final.calculate_pol_angle(ft, fr, fth, fph, 35., 1., data['bha'], data['alpha'], data['beta'],dt, dr, dth, dph), himwich, ft, fr, fth, fph



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
