from load import prepare
from angle import equ, initial, final, analytical, numerical
from one_lightray import solver, position

import numpy as np
from scipy.optimize import fsolve


def main(data, u1, u3, v, ft=None, fr=None, fth=None, fph=None, keys=None):
    # Step 1: Prepare the data
    data = prepare.prepare_data(data)

    # Step 2:
    if type(None) == type(ft):
        ft, fr, fth, fph, keys = equ.my_fav_fun()

    # Step 3: Get the initial values of the Penrose-Walker constant:
    ft, fr, fth, fph, K1, K2 = initial.compute_f_initially(data['r_em'], data['th_em'], data['bha'],
                                                     data['dt_em'], data['dr_em'], data['dth_em'], data['dph_em'],
                                                     u1, u3, v)
    print(numerical.kerr([ft, fr, fth, fph], data['r_em'], data['th_em'], 0., data['dt_em'], data['dr_em'], data['dth_em'], data['dph_em'], K1, K2))

    # Step 4: Calculate the components of the light ray
    salvation = solver.ODESolverPolAngle(data['r_em'], data['th_em'], data['ph_em'],
                                         data['dt_em'], data['dr_em'], data['dth_em'], data['dph_em'],
                                         bha=data['bha'])

    s, res = salvation.solve()

    # Step 5: Get the light ray components at the observer:
    dt, dr, dth, dph = position.get_values_at_obs(res[:, 2], res[:, 4], res[:, 6],
                                                  res[:, 1], res[:, 3], res[:, 5], res[:, 7])

    # Step 6: Evaluate f at observer:
    values = [35., K1, K2, dt, dr, dth, dph, (1 - 2 / 35.), np.sin(1.) ** 2]

    try:
        ft = float(ft.subs({keys[i]: values[i] for i in range(len(values))}))
        fr = float(fr.subs({keys[i]: values[i] for i in range(len(values))}))
        fth = float(fth.subs({keys[i]: values[i] for i in range(len(values))}))
        fph = float(fph.subs({keys[i]: values[i] for i in range(len(values))}))
    except:
        #ft = float(ft.subs({keys[i]: values[i] for i in range(len(values))}))
        #fr = float(fr.subs({keys[i]: values[i] for i in range(len(values))}))
        #fth = float(fth.subs({keys[i]: values[i] for i in range(len(values))}))
        #fph = float(fph.subs({keys[i]: values[i] for i in range(len(values))}))
        return 0.

    re = fsolve(numerical.kerr, [ft, fr, fth, fph], args=(35., 1., 0.0, dt, dr, dth, dph, K1, K2))

    # Step 7: Calculate polarization angle at observer:
    pol_angle = final.calculate_pol_angle(ft, fr, fth, fph, 35., 1., data['bha'], data['alpha'], data['beta'])
    print(re[0], re[1], re[2], re[3])
    print(ft, fr, fth, fph)
    pol_angle = final.calculate_pol_angle(re[0], re[1], re[2], re[3], 35., 1., data['bha'], data['alpha'], data['beta'])
    print(pol_angle)
    #a = data['alpha']
    #b = data['beta']
    #pol_angle = np.arctan(-(b * K2 - a * K1)/(b * K1 + a * K2))
    #print(pol_angle)

    return pol_angle


if __name__ == '__main__':
    fp = ['/home/jan-menno/Data/Schwarzschild/bigger_sample_3/0.0/data/0.0_0.00405671822081923_-5.30838921370405.json',
          'Z:/Data/v/0.0/data/-0.0_0.004194273846645237_-5.310068737755823.json',
          'Z:/Data/s_015/0.0/data/-0.0015_0.004194273846645237_-5.310068737755823.json']

    for f in fp:
        print('----------------------------')
        print('Starting run for ')
        print('{}\n'.format(f))
        main(f)
        print('----------------------------')
