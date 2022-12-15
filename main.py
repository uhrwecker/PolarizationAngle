from load import prepare
from angle import equ, initial, final
from one_lightray import solver, position

import numpy as np


def main(fp, ft=None, fr=None, fth=None, fph=None, keys=None):
    # Step 1: Prepare the data
    print('(1) Preparing data ...')
    data = prepare.prepare_data(fp)

    # Step 2:
    print('(2) Calculating f analytically ...')
    if type(None) == type(ft):
        ft, fr, fth, fph, keys = equ.my_fav_fun()

    # Step 3: Get the initial values of the Penrose-Walker constant:
    print('(3) Calculating the Penrose Walker constant ...')
    _, _, _, _, K1, K2 = initial.compute_f_initially(data['r_em'], data['th_em'], data['bha'],
                                                     data['dt_em'], data['dr_em'], data['dth_em'], data['dph_em'],
                                                     data['u1'], data['u3'], data['v'])
    print(' -  K1: {}'.format(K1))
    print(' -  K2: {}'.format(K2))

    # Step 4: Calculate the components of the light ray
    print('(4) Solving the geod. eq. ...')
    salvation = solver.ODESolverPolAngle(data['r_em'], data['th_em'], data['ph_em'],
                                         data['dt_em'], data['dr_em'], data['dth_em'], data['dph_em'],
                                         bha=data['bha'])

    s, res = salvation.solve()

    # Step 5: Get the light ray components at the observer:
    print('(5) Get the velocities of the light ray at the observer ...')
    dt, dr, dth, dph = position.get_values_at_obs(res[:, 2], res[:, 4], res[:, 6],
                                                  res[:, 1], res[:, 3], res[:, 5], res[:, 7])
    print(' -  dt  : {}'.format(dt))
    print(' -  dr  : {}'.format(dr))
    print(' -  dth : {}'.format(dth))
    print(' -  dph : {}'.format(dph))
    
    # Step 6: Evaluate f at observer:
    print('(6) Get f at the observer ...')
    values = [35., K1, K2, dt, dr, dth, dph, -(1 - 2 / 35.), np.sin(1.) ** 2]
    
    ft = float(ft.subs({keys[i]: values[i] for i in range(len(values))}))
    fr = float(fr.subs({keys[i]: values[i] for i in range(len(values))}))
    fth = float(fth.subs({keys[i]: values[i] for i in range(len(values))}))
    fph = float(fph.subs({keys[i]: values[i] for i in range(len(values))}))

    print(' -  ft  : {}'.format(ft))
    print(' -  fr  : {}'.format(fr))
    print(' -  fth : {}'.format(fth))
    print(' -  fph : {}'.format(fph))

    # Step 7: Calculate polarization angle at observer:
    print('(7) Calculate the polarization angle ...')
    pol_angle = final.calculate_pol_angle(ft, fr, fth, fph, 35., 1., data['bha'])
    print(' -  psi     : {}'.format(pol_angle))

    return pol_angle


if __name__ == '__main__':
    fp = ['Z:/Data/v0/0.0/data/-0.0_0.004194273846645237_-5.310068737755823.json',
          'Z:/Data/v/0.0/data/-0.0_0.004194273846645237_-5.310068737755823.json',
          'Z:/Data/s_015/0.0/data/-0.0015_0.004194273846645237_-5.310068737755823.json']

    for f in fp:
        print('----------------------------')
        print('Starting run for ')
        print('{}\n'.format(f))
        main(f)
        print('----------------------------')
