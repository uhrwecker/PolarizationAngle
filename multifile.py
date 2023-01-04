import numpy as np
import os
import pandas as pd
import json
import single_angle_solution as sas
from angle import equ


def evaluate(fp_to_redshift, fp_to_save, ft, fr, fth, fphi, keys, s):
    # step 0: reformalization
    fp_to_json = fp_to_redshift + 'data/'

    # step 1: get a list of every json file within
    filenames = next(os.walk(fp_to_json), (None, None, []))[2]

    # step 2: load .csv of redshift
    redshift_data = np.loadtxt(fp_to_redshift + 'redshift.csv', delimiter=',', skiprows=1)

    result = []

    # step 3: iterate over redshift data
    for n, row in enumerate(redshift_data):
        # step 3a: exclude data when there is no hit:
        if row[-1] == 0.:
            continue

        # step 4: load all json files one by one
        for file in filenames:
            with open(fp_to_json + file, 'r') as f:
                config = json.load(f)

            if config['OBSERVER']['alpha'] != row[0] or config['OBSERVER']['beta'] != row[1]:
                del config
                continue

            # here goes the programming stuff

            # the surface velocity:
            try:
                rho = config['EMITTER']['rho']
            except:
                rho = config['EMITTER']['a']
            T = config['EMITTER']['Theta']
            P = config['EMITTER']['Phi']

            u1 = 5 * s / (2 * rho) * np.sin(P) * np.sin(T)
            u3 = 5 * s / (2 * rho) * np.cos(P) * np.sin(T)

            v = config['VELOCITIES']['orbit']

            pa = sas.main(config, u1, u3, v, ft, fr, fth, fphi, keys)

            result.append(pa)

            break

        row[-1] = pa
        redshift_data[n] = row

        if not filenames == []:
            filenames.remove(file)

    np.savetxt(fp_to_save + 'polarization.csv', redshift_data, delimiter=',', header='alpha,beta,pol_angle')

    return result


def main(fp_data, fp_save, s):
    print('Starting with the analytical stuff ...')
    ft, fr, fth, fph, keys = equ.my_fav_fun()

    phis = [x[0] for x in os.walk(fp_data)]
    phis = [phi for phi in phis if not (phi.endswith('data') or phi.endswith('extra'))]
    phis = [phi + '/' for phi in phis if not phi == fp_data]
    phis.sort()

    for n, file in enumerate(phis):
        phi = file[len(fp_data):-1]
        print(f'Now at {phi} ... ({n+1} / {len(phis)})')
        if not os.path.isdir(fp_save+phi):
            os.mkdir(fp_save + phi)

        res = evaluate(file, fp_save + phi, ft, fr, fth, fph, keys, s)


if __name__ == '__main__':
    fp_data = '/home/jan-menno/Data/redshift/'
    fp_save = '/home/jan-menno/Data/s0/'

    s = 0.

    main(fp_data, fp_save, s)
