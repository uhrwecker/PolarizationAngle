import numpy as np
import os
import pandas as pd
import json
import single_angle_solution as sas
from angle import equ

from tqdm import tqdm
import multiprocessing as mp
import pathos as pt


def get_angle(data):
    redshift_data, fp_to_json, file, f1, f2, keys, s = data
    ft, fr, fth, fphi = f1
    ft2, fr2, fth2, fphi2 = f2

    with open(fp_to_json + file, 'r') as f:
        config = json.load(f)
    idx1 = np.where(redshift_data[:, 0] == config['OBSERVER']['alpha'])[0]
    idx2 = np.where(redshift_data[:, 1] == config['OBSERVER']['beta'])[0]
    try:
        idx = idx1[np.in1d(idx1, idx2)][0]
    except:
        return 0, 0.

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
    if pa == 0.0:
        pa = sas.main(config, u1, u3, v, ft2, fr2, fth2, fphi2, keys)

    #print(pa)

    return idx, pa


def evaluate(fp_to_redshift, fp_to_save, f1, f2, keys, s):
    # step 0: reformalization
    fp_to_json = fp_to_redshift + 'data/'

    # step 1: get a list of every json file within
    filenames = next(os.walk(fp_to_json), (None, None, []))[2]

    # step 2: load .csv of redshift
    redshift_data = np.loadtxt(fp_to_redshift + 'redshift.csv', delimiter=',', skiprows=1)
    redshift_data2 = np.loadtxt(fp_to_redshift + 'redshift.csv', delimiter=',', skiprows=1)

    mp_data = [[redshift_data, fp_to_json, file, f1, f2, keys, s] for file in filenames]
    pool = pt.pools.ProcessPool(mp.cpu_count()-1)
    for result in tqdm(pool.uimap(get_angle, mp_data), total=len(mp_data)):
        idx, pa = result
        #print(idx, pa)

        if idx != 0:
            redshift_data[idx][-1] = pa[0]
            redshift_data2[idx][-1] = pa[1]

    #print(pa)
    np.savetxt(fp_to_save + 'polarization.csv', redshift_data, delimiter=',', header='alpha,beta,pol_angle')
    np.savetxt(fp_to_save + 'polarization2.csv', redshift_data2, delimiter=',', header='alpha,beta,pol_angle')

    #for file in filenames:
    #    with open(fp_to_json + file, 'r') as f:
    #        config = json.load(f)

    #    pa, idx = get_angle(redshift_data, config, ft, fr, fth, fphi, keys, s)#

    #    redshift_data[idx][-1] = pa

    #    if not filenames == []:
    #        filenames.remove(file)

    #np.savetxt(fp_to_save + 'polarization.csv', redshift_data, delimiter=',', header='alpha,beta,pol_angle')

    # step 3: iterate over redshift data
    #for n, row in enumerate(redshift_data):
    #    # step 3a: exclude data when there is no hit:
    #    if row[-1] == 0.:
    #        continue

    #    # step 4: load all json files one by one
    #    for file in filenames:
    #        with open(fp_to_json + file, 'r') as f:
    #            config = json.load(f)

    #        if config['OBSERVER']['alpha'] != row[0] or config['OBSERVER']['beta'] != row[1]:
    #            del config
    #            continue

            # here goes the programming stuff

            # the surface velocity:
    #        try:
    ##            rho = config['EMITTER']['rho']
     #       except:
    #            rho = config['EMITTER']['a']
    #        T = config['EMITTER']['Theta']
    #        P = config['EMITTER']['Phi']

    #        u1 = 5 * s / (2 * rho) * np.sin(P) * np.sin(T)
    #        u3 = 5 * s / (2 * rho) * np.cos(P) * np.sin(T)

    #        v = config['VELOCITIES']['orbit']

    #        pa = sas.main(config, u1, u3, v, ft, fr, fth, fphi, keys)

    #        result.append(pa)

    #        break

    #    row[-1] = pa
    #    redshift_data[n] = row

    #    if not filenames == []:
    #        filenames.remove(file)

    #np.savetxt(fp_to_save + 'polarization.csv', redshift_data, delimiter=',', header='alpha,beta,pol_angle')

    #return result


def main(fp_data, fp_save, s):
    print('Starting with the analytical stuff ...')
    #ft, fr, fth, fph, keys = equ.my_fav_fun(1)
    #ft2, fr2, fth2, fph2, keys2 = equ.my_fav_fun(0)
    ft, fr, fth, fph, keys, ft2, fr2, fth2, fph2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    print([x[0] for x in os.walk(fp_data)])
    phis = [x[0] for x in os.walk(fp_data)]
    phis = [phi for phi in phis if not (phi.endswith('data') or phi.endswith('extra'))]
    phis = [phi + '/' for phi in phis if not phi == fp_data]
    phis.sort()

    print(phis)
    for n, file in enumerate(phis):
        phi = file[len(fp_data):-1]
        print(f'Now at {phi} ... ({n+1} / {len(phis)})')
        print(os.path.isdir(fp_save+phi))
        if not os.path.isdir(fp_save+phi):
            os.mkdir(fp_save + phi)
            res = evaluate(file, fp_save + phi + '/', [ft, fr, fth, fph], [ft2, fr2, fth2, fph2], keys, s)


if __name__ == '__main__':
    fp_data = '/media/jan-menno/T7/Schwarzschild/higher_resolution/redshift_dist_3pi-2_sphere/s0/'
    fp_save = '/home/jan-menno/Data/Schwarzschild/depre_2/'

    s = 0.00175
    #print(os.listdir('Z:/Data/'))#os.path.abspath(fp_data))

    main(fp_data, fp_save, s)
