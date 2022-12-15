import json


def prepare_data(fp):
    with open(fp, 'r') as f:
        data = json.load(f)

    new_data = {}

    new_data['r_em'] = data['INITIAL_DATA']['r0']
    new_data['th_em'] = data['INITIAL_DATA']['theta0']
    new_data['ph_em'] = data['INITIAL_DATA']['phi0']

    new_data['dt_em'] = data['INITIAL_DATA']['dt']
    new_data['dr_em'] = data['INITIAL_DATA']['dr']
    new_data['dth_em'] = data['INITIAL_DATA']['dtheta']
    new_data['dph_em'] = data['INITIAL_DATA']['dphi']

    new_data['bha'] = data['EMITTER']['bh_a']

    new_data['u1'] = data['VELOCITIES']['surf_u1']
    new_data['u3'] = data['VELOCITIES']['surf_u3']
    new_data['v'] = data['VELOCITIES']['orbit']

    new_data['alpha'] = data['OBSERVER']['alpha']
    new_data['beta'] = data['OBSERVER']['beta']

    return new_data
