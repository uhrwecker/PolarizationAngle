import numpy as np
import matplotlib.pyplot as pl

import matplotlib as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable


def draw_arrows(ax, g, alpha, beta, margin):
    interval = 12
    n_sample = np.arange(0, len(g))[::interval]
    g_sample = g[::interval]

    alpha += np.abs(np.amax(alpha) - np.amin(alpha)) / int(np.sqrt(len(g)))
    # beta += np.abs(np.amax(beta) - np.amin(beta)) / int(np.sqrt(len(g)))

    factor = 0.000066  # 18

    for n, sample in zip(n_sample, g_sample):
        marg = 2 * margin * n / len(n_sample) - margin
        if np.isnan(sample):
            continue

        sample -= np.pi / 2
        if sample < 0:
            sample += 2 * np.pi

        a = alpha[n]  # + marg
        b = beta[n]  # + marg
        ax.arrow(a, b, -factor * np.sin(sample), -factor * np.cos(sample), head_length=0.,
                 head_width=0.)
        # ax.annotate('', xy=(a, b), xytext=(a+factor*np.cos(sample), b+factor*np.sin(sample)), arrowprops=dict(arrowstyle='->'))


def plot_redshift_distribution(fp, ax, s, fig, flag=False, pi_factor=0.):
    data = np.loadtxt(fp + '/polarization.csv', delimiter=',', skiprows=1)
    al = data[:, 0]
    be = data[:, 1]
    g = data[:, 2]
    g[g == 0] = np.nan
    if np.nanmax(be) < 0:
        g[g < 3] = g[g < 3] + np.pi
    g += 0.5 * np.pi
    draw_arrows(ax, g, data[:, 0], data[:, 1], margin=0.6)



    # g = - g

    g = g % (2 * np.pi)

    # g *= 180 / np.pi
    a_gmin = al[g == np.nanmin(g)]
    b_gmin = be[g == np.nanmin(g)]
    # ax.scatter(a_gmin, b_gmin, s=10, color='green')

    a_gmax = al[g == np.nanmax(g)]
    b_gmax = be[g == np.nanmax(g)]
    # ax.scatter(a_gmax, b_gmax, s=10, color='green')

    n = int(np.sqrt(len(g)))
    g = g.reshape(n, n).T[::-1]
    # if np.nanmin(g) < 0:
    #    g *= -1
    # g[np.isnan(g)] = 0

    print(np.nanmin(g), np.nanmax(g))

    cmap = pl.cm.hsv
    norm = mp.colors.Normalize(0, 2 * np.pi)
    # norm = mp.colors.Normalize(np.nanmin(g), np.nanmax(g))

    margin = 0#1
    im = ax.imshow(g, extent=(np.amin(data[:, 0]) - margin, np.amax(data[:, 0]) + margin,
                              np.amin(data[:, 1]) - margin, np.amax(data[:, 1]) + margin), norm=norm, cmap=cmap)

    # if s == 0.001 or s == -0.0005 or s == -0.00175 or flag:
    #    divider = make_axes_locatable(ax)
    #    cax = divider.append_axes('right', size='2%', pad=0.05)

    # fig.colorbar(im, cax=cax, orientation='vertical')

    if s == 0.00175 or s == 0.0005 or s == -0.001:
        ax.set_ylabel(r'$\beta$')

    if s == -0.001 or s == -0.0015 or s == -0.00175:
        ax.set_xlabel(r'$\alpha$')

    ax.scatter(0, 0, label=f's ={s}', s=0)
    ax.legend()
    # ax.set_xlim(-10, 10)
    # ax.set_ylim(-10, 10)
    ax.set_xlim(np.amin(data[:, 0]), np.amax(data[:, 0]))
    ax.set_ylim(np.amin(data[:, 1]), np.amax(data[:, 1]))

    return np.nanmin(g), np.nanmax(g), im


def main():
    fp1 = '/home/jan-menno/Data/spin_variation/s-0175/4.601072705257493'
    fp2 = '/home/jan-menno/Data/spin_variation/s0/4.601072705257493'
    fp3 = '/home/jan-menno/Data/spin_variation/s0175/4.601072705257493'
    #fp = '/home/jan-menno/Data/spin_variation/s-0175/4.601072705257493'
    #fp = '/home/jan-menno/Data/spin_variation/s-0175/6.283185307179586'
    #fp = 'Z:/Data/06022023/polarization/1.0389519011871757'
    #fp = 'Z:/Data/06022023/polarization/2.028429902317819'
    #fp = 'Z:/Data/06022023/polarization/3.017907903448463'
    #fp = 'Z:/Data/06022023/polarization/1.0389519011871757'
    #fp = 'Z:/Data/06022023/polarization/1.0389519011871757'
    fps = [fp1, fp2, fp3]
    ss = ['-0.00175', ' 0.00000', ' 0.00175']

    fig, axes = pl.subplots(1, 3, figsize=(13, 7))

    for fp, s, ax in zip(fps, ss, axes):

        gn, gx, im = plot_redshift_distribution(fp, ax, s, fig, flag=True, pi_factor=0)
    #print(fp)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)

    fig.colorbar(im, cax=cax, orientation='vertical')

    fig.set_tight_layout(True)
    #ax.set_xlim(-10, 10)
    #ax.set_ylim(-10, 10)

    pl.show()

if __name__ == '__main__':
    main()