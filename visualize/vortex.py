import numpy as np
import matplotlib.pyplot as pl

import matplotlib as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_redshift_distribution(fp, ax, s, norm_color=(0, 0)):
    #data = np.loadtxt(fp + '/redshift.csv', delimiter=',', skiprows=1)
    data = np.loadtxt(fp + '/polarization.csv', delimiter=',', skiprows=1)
    al = data[:, 0]
    be = data[:, 1]

    g = data[:, 2]

    #g[g != 0] += np.pi / 2
    g[g == 0] = np.nan

    n = int(np.sqrt(len(g)))
    g = g.reshape(n, n).T[::-1] #- 1

    print(np.nanmin(g), np.nanmax(g))

    normx, normy = norm_color
    #if not normx:
    #    normx = np.nanmin(g)
    #if not normy:
    #    normy = np.nanmax(g)

    cmap = pl.cm.coolwarm.reversed()
    cmap = pl.hsv()
    norm = mp.colors.Normalize(normx, normy)

    im = ax.imshow(g, extent=(np.amin(data[:, 0]), np.amax(data[:, 0]),
                   np.amin(data[:, 1]), np.amax(data[:, 1])), norm=norm, cmap=cmap)

    ax.scatter(0, 0, label=f's = {s}', s=0)
    ax.legend()
    ax.set_xlim(np.amin(data[:, 0]), np.amax(data[:, 0]))
    ax.set_ylim(np.amin(data[:, 1]), np.amax(data[:, 1]))

    return im


def draw_arrows(ax, g, alpha, beta, margin):
    interval = 111
    scale = 1.000#01
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
        ax.arrow(a*scale, b, -factor * np.sin(sample), -factor * np.cos(sample), head_length=0.,
                 head_width=0., alpha=0.3)
        # ax.annotate('', xy=(a, b), xytext=(a+factor*np.cos(sample), b+factor*np.sin(sample)), arrowprops=dict(arrowstyle='->'))


def plot_pol_angle(fp, ax, s, fig, flag=False, pi_factor=0.):
    data = np.loadtxt(fp + '/polarization.csv', delimiter=',', skiprows=1)
    al = data[:, 0]
    be = data[:, 1]
    g = data[:, 2]
    g[g == 0] = np.nan
    if np.nanmax(be) < 0:
        g[g < 3] = g[g < 3] + np.pi
    g += 0.5 * np.pi
    draw_arrows(ax, g, data[:, 0], data[:, 1], margin=0.6)

    return np.nanmin(g), np.nanmax(g)


def main():
    phi = 3.141592653589793
    #phi = 0.0
    #phi = 4.700020505370556
    #base = '/media/jan-menno/T7/Schwarzschild/higher_resolution/redshift_dist_3pi-2_sphere/'
    #base2 = '/home/jan-menno/Data/Schwarzschild/depre_2/'
    base = "Z:/Polarization/Schwarzschild/phipi/"
    base2 = "E:/Schwarzschild/higher_resolution/redshift_dist_pi_sphere/"
    fps = [(base + f'stereo/{phi}', base + f'stereo/{phi}', 0.00175),
           (base + f's0/{phi}', base + f's0/{phi}', 0.00),
           (base + f's-0175/{phi}', base + f's-0175/{phi}', -0.00175),
           (base + f's0175/{phi}', base + f's0175/{phi}', 0.00175),
           #(f'/home/jan-menno/Data/Schwarzschild/bigger_sample_4/{phi}', base2 + f'{phi}', 0.0019)
            ]

    fig, axes = pl.subplots(1, len(fps), figsize=(13, 5), sharex=True, sharey=True)

    for fp, ax in zip(fps, axes.flatten()):
        fp0, fp1, s = fp

        nmin = -0.8426533109584751
        nmax = 0.8426533109584751

        nmin = 0.
        nmax = np.pi * 2

        im = plot_redshift_distribution(fp1, ax, s, norm_color=(nmin, nmax))

        try:
            plot_pol_angle(fp0, ax, s, fig)
        except:
            print('NOPE')
            plot_pol_angle(fp0, ax, s, fig)

        if fp == fps[-1]:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='2%', pad=0.05)

            fig.colorbar(im, cax=cax, orientation='vertical')

        if fp == fps[0]:
            ax.set_ylabel(r'$\beta$')

        if fp == fps[-1]:
            ax.set_xlabel(r'$\alpha$')

    fig.set_tight_layout(True)
    pl.show()


if __name__ == '__main__':
    main()