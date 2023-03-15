import numpy as np
import matplotlib.pyplot as pl

import matplotlib as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_redshift_distribution(fp, ax, s, norm_color=(0, 0)):
    data = np.loadtxt(fp + '/redshift.csv', delimiter=',', skiprows=1)
    al = data[:, 0]
    be = data[:, 1]

    g = data[:, 2]

    g[g == 0] = np.nan

    n = int(np.sqrt(len(g)))
    g = g.reshape(n, n).T[::-1] - 1

    print(np.nanmin(g), np.nanmax(g))

    normx, normy = norm_color
    if not normx:
        normx = np.nanmin(g)
    if not normy:
        normy = np.nanmax(g)

    cmap = pl.cm.coolwarm.reversed()
    norm = mp.colors.Normalize(normx, normy)

    im = ax.imshow(g, extent=(np.amin(data[:, 0]), np.amax(data[:, 0]),
                   np.amin(data[:, 1]), np.amax(data[:, 1])), norm=norm, cmap=cmap)

    ax.scatter(0, 0, label=f's = {s}', s=0)
    ax.legend()
    ax.set_xlim(np.amin(data[:, 0]), np.amax(data[:, 0]))
    ax.set_ylim(np.amin(data[:, 1]), np.amax(data[:, 1]))

    return im


def draw_arrows(ax, g, alpha, beta, margin):
    interval = 11
    scale = 1.00001
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
    phi = 4.700020505370556
    base = '/media/jan-menno/T7/Schwarzschild/higher_resolution/redshift_dist_3pi-2_sphere/'
    fps = [(base + f's-0175/{phi}', -0.00175),
           (base + f's-015/{phi}', -0.00150),
           (base + f's-01/{phi}', -0.00100),
           (base + f's-005/{phi}', -0.0005),
           (base + f's0/{phi}', 0.00),
           (base + f's005/{phi}', 0.0005),
           (base + f's01/{phi}', 0.001),
           (base + f's015/{phi}', 0.0015),
           (base + f's0175/{phi}', 0.00175)
            ]

    fig, axes = pl.subplots(3, 3, figsize=(11, 10), sharex=True, sharey=True)

    for fp, ax in zip(fps, axes.flatten()):
        fp0, s = fp

        im = plot_redshift_distribution(fp0, ax, s, norm_color=(-0.8426533109584751, 0.8426533109584751))

        try:
            plot_pol_angle(fp0, ax, s, fig)
        except:
            continue

        if fp == fps[2] or fp == fps[5] or fp == fps[8]:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='2%', pad=0.05)

            fig.colorbar(im, cax=cax, orientation='vertical')

        if fp == fps[0] or fp == fps[3] or fp == fps[6]:
            ax.set_ylabel(r'$\beta$')

        if fp == fps[6] or fp == fps[7] or fp == fps[8]:
            ax.set_xlabel(r'$\alpha$')

    fig.set_tight_layout(True)
    pl.show()


if __name__ == '__main__':
    main()