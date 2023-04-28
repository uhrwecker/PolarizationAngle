import numpy as np
import matplotlib.pyplot as pl

import matplotlib as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable


def clean_data(g, norm):
    limit = 0.6
    gmin, gmax = norm
    for m, row in enumerate(g):
        for n in range(len(row)-1):
            if not np.isnan(row[n]):
                if np.abs(row[n+1] - row[n]) > limit or row[n+1] < gmin or row[n+1] > gmax:
                    row[n+1] = row[n]
                if np.abs(row[n+1] - row[n]) > limit or row[n+1] < gmin or row[n+1] > gmax:
                    g[m][n+1] = g[m+1][n]

    return g


def plot_redshift_distribution(fp, ax, s, norm_color=(0, 0), margin=0.01):
    data = np.loadtxt(fp + '/redshift.csv', delimiter=',', skiprows=1)
    #data = np.loadtxt(fp + '/polarization.csv', delimiter=',', skiprows=1)
    al = data[:, 0]
    be = data[:, 1]

    g = data[:, 2]
    #g[g != 0] -= np.pi / 2
    #g[g < 0] += np.pi * 2

    #g[g != 0] += np.pi / 2
    g[g == 0] = np.nan

    n = int(np.sqrt(len(g)))
    g = g.reshape(n, n).T[::-1] - 1

    g = clean_data(g, norm_color)

    print(np.nanmin(g), np.nanmax(g))

    normx, normy = norm_color
    #if not normx:
    #    normx = np.nanmin(g)
    #if not normy:
    #    normy = np.nanmax(g)

    cmap = pl.cm.coolwarm.reversed()
    #cmap = pl.hsv()
    norm = mp.colors.Normalize(normx, normy)

    im = ax.imshow(g, extent=(np.amin(data[:, 0]), np.amax(data[:, 0]),
                   np.amin(data[:, 1]), np.amax(data[:, 1])), norm=norm, cmap=cmap)

    ax.scatter(0, 0, label=f's = {s}', s=0)
    ax.legend()
    ax.set_xlim(np.amin(data[:, 0]).round(3), np.amax(data[:, 0]).round(3))
    ax.set_ylim(np.amin(data[:, 1]).round(3), np.amax(data[:, 1]).round(3))

    return im


def draw_arrows(ax, g, alpha, beta, margin, index):
    interval = 793 + index # 827 # 1127 # 1264
    print('Interval at ' , interval)
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

        #print(sample)
        #if np.abs(sample - np.pi) < 0.2:
        #    print(sample)
        #    sample += 1.6

        a = alpha[n]  # + marg
        b = beta[n]  # + marg
        ax.arrow(a*scale, b, -factor * np.sin(sample), -factor * np.cos(sample), head_length=0.,
                 head_width=0., alpha=0.3)
        # ax.annotate('', xy=(a, b), xytext=(a+factor*np.cos(sample), b+factor*np.sin(sample)), arrowprops=dict(arrowstyle='->'))


def plot_pol_angle(fp, ax, s, fig, flag=False, pi_factor=0., index=0):
    data = np.loadtxt(fp + '/polarization.csv', delimiter=',', skiprows=1)
    al = data[:, 0]
    be = data[:, 1]
    g = data[:, 2]
    g[g == 0] = np.nan
    if np.nanmax(be) < 0:
        g[g < 3] = g[g < 3] + np.pi
    g += 0.5 * np.pi
    draw_arrows(ax, g, data[:, 0], data[:, 1], margin=0.6, index=index)

    return np.nanmin(g), np.nanmax(g)


def main():
    phi = (0.0, '0')
    phi = (1.5831648018090296, 'pi-2')
    phi = (3.141592653589793, 'pi')
    phi = (4.700020505370556, '3pi-2')

    typ = 'sphere'
    typ = 'maclaurin'

    xn = 5
    yn = 6

    if typ == 'sphere':
        fps = [(f"Z:/Polarization/Schwarzschild/{typ}/s0175/{phi[0]}",
               f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s0175/{phi[0]}", 0.00175),
               (f"Z:/Polarization/Schwarzschild/{typ}/s015/{phi[0]}",
                f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s015/{phi[0]}", 0.00150),
               (f"Z:/Polarization/Schwarzschild/{typ}/s01/{phi[0]}",
                f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s01/{phi[0]}", 0.00100),
               (f"Z:/Polarization/Schwarzschild/{typ}/s005/{phi[0]}",
                f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s005/{phi[0]}", 0.00050),
               (f"Z:/Polarization/Schwarzschild/{typ}/s0/{phi[0]}",
                f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s0/{phi[0]}", 0.00000),
               (f"Z:/Polarization/Schwarzschild/{typ}/s-005/{phi[0]}",
                f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s-005/{phi[0]}", -0.00050),
               (f"Z:/Polarization/Schwarzschild/{typ}/s-01/{phi[0]}",
                f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s-01/{phi[0]}", -0.00100),
               (f"Z:/Polarization/Schwarzschild/{typ}/s-015/{phi[0]}",
                f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s-015/{phi[0]}", -0.00150),
              (f"Z:/Polarization/Schwarzschild/{typ}/s-0175/{phi[0]}",
                f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s-0175/{phi[0]}", -0.00175)
              ]

    if typ == 'maclaurin':
        fps = [(f"Z:/Polarization/Schwarzschild/maclaurin/s0175/{phi[0]}",
                    f"Z:/Polarization/Schwarzschild/redshift/s0175/{phi[0]}", 0.00175),
                   (f"Z:/Polarization/Schwarzschild/maclaurin/s015/{phi[0]}",
                    f"Z:/Polarization/Schwarzschild/redshift/s015/{phi[0]}", 0.00150),
                   (f"Z:/Polarization/Schwarzschild/maclaurin/s01/{phi[0]}",
                    f"Z:/Polarization/Schwarzschild/redshift/s01/{phi[0]}", 0.00100),
                   (f"Z:/Polarization/Schwarzschild/maclaurin/s005/{phi[0]}",
                    f"Z:/Polarization/Schwarzschild/redshift/s005/{phi[0]}", 0.00050),
                   (f"Z:/Polarization/Schwarzschild/sphere/s0/{phi[0]}",
                    f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_sphere/s0/{phi[0]}", 0.00000),
                   (f"Z:/Polarization/Schwarzschild/maclaurin/s-005/{phi[0]}",
                    f"Z:/Polarization/Schwarzschild/redshift/s-005/{phi[0]}", -0.00050),
                   (f"Z:/Polarization/Schwarzschild/maclaurin/s-01/{phi[0]}",
                    f"Z:/Polarization/Schwarzschild/redshift/s-01/{phi[0]}", -0.00100),
                   (f"Z:/Polarization/Schwarzschild/maclaurin/s-015/{phi[0]}",
                    f"Z:/Polarization/Schwarzschild/redshift/s-015/{phi[0]}", -0.00150),
                   (f"Z:/Polarization/Schwarzschild/maclaurin/s-0175/{phi[0]}",
                    f"Z:/Polarization/Schwarzschild/redshift/s-0175/{phi[0]}", -0.00175)
                   ]

    fig, axes = pl.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
    pl.rcParams.update({'font.size': 12})

    for fp, ax in zip(fps, axes.flatten()):
        fp0, fp1, s = fp

        nmin = -0.95
        nmax = 0.95

        #nmin = 0.
        #nmax = np.pi * 2

        im = plot_redshift_distribution(fp1, ax, s, norm_color=(nmin, nmax))

        if fp1 == f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_sphere/s0/{phi[0]}" and typ == 'maclaurin':
            plot_pol_angle(fp0, ax, s, fig, index=175)
        else:
            try:
                plot_pol_angle(fp0, ax, s, fig)
            except:
                print('NOPE')
                plot_pol_angle(fp0, ax, s, fig)

        if fp == fps[-1] or fp == fps[2] or fp == fps[5]:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='2%', pad=0.05)

            fig.colorbar(im, cax=cax, orientation='vertical')

        if fp == fps[0] or fp == fps[3] or fp == fps[6]:
            ax.set_ylabel(r'$\beta$')
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            ax.yaxis.label.set_size(12)

        if fp == fps[-1] or fp == fps[-2] or fp == fps[-3]:
            ax.set_xlabel(r'$\alpha$')
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            ax.xaxis.label.set_size(12)

    fig.set_tight_layout(True)
    from matplotlib.ticker import FormatStrFormatter

    pl.subplots_adjust(wspace=1.8)

    ax.xaxis.set_major_locator(pl.MaxNLocator(xn))
    ax.yaxis.set_major_locator(pl.MaxNLocator(yn))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    pl.savefig(f'E:/Results/Schwarzschild/0_{typ}_{phi[1]}.png')
    #pl.show()


def find_index(index):
    phi = (0.0, '0')
    # phi = (1.5831648018090296, 'pi-2')
    # phi = (3.14159265359, 'pi')#(3.141592653589793, 'pi')
    # phi = (4.700020505370556, '3pi-2')

    typ = 'sphere'
    typ = 'maclaurin'

    xn = 5
    yn = 6

    fps = [(f"Z:/Polarization/Schwarzschild/{typ}/s0175/{phi[0]}",
            f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s0175/{phi[0]}", 0.00175),
           (f"Z:/Polarization/Schwarzschild/{typ}/s015/{phi[0]}",
            f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s015/{phi[0]}", 0.00150),
           (f"Z:/Polarization/Schwarzschild/{typ}/s01/{phi[0]}",
            f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s01/{phi[0]}", 0.00100),
           (f"Z:/Polarization/Schwarzschild/{typ}/s005/{phi[0]}",
            f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s005/{phi[0]}", 0.00050),
           (f"Z:/Polarization/Schwarzschild/{typ}/s0/{phi[0]}",
            f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s0/{phi[0]}", 0.00000),
           (f"Z:/Polarization/Schwarzschild/{typ}/s-005/{phi[0]}",
            f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s-005/{phi[0]}", -0.00050),
           (f"Z:/Polarization/Schwarzschild/{typ}/s-01/{phi[0]}",
            f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s-01/{phi[0]}", -0.00100),
           (f"Z:/Polarization/Schwarzschild/{typ}/s-015/{phi[0]}",
            f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s-015/{phi[0]}", -0.00150),
           (f"Z:/Polarization/Schwarzschild/{typ}/s-0175/{phi[0]}",
            f"E:/Schwarzschild/higher_resolution/redshift_dist_{phi[1]}_{typ}/s-0175/{phi[0]}", -0.00175)
           ]

    fig, axes = pl.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
    pl.rcParams.update({'font.size': 12})

    for fp, ax in zip(fps, axes.flatten()):
        fp0, fp1, s = fp

        nmin = -0.95
        nmax = 0.95

        # nmin = 0.
        # nmax = np.pi * 2

        im = plot_redshift_distribution(fp1, ax, s, norm_color=(nmin, nmax))

        try:
            plot_pol_angle(fp0, ax, s, fig, index=index)
        except:
            print('NOPE')
            plot_pol_angle(fp0, ax, s, fig)

        if fp == fps[-1] or fp == fps[2] or fp == fps[5]:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='2%', pad=0.05)

            fig.colorbar(im, cax=cax, orientation='vertical')

        if fp == fps[0] or fp == fps[3] or fp == fps[6]:
            ax.set_ylabel(r'$\beta$')
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            ax.yaxis.label.set_size(12)

        if fp == fps[-1] or fp == fps[-2] or fp == fps[-3]:
            ax.set_xlabel(r'$\alpha$')
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            ax.xaxis.label.set_size(12)

    fig.set_tight_layout(True)
    from matplotlib.ticker import FormatStrFormatter

    pl.subplots_adjust(wspace=1.8)

    ax.xaxis.set_major_locator(pl.MaxNLocator(xn))
    ax.yaxis.set_major_locator(pl.MaxNLocator(yn))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # pl.savefig(f'E:/Results/Schwarzschild/0_{typ}_{phi[1]}.png')
    pl.show()


if __name__ == '__main__':
    main()
    #for i in range(100):
    #    find_index(i)