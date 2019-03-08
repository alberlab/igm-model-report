import numpy as np
import matplotlib.pyplot as plt


def logloghist2d(d1, d2, bins=(100, 100), ranges=((1e-3, 1), (1e-3, 1)), outfile=None, vmin=1e2, vmax=1e5, nlevels=5,
                 sigma=None, xlabel='in', ylabel='out', **kwargs):
    from matplotlib.colors import LogNorm

    if ranges[0] is None:
        x0, x1 = np.min(d1), np.max(d1)
    else:
        x0, x1 = ranges[0]

    if ranges[1] is None:
        y0, y1 = np.min(d2), np.max(d2)
    else:
        y0, y1 = ranges[1]

    xx = np.logspace(np.log10(x0), np.log10(x1), bins[0], base=10)
    yy = np.logspace(np.log10(y0), np.log10(y1), bins[1], base=10)

    h, e1, e2 = np.histogram2d(d1, d2, bins=(xx, yy))
    f = plt.figure(figsize=(10, 10))
    # p = plt.pcolormesh(xx, yy, h, **kwargs)
    X, Y = np.meshgrid(xx[:-1], yy[:-1])
    q = h.copy()
    h[h > vmax] = vmax
    levels = np.logspace(np.log10(vmin), np.log10(vmax), nlevels, base=10)
    p = plt.contourf(X, Y, h, norm=LogNorm(vmin, vmax), cmap='Greys', levels=levels, **kwargs)
    if sigma is not None:
        plt.axvline(x=sigma, ls='--', c='green')

    plt.plot([vmin, vmax], [vmin, vmax], 'k--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile)
    return f, p, q, e1, e2
