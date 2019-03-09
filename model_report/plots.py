import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def logloghist2d(d1, d2, bins=(100, 100), ranges=((1e-3, 1), (1e-3, 1)), outfile=None, vmin=1e2, vmax=1e5, nlevels=5,
                 sigma=None, xlabel='in', ylabel='out', smooth=None, **kwargs):
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
    bottom_left = max(xx[0], yy[0])
    top_right = min(xx[-1], yy[-1])

    h, e1, e2 = np.histogram2d(d1, d2, bins=(xx, yy))

    if smooth:
        h = gaussian_filter(h, **smooth)

    f = plt.figure(figsize=(5, 5))
    # p = plt.pcolormesh(xx, yy, h, **kwargs)
    X, Y = np.meshgrid(xx[1:], yy[1:])
    q = h.copy()
    h[h > vmax] = vmax
    levels = np.logspace(np.log10(vmin), np.log10(vmax), nlevels, base=10)
    p = plt.contourf(X, Y, h, norm=LogNorm(vmin, vmax), cmap='Greys', levels=levels, **kwargs)
    if sigma is not None:
        plt.axvline(x=sigma, ls='--', c='green')

    plt.plot([bottom_left, top_right], [bottom_left, top_right], 'k--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xx[0], xx[-1])
    plt.ylim(yy[0], yy[-1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile)
    return f, p, q, e1, e2


def density_histogram_2d(d1, d2, bins=(100, 100), ranges=[(1e-3, 1), (1e-3, 1)],
                  outfile=None, vmin=1e2, vmax=1e5, nlevels=5,
                  xlabel='in', ylabel='out', smooth=None, **kwargs):
    import numpy as np
    from matplotlib.colors import LogNorm

    if ranges[0] is None:
        x0, x1 = np.min(d1), np.max(d1)
    else:
        x0, x1 = ranges[0]

    if ranges[1] is None:
        y0, y1 = np.min(d2), np.max(d2)
    else:
        y0, y1 = ranges[1]

    xx = np.linspace(x0, x1, bins[0])
    yy = np.linspace(y0, y1, bins[1])

    bottom_left = max(xx[0], yy[0])
    top_right = min(xx[-1], yy[-1])

    h, e1, e2 = np.histogram2d(d1, d2, bins=(xx, yy))

    if smooth:
        h = gaussian_filter(h, **smooth)

    f = plt.figure(figsize=(4, 4))
    # p = plt.pcolormesh(xx, yy, h, **kwargs)
    X, Y = np.meshgrid(xx[1:], yy[1:])
    q = h.copy()
    h[h > vmax] = vmax
    levels = np.logspace(np.log10(vmin), np.log10(vmax), nlevels, base=10)
    p = plt.contourf(X, Y, h, norm=LogNorm(vmin, vmax),
                     cmap='Greys', levels=levels, **kwargs)
    plt.plot([bottom_left, top_right], [bottom_left, top_right], 'k--')
    plt.xlim(xx[0], xx[-1])
    plt.ylim(yy[0], yy[-1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile)
    return f, p, q, e1, e2