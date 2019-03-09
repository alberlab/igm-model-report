import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap
from matplotlib.patches import Circle
import logging
import traceback
import os.path
from alabtools import HssFile
from alabtools.plots import plot_by_chromosome

from .utils import create_folder, average_copies


def get_radial_level(crd, index, semiaxes):
    '''
    Use coordinates in bead-major format to extract average radial levels.
    Levels are defined for a point (x, y, z) as the square root of
    (x/a)^2 + (y/b)^2 + (z/c)^2
    where a, b, c are the three semiaxes.
    That generalizes to the fraction of the nucleus radius for spheres

    Parameters
    ----------
        crd: np.ndarray
            coordinates in bead-major format (n_beads x n_structures x 3)
        index: alabtools.Index
            index of genomic locations
        semiaxes: np.ndarray
            the 3 semiaxes of the envelope
    '''

    semiaxes = np.array(semiaxes)

    radials = np.array([
        np.sqrt(np.sum(np.square(crd[i] / semiaxes), axis=1)).mean() for i in range(len(index))
    ])

    return average_copies(radials, index)


def radial_plot_p(edges, val, cmap='Greys', **kwargs):
    '''
    Plots radial densities on a sphere, colorcoded
    '''
    fig = plt.figure()
    ax = fig.gca()
    vmax = kwargs.get('vmax', max(val))
    vmin = kwargs.get('vmin', min(val))
    maxe = edges[-1]
    plt.axis('equal')
    plt.xlim(-maxe, maxe)
    plt.ylim(-maxe, maxe)

    if not isinstance(cmap, Colormap):
        cmap = get_cmap(cmap)

    def get_color(v):
        rng = vmax - vmin
        d = np.clip((v - vmin) / rng, 0, 0.999)
        idx = int(d * cmap.N)
        return cmap.colors[idx]

    for i in reversed(range(len(val))):
        c = Circle((0, 0), edges[i + 1], facecolor=get_color(val[i]))
        ax.add_patch(c)


def report_radials(hssfname, semiaxes=None, run_label=''):
    if run_label:
        run_label = '-' + run_label
    logger = logging.getLogger('Radials')
    logger.info('Executing Radials report...')
    try:
        create_folder("radials")
        with HssFile(hssfname, 'r') as hss:
            index = hss.index
            if semiaxes is None:
                # see if we have information about semiaxes in the file
                try:
                    semiaxes = hss['envelope']['params'][()]
                    if len(semiaxes.shape) == 0:  # is scalar
                        semiaxes = np.array([semiaxes, semiaxes, semiaxes])
                except:
                    semiaxes = np.array([5000., 5000., 5000.])
            radials = get_radial_level(hss.coordinates, index, semiaxes)
        np.savetxt(f'radials/radials{run_label}.txt', radials)
        f, _ = plot_by_chromosome(radials, index.get_haploid(), vmin=.4, vmax=1.0)
        f.suptitle(f'Radial position per bead {run_label}')
        plt.savefig(f'radials/radials{run_label}.pdf')
        plt.savefig(f'radials/radials{run_label}.png')

        if os.path.isfile(f'shells/ave_radial{run_label}.txt'):
            logger.info('Note: normalizing with respect to last shell')
            n = np.loadtxt(f'shells/ave_radial{run_label}.txt')[-1]
            np.savetxt(f'radials/radials_norm{run_label}.txt', radials / n)
            plot_by_chromosome(radials / n, index.get_haploid(), vmin=.4, vmax=1.0)
            plt.savefig(f'radials/radials_norm{run_label}.pdf')
            plt.savefig(f'radials/radials_norm{run_label}.png')
        logger.info('Done.')

    except KeyboardInterrupt:
        logger.error('User interrupt. Exiting.')
        exit(1)

    except:
        traceback.print_exc()
        logger.error('Error in radials step\n==============================')