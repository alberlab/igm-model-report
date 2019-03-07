'''
Reporting tool
==============

reporting based on a structure:

- create summary stats:
    - imposed hic restraints: cis / trans / distribution by chromosome / distribution by genomic distance
    - violations histograms by type
    - histogram of number of restraint per bead

- create contact map and compare with the input
    - visual matrix images
    - in/out scatter plot
    - compute correlation

- structural features:
    - summary stats:
        - average radius of each shell
        - total radius of gyration
    - plot of average number of neighbors per bead
    - histogram of number of neighbors
    - radius of gyration per chromosome

- radial positions
    - plots of radial positions per bead
    - plots of radial position per chromosome
    - identification of peaks and dips

- five shells analysis
    - ?

- damid
    - scatter plot in/out (or expected/out)
    - plots of damid profile by chromosome (so, what are we using here?)

'''

import os, argparse
import traceback
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from alabtools import *
from alabtools.analysis import get_simulated_hic
from alabtools.plots import plot_comparison, red, plot_by_chromosome
from scipy.stats import pearsonr
from matplotlib.patches import Circle
from matplotlib.ticker import PercentFormatter

from igm import Config

logging.basicConfig()
logger = logging.getLogger()


def average_copies(v, index):
    ci = index.copy_index
    x = np.zeros(len(ci))
    for i in range(len(ci)):
        x[i] = np.mean(v[ci[i]])
    return x


def sum_copies(v, index):
    ci = index.copy_index
    x = np.zeros(len(ci))
    for i in range(len(ci)):
        x[i] = np.sum(v[ci[i]])
    return x


def get_radial_level(crd, index, semiaxes):
    semiaxes = np.array(semiaxes)

    radials = np.array([
        np.sqrt(np.sum(np.square(crd[i] / semiaxes), axis=1)).mean() for i in range(len(index))
    ])

    return average_copies(radials, index)


def create_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def plot_violation_histogram(h, edges, tol=0.05, nticks=20, title='', outfile=None):
    from matplotlib.patches import Rectangle
    plt.figure(figsize=(10, 4))
    step = edges[1] - edges[0]
    tick_step = int(len(edges) / nticks)

    # transform to percentage
    totsum = np.sum(h)
    h = h / totsum

    xx = np.arange(len(h) - 1) + 0.5
    xx = np.concatenate([xx, [len(h) + tick_step + 0.5]])

    tick_pos = list(range(len(edges))[::tick_step])
    tick_labels = ['{:.2f}'.format(edges[i]) for i in tick_pos]
    tick_pos.append(len(h) + tick_step + 0.5)
    tick_labels.append('>{:.2f}'.format(edges[-2]))

    # ignore the first bin to determine height
    vmax = np.max(h[1:]) * 1.1

    plt.title(title)

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

    plt.axvline(x=tol / step, ls='--', c='green')
    plt.gca().add_patch(Rectangle([0, 0], width=tol / step, height=vmax, fill=False, hatch='/'))
    plt.ylim(0, vmax)

    plt.bar(xx, height=h, width=1, color='grey')
    plt.xticks(tick_pos, tick_labels, rotation=60)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile)


def logloghist2d(d1, d2, bins=(100, 100), ranges=[(1e-3, 1), (1e-3, 1)], outfile=None, vmin=1e2, vmax=1e5, nlevels=5, sigma=None, xlabel='in', ylabel='out', **kwargs):
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
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile)
    return f, p, q, e1, e2


def rg(crds):
    ave = np.average(crds, axis=0)
    v = crds - ave
    return np.sqrt(np.sum(np.square(v)) / len(v))


def get_chroms_rgs(crds, index):
    rgs = []
    for chrom in index.get_chromosomes():
        copies = index.get_chrom_copies(chrom)
        data = list()
        for copy in copies:
            ii = index.get_chrom_pos(chrom, copy)
            for crd in crds[ii, :].swapaxes(0, 1):
                data.append(rg(crd))
        rgs.append(np.array(data))
    return rgs


def setBoxColors(bp, clr):
    if not isinstance(clr, (list, tuple, np.ndarray)):
        clr = [clr] * len(bp['boxes'])
    for i in range(len(clr)):
        plt.setp(bp['boxes'][i], facecolor=clr[i])
        plt.setp(bp['medians'][i], color='black')


def rg_boxplot(data, chroms, n_per_row=6, subplot_width=10, subplot_height=2.5, vmin=800, vmax=2500, outfile=None):

    n_chroms = len(chroms)
    n_rows = n_chroms // n_per_row if n_chroms % n_per_row == 0 else n_chroms // n_per_row + 1
    f, plots = plt.subplots(n_rows, 1, figsize=(subplot_width, subplot_height * n_rows), sharey=True)
    for ip, i in enumerate(range(0, n_chroms, n_per_row)):

        # select chromosome data
        boxdata = data[i:i + n_per_row]
        plots[ip].set_ylim(vmin, vmax)
        bp = plots[ip].boxplot(boxdata, labels=chroms[i:i + n_per_row], patch_artist=True, showfliers=False)
        setBoxColors(bp, '#fb7b04')

    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile)
    return f, plots


def create_average_distance_map(hssfname):
    with HssFile(hssfname, 'r') as h:
        crd = h.coordinates
        index = h.index
    N = len(index.copy_index)
    fmap = np.empty((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            aves = []
            if index.chrom[i] == index.chrom[j]:
                for k, m in zip(index.copy_index[i], index.copy_index[j]):
                    aves.append(np.linalg.norm(crd[k] - crd[m], axis=1).mean())
            else:
                for k in index.copy_index[i]:
                    for m in index.copy_index[j]:
                        aves.append(np.linalg.norm(crd[k] - crd[m], axis=1).mean())
            fmap[i, j] = np.mean(aves)
    return fmap


def radial_plot_p(edges, val, **kwargs):
    from matplotlib.cm import get_cmap
    fig = plt.figure()
    ax = fig.gca()
    cmap = kwargs.get('cmap', get_cmap())
    vmax = kwargs.get('vmax', max(val))
    vmin = kwargs.get('vmin', min(val))
    maxe = edges[-1]
    plt.axis('equal')
    plt.xlim(-maxe, maxe)
    plt.ylim(-maxe, maxe)

    def get_color(v):
        rng = vmax - vmin
        d = np.clip((val[i] - vmin) / rng, 0, 0.999)
        idx = int(d * cmap.N)
        return cmap.colors[idx]

    for i in reversed(range(len(val))):
        c = Circle((0, 0), edges[i + 1], facecolor=get_color(val[i]))
        ax.add_patch(c)


def snormsq_ellipse(x, semiaxes, r):
    a, b, c = np.array(semiaxes) - r
    sq = np.square(x)
    return sq[:, 0] / (a**2) + sq[:, 1] / (b**2) + sq[:, 2] / (c**2)


# read options
parser = argparse.ArgumentParser(description='Run population analysis pipeline')

parser.add_argument('hss', help='Hss file for population')
parser.add_argument('-c', '--config', default='igm-config.json', help='Config file for IGM run (infers some parameters)')

parser.add_argument('--hic', help='Input probability matrix for HiC')
parser.add_argument('--hic-sigma', type=float, help='Probability cutoff for HiC')
parser.add_argument('--hic-contact-range', type=float, help='Probability cutoff for HiC', default=0.01)

parser.add_argument('--damid', help='Input probability matrix for DamID')
parser.add_argument('--damid-sigma', type=float, help='Probability cutoff for DamID', default=0.05)
parser.add_argument('--damid-contact-range', type=float, help='Probability cutoff for DamID')

parser.add_argument('--violation-tolerance', type=float, help='Violation tolerance')

parser.add_argument('--semiaxes', nargs=3, type=float, help='Specify semiaxes of the envelope')

parser.add_argument('--steps', help='comma separated list of steps to perform. Perform all of the applicable ones by default.'
                    ' Possible values: radius_of_gyration, violations, five_shells, radials, radial_density, damid',
                    default='radius_of_gyration,violations,hic,five_shells,radials,radial_density,damid')

parser.add_argument('--no-config', action='store_true', help='do not read parameters from config file')

parser.add_argument('-o', '--out-dir', help='Output directory')

args = parser.parse_args()


# load stuff
tol = 0.05
damid_file = None
cm = False
if not args.no_config:
    logger.info('Reading config from: %s' % args.config)
    cfg = Config(args.config)
    cm = cfg.get('restraints/Hi-C/input_matrix', False)
    hic_contact_range = cfg.get('restraints/Hi-C/contact_range', False)
    damid_file = cfg.get('restraints/DamID/input_profile', False)
    damid_contact_range = cfg.get('restraints/DamID/contact_range', False)
    tol = cfg.get('optimization/violation_tolerance', 0.05)

hssfname = os.path.realpath(args.hss)

if args.out_dir is None:
    d, f = os.path.split(hssfname)
    b, e = os.path.splitext(f)
    args.out_dir = os.path.join(d, 'QC_' + b)

if args.hic is not None:
    cm = args.hic

if args.damid is not None:
    damid_file = args.damid

if cm:
    cm = os.path.abspath(cm)

if damid_file:
    damid_file = os.path.abspath(damid_file)

if args.hic_sigma is not None:
    sigma = args.hic_sigma

if damid_contact_range is False:
    damid_contact_range = args.damid_contact_range

if hic_contact_range is False:
    hic_contact_range = args.hic_contact_range

if args.violation_tolerance is not None:
    tol = args.violation_tolerance

semiaxes = args.semiaxes
out_folder = args.out_dir
steps = args.steps.split(',')

# do not show interactive graphs
plt.switch_backend('agg')

# Prepare output directory
# ========================
call_dir = os.getcwd()
try:
    create_folder(out_folder)
    os.chdir(out_folder)

    # Summary stats
    # =============

    if 'violations' in steps:
        logger.info('Step: violations')
        try:
            #
            # n_expected_hic_trans = cm.expectedRestraints(cut=sigma, which='inter')
            # n_expected_hic_cis = cm.expectedRestraints(cut=sigma, which='intra')
            # n_expected_hic = n_expected_hic_cis + n_expected_hic_trans

            create_folder('violations')

            # with open('violations/expected_hic_restraints.txt', 'w') as f:
            #     f.write('# total intra inter\n')
            #     f.write('# {} {} {}\n'.format(n_expected_hic, n_expected_hic_cis, n_expected_hic_trans))
            with HssFile(hssfname, 'r') as hss:
                stats = json.loads(hss['summary'][()])
            # save a copy of the data
            with open('violations/stats.json', 'w') as f:
                json.dump(stats, f, indent=4)

            n_rest_types = len(stats['byrestraint'])

            with open('violations/restraints_summary.txt', 'w') as f:
                f.write('# type imposed violated\n')
                f.write('"all" {} {}\n'.format(
                    stats['n_imposed'],
                    stats['n_violations'],
                ))
                for k, ss in stats['byrestraint'].items():
                    f.write('"{}" {} {}\n'.format(
                        k,
                        ss['n_imposed'],
                        ss['n_violations'],
                    ))

            create_folder('violations/histograms')

            h = stats['histogram']['counts']
            edges = stats['histogram']['edges']
            plot_violation_histogram(h, edges, tol, nticks=10, title="all violations", outfile="violations/histograms/summary.pdf")
            for k, v in stats['byrestraint'].items():
                h = v['histogram']['counts']
                plot_violation_histogram(h, edges, tol, nticks=10, title=k,
                                         outfile="violations/histograms/{}.pdf".format(k))

            # TODO: energies and stuff

        except:
            traceback.print_exc()
            logger.error('Error trying to compute violation statistics\n==============================')

    # Compare matrices
    # ================
    if 'hic' in steps and cm:

        logger.info('Step: hic')
        try:
            sigma
            cm = Contactmatrix(cm)
            create_folder("matrix_comparison")
            print(hic_contact_range)
            outmap = get_simulated_hic(hssfname, float(hic_contact_range))
            outmap.save("matrix_comparison/outmap.hcs")

            with HssFile(hssfname, 'r') as hss:
                genome = hss.genome
            corrs_all = []
            corrs_imposed = []
            for c in genome.chroms:
                x1 = cm[c]
                x2 = outmap[c]
                x1d = x1.matrix.toarray()
                x2d = x2.matrix.toarray()

                corrs_all.append(pearsonr(x1d.ravel(), x2d.ravel())[0])

                mask = x1d > sigma
                corrs_imposed.append(pearsonr(x1d[mask].ravel(), x2d[mask].ravel())[0])

                plot_comparison(x1, x2, file='matrix_comparison/{}.pdf'.format(c),
                                labels=['INPUT', 'OUTPUT'], title=c, cmap=red, vmax=0.2)

            with open('matrix_comparison/correlations.txt', 'w') as f:
                f.write('# chrom all imposed\n')
                for c, x, y in zip(genome.chroms, corrs_all, corrs_imposed):
                    f.write('{} {} {}\n'.format(c, x, y))

            # create a scatter plot of probabilities:
            plt.figure(figsize=(10, 10))
            logloghist2d(
                cm.matrix.toarray().ravel(),
                outmap.matrix.toarray().ravel(),
                bins=(100, 100),
                outfile='matrix_comparison/histogram2d.pdf',
                nlevels=10,
            )

        except:
            traceback.print_exc()
            logger.error('Error in matrix comparison step\n==============================')

    # Radius of gyration
    # ==================
    if 'radius_of_gyration' in steps:
        try:

            logger.info('Step: radius_of_gyration')
            create_folder("radius_of_gyration")
            with HssFile(hssfname, 'r') as hss:
                chroms = hss.genome.chroms
                rgs = get_chroms_rgs(hss.coordinates, hss.index)
                np.savez('radius_of_gyration/chromosomes.npz', **{c: arr for c, arr in zip(hss.genome.chroms, rgs)})
            rg_boxplot(rgs, chroms, outfile='radius_of_gyration/rgs.pdf')

        except:
            traceback.print_exc()
            logger.error('Error in radius of gyration step\n==============================')

    # Five shells
    # ===========
    if 'five_shells' in steps:
        try:
            logger.info('Step: five_shells')
            create_folder("five_shells")
            with HssFile(hssfname, 'r') as hss:
                genome = hss.genome
                index = hss.index
                crd = np.swapaxes(hss.coordinates, 0, 1)
                if semiaxes is None:
                    # see if we have information about semiaxes in the file
                    try:
                        semiaxes = hss['envelope']['params'][()]
                        if len(semiaxes.shape) == 0:  # is scalar
                            semiaxes = np.array([semiaxes, semiaxes, semiaxes])
                    except:
                        semiaxes = np.array([5000., 5000., 5000.])

                n = hss.nbead
                kth = [int(k * n) for k in [0.2, 0.4, 0.6, 0.8]]
                bds = [0] + kth + [n]
                ave_shell_rad = np.empty((hss.nstruct, 5))
                for i in range(hss.nstruct):
                    radials = np.sqrt(np.sum(np.square(crd[i] / semiaxes), axis=1))
                    radials = np.partition(radials, kth)
                    for j in range(5):
                        ave_shell_rad[i][j] = np.average(radials[bds[j]:bds[j + 1]])

                np.savetxt('five_shells/ave_radial.txt', np.average(ave_shell_rad, axis=0))

        except:
            traceback.print_exc()
            logger.error('Error in radials step\n==============================')

    # Radial positions
    # ================
    if 'radials' in steps:

        logger.info('Step: radials')
        try:
            create_folder("radials")
            with HssFile(hssfname, 'r') as hss:
                genome = hss.genome
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
            np.savetxt('radials/radials.txt', radials)
            f, p = plot_by_chromosome(radials, index.get_haploid())
            plt.savefig('radials/radials.pdf')

            if os.path.isfile('five_shells/ave_radial.txt'):
                N = np.loadtxt('five_shells/ave_radial.txt')[-1]
                np.savetxt('radials/radials_norm.txt', radials / N)
                f, p = plot_by_chromosome(radials / N, index.get_haploid())
                plt.savefig('radials/radials_norm.pdf')

        except:
            traceback.print_exc()
            logger.error('Error in radials step\n==============================')

    # Density distribution
    # ====================
    if 'radial_density' in steps:

        logger.info('Step: radial_density')
        try:
            create_folder("radial_density")
            with HssFile(hssfname, 'r') as hss:
                genome = hss.genome
                index = hss.index
                crd = hss.coordinates.reshape((hss.nstruct * hss.nbead, 3))
                if semiaxes is None:
                    # see if we have information about semiaxes in the file
                    try:
                        semiaxes = hss['envelope']['params'][()]
                        if len(semiaxes.shape) == 0:  # is scalar
                            semiaxes = np.array([semiaxes, semiaxes, semiaxes])
                    except:
                        semiaxes = np.array([5000., 5000., 5000.])

                radials = np.sqrt(np.sum(np.square(crd / semiaxes), axis=1))
                n = 11
                vmax = 1.1
                counts, edges = np.histogram(radials, bins=n, range=(0, vmax))
                volumes = np.array([edges[i + 1]**3 - edges[i]**3 for i in range(n)])
                plt.figure()
                plt.bar(np.arange(n) + 0.5, height=counts / volumes, width=1)
                plt.xticks(range(n + 1), ['{:2f}'.format(x) for x in edges], rotation=60)
                plt.tight_layout()
                plt.savefig('radial_density/density_histo.pdf')

                np.savetxt('radial_density/density_histo.txt', counts / volumes)

                plt.figure()
                radial_plot_p(edges, counts / volumes)
                plt.savefig('radial_density/circular_plot.pdf')

        except:
            traceback.print_exc()
            logger.error('Error in radials step\n==============================')

    # Damid
    # =====
    if 'damid' in steps:

        logger.info('Step: damid')
        try:
            create_folder("damid")

            with HssFile(hssfname, 'r') as hss:
                genome = hss.genome
                index = hss.index
                radii = hss.radii
                if semiaxes is None:
                    # see if we have information about semiaxes in the file
                    try:
                        semiaxes = hss['envelope']['params'][()]
                        if len(semiaxes.shape) == 0:  # is scalar
                            semiaxes = np.array([semiaxes, semiaxes, semiaxes])
                    except:
                        semiaxes = np.array([5000., 5000., 5000.])

                out_damid_prob = np.zeros(len(index.copy_index))
                for locid in index.copy_index.keys():
                    ii = index.copy_index[locid]
                    n_copies = len(ii)

                    r = radii[ii[0]]

                    # rescale pwish considering the number of copies
                    # pwish = np.clip(pwish/n_copies, 0, 1)

                    d_sq = np.empty(n_copies * hss.nstruct)

                    for i in range(n_copies):
                        x = hss.get_bead_crd(ii[i])
                        R = np.array(semiaxes) * (1 - damid_contact_range)
                        d_sq[i * hss.nstruct:(i + 1) * hss.nstruct] = snormsq_ellipse(x, R, r)

                    contact_count = np.count_nonzero(d_sq >= 1)
                    out_damid_prob[locid] = float(contact_count) / hss.nstruct / n_copies
                np.savetxt('damid/output.txt', out_damid_prob)

            if damid_file:
                damid_profile = np.loadtxt(damid_file, dtype='float32')
                np.savetxt('damid/input.txt', damid_profile)
                plt.figure(figsize=(10, 10))
                plt.title('DamID')
                plt.scatter(damid_profile, out_damid_prob, s=6)
                plt.xlabel('input')
                plt.ylabel('output')
                plt.savefig('damid/scatter.pdf')

        except:
            traceback.print_exc()
            logger.error('Error in DamID step\n==============================')

finally:
    os.chdir(call_dir)

