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


from igm import Config

logging.basicConfig()
logger = logging.getLogger()


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
cm = None
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
        pass

    # Radius of gyration
    # ==================
    if 'radius_of_gyration' in steps:
        pass

    # Five shells
    # ===========
    if 'five_shells' in steps:
        pass

    # Radial positions
    # ================
    if 'radials' in steps:
        pass

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

    # Compare matrices
    # ================
    if 'hic' in steps and cm is not None:

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

finally:
    os.chdir(call_dir)

