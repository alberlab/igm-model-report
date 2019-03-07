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


# get the absolute filename
hssfname = os.path.realpath(args.hss)


# we need some parameters info about the file. There are several ways to obtain it: one is to write
# it directly on the file, and I just pushed a commit to do that on igm.
# If we don't find the data on the file itself, we may read it from a configuration file, or from the step
# database.

def get_parameters_from_igm_config(cfg):
    report_config = {
        'hic': {},
        'damid': {},
    }
    report_config['hic'] = cfg.get('restraints/Hi-C')
    report_config['damid'] = cfg.get('restraints/DamID')
    report_config['tol'] = cfg.get('optimization/violation_tolerance', 0.05)

    # check if we have runtime information in this config
    if 'runtime' in cfg:
        if cfg.get('runtime/Hi-C/sigma', False):
            report_config['hic']['inter_sigma'] = cfg.get('runtime/Hi-C/sigma')
            report_config['hic']['intra_sigma'] = cfg.get('runtime/Hi-C/sigma')

    # try to figure out info from the database file

    # if it fails, try to guess a sigma directly from the config
    if cfg.get('restraints/Hi-C/sigma_list', False):  # old style sigma lists
        report_config['hic']['inter_sigma'] = cfg.get('restraints/Hi-C/sigma_list')[-1]
        report_config['hic']['intra_sigma'] = cfg.get('restraints/Hi-C/sigma_list')[-1]
    if cfg.get('restraints/Hi-C/inter_sigma_list', False):  # new style
        l1 = cfg.get('restraints/Hi-C/inter_sigma_list', False)
        l2 = cfg.get('restraints/Hi-C/intra_sigma_list', False)
        report_config['hic']['sigm'] = cfg.get('restraints/Hi-C/sigma_list')[-1]


# unless otherwise specified, we read details from a config file
if not args.no_config:
    logger.info('Reading config from: %s' % args.config)
    cfg = Config(args.config)
    parm = get_parameters_from_igm_config(cfg)

# if not specified, set a default output directory
if args.out_dir is None:
    d, f = os.path.split(hssfname)
    b, e = os.path.splitext(f)
    args.out_dir = os.path.join(d, 'QC_' + b)

# overwrite arguments if specified from command line
if args.hic is not None:
    report_config['hic']['input_matrix'] = args.hic

if args.damid is not None:
    report_config['damid']['input_profile'] = args.damid




if args.hic_sigma is not None:
    sigma = args.hic_sigma


if cm:
    cm = os.path.abspath(cm)

if damid_file:
    damid_file = os.path.abspath(damid_file)


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
        pass

    # Compare matrices
    # ================
    if 'hic' in steps and cm is not None:


finally:
    os.chdir(call_dir)

