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

import os
import argparse
import logging
import json
import numpy as np
import os.path as op

from alabtools import *
from igm import Config

# do not show interactive graphs
import matplotlib
matplotlib.use('agg')

from model_report.violations import report_violations
from model_report.radius_of_gyration import report_radius_of_gyration
from model_report.damid import report_damid
from model_report.hic import report_hic
from model_report.radials import report_radials
from model_report.shells import report_shells
from model_report.images import render_structures

# prepare a logger
logging.basicConfig()
logger = logging.getLogger('IGM-Report')


# read options
parser = argparse.ArgumentParser(description='Run population analysis pipeline')

parser.add_argument('hss', help='Hss file for population')
parser.add_argument('-c', '--config', help='Config file for IGM run (infers some parameters)')
parser.add_argument('-l', '--label', help='Run label to be applied in titles and filenames', default='')

parser.add_argument('--hic', help='Input probability matrix for HiC')
parser.add_argument('--hic-sigma', type=float, help='Probability cutoff for HiC')
parser.add_argument('--hic-inter-sigma', type=float, help='Inter-chromosomal probability cutoff for HiC')
parser.add_argument('--hic-intra-sigma', type=float, help='Intra-chromosomal probability cutoff for HiC')
parser.add_argument('--hic-contact-range', type=float, help='Probability cutoff for HiC')

parser.add_argument('--damid', help='Input probability matrix for DamID')
parser.add_argument('--damid-sigma', type=float, help='Probability cutoff for DamID')
parser.add_argument('--damid-contact-range', type=float, help='Probability cutoff for DamID')

parser.add_argument('--violation-tolerance', type=float, help='Violation tolerance')

parser.add_argument('--semiaxes', nargs=3, type=float, help='Specify semiaxes of the envelope')

parser.add_argument('--steps', help='comma separated list of steps to perform. Perform all of the applicable ones by '
                                    'default. '
                    ' Possible values: radius_of_gyration, violations, five_shells, radials, radial_density, damid',
                    default='radius_of_gyration,violations,hic,five_shells,radials,radial_density,damid,images')

parser.add_argument('-o', '--out-dir', help='Output directory')


# we need some parameters info about the file. There are several ways to obtain it: one is to write
# it directly on the file, and I just pushed a commit to do that on igm.
# If we don't find the data on the file itself, we may read it from a configuration file, or from the step
# database.
def get_parameters_from_igm_config(cfg: dict, basedir='.') -> dict:
    report_config = {}
    if cfg.get('restraints/Hi-C', False):
        report_config['hic'] = {}
        fpath = cfg.get('restraints/Hi-C/input_matrix')
        if not op.isabs(fpath):
            fpath = op.abspath(op.join(basedir, fpath))
        report_config['hic']['input_matrix'] = fpath
        report_config['hic']['contact_range'] = cfg.get('restraints/Hi-C/contact_range')
    if cfg.get('restraints/DamID', False):
        report_config['damid'] = {}
        fpath = cfg.get('restraints/DamID/input_profile')
        if not op.isabs(fpath):
            fpath = op.abspath(op.join(basedir, fpath))
        report_config['damid']['input_profile'] = fpath
        report_config['damid']['contact_range'] = cfg.get('restraints/DamID/contact_range')
    report_config['tol'] = cfg.get('optimization/violation_tolerance', 0.05)

    if cfg.get('model/restraints/envelope/nucleus_semiaxes', False):
        report_config['semiaxes'] = np.array(cfg.get('model/restraints/envelope/nucleus_semiaxes'))
    elif cfg.get('model/restraints/envelope/nucleus_radius', False):
        report_config['semiaxes'] = np.array(
            [cfg.get('model/restraints/envelope/nucleus_radius')] * 3
        )
    else:
        report_config['semiaxes'] = np.array([5000.0, 5000.0, 5000.0])

    # check if we have runtime information in this config
    hic_ok = 'hic' not in report_config
    damid_ok = 'damid' not in report_config
    if 'runtime' in cfg:
        if cfg.get('runtime/Hi-C/sigma', False):
            report_config['hic']['inter_sigma'] = cfg.get('runtime/Hi-C/sigma')
            report_config['hic']['intra_sigma'] = cfg.get('runtime/Hi-C/sigma')
            hic_ok = True
        elif cfg.get('runtime/Hi-C/inter_sigma', False):
            report_config['hic']['inter_sigma'] = cfg.get('runtime/Hi-C/inter_sigma')
            report_config['hic']['intra_sigma'] = cfg.get('runtime/Hi-C/intra_sigma')
            hic_ok = True

        if cfg.get('runtime/DamID/sigma', False):
            report_config['damid']['sigma'] = cfg.get('runtime/DamID/sigma')
            damid_ok = True

    if hic_ok and damid_ok:
        return report_config

    # if it fails, try to guess a sigma directly from the config
    if not hic_ok:
        report_config['hic']['inter_sigma'] = None
        report_config['hic']['intra_sigma'] = None

        if cfg.get('restraints/Hi-C/sigma_list', False):  # old style sigma lists
            l1 = cfg.get('restraints/Hi-C/sigma_list')
            if len(l1):
                report_config['hic']['inter_sigma'] = l1[-1]
                report_config['hic']['intra_sigma'] = l1[-1]

        if cfg.get('restraints/Hi-C/inter_sigma_list', False):  # new style
            l1 = cfg.get('restraints/Hi-C/inter_sigma_list', False)
            l2 = cfg.get('restraints/Hi-C/intra_sigma_list', False)
            if len(l1):
                report_config['hic']['inter_sigma'] = l1[-1]
            if len(l2):
                report_config['hic']['intra_sigma'] = l2[-1]

    if not damid_ok:
        if cfg.get('restraints/DamID/sigma_list', False):  # old style sigma lists
            report_config['damid']['sigma'] = cfg.get('restraints/DamID/sigma_list')[-1]
    return report_config


def process_user_args(cfg, args):
    # overwrite arguments if specified from command line

    if args.hic is not None:
        if 'hic' not in cfg:
            cfg['hic'] = {}
        cfg['hic']['input_matrix'] = os.path.abspath(args.hic)

    if args.damid is not None:
        if 'damid' not in cfg:
            cfg['damid'] = {}
        cfg['damid']['input_profile'] = os.path.abspath(args.damid)

    if args.hic_sigma is not None:
        if 'hic' not in cfg:
            logger.warning('HiC sigma specified, but no hic map in config. Ignoring.')
        else:
            cfg['hic']['inter_sigma'] = args.hic_sigma
            cfg['hic']['intra_sigma'] = args.hic_sigma

    if args.hic_inter_sigma is not None:
        if 'hic' not in cfg:
            logger.warning('HiC inter sigma specified, but no hic map in config. Ignoring.')
        else:
            cfg['hic']['inter_sigma'] = args.hic_inter_sigma

    if args.hic_intra_sigma is not None:
        if 'hic' not in cfg:
            logger.warning('HiC intra sigma specified, but no hic map in config. Ignoring.')
        else:
            cfg['hic']['intra_sigma'] = args.hic_intra_sigma

    if args.hic_contact_range is not None:
        if 'hic' not in cfg:
            logger.warning('HiC contact range specified, but no hic map in config. Ignoring.')
        else:
            cfg['hic']['contact_range'] = args.hic_contact_range

    if args.damid_contact_range is not None:
        if 'damid' not in cfg:
            logger.warning('DamID contact range specified, but no DamID restraints in config. Ignoring.')
        else:
            cfg['damid']['contact_range'] = args.hic_contact_range

    if args.violation_tolerance is not None:
        cfg['tol'] = args.violation_tolerance

    if args.semiaxes is not None:
        cfg['semiaxes'] = np.array(args.semiaxes)


if __name__ == '__main__':

    args = parser.parse_args()

    # get the absolute filename
    hssfname = os.path.realpath(args.hss)

    cfg = None

    # if the user explicitly specified a configuration file, we read it

    if args.config:
        logger.info(f'Reading config from {args.config}')
        cfg = get_parameters_from_igm_config(Config(args.config))
    else:
        # else, let's first see if the hss file contains configuration infos

        with HssFile(hssfname) as f:
            if 'config_data' in f:
                d, _ = os.path.split(hssfname)
                logger.info('Reading stored config from HSS file')
                cfg = json.loads(f['config_data'][()])
                cfg = get_parameters_from_igm_config(Config(cfg), basedir=d)

        if cfg is None:
            # no config in the hss, let's try to read from a configuration file
            # we try to look in the hss file directory

            d, f = op.split(hssfname)
            cpath = op.join(d, 'igm-config.json')
            if op.isfile(cpath):
                logger.info(f'Reading config from {cpath}')
                cfg = get_parameters_from_igm_config(Config(cpath), basedir=d)
            else:
                # we do not have any configuration, let's fall back to an empty one
                # and leave to the user to actually specify all the parameters

                logger.warning('No configuration found, using only command line configuration')
                cfg = {}

    # if not specified, set a default output directory in the same of the hss file
    if args.out_dir is None:
        d, f = os.path.split(hssfname)
        b, e = os.path.splitext(f)
        if args.label:
            args.out_dir = os.path.join(d, 'QC_' + args.label)
        else:
            args.out_dir = os.path.join(d, 'QC_' + b)
    logger.info(f'Output will be written to: {args.out_dir}')

    process_user_args(cfg, args)

    logger.info('CONFIGURATION:')
    logger.info('=' * 60)
    cfgrepr = json.dumps(cfg, indent=2, default=lambda x: x.tolist())
    for line in cfgrepr.split('\n'):
        logger.info(line)
    logger.info('=' * 60)

    # get the list of the steps to run
    steps = args.steps.split(',')

    # Prepare output directory
    # ========================
    call_dir = os.getcwd()
    try:
        if not op.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        os.chdir(args.out_dir)

        # Dump label for future automated processing
        # ==========================================
        with open('label.txt', 'w') as f:
            f.write(args.label)

        # Summary stats
        # =============

        if 'violations' in steps:
            report_violations(hssfname, violation_tolerance=cfg['tol'], run_label=args.label)

        # Radius of gyration
        # ==================
        if 'radius_of_gyration' in steps:
            report_radius_of_gyration(hssfname, run_label=args.label)

        # Five shells
        # ===========
        if 'five_shells' in steps:
            report_shells(hssfname, semiaxes=cfg['semiaxes'], run_label=args.label)

        # Radial positions
        # ================
        if 'radials' in steps:
            report_radials(hssfname, semiaxes=cfg['semiaxes'], run_label=args.label)

        # Damid
        # =====
        if 'damid' in steps:
            if 'damid' not in cfg:
                logger.warning('No DamID in configuration, skipping DamID step')
            else:
                report_damid(hssfname,
                             damid_file=cfg['damid']['input_profile'],
                             contact_range=cfg['damid']['contact_range'],
                             semiaxes=cfg['semiaxes'],
                             run_label=args.label)

        # Compare matrices
        # ================
        if 'hic' in steps:
            if 'hic' not in cfg:
                logger.warning('No HiC in configuration, skipping HiC step.')
            else:
                report_hic(hssfname,
                           input_matrix=cfg['hic']['input_matrix'],
                           inter_sigma=cfg['hic']['inter_sigma'],
                           intra_sigma=cfg['hic']['intra_sigma'],
                           contact_range=cfg['hic']['contact_range'],
                           run_label=args.label)

        logger.info('Done.')

        if 'images' in steps:
            render_structures(hssfname)

    finally:
        os.chdir(call_dir)

