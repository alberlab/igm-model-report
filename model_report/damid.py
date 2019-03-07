import numpy as np
import matplotlib.pyplot as plt
import logging
import traceback
from alabtools import HssFile
from .utils import create_folder, snormsq_ellipse


def report_damid(hssfname, damid_file, contact_range, semiaxes=None):
    logger = logging.getLogger("DamID")
    logger.info('Executing DamID report')
    try:
        create_folder("damid")

        with HssFile(hssfname, 'r') as hss:
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
                    R = np.array(semiaxes) * (1 - contact_range)
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