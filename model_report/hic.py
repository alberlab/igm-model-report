import logging
import traceback
from scipy.stats import pearsonr
from alabtools import Contactmatrix, HssFile
from alabtools.analysis import get_simulated_hic

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from alabtools.plots import plot_comparison, red

from .plots import logloghist2d
from .utils import create_folder


def report_hic(hssfname, input_matrix, theta, hic_contact_range):
    logger = logging.getLogger("HiC")
    logger.info('Executing HiC report')
    try:
        cm = Contactmatrix(input_matrix)
        create_folder("matrix_comparison")
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

            mask = x1d > theta
            corrs_imposed.append(pearsonr(x1d[mask].ravel(), x2d[mask].ravel())[0])

            plot_comparison(x1, x2, file='matrix_comparison/{}.pdf'.format(c),
                            labels=['INPUT', 'OUTPUT'], title=c, cmap=red, vmax=0.2)

            x1.matrix.data[x1.matrix.data < theta] = 0
            plot_comparison(x1, x2, file='matrix_comparison/imposed_{}.pdf'.format(c),
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
        logger.error('Error in HiC step\n==============================')
