import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import logging
import traceback
import json
from alabtools import HssFile

from .utils import create_folder


def plot_violation_histogram(h, edges, tol=0.05, nticks=20, title='', figsize=(10, 4), outfile=None):
    plt.figure(figsize=figsize)
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
    plt.gca().add_patch(Rectangle((0, 0), width=tol / step, height=vmax, fill=False, hatch='/'))
    plt.ylim(0, vmax)

    plt.bar(xx, height=h, width=1, color='grey')
    plt.xticks(tick_pos, tick_labels, rotation=60)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile)


def report_violations(hssfname, violation_tolerance, run_label=''):
    logger = logging.getLogger('Violations')
    logger.info('Executing violation report...')
    if run_label:
        run_label = '-' + run_label
    try:

        create_folder('violations')

        with HssFile(hssfname, 'r') as hss:
            stats = json.loads(hss['summary'][()])

        # save a copy of the data
        with open(f'violations/stats{run_label}.json', 'w') as f:
            json.dump(stats, f, indent=4)

        with open(f'violations/restraints_summary{run_label}.txt', 'w') as f:
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
        plot_violation_histogram(h, edges, violation_tolerance, nticks=10,
                                 title="Histogram of all Violations",
                                 outfile=f"violations/histograms/summary{run_label}.pdf")
        plt.savefig(f"violations/histograms/summary{run_label}.png")
        for k, v in stats['byrestraint'].items():
            h = v['histogram']['counts']
            plot_violation_histogram(h, edges, violation_tolerance, nticks=10,
                                     title='Histogram of Violations for ' + k,
                                     outfile=f"violations/histograms/{k}{run_label}.pdf")
            plt.savefig(f"violations/histograms/{k}{run_label}.png")

        # TODO: energies and stuff
        logger.info('Done.')

    except KeyboardInterrupt:
        logger.error('User interrupt. Exiting.')
        exit(1)

    except:
        traceback.print_exc()
        logger.error('Error trying to compute violation statistics\n==============================')