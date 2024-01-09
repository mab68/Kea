
"""
plot.py

Provides helper functions for plotting in both matplotlib and plotly

Functions:
----------
imshow_cbar\n
get_symmetric_minmax\n
get_fig\n
"""

import numpy as np

import matplotlib as mpl

import seaborn as sns


COLOR_CYCLE_1 = [
    '#377eb8', '#ff7f00', '#4daf4a',
    '#f781bf', '#a65628', '#984ea3',
    '#999999', '#e41a1c', '#dede00'
]
COLOR_CYCLE_2 = [
    '#332288', '#88CCEE', '#117733',
    '#999933', '#DDCC77', '#CC6677',
    '#882255', '#AA4499'
]
COLOR_CYCLE_3 = [
    '#E69F00', '#56B4E9', '#009E73',
    '#0072B2', '#D55E00', '#CC79A7',
    "#F0E442"
]
COLOR_CYCLE_4 = sns.color_palette('tab10')
COLOR_CYCLE_5 = [
    '#0072bd', '#d95319', '#edb120',
    '#7e2f8e', '#77ac30', '#4dbeee',
    '#a2142f'
]
COLOR_CYCLE_6 = [
    '#0072bd',
    '#d95319',
    '#33a02c',
    '#fb9a99',
    '#e31a1c',
    '#fdbf6f',
    '#ff7f00',
    '#33a02c'
]

LINESTYLE_CYCLE_1 = [
    '-', 'dashed', 'dashdot',
    (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10)), ':'
]
LINESTYLE_CYCLE_2 = [
    (0, (5, 5)), ':'
]

IMG_CBARS = {
    'divering': ['RdBu', 'bwr', 'seismic', 'coolwarm'],
    'sequential': ['viridis', 'afmhot']
}

def savefig(fig, filename):
    """savefig(fig, filename)
    
    Save the figure to filename as a pdf, and transparent png
    """
    fig.savefig(filename + '.pdf', transparent=True, bbox_inches='tight', pad_inches=0)
    fig.savefig(filename + '.png', transparent=True, bbox_inches='tight', pad_inches=0)

def modify_rc():
    """modify_rc()

    Modifies the matplotlib rcparams
    """
    mpl.rcParams['figure.dpi'] = 250
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]
    mpl.rc('font', family='serif', serif='cm10', size=8)

def set_logticks(ax, x=False, y=True, numticks=10):
    """set_logticks(ax)
    
    Sets the x/y-axis to have nice ticks for a log format
    """
    locmaj = mpl.ticker.LogLocator(base=10.0, numticks=numticks)
    if x:
        ax.xaxis.set_major_locator(locmaj)
    if y:
        ax.yaxis.set_major_locator(locmaj)
    locmin = mpl.ticker.LogLocator(base=10.0, subs=[i/(10.) for i in range(1,numticks)], numticks=numticks)
    if x:
        ax.xaxis.set_minor_locator(locmin)
    if y:
        ax.yaxis.set_minor_locator(locmin)

def set_format(ax, format='%.1f'):
    """set_format(ax, format)
    
    Sets the string format for the x/y-axis
    """
    ax.set_major_formatter(mpl.ticker.FormatStrFormatter(format))

def make_bad_cmap(cmap, colour='black'):
    """make_bad_cmap(cmap, colour)
    
    Returns a cmap object with bad pixels set to `color`

    Args:
        cmap (str): Name of the cmap
        colour (str): Name of the colour
    Returns:
        matplotlib.cm
    """
    cmap_obj = mpl.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(colour)
    return cmap_obj

def get_minmax(ar, p=0, sym=True):
    """get_minmax(ar)

    Args:
        ar (np.ndarray): Array to find the symmetric min/max
        p (int): Percentile to take
        sym (bool): If true, make it symmetric
    Returns:
        float, float: minimum and maximum value
    """
    if p != 0:
        cmin, cmax = np.nanpercentile(ar, p), np.nanpercentile(ar, 100-p)
    else:
        cmin, cmax = np.nanmin(ar), np.nanmax(ar)
    if sym:
        vmax = np.nanmax([np.abs(cmin), np.abs(cmax)])
        vmin = -vmax
    else:
        vmin, vmax = cmin, cmax
    return vmin, vmax

def get_default_imshow_kwargs():
    """get_default_imshow_kwargs()

    Returns a default set of kwargs
    """
    return {
        'interpolation': 'none',
        'origin': 'lower'
    }
