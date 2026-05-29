"""
plotting.py
Provides useful style functions for matplotlib.

Functions
---------
modify_rc\n
pretty_axes\n
set_logticks\n
make_ax_step\n
"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

def modify_rc():
    """Modifies matplotlib's rcParams to use latex fonts"""
    mpl.rcParams['figure.dpi'] = 250
    mpl.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r'''\usepackage{bm}
\usepackage{xcolor}
\usepackage{mathrsfs, amsmath, amssymb, amsthm, amsxtra}''')
    #mpl.rcParams['text.latex.preamble']=[r"\usepackage{bm}", r"\usepackage{xcolor}"]
    mpl.rc('font', family='serif', serif='Computer Modern', size=8)

def pretty_axes(axx):
    """Makes axes ticks inside and on both sides"""
    axx.yaxis.set_ticks_position('both')
    axx.xaxis.set_ticks_position('both')
    axx.tick_params(axis='y', direction='in')
    axx.tick_params(axis='y', direction='in', which='minor')
    axx.tick_params(axis='x', direction='in')
    axx.tick_params(axis='x', direction='in', which='minor')
    axx.grid(linestyle=':', alpha=0.3, linewidth=0.5, color='gray')

def set_logticks(ax, x=False, y=True, numticks=10):
    """Enforce log ticks to have minor ticks (if possible)"""
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

def make_ax_step(N, step=None):
    """For imshow, make grid ticks evenly spaced"""
    if step is None:
        step = N//4
    tick = np.arange(0, N+1, step)
    label = np.arange(-N//2, N//2+1, step)
    return tick, label
