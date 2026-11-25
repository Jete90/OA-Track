#!/usr/bin/env python
"""
fig_box_schematic.py  (the schematic of Box 1: the method and what can go wrong)
====================
a  the method: observed state of an fCO2-product minus the change simulated by each ESM gives one preindustrial state
   per ESM; the emergent constraint (observed dSST and CUC) selects the constrained 1750 state; both states on the
   same area with the same equilibrium constants, so that an error of the constants or of the observational
   baseline shifts both alike and cancels in the decline.
b  what can go wrong: four inconsistencies between the two states, each with the shift of the decline found for the
   Planetary Health Check (main text, Fig. 3c in nine steps): area +0.2 (the 5 % of cells only the 2024
   state covers), observational baseline +0.8, constants -0.3 (net, in the other direction), CO2 of 1750 (272 instead
   of 278.3 ppm) +1.0 percentage points. Bar heights are schematic, the
   shifts are exaggerated so that they can be seen; the numbers give the real shifts.
No data is read. Usage:  python fig_box_schematic.py [--out DIR]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'font.size': 7, 'axes.linewidth': 0.6, 'pdf.fonttype': 42})
C_OBS, C_ESM, C_CON, C_PI, C_NOW, C_BAD = '#2b6cb0', '#8a8f98', '#c05621', '#2f855a', '#2b6cb0', '#c53030'
C_BOX, C_BG, C_PROD, C_PRE = '#1d4f91', '#e9eff7', '#2b7f8e', '#a68fc4'   # panel a: 2024 box, background,
# the fCO2-product (teal), the 18 preindustrial states (purple)
O17, D = 3.60, 18.3                    # schematic 1750 value and decline (%)
EXAG = 6.0                             # exaggeration of the shifts in panel b


def box(ax, x, y, w, h, text, fc, ec=None, tc='white', fs=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.02', fc=fc,
                                ec=ec or fc, lw=0.6, transform=ax.transAxes, clip_on=False, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', color=tc, fontsize=fs,
            fontweight='bold' if bold else 'normal', transform=ax.transAxes, linespacing=1.25, zorder=3)


def arrow(ax, p0, p1, color='k', lw=0.8, style='-|>', ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=7, color=color, lw=lw, ls=ls,
                                 transform=ax.transAxes, shrinkA=0, shrinkB=0, clip_on=False, zorder=3))


def panel_a(ax):
    ax.set_axis_off()
    ax.text(0.0, 1.0, 'a', fontsize=9, fontweight='bold', va='top', transform=ax.transAxes)
    ax.text(0.03, 1.0, 'One consistent calculation for each fCO$_2$-product', fontsize=7.5, va='top',
            transform=ax.transAxes)
    # background of the calculation
    ax.add_patch(FancyBboxPatch((-0.012, 0.0), 1.024, 0.90, boxstyle='round,pad=0.0,rounding_size=0.04', fc=C_BG,
                                ec='none', transform=ax.transAxes, clip_on=False, zorder=0))
    # four columns; arrows run between the padded box edges (PAD) with a small gap, never into a box
    W, X, PAD, G = 0.19, (0.02, 0.28, 0.54, 0.80), 0.008, 0.012
    yt, ht, yb, hb = 0.60, 0.24, 0.06, 0.30                 # rows: top boxes, bottom boxes
    mt, mb = yt + ht / 2, yb + hb / 2
    left = lambda k: X[k] - PAD - G
    right = lambda k: X[k] + W + PAD + G
    # top row: the observed state is the present-day state
    box(ax, X[0], yt, W, ht, 'fCO$_2$-product\nobserved state\n(fCO$_2$, TA, T, S)', C_PROD, fs=6.6)
    box(ax, X[3], yt + 0.02, W, ht - 0.04, 'observed\n2024 state', C_BOX, bold=True)
    arrow(ax, (right(0), mt), (left(3), mt))
    ax.text((right(0) + left(3)) / 2, mt + 0.03, 'trend fit through the record of each cell', ha='center', va='bottom',
            fontsize=6.5, color=C_PROD, transform=ax.transAxes)
    # bottom row: the preindustrial state
    box(ax, X[0], yb, W, hb, '18 ESMs\nsimulated change\nfrom 1750 to the\nobserved years', C_ESM, fs=6.6)
    box(ax, X[1], yb, W, hb, 'observed state\nminus change:\n18 preindustrial\nstates', C_PRE, fs=6.6)
    box(ax, X[2], yb, W, hb, 'emergent\nconstraint\n(observed ΔSST\nand CUC)', C_CON, fs=6.6)
    box(ax, X[3], yb + 0.05, W, hb - 0.10, 'constrained\n1750 state', C_PI, bold=True)
    for k in range(3):
        arrow(ax, (right(k), mb), (left(k + 1), mb))
    # from the fCO2-product down to the preindustrial states: below the one box, above the other
    arrow(ax, (X[0] + 0.62 * W, yt - PAD - 0.03), (X[1] + 0.38 * W, yb + hb + PAD + 0.04))
    # both states treated alike
    ax.text(X[3] + W / 2, 0.465, 'same area\nsame constants\nsame baseline', ha='center', va='center', fontsize=6.5,
            transform=ax.transAxes, style='italic', linespacing=1.3)


def panel_a2(ax):
    """an error of the constants or the baseline shifts both states alike"""
    o24 = O17 * (1 - D / 100)
    f = 0.93                                   # schematic shift of both states (exaggerated)
    x = [0, 0.55, 1.6, 2.15]
    h = [O17, o24, O17 * f, o24 * f]
    cols = [C_PI, C_NOW, C_PI, C_NOW]
    ax.bar(x, h, width=0.5, color=cols, alpha=0.95)
    ax.bar(x[2:], h[:2], width=0.5, fill=False, ec='k', lw=0.5, ls=':')
    for xi, lab in zip(x, ['1750', '2024', '1750', '2024']):
        ax.text(xi, 0.08, lab, ha='center', va='bottom', color='white', fontsize=6, rotation=90)
    ax.text(0.275, O17 + 0.15, f'{D:.1f}%', ha='center', fontsize=6.5)
    ax.text(1.875, O17 + 0.15, f'{D:.1f}%', ha='center', fontsize=6.5)
    ax.set_xlim(-0.4, 2.55); ax.set_ylim(0, 4.3)
    ax.set_xticks([0.275, 1.875]); ax.set_xticklabels(['reference', 'other constants\nor baseline'], fontsize=6)
    ax.set_yticks([]); ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_title('decline = 1 − Ω$_{2024}$ / Ω$_{1750}$', fontsize=7, fontweight='bold')
    ax.set_xlabel('an error shifts both states\nand cancels in the decline', fontsize=6, labelpad=3)


def panel_b(fig, gs):
    o24 = O17 * (1 - D / 100)
    cases = [  # title, which bar is wrong, real shift of the decline (points), sign of the bar shift
        ('Different areas', '2024 on a larger area\nwith polar cells', '2024', +0.2),
        ('Different baselines', '1750 from another\nobserved state', '1750', +0.8),
        ('Different constants', '1750 and 2024 with\nother constants', '2024', -0.3),
        ('CO$_2$ of 1750', '272 instead of\n278.3 ppm', '1750', +1.0)]
    axes = []
    for k, (title, sub, which, dd) in enumerate(cases):
        ax = fig.add_subplot(gs[k])
        axes.append(ax)
        dex = EXAG * dd                                   # exaggerated shift of the decline
        if which == '2024':
            p17, p24 = O17, O17 * (1 - (D + dex) / 100)
        else:
            p17, p24 = o24 / (1 - (D + dex) / 100), o24
        x = [0, 0.55, 1.5, 2.05]
        ax.bar(x[:2], [O17, o24], width=0.5, color=[C_PI, C_NOW])
        ax.bar(x[2:], [p17, p24], width=0.5, color=[C_PI, C_NOW], alpha=0.95)
        j = 2 if which == '1750' else 3
        ax.bar(x[j], [p17, p24][j - 2], width=0.5, fill=False, ec=C_BAD, lw=1.0)
        ax.plot([x[j] - 0.25, x[j] + 0.25], [[O17, o24][j - 2]] * 2, color='k', lw=0.6, ls=':')
        ax.text(0.275, 4.55, f'{D:.1f}%', ha='center', fontsize=6.5)
        ax.text(1.775, 4.55, f'{D + dd:.1f}%', ha='center', fontsize=6.5, color=C_BAD, fontweight='bold')
        ax.text(1.775, 4.32, f'({dd:+.1f} points)', ha='center', fontsize=6, color=C_BAD)
        ax.set_xlim(-0.4, 2.45); ax.set_ylim(2.2, 4.8)
        ax.set_xticks([0.275, 1.775]); ax.set_xticklabels(['consistent', 'inconsistent'], fontsize=6)
        ax.set_yticks([]); ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.text(0.5, 1.30, title, transform=ax.transAxes, ha='center', va='bottom', fontsize=7, fontweight='bold')
        ax.text(0.5, 1.03, sub, transform=ax.transAxes, ha='center', va='bottom', fontsize=6)
    axes[0].text(-0.12, 1.58, 'b', transform=axes[0].transAxes, fontsize=9, fontweight='bold', va='top')
    axes[0].text(-0.02, 1.58, 'What goes wrong when the two states come from different calculations',
                 transform=axes[0].transAxes, fontsize=7.5, va='top')
    return axes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    fig = plt.figure(figsize=(7.2, 5.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.9, left=0.03, right=0.98, top=0.95,
                             bottom=0.07)
    top = outer[0].subgridspec(1, 2, width_ratios=[4.2, 1.0], wspace=0.08)
    panel_a(fig.add_subplot(top[0]))
    panel_a2(fig.add_subplot(top[1]))
    panel_b(fig, outer[1].subgridspec(1, 4, wspace=0.35))
    out = Path(a.out)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'Box1_schematic_draft.{ext}', dpi=300)
    print('written', out / 'Box1_schematic_draft.png')


if __name__ == '__main__':
    main()
