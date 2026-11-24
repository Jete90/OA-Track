#!/usr/bin/env python
"""
test_access_map.py  (why does ACCESS-ESM1-5 give a smaller decline over the whole ocean than on the common area?)
==================
In Extended Data Fig. 5c,d ACCESS-ESM1-5 is the only model whose decline 1850 -> 2024 is smaller over the whole ocean
than on the common area (-0.20 points against +0.09 to +0.34), and its whole-ocean level of 1850 lies only 1.7 %
below the common area (the others 4 to 5 %). Here the maps behind these numbers, with the states of
test_polar_shift_models.py (smoothed, drift-corrected annual-mean states of 1850 and 2024, mocsy, the same cells in
both years), every model put onto the product grid by nearest neighbour.
Printed: per model the coverage of the ocean (share of the ocean of the paper with a value in both years) and the
declines on the common area and outside it; for the model and the mean of the others, per region outside the common
area (Arctic north of 60 N, Southern Ocean south of 50 S, the rest), the area share, the level of 1850 and the decline.
Figure logs/<run>/test_access_map.png: the model, the mean of the other models and their difference, for the decline
1850 -> 2024 (%) and the level of 1850; cells of the ocean without a model value in grey, the common area outlined.
Only for us, no figures of the paper.
Usage (code/):  OA_RUN=u74 python -u test_access_map.py [--model ACCESS-ESM1-5]
"""
import argparse
import pickle
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import oa_config
import calc_omega_arag_obs as co
import calc_pi_xco2 as cpx
import regional_common as rc
import test_polar_shift_models as tp

Y0, Y1 = 1850, 2024


def on_grid(a, la, lo, lat, lon):
    """field a(la, lo) of a model put onto the grid (lat, lon) by nearest neighbour"""
    jy = np.abs(lat[:, None] - la[None, :]).argmin(1)
    ix = np.abs(((lon[:, None] - lo[None, :]) + 180) % 360 - 180).argmin(1)
    return a[np.ix_(jy, ix)]


def amean(a, m, w):
    k = m & np.isfinite(a)
    return float(np.sum(a[k] * w[k]) / np.sum(w[k])) if k.any() else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='ACCESS-ESM1-5')
    a = ap.parse_args()
    warnings.filterwarnings('ignore', message='Mean of empty slice')
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    r = pk['results_common']['UExP']
    lat, lon, cmask = np.asarray(r['lat'], 'f8'), np.asarray(r['lon'], 'f8'), np.asarray(r['mask'], bool)
    ocean = rc.ocean_on(lat, lon).astype(bool)
    w = oa_config.area_weights(lat, lon)
    la2 = np.broadcast_to(lat[:, None], cmask.shape)
    regions = {'Arctic (> 60N)': ocean & ~cmask & (la2 > 60), 'Southern Ocean (< 50S)': ocean & ~cmask & (la2 < -50),
               'rest outside': ocean & ~cmask & (la2 >= -50) & (la2 <= 60)}

    O0, O1 = {}, {}
    print(f'{"model":16s} {"ocean covered":>13s} {"common":>7s} {"outside":>8s} {"ocean":>7s}  (decline {Y0} -> {Y1}, %)')
    for model, member in cpx.find_runs(None):
        try:
            st, mla, mlo, _, _ = tp.annual_states(model, member)
            si, po4 = co.load_nutrients(mla, mlo)
            mla2 = np.broadcast_to(mla[:, None], (mla.size, mlo.size))
            om = {y: tp.omega(st[y], si, po4, mla2) for y in (Y0, Y1)}
        except Exception as e:
            print(f'{model:16s} failed: {type(e).__name__}: {e}', flush=True)
            continue
        both = np.isfinite(om[Y0]) & np.isfinite(om[Y1])
        o0 = on_grid(np.where(both, om[Y0], np.nan), mla, mlo, lat, lon)
        o1 = on_grid(np.where(both, om[Y1], np.nan), mla, mlo, lat, lon)
        o0[~ocean], o1[~ocean] = np.nan, np.nan
        O0[model], O1[model] = o0, o1
        cov = 100 * w[ocean & np.isfinite(o0)].sum() / w[ocean].sum()
        dec = lambda m: 100 * (1 - amean(o1, m, w) / amean(o0, m, w))
        print(f'{model:16s} {cov:12.1f}% {dec(cmask):7.2f} {dec(ocean & ~cmask):8.2f} {dec(ocean):7.2f}', flush=True)
    assert a.model in O0, f'{a.model} not among the models: {list(O0)}'

    others = [m for m in O0 if m != a.model]
    D = {m: 100 * (1 - O1[m] / O0[m]) for m in O0}
    mean_d = np.nanmean([D[m] for m in others], axis=0)
    mean_o = np.nanmean([O0[m] for m in others], axis=0)
    print(f'\nregions outside the common area ({a.model} | mean of the {len(others)} others): share of the ocean, share '
          f'covered by the model, level {Y0} relative to the common area (%), decline {Y0} -> {Y1} (%)')
    for lab, sel in [('common area', ocean & cmask)] + list(regions.items()):
        sh = 100 * w[sel].sum() / w[ocean].sum()
        row = []
        for ms in ([a.model], others):
            cv = np.nanmean([100 * w[sel & np.isfinite(O0[m])].sum() / w[sel].sum() for m in ms])
            lev = np.nanmean([100 * (amean(O0[m], sel, w) / amean(O0[m], ocean & cmask, w) - 1) for m in ms])
            dc = np.nanmean([100 * (1 - amean(O1[m], sel, w) / amean(O0[m], sel, w)) for m in ms])
            row.append(f'covered {cv:5.1f}%, level {lev:+6.1f}%, decline {dc:5.2f}%')
        print(f'  {lab:24s} {sh:5.1f}% of the ocean | ' + ' | '.join(row))

    fig, axs = plt.subplots(2, 3, figsize=(17, 7))
    panels = [(D[a.model], f'{a.model}: decline {Y0}-{Y1} (%)', 'viridis', 0, 35),
              (mean_d, f'mean of {len(others)} other ESMs: decline (%)', 'viridis', 0, 35),
              (D[a.model] - mean_d, f'{a.model} minus mean (points)', 'RdBu_r', -10, 10),
              (O0[a.model], f'{a.model}: Omega_arag {Y0}', 'magma', 0.5, 4.5),
              (mean_o, f'mean of the others: Omega_arag {Y0}', 'magma', 0.5, 4.5),
              (O0[a.model] - mean_o, f'{a.model} minus mean, Omega_arag {Y0}', 'RdBu_r', -0.6, 0.6)]
    gap = ocean & ~np.isfinite(O0[a.model])
    for ax, (fld, title, cm, lo_, hi_) in zip(axs.ravel(), panels):
        rc.map_mesh(ax, lon, lat, np.where(gap, np.nan, fld), cmap=cm, vmin=lo_, vmax=hi_)
        rc.map_mesh(ax, lon, lat, np.where(gap, 1.0, np.nan), cmap='Greys', vmin=0, vmax=2.5)
        rc.map_contour(ax, lon, lat, cmask.astype(float), levels=[0.5], colors='k', linewidths=0.6)
        plt.colorbar(ax.collections[0], ax=ax, shrink=0.7, pad=0.02)
        ax.set_title(title, loc='left', fontsize=9)
    fig.text(0.01, 0.01, f'grey: ocean cells without a value of {a.model}; black line: common area', fontsize=8)
    f = oa_config.BASE / 'logs' / oa_config.RUN / 'test_access_map.png'
    f.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f, dpi=130, bbox_inches='tight')
    print(f'\nwritten {f}')


if __name__ == '__main__':
    main()
