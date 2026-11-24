#!/usr/bin/env python
"""
test_cell_constraint_terms.py  (17 November 2026)
=============================
Which of the two predictors moves the cell-by-cell constrained 1750 Omega_arag, and in which direction?
The local constraint is an ordinary least-squares regression across the 18 ESMs in every cell (rc.local_constraint),
so the constrained value minus the multi-model mean splits exactly into
    dY = b_dSST (dSST_obs - mean dSST_models) + b_CUC (CUC_obs - mean CUC_models).
A higher 1750 state means a larger decline and an earlier crossing of the boundary (80 % of the 1750 state).
For each product on its own area (constraint_results.pkl, 'results', as test_cell_crossing.py), area-weighted:
  - the share of the area where the constraint raises the 1750 state (earlier crossing) or lowers it (later),
  - the mean of dY and of the two terms, and the share of the area where the dSST term is the larger one,
in the whole area, on the Arctic shelves and in the Arctic (north of 66N), in the northwest Atlantic (40-56N,
75-45W) and elsewhere, and in the cells with a large change (|dY| >= 0.05, about a 1.5-point change of the decline).
Checks that the two terms add up to the constrained value of the chain (pi_full - pi_mean of rc.load).
Usage (code/):  OA_RUN=u74 python -u test_cell_constraint_terms.py
"""
import pickle

import numpy as np

import oa_config
import regional_common as rc

with open(f'{rc.ROOT}/constraint_results.pkl', 'rb') as f:
    RES = pickle.load(f)['results']
print(f'run {rc.ROOT}; each product on its own area; area weights: {oa_config.WEIGHTS_TEXT}')
BIG = 0.05


def wmean(a, w, m):
    return float(np.sum(a[m] * w[m]) / np.sum(w[m])) if m.any() else np.nan


for ref, r in RES.items():
    F = r['fields']
    lat, lon = np.asarray(r['lat'], 'f8'), np.asarray(r['lon'], 'f8')
    mask = np.asarray(r['mask']).astype(bool)
    Y, DT, CU = (F[k].astype('f8') for k in ('y1750', 'dT', 'CUC'))
    n = Y.shape[0]
    dobs = np.array([F['dsst_obs'][k] for k in F['dsst_obs']], 'f8')
    dTo, dTs = dobs.mean(0), dobs.std(0, ddof=1)
    CUo = F['CUC_obs'].astype('f8')
    CUs_f = rc.cuc_sigma(ref, 'results')[1]
    with __import__('netCDF4').Dataset(rc.OBS[ref]) as f_:
        yo = np.asarray(f_['year'][:])
        om = np.ma.filled(f_['omega_arag'][:].astype('f8'), np.nan)[yo <= rc.LAST_YEAR]
    # the cells of the chain's cell-by-cell constraint (as rc.load: all model values, observations and predictors)
    base = mask & np.all(np.isfinite(Y), 0) & np.all(np.isfinite(om), 0)
    ok = base & np.all(np.isfinite(DT), 0) & np.all(np.isfinite(CU), 0) & np.isfinite(dTo) & np.isfinite(CUo)
    idx = np.where(ok.ravel())[0]
    sel = lambda a: a.reshape(n, -1)[:, idx] if a.ndim == 3 else a.ravel()[idx]
    y, dt, cu = sel(Y), sel(DT), sel(CU)
    yc, st, beta = rc.local_constraint(y, dt, cu, sel(dTo), sel(dTs), sel(CUo), ok, return_beta=True, CUs=sel(CUs_f))
    tT = beta[:, 1] * (sel(dTo) - dt.mean(0))
    tC = beta[:, 2] * (sel(CUo) - cu.mean(0))
    dY = yc - y.mean(0)
    err = np.max(np.abs(dY - (tT + tC)))
    w = oa_config.area_weights(lat, lon).ravel()[idx]
    la = np.broadcast_to(lat[:, None], mask.shape).ravel()[idx]
    lo = ((np.broadcast_to(lon[None, :], mask.shape).ravel()[idx] + 180.0) % 360.0) - 180.0
    regions = {'all': np.ones(idx.size, bool), 'Arctic (N of 66N)': la >= 66,
               'NW Atlantic (40-56N, 75-45W)': (la >= 40) & (la < 56) & (lo >= -75) & (lo <= -45)}
    regions['elsewhere'] = ~(regions['Arctic (N of 66N)'] | regions['NW Atlantic (40-56N, 75-45W)'])
    print(f'\n=== {ref}: {idx.size} cells; dY = dSST term + CUC term to within {err:.1e}; '
          f'dSST observed minus model mean, area mean {wmean(sel(dTo) - dt.mean(0), w, regions["all"]):+.3f} degC, '
          f'CUC {wmean(sel(CUo) - cu.mean(0), w, regions["all"]):+.4f}')
    for name, m in regions.items():
        for lab, mm in (('', m), (f' |dY| >= {BIG}', m & (np.abs(dY) >= BIG))):
            if not mm.any():
                continue
            a = 100 * w[mm].sum() / w.sum()
            up = 100 * w[mm & (dY > 0)].sum() / w[mm].sum()
            domT = 100 * w[mm & (np.abs(tT) > np.abs(tC))].sum() / w[mm].sum()
            print(f'  {name + lab:40s} {a:5.1f} % of the area: 1750 raised (earlier crossing) on {up:3.0f} %, '
                  f'lowered on {100 - up:3.0f} %; mean dY {wmean(dY, w, mm):+.4f} = dSST {wmean(tT, w, mm):+.4f} '
                  f'+ CUC {wmean(tC, w, mm):+.4f}; mean |dSST term| {wmean(np.abs(tT), w, mm):.4f}, '
                  f'|CUC term| {wmean(np.abs(tC), w, mm):.4f}; dSST term larger on {domT:3.0f} %')
