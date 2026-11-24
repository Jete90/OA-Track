#!/usr/bin/env python
"""
test_r2_mechanism.py  (how much of the model spread do CUC and dSST explain: of the 1750 Omega_arag, and of the change
====================   of DIC minus alkalinity?)
For OceanSODA-ETHZ on its constraint area (constraint_results.pkl, results; and results_common): across the 18 ESMs,
the bias-adjusted 1750 Omega_arag (y1750), the observed-period predictors CUC and dSST of every model, and from the
mechanism files (data/mechanism/state_<model>_<member>.nc, calc_mechanism.py) the area mean of the change of DIC minus
alkalinity between the piControl and 1983-2014 (umol/kg). Prints R2 of the two-predictor regressions
  y1750 ~ CUC + dSST,   d(DIC-ALK) ~ CUC + dSST,   y1750 ~ d(DIC-ALK)
and the single-predictor R2. Area weights of the run (oa_config.area_weights).
Usage (code/):  OA_RUN=u74 python -u test_r2_mechanism.py
"""
import glob
import pickle

import netCDF4
import numpy as np

import oa_config


def r2(y, *xs):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in xs])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    return 1.0 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))


with open(oa_config.DATA / 'constraint_results.pkl', 'rb') as f:
    P = pickle.load(f)

for which in ('results', 'results_common'):
    r = P[which]['OceanSODA']
    names = list(r['names'])
    y = np.asarray(r['y1750'], float)
    cuc, dt = np.asarray(r['CUC'], float), np.asarray(r['dT'], float)
    lat, lon = np.asarray(r['lat'], float), np.asarray(r['lon'], float)
    mask = np.asarray(r['mask']).astype(bool)
    w = oa_config.area_weights(lat, lon)
    dda = np.full(len(names), np.nan)
    for i, m in enumerate(names):
        hits = sorted(glob.glob(str(oa_config.DATA / 'mechanism' / f'state_{m}_*.nc')))
        if not hits:
            print(f'  {m}: no mechanism file')
            continue
        with netCDF4.Dataset(hits[0]) as g:
            mlat, mlon = np.asarray(g['lat'][:], float), np.asarray(g['lon'][:], float)
            get = lambda k: np.ma.filled(g[k][:].astype('f8'), np.nan)
            d = (get('dissic_hist') - get('talk_hist')) - (get('dissic_pi') - get('talk_pi'))
        if mlon.max() > 180 and lon.min() < 0:            # mechanism grid 0..360, product grid -180..180
            d = np.roll(d, -int(np.argmin(np.abs(mlon - 180.5))), axis=1)
        ok = mask & np.isfinite(d)
        dda[i] = np.sum(d[ok] * w[ok]) / np.sum(w[ok])
    good = np.isfinite(dda)
    print(f'\n=== OceanSODA, {"own area" if which == "results" else "common area"}: {int(good.sum())} of {len(names)} models')
    print(f'  y1750 ~ CUC + dSST          R2 {r2(y, cuc, dt):.2f}   (CUC alone {r2(y, cuc):.2f}, dSST alone {r2(y, dt):.2f})')
    if good.sum() > 3:
        print(f'  d(DIC-ALK) ~ CUC + dSST     R2 {r2(dda[good], cuc[good], dt[good]):.2f}   '
              f'(CUC alone {r2(dda[good], cuc[good]):.2f}, dSST alone {r2(dda[good], dt[good]):.2f})')
        print(f'  y1750 ~ d(DIC-ALK)          R2 {r2(y[good], dda[good]):.2f}')
        print(f'  d(DIC-ALK) across models: {np.nanmean(dda):.1f} +- {np.nanstd(dda, ddof=1):.1f} umol/kg')
