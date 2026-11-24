#!/usr/bin/env python
"""
test_alk_before_constraint.py  (does the order of emergent constraint and alkalinity adjustment matter?)
=============================
The chain applies the alkalinity adjustment after the emergent constraint: the constrained 1750 state of each
fCO2-product is lowered by the change of Omega_arag for its missing alkalinity m(1750) x pattern
(make_product_alkalinity.py). Here, on the common area of the paper, the alternative: the same adjustment applied to
the bias-adjusted 1750 state of every ESM before the constraint, then the same regression (dT + CUC, the observed
predictors of the product). The predictors are left unchanged, as the CUC of the window state would need the
alkalinity of the window, which is the observed one.
    delta_m   change of the global mean Omega_arag of 1750 of model m for TA + m(1750) x pattern at constant pCO2
              (make_product_alkalinity.Chem.deltas on the time-mean 1750 state of its xco2 file: T, S, TA of the
              adjusted state at 284.317 ppm, DIC + ddic_1750), area mean on the constraint mask
    after     constrained 1750 + mean of delta_m (the chain: the change on the constrained state, of which the mean of
              delta_m is a close estimate; both printed for the same cells)
    before    the regression on y1750_m + delta_m, evaluated at the observed predictors
Printed per product: mean and spread of delta_m, the regression coefficient of delta_m on the predictors, before - after
in Omega_arag and in percentage points of the decline 1750 -> 2024 (forced 2024 state of the product, unchanged).
Usage (code/):  OA_RUN=u74 OMP_NUM_THREADS=2 python -u test_alk_before_constraint.py 2>&1 | tee ../logs/u74/test_alk_before_constraint.log
"""
import glob
import pickle

import netCDF4
import numpy as np

import oa_config
import make_product_alkalinity as ma

PRODUCTS = ('OceanSODA', 'UExP', 'JMA', 'CMEMS')
LAB = 'dT + CUC'


def fit(X, y, x0):
    """as constraint_refs.fit: least squares, value and fit sigma at x0"""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = res @ res / (X.shape[0] - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta, x0 @ beta, np.sqrt(s2 + x0 @ cov @ x0)


def main():
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    P = ma.Params()
    print(f'weights {oa_config.WEIGHTS}; structural_adjustments.npz {P.created}')
    for ref in PRODUCTS:
        r = pk['results_common'][ref]
        lat, lon, mask = np.asarray(r['lat'], 'f8'), np.asarray(r['lon'], 'f8'), np.asarray(r['mask'], bool)
        w = oa_config.area_weights(lat, lon) * mask
        pattern, _ = ma.load_pattern(lat, lon)
        m1750 = float(P.on(P.m_best[ref], np.array([1750]))[0])
        dta = pattern * m1750
        chem = ma.Chem(lat, lon)
        adj = oa_config.DATA / ('omega_arag_biasadj' + ('' if ref == 'OceanSODA' else f'_{ref}'))
        delta, cells = [], None
        for m in r['names']:
            fx = glob.glob(f'{adj}/piControl/omega_arag_biasadj_xco2_Oyr_{m}_piControl_*.nc')
            assert len(fx) == 1, f'{ref} {m}: {len(fx)} xco2 files'
            with netCDF4.Dataset(fx[0]) as f:
                g = {v: np.ma.filled(f[v][:].astype('f8'), np.nan)
                     for v in ('thetao_mean', 'so_mean', 'talk_mean', 'dissic_mean', 'ddic_1750_mean')}
            D = chem.deltas(g['thetao_mean'], g['so_mean'], g['talk_mean'], g['dissic_mean'] + g['ddic_1750_mean'], dta)
            d = D['omega_arag']
            ok = np.isfinite(d) & mask
            cells = ok if cells is None else cells & ok
            delta.append(d)
        dm = np.array([np.sum(d[cells] * w[cells]) / np.sum(w[cells]) for d in delta])
        dT, CUC, y = np.asarray(r['dT'], 'f8'), np.asarray(r['CUC'], 'f8'), np.asarray(r['y1750'], 'f8')
        X = np.column_stack([np.ones(y.size), dT, CUC])
        x0 = np.array([1.0, float(r['dT_obs'][0]), float(r['CUC_obs'])])
        _, yc, _ = fit(X, y, x0)
        bd, _, _ = fit(X, dm, x0)
        _, yb, _ = fit(X, y + dm, x0)
        ya = yc + dm.mean()
        now = float(r['now'])
        dec = lambda y1750: 100 * (1 - now / y1750)
        frac = 100 * cells.sum() / mask.sum()
        print(f'\n=== {ref}: m(1750) {m1750:+.2f} umol/kg; delta of the 18 ESMs {dm.mean():+.4f} (sd {dm.std(ddof=1):.4f}, '
              f'{dm.min():+.4f} to {dm.max():+.4f}) on {frac:.0f} % of the common-area cells')
        print(f'  regression of delta on the predictors: dT {bd[1]:+.5f} per degC, CUC {bd[2]:+.5f} per unit')
        print(f'  constrained 1750 {yc:.4f}; after (chain) {ya:.4f}; before {yb:.4f}; before - after {yb - ya:+.5f} '
              f'Omega_arag, decline {dec(yb) - dec(ya):+.4f} percentage points (decline after {dec(ya):.2f} %)')
        print(f'  delta at the predictors (before) {yb - yc:+.4f} against the mean of delta {dm.mean():+.4f}')


if __name__ == '__main__':
    main()
