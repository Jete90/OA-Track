#!/usr/bin/env python
"""
check_alk_trend.py  (exploratory, not part of the chain)
==================
Is the models' recent change of surface alkalinity related to their constrained quantity, the bias-adjusted
1750 Omega_arag? Only a diagnostic: the relationship and its sign, not a new predictor.

Per model (the models of constraint_results.pkl, per reference, common area of the paper):
    trend    linear trend of the area-mean surface ALK over PERIOD (umol/kg per decade), from the drift-removed
             smoothed states of calc_smooth_states.py (forced change; historical continued with the model's
             default scenario after 2014, SSP2-4.5 where it exists); also the unsmoothed annual means
    dA       the change of ALK from the piControl state to the reference window (the dALK of the chain): does the
             recent trend stand for the long-term change?
    y1750    the bias-adjusted 1750 Omega_arag;  resid  y1750 minus its regression on dSST and CUC
Printed: per-model table, correlations (r, p) of the trend with y1750, with the residual and with dA, the slope
of y1750 on the trend (Omega per umol/kg per decade), and the regression on dSST, CUC and the trend (coefficient,
t value, R2 against dSST and CUC alone). For context, the ALK trend of the product over the same period
(obs_<ref>.nc of calc_product_obs.py, if it exists): the products' ALK is mapped largely from salinity, so this is
not an independent observation of the ALK change.

Needs: data/smoothed (P0 of rerun_product.sh, marker COMPLETE), data/<run>/constraint_results.pkl.
Usage:  OA_RUN=u74 python check_alk_trend.py                    # 1985-2024
        OA_RUN=u74 python check_alk_trend.py --period 1983 2014 --reference OceanSODA
"""
from __future__ import annotations

import argparse
import glob
import pickle
import warnings

import netCDF4
import numpy as np
from scipy import stats

import oa_config
import calc_omega_arag_obs as co
import state_constraint as sc

SMOOTH = oa_config.BASE / 'data' / 'smoothed'


def model_alk(model, lat, lon, mask, years):
    """area-mean surface ALK per year on the mask: (smoothed, unsmoothed) arrays over `years`"""
    fs = sorted(glob.glob(str(SMOOTH / f'smoothed_{model}_*.nc')))
    if not fs:
        raise FileNotFoundError(f'no smoothed file for {model} in {SMOOTH}')
    with netCDF4.Dataset(fs[0]) as f:
        cont = f.continuation
        yh = np.asarray(f['year_hist'][:]).astype(int)
        ys = np.asarray(f['year_scen'][:]).astype(int)
        jy = co.nearest_index(np.asarray(f['lat'][:], 'f8'), lat)
        ix = co.nearest_index(np.asarray(f['lon'][:], 'f8'), lon, periodic=True)
        sm, raw = [], []
        for y in years:
            if y <= 2014:
                i = int(np.searchsorted(yh, y)); a = f['talk_hist'][i]; r = f['talk_hist_raw_annual'][i]
            else:
                i = int(np.searchsorted(ys, y)); a = f[f'talk_{cont}'][i]; r = f[f'talk_{cont}_raw_annual'][i]
            a = np.ma.filled(a.astype('f8'), np.nan).mean(axis=0)[jy][:, ix]
            r = np.ma.filled(r.astype('f8'), np.nan)[jy][:, ix]
            sm.append(float(sc.area_mean(a, lat, lon, mask)))
            raw.append(float(sc.area_mean(r, lat, lon, mask)))
    return np.array(sm), np.array(raw), cont


def trend(years, x):
    return 10.0 * np.polyfit(np.asarray(years, 'f8'), x, 1)[0]


def corr(a, b):
    r, p = stats.pearsonr(a, b)
    return f'r {r:+.2f} (p {p:.2f})'


def regress(y, X):
    """OLS of y on X (n, k) with intercept: coefficients, t values, R2"""
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, 'f8') for c in X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ beta
    dof = len(y) - A.shape[1]
    s2 = e @ e / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    r2 = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    return beta, beta / se, r2, e


def product_trend(ref, lat, lon, mask, years):
    p = oa_config.DATA / 'product' / f'obs_{ref}.nc'
    if not p.exists():
        return None
    try:
        f = netCDF4.Dataset(p)
    except OSError:
        print(f'  ({p.name} not readable: still being written by calc_product_obs.py?)')
        return None
    with f:
        yo = np.asarray(f['year'][:]).astype(int)
        sel = [y for y in years if y in yo]
        if len(sel) < 10:
            return None
        jy = co.nearest_index(np.asarray(f['lat'][:], 'f8'), lat)
        ix = co.nearest_index(np.asarray(f['lon'][:], 'f8'), lon, periodic=True)
        x = [float(sc.area_mean(np.ma.filled(f['talk'][int(np.where(yo == y)[0][0])].astype('f8'), np.nan)[jy][:, ix], lat, lon, mask))
             for y in sel]
    return trend(sel, np.array(x)), sel[0], sel[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--period', nargs=2, type=int, default=[1985, 2024])
    ap.add_argument('--reference', nargs='+', default=['OceanSODA', 'UExP', 'JMA', 'CMEMS'])
    a = ap.parse_args()
    if not (SMOOTH / 'COMPLETE').exists():
        raise SystemExit(f'{SMOOTH} not complete (P0 of rerun_product.sh): run that first')
    years = np.arange(a.period[0], a.period[1] + 1)
    with open(oa_config.DATA / 'constraint_results.pkl', 'rb') as f:
        P = pickle.load(f)
    print(f'run {oa_config.RUN}: surface ALK trend {years[0]}-{years[-1]} of the models against their 1750 Omega_arag '
          f'(common area of the paper)')
    cache = {}
    for ref in a.reference:
        if ref not in P['results_common']:
            continue
        R = P['results_common'][ref]
        names = list(R['names'])
        lat, lon = np.asarray(R['lat'], 'f8'), np.asarray(R['lon'], 'f8')
        mask = np.asarray(R['mask']).astype(bool)
        dT, CUC, dA, y = (np.asarray(R[k], 'f8') for k in ('dT', 'CUC', 'dA', 'y1750'))
        tr, trr = [], []
        for m in names:
            key = (m, mask.tobytes())
            if key not in cache:
                s, r, cont = model_alk(m, lat, lon, mask, years)
                cache[key] = (trend(years, s), trend(years, r), cont)
            tr.append(cache[key][0]); trr.append(cache[key][1])
        tr, trr = np.array(tr), np.array(trr)
        _, _, r2_0, res = regress(y, [dT, CUC])
        beta, t, r2_1, _ = regress(y, [dT, CUC, tr])
        print(f'\n===== {ref}: {len(names)} models')
        print(f'  {"model":16s} {"ALK trend":>10s} {"(raw)":>7s} {"dA pi->W":>9s} {"1750":>7s} {"resid":>7s}   umol/kg per decade, umol/kg, Omega')
        for i, m in enumerate(names):
            print(f'  {m:16s} {tr[i]:+10.2f} {trr[i]:+7.2f} {dA[i]:+9.2f} {y[i]:7.3f} {res[i]:+7.3f}   ({cache[(m, mask.tobytes())][2]} after 2014)')
        print(f'  model range of the trend {tr.min():+.2f} to {tr.max():+.2f}, mean {tr.mean():+.2f}, std {tr.std(ddof=1):.2f} umol/kg per decade')
        print(f'  trend vs 1750 Omega_arag:             {corr(tr, y)}, slope {np.polyfit(tr, y, 1)[0]:+.4f} per (umol/kg per decade)')
        print(f'  trend vs residual after dSST and CUC: {corr(tr, res)}')
        print(f'  trend vs dA (piControl -> window):    {corr(tr, dA)}')
        print(f'  dA vs 1750 Omega_arag:                {corr(dA, y)}')
        print(f'  y1750 ~ dSST + CUC + trend: trend coefficient {beta[3]:+.4f} (t {t[3]:+.2f}), R2 {r2_1:.2f} against {r2_0:.2f} '
              f'with dSST and CUC alone')
        pt = product_trend(ref, lat, lon, mask, years)
        if pt:
            print(f'  for context, the product\'s own ALK trend {pt[1]}-{pt[2]}: {pt[0]:+.2f} umol/kg per decade '
                  f'(mapped largely from salinity: not an independent observation)')


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
