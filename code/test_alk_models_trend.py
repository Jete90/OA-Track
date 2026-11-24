#!/usr/bin/env python
"""
test_alk_models_trend.py
========================
Is the calcification feedback in surface alkalinity (Barrett et al. 2025: +0.072 +- 0.023 umol/kg per year in
global surface TA over about 30 years, measured minus S/T/nutrient-predicted TA) already part of the model spread
of the constraint, and do the observations single out some models?

 1  Per model (raw output, surface, common area of the paper, cos(latitude), drift-corrected with a linear fit to
    the parallel piControl years, 1025 kg m-3): salinity-normalized TA (nTA = TA x 35 / S)
      - linear trend 1990-2020 (historical to 2014, SSP2-4.5 from 2015), umol/kg per year, against Barrett et al.
      - change 1850-1869 -> 2015-2024 (the period closest to 1750 -> 2024 with model output)
 2  Spread: standard deviation of the nTA change across the models x the sensitivity of the decline
    (0.055 points per umol/kg, test_disequilibrium_alk.py): the part of the model spread that TA creates
 3  constraint_results.pkl: structure of the per-model information of the common-area results (printed), so that
    the residuals of the constraint can be compared with the nTA changes in a second step

Usage (code/):  OA_RUN=u74 python -u test_alk_models_trend.py
"""
import pickle

import netCDF4
import numpy as np

import oa_config
import calc_omega_arag as cm
import calc_pi_xco2 as cpx
import drift_tools as dt
from test_alk_diseq_models import common_mask_on, gm, CALC_OMEGA, RHO

BARRETT = (0.072, 0.023)                  # umol/kg per year, 1 sigma
SENS = 0.055                              # points of decline per umol/kg (test C)
TREND = (1990, 2020)
W0, W1 = (1850, 1869), (2015, 2024)


def years_of(f):
    t = f['time']
    u = t.units.lower()
    if u.startswith('years since'):
        y0 = int(u.split('since')[1].strip()[:4])
        return np.floor(np.asarray(t[:], 'f8')).astype(int) + y0
    d = netCDF4.num2date(t[:], t.units, getattr(t, 'calendar', 'standard'))
    return np.array([x.year for x in d])


def surface(path, var):
    with netCDF4.Dataset(path) as f:
        v = f[var]
        a = v[:, 0] if v.ndim == 4 else v[:]
        y = years_of(f)
        lat, lon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
    return y, np.ma.filled(np.ma.asarray(a).astype('f8'), np.nan), lat, lon


def nta_series(model, member, exp):
    p = {v: cm.input_path(v, model, exp, member) for v in ('talk', 'so')}
    if None in p.values():
        return None
    y, ta, lat, lon = surface(p['talk'], 'talk')
    _, so, _, _ = surface(p['so'], 'so')
    m = common_mask_on(lat, lon)
    return y, np.array([gm(ta[i] * 35 / so[i], m, lat) for i in range(y.size)]) * 1e6 / RHO


def part1():
    print(f'\n===== 1  nTA per model: trend {TREND[0]}-{TREND[1]} (umol/kg per year) against Barrett et al. '
          f'{BARRETT[0]} +- {BARRETT[1]}, and change {W0[0]}-{W0[1]} -> {W1[0]}-{W1[1]} (umol/kg), drift-corrected')
    print(f'{"model":16s} {"trend":>7s} {"z":>6s} {"change":>8s} {"drift(chg)":>10s}  calcification')
    rows = {}
    for model, member in cpx.find_runs(None):
        try:
            h = nta_series(model, member, 'historical')
            s = nta_series(model, member, 'ssp245')
            c = nta_series(model, member, 'piControl')
            if h is None or s is None or c is None:
                print(f'{model:16s} files missing (historical {h is not None}, ssp245 {s is not None}, '
                      f'piControl {c is not None})', flush=True)
                continue
            y = np.concatenate([h[0], s[0][s[0] > h[0][-1]]])
            a = np.concatenate([h[1], s[1][s[0] > h[0][-1]]])
            d = dt.Drift(model, member)
            par = d.by + (y - 1850)
            ok = (par >= c[0].min()) & (par <= c[0].max())
            ic = np.array([np.argmin(np.abs(c[0] - p_)) for p_ in par[ok]])
            b = np.polyfit(par[ok], c[1][ic], 1)
            drift = np.polyval(b, par)                          # extrapolated where the control ends
            ac = a - (drift - drift[0])
            it = (y >= TREND[0]) & (y <= TREND[1])
            tr = np.polyfit(y[it], ac[it], 1)[0]
            w0 = (y >= W0[0]) & (y <= W0[1]); w1 = (y >= W1[0]) & (y <= W1[1])
            chg = ac[w1].mean() - ac[w0].mean()
            z = (tr - BARRETT[0]) / BARRETT[1]
            rows[model] = (tr, chg)
            print(f'{model:16s} {tr:+7.3f} {z:+6.1f} {chg:+8.2f} {drift[w1].mean() - drift[w0].mean():+10.2f}  '
                  f'{"Omega-dependent" if model in CALC_OMEGA else ""}{"  (control shorter than needed)" if not ok.all() else ""}',
                  flush=True)
        except Exception as e:
            print(f'{model:16s} failed: {type(e).__name__}: {e}', flush=True)
    return rows


def part2(rows):
    if not rows:
        return
    tr = np.array([v[0] for v in rows.values()]); ch = np.array([v[1] for v in rows.values()])
    fb = np.array([k in CALC_OMEGA for k in rows])
    print(f'\n===== 2  spread ({len(rows)} models)')
    print(f'  nTA change: mean {ch.mean():+.2f}, sd {ch.std(ddof=1):.2f}; feedback models {ch[fb].mean():+.2f}, '
          f'others {ch[~fb].mean():+.2f} umol/kg')
    print(f'  nTA trend:  mean {tr.mean():+.3f}, sd {tr.std(ddof=1):.3f}; feedback models {tr[fb].mean():+.3f}, '
          f'others {tr[~fb].mean():+.3f} umol/kg per year; Barrett {BARRETT[0]} +- {BARRETT[1]}')
    print(f'  spread of the decline from TA alone: {SENS} x {ch.std(ddof=1):.2f} = {SENS * ch.std(ddof=1):.3f} points')
    print(f'  feedback models minus mean: {SENS} x {ch[fb].mean() - ch.mean():+.2f} = '
          f'{-SENS * (ch[fb].mean() - ch.mean()):+.3f} points of decline (negative: smaller decline)')
    inside = np.sum(np.abs(tr - BARRETT[0]) <= 2 * BARRETT[1])
    print(f'  models within 2 sigma of Barrett: {inside} of {len(tr)} '
          f'({", ".join(k for k, v in rows.items() if abs(v[0] - BARRETT[0]) <= 2 * BARRETT[1])})')


def part3():
    print('\n===== 3  constraint_results.pkl, common-area results of UExP: keys, types, shapes')
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    print(f'  top level: {list(pk)}')
    r = pk['results_common']['UExP']

    def show(d, pre='  ', depth=0):
        for k, v in d.items():
            if isinstance(v, dict) and depth < 2:
                print(f'{pre}{k}: dict with {len(v)} keys {list(v)[:8]}')
                show(v, pre + '    ', depth + 1)
            else:
                sh = getattr(v, 'shape', None)
                ln = len(v) if hasattr(v, '__len__') and not isinstance(v, str) else None
                print(f'{pre}{k}: {type(v).__name__} shape {sh} len {ln}'
                      + (f' -> {list(v)[:20]}' if isinstance(v, (list, tuple)) and ln and ln <= 30 else ''))
    show(r)


if __name__ == '__main__':
    part2(part1())
    part3()
