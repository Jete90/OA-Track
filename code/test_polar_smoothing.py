#!/usr/bin/env python
"""
test_polar_smoothing.py  (does the whole-ocean shift of the ESMs depend on the smoothing of their states?)
=======================
structural_adjustments.py takes the whole-ocean shift of the 18 ESMs from test_polar_shift_models.py: the decline of
Omega_arag 1850 -> 2024 over the whole ocean minus that on the common area, relative to the decline (rel), scaled
with the decline of the paper (rel x 18.4 % = 0.21 points). The states there are the smoothed states of
calc_smooth_states.py (drift-removed, 30-year Enting spline over the piControl years in front, the historical run and
the scenario, plus the drift window mean). Here the same comparison with other states of the same models, all from
the smoothed files (no model file is read):
    chain        the states as test_polar_shift_models.annual_states (30-year spline with the piControl years in
                 front), read here for every year needed
    spline 10/20/30/50   the stored unsmoothed annual means (<var>_hist_raw_annual, <var>_<scenario>_raw_annual,
                 drift-removed) smoothed here with an Enting spline of that cutoff over 1850 to the end of the scenario
                 (without the piControl years in front, so 30 here against the chain shows that end effect at 1850)
    single year  the unsmoothed annual means of 1850 and 2024 themselves (internal variability not removed)
Every variant gets the same drift window mean (feedback_pattern.DRIFT_WINDOW) as the chain, so the level is the same.
Omega_arag with mocsy as the chain (test_polar_shift_models.omega), the same cells in every variant and year, the
weights of the run. Also the relative shift over two parts of the period, 1850 -> 1950 and 1950 -> 2024: if it is
about the same in both, scaling it to the decline from 1750 (whose first step, 1750 -> 1850, no model simulates)
is justified.
Also (16 Nov 2026) the shift from 1750 itself: the 1750 state of every variant from its 1850 state with surface pCO2
lowered from 284.317 to 278.3 ppm at constant alkalinity, temperature and salinity (calc_pi_xco2.target_states, as the
1750 states of the chain), so that rel 1750 -> 2024 needs no scaling of 1850 -> 2024 to 1750.
Printed per model and as mean and s.d. over the models: rel (% of the decline) and rel x DECLINE (points).
Only for us, no figures.
Usage (code/):  OA_RUN=u74 python -u test_polar_smoothing.py 2>&1 | tee ../logs/u74/test_polar_smoothing.log
"""
import pickle
import warnings

import netCDF4
import numpy as np

import oa_config
import calc_omega_arag_obs as co
import calc_pi_xco2 as cpx
import calc_smooth_states as css
import drift_tools as dt
import regional_common as rc
import test_polar_shift_models as tp
from feedback_pattern import DRIFT_WINDOW as WINDOW

YEARS = (1850, 1950, 2014, 2024)
CUTOFFS = (10.0, 20.0, 30.0, 50.0)
VARIANTS = ('chain',) + tuple(f'spline {c:g}' for c in CUTOFFS) + ('single year',)
X1750 = 278.3              # ppm, AR6 (Gulev et al. 2021), the 1750 state of the paper
DECLINE = 18.37            # %, the decline of the paper on the common area (mean of the four products), for the points
VARS = tp.VARS


def states(model, member):
    """{variant: {year: {var: (ny, nx)}}}, lat, lon"""
    d = dt.Drift(model, member)
    out = {v_: {y: {} for y in YEARS} for v_ in VARIANTS}
    with netCDF4.Dataset(css.out_path(model, member)) as f:
        lat, lon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        cont = str(f.continuation)
        yh = np.asarray(f['year_hist'][:]).astype(int)
        ys = np.asarray(f['year_scen'][:]).astype(int)
        for v in VARS:
            off = d.window_mean(v, WINDOW)
            for y in YEARS:                     # the chain: as test_polar_shift_models.annual_states, any year
                if y in yh:
                    a = f[f'{v}_hist'][int(np.searchsorted(yh, y))]
                elif cont != 'none' and y in ys:
                    a = f[f'{v}_{cont}'][int(np.searchsorted(ys, y))]
                else:
                    raise ValueError(f'{model}: no {y} in the smoothed file (continuation {cont})')
                out['chain'][y][v] = np.ma.filled(a, np.nan).astype('f8').mean(axis=0) + off
            raw_h = np.ma.filled(f[f'{v}_hist_raw_annual'][:].astype('f8'), np.nan)
            if cont != 'none':
                raw_s = np.ma.filled(f[f'{v}_{cont}_raw_annual'][:].astype('f8'), np.nan)
                ks = ys > yh[-1]
                years = np.concatenate([yh, ys[ks]])
                raw = np.concatenate([raw_h, raw_s[ks]], axis=0)
            else:
                years, raw = yh, raw_h
            assert np.all(np.diff(years) == 1), f'{model}: years not continuous'
            have = np.isfinite(raw.reshape(years.size, -1)).any(axis=1)   # trailing empty years (runs ending in 2099)
            last = int(np.nonzero(have)[0][-1]) + 1
            years, raw = years[:last], raw[:last]
            idx = {y: int(np.searchsorted(years, y)) for y in YEARS}
            n = years.size
            flat = raw.reshape(n, -1)
            ok = np.all(np.isfinite(flat), axis=0)
            for c in CUTOFFS:
                S = dt.smoother(n, c)
                rows = S[[idx[y] for y in YEARS]]                  # only the rows of the years needed
                sm = np.full((len(YEARS), flat.shape[1]), np.nan)
                sm[:, ok] = rows @ flat[:, ok]
                for i, y in enumerate(YEARS):
                    out[f'spline {c:g}'][y][v] = sm[i].reshape(lat.size, lon.size) + off
            for y in YEARS:
                out['single year'][y][v] = raw[idx[y]] + off
    return out, lat, lon


def main():
    warnings.filterwarnings('ignore', message='Mean of empty slice')
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    rcm = pk['results_common']['UExP']
    clat, clon, cmask = np.asarray(rcm['lat'], 'f8'), np.asarray(rcm['lon'], 'f8'), np.asarray(rcm['mask'], bool)
    print(f'weights {oa_config.WEIGHTS}; drift window {WINDOW}; scaled with the decline of the paper {DECLINE} %')
    R = {v_: [] for v_ in VARIANTS}            # per model: (rel 1850-2024, rel 1850-1950, rel 1950-2024, points)
    names = []
    for model, member in cpx.find_runs(None):
        try:
            S, lat, lon = states(model, member)
            jy = np.abs(lat[:, None] - clat[None, :]).argmin(1)
            ix = np.abs(((lon[:, None] - clon[None, :]) + 180) % 360 - 180).argmin(1)
            m_c = cmask[np.ix_(jy, ix)]
            m_o = rc.ocean_on(lat, lon).astype(bool)
            si, po4 = co.load_nutrients(lat, lon)
            la = np.broadcast_to(lat[:, None], m_c.shape)
            w = oa_config.area_weights(lat, lon)
            om = {v_: {y: tp.omega(S[v_][y], si, po4, la) for y in YEARS} for v_ in VARIANTS}
            for v_ in VARIANTS:                    # 1750: the 1850 state at 278.3 ppm (pCO2 lowered, TA, T, S constant)
                fields = {v: S[v_][1850][v][None] for v in VARS}
                _, res = next(cpx.target_states(fields, si, po4, la, {'1750': X1750}, [0.0]))
                om[v_][1750] = res['omega_arag'][0]
            YRS = (1750,) + YEARS
            same = np.all([np.isfinite(om[v_][y]) for v_ in VARIANTS for y in YRS], axis=0)
            am = lambda a, m: float(np.sum(a[m & same] * w[m & same]) / np.sum(w[m & same]))
            row = []
            for v_ in VARIANTS:
                L = {(mk, y): am(om[v_][y], m) for mk, m in (('c', m_c), ('o', m_o)) for y in YRS}
                dec = lambda mk, y0, y1: 100 * (1 - L[(mk, y1)] / L[(mk, y0)])
                rel = lambda y0, y1: (dec('o', y0, y1) - dec('c', y0, y1)) / dec('c', y0, y1)
                r = (rel(1850, 2024), rel(1850, 1950), rel(1950, 2024), rel(1750, 2024))
                R[v_].append(r)
                row.append(f'{100 * r[0]:+5.2f}')
        except Exception as e:
            print(f'{model:16s} failed: {type(e).__name__}: {e}', flush=True)
            continue
        names.append(model)
        print(f'{model:16s} rel 1850-2024 (%): ' + ', '.join(f'{v_} {x}' for v_, x in zip(VARIANTS, row)), flush=True)
        print(f'{"":16s} rel 1750-2024 (%): ' + ', '.join(f'{v_} {100 * R[v_][-1][3]:+5.2f}' for v_ in VARIANTS), flush=True)
    if not names:
        raise SystemExit('no model')
    sd = lambda a: a.std(ddof=1) if a.size > 1 else np.nan
    print(f'\nacross {len(names)} models: relative shift of the decline, whole ocean minus common area (% of the decline; '
          f'points at a decline of {DECLINE} %), mean +- s.d.')
    print(f'  {"variant":12s} {"1850-2024":>17s} {"points":>15s} {"1850-1950":>17s} {"1950-2024":>17s} {"1750-2024":>17s} {"points":>15s}')
    for v_ in VARIANTS:
        a = 100 * np.array(R[v_])
        print(f'  {v_:12s} {a[:, 0].mean():+7.2f} +- {sd(a[:, 0]):5.2f} '
              f'{a[:, 0].mean() * DECLINE / 100:+7.3f} +- {sd(a[:, 0]) * DECLINE / 100:5.3f} '
              f'{a[:, 1].mean():+7.2f} +- {sd(a[:, 1]):5.2f} {a[:, 2].mean():+7.2f} +- {sd(a[:, 2]):5.2f} '
              f'{a[:, 3].mean():+7.2f} +- {sd(a[:, 3]):5.2f} {a[:, 3].mean() * DECLINE / 100:+7.3f} +- {sd(a[:, 3]) * DECLINE / 100:5.3f}')
    c = 100 * np.array(R['chain'])[:, 0]
    for v_ in VARIANTS[1:]:
        dd = 100 * np.array(R[v_])[:, 0] - c
        print(f'  {v_:12s} minus chain: mean {dd.mean() * DECLINE / 100:+.3f} points, largest |.| '
              f'{np.abs(dd).max() * DECLINE / 100:.3f} ({names[int(np.abs(dd).argmax())]})')


if __name__ == '__main__':
    main()
