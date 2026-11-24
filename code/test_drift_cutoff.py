#!/usr/bin/env python
"""
test_drift_cutoff.py  (does the 50-year cutoff of the drift spline matter for the decline of the models?)
====================
The chain removes the piControl drift with an Enting spline of 50-year cutoff (calc_drift.py, drift_tools.CUTOFF):
    X'(t) = X(t) - [a(k(t)) - mean of a over W']        a = s - mean(piControl), W' parallel to the window W
Here the same with other drift estimates, per model and cell, and the decline of Omega_arag on the common area:
    Enting spline with 30, 50 (the chain), 100 and 200 years cutoff; a linear and a quadratic fit over the whole
    piControl; no drift removal at all (how large the drift is in the first place).
States as test_polar_shift_models.annual_states (the smoothed states of calc_smooth_states.py plus the mean of a over
W' = WINDOW of feedback_pattern), with the drift exchanged exactly:
    state_C(y) = state_50(y) + smooth30[a_50 - a_C](y) + mean of (a_C - a_50) over W'
(smooth30: the Enting spline of calc_smooth_states over its series, piControl years in front, historical, the
scenario of the file to 2100; the drift is the same in every month, so the annual mean is exact). The preindustrial
state is the piControl mean with the drift removed, mean(piControl) + mean of a_C over W' (the state the bias-adjusted
piControl averages to, at 284 ppm; the step to 1750 is the same for every drift estimate and left out).
The raw annual means of the piControl are read from the model files (as calc_drift.py, the only slow part); the
50-year spline recomputed from them is compared with the stored drift file (check of the reading and the alignment).
Printed per model: piControl length and how close the year parallel to 2024 is to its end (end effect of the
spline), the decline piControl mean -> 2014 and -> 2024 (%) with the 50-year spline and the change with every other
drift estimate (points); then mean and s.d. across the models, and the same from 1850 -> 2024 (the definition of
test_polar_shift_models). Weights of the run (OA_WEIGHTS), Omega_arag as the chain (mocsy, GLODAP Si and PO4, the
annual-mean state). Only for us, no figures.
Output: data/<run>/structural/drift_cutoff.npz (levels per model, drift estimate and state).
Usage (code/):  OA_RUN=u74 python -u test_drift_cutoff.py --jobs 4 2>&1 | tee ../logs/u74/test_drift_cutoff.log
"""
import argparse
import pickle
import traceback
import warnings
from multiprocessing import Pool

import netCDF4
import numpy as np

import oa_config
import calc_bias_adjusted as cb
import calc_omega_arag_obs as co
import calc_pi_xco2 as cpx
import calc_smooth_states as css
import drift_tools as dt
import test_polar_shift_models as tp
from feedback_pattern import DRIFT_WINDOW as WINDOW

ALTS = ('enting30', 'enting50', 'enting100', 'enting200', 'linear', 'quadratic', 'none')
STATES = ('pi', 1850, 2014, 2024)
VARS = tp.VARS
DEST = oa_config.DATA / 'structural'


def raw_annual(model, member, lat, lon):
    """annual means of the piControl on the model grid {var: (n, ny, nx)} float32 and the piControl years"""
    run = cb.ModelRun(model, 'piControl', member)
    try:
        assert np.allclose(run.lat, lat) and np.allclose(run.lon, lon), f'{model}: piControl grid differs from the drift file'
        jy, ix = np.arange(lat.size), np.arange(lon.size)
        lon2, lat2 = np.meshgrid(lon, lat)
        years = run.years[::run.step][:run.nt // run.step]
        out = {v: np.full((years.size, lat.size, lon.size), np.nan, 'f4') for v in VARS}
        for i0 in range(0, run.nt, cb.CHUNK):
            i1 = min(i0 + cb.CHUNK, run.nt)
            f = run.read(i0, i1, jy, ix, lon2, lat2)
            k0 = i0 // run.step
            for v in VARS:
                out[v][k0:k0 + (i1 - i0) // run.step] = f[v].reshape(-1, run.step, lat.size, lon.size).mean(axis=1)
    finally:
        run.close()
    return out, years


def drift_anomaly(P, alt):
    """a_C = drift estimate minus the mean of the piControl, (n, cells)"""
    n = P.shape[0]
    if alt.startswith('enting'):
        s = dt.smoother(n, float(alt[6:])) @ P
    elif alt in ('linear', 'quadratic'):
        k = np.arange(n, dtype='f8')
        V = np.vander((k - k.mean()) / n, 2 if alt == 'linear' else 3)
        s = V @ np.linalg.lstsq(V, P, rcond=None)[0]
    else:
        return np.zeros_like(P)
    return s - P.mean(axis=0)


def process(model, member, cmask, clat, clon):
    d = dt.Drift(model, member)
    st, lat, lon, cont, cutoff = tp.annual_states(model, member)
    assert np.allclose(d.lat, lat) and np.allclose(d.lon, lon), f'{model}: drift and smoothed grids differ'
    with netCDF4.Dataset(css.out_path(model, member)) as f:
        n_pre = int(f.n_pre)
        yh = np.asarray(f['year_hist'][:]).astype(int)
        ys = np.asarray(f['year_scen'][:]).astype(int)
    raw, pyears = raw_annual(model, member, lat, lon)
    assert np.array_equal(pyears, d.years), f'{model}: piControl years {pyears[[0, -1]]} differ from the drift file {d.years[[0, -1]]}'
    n = pyears.size
    ny, nx = lat.size, lon.size

    # the cells: common area, valid in every state and every piControl year
    jy = np.abs(lat[:, None] - clat[None, :]).argmin(1)
    ix = np.abs(((lon[:, None] - clon[None, :]) + 180) % 360 - 180).argmin(1)
    m_c = cmask[np.ix_(jy, ix)]
    ok = m_c.copy()
    for v in VARS:
        ok &= np.all(np.isfinite(raw[v]), axis=0) & np.all([np.isfinite(st[y][v]) for y in (1850, 2014, 2024)], axis=0)
    cells = np.flatnonzero(ok)

    # the series of calc_smooth_states: piControl years in front, historical, the scenario to its last year
    k_pre = np.arange(d.by - n_pre, d.by)
    y_ser = np.arange(yh[0], (ys[-1] if cont != 'none' else yh[-1]) + 1)
    idx = np.concatenate([d.index('piControl', k_pre), d.index('historical', y_ser)])
    S30 = css.smoother(idx.size, cutoff)
    rows = {y: n_pre + int(np.searchsorted(y_ser, y)) for y in (1850, 2014, 2024)}
    q0, q1 = dt.parallel_years(d.by, d.hist0, *WINDOW)
    k0, k1 = dt.window_slice(d.years, q0, q1)

    si, po4 = co.load_nutrients(lat, lon)
    la = np.broadcast_to(lat[:, None], (ny, nx))
    w = oa_config.area_weights(lat, lon).ravel()[cells]
    sub = lambda a: a.reshape(-1)[cells]
    # per variable and drift estimate only what the states need (keeps the memory at one (n, cells) array per step):
    #   corr[alt][v][y] = smooth30[a_50 - a_C](y) + mean of (a_C - a_50) over W'   for y = 1850, 2014, 2024
    #   corr[alt][v]['pi'] = mean of a_C over W'
    corr = {alt: {} for alt in ALTS}
    mean, check = {}, {}
    for v in VARS:
        P = raw.pop(v).reshape(n, -1)[:, cells].astype('f8')
        mean[v] = P.mean(axis=0)
        a50 = drift_anomaly(P, 'enting50')
        stored = d.anom[v].reshape(n, -1)[:, cells].astype('f8')
        check[v] = float(np.max(np.abs((a50 - stored) @ w / w.sum())))
        del stored
        w50 = a50[k0:k1].mean(axis=0)
        for alt in ALTS:
            aC = a50 if alt == 'enting50' else drift_anomaly(P, alt)
            wC = aC[k0:k1].mean(axis=0)
            diff = (a50 - aC)[idx]                                                # a_50 - a_C along the series
            corr[alt][v] = {y: S30[rows[y]] @ diff + wC - w50 for y in (1850, 2014, 2024)}
            corr[alt][v]['pi'] = wC
            del aC, diff
        del P, a50
    ww = oa_config.area_weights(lat, lon)
    levels = np.full((len(ALTS), len(STATES)), np.nan)
    for i, alt in enumerate(ALTS):
        stc = {y: {} for y in STATES}
        for v in VARS:
            for y in STATES:
                full = np.full(ny * nx, np.nan)
                full[cells] = (mean[v] if y == 'pi' else sub(st[y][v])) + corr[alt][v][y]
                stc[y][v] = full.reshape(ny, nx)
        om = {y: tp.omega(stc[y], si, po4, la) for y in STATES}
        both = np.all([np.isfinite(om[y]) for y in STATES], axis=0) & ok
        levels[i] = [np.sum(om[y][both] * ww[both]) / np.sum(ww[both]) for y in STATES]
    k24 = int(d.by + (2024 - d.hist0))
    meta = dict(n=n, years=f'{pyears[0]}-{pyears[-1]}', to_end=int(pyears[-1] - k24), check=check, cont=cont)
    return model, member, levels, meta


def run_one(args):
    try:
        return process(*args)
    except Exception:
        print(f'[FAILED] {args[0]} {args[1]}\n{traceback.format_exc()}', flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+')
    ap.add_argument('--jobs', type=int, default=1)
    a = ap.parse_args()
    warnings.filterwarnings('ignore', message='Mean of empty slice')
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    r = pk['results_common']['UExP']
    clat, clon, cmask = np.asarray(r['lat'], 'f8'), np.asarray(r['lon'], 'f8'), np.asarray(r['mask'], bool)
    runs = [(m, mem, cmask, clat, clon) for m, mem in cpx.find_runs(a.models)]
    print(f'{len(runs)} models; drift window {WINDOW[0]}-{WINDOW[1]}; weights {oa_config.WEIGHTS}; decline of the '
          f'common-area mean Omega_arag (%), the changes in points against the 50-year spline', flush=True)
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            res = pool.map(run_one, runs, chunksize=1)
    else:
        res = [run_one(x) for x in runs]
    res = [x for x in res if x is not None]
    if not res:
        raise SystemExit('no model')

    ipi, i50, i14, i24 = STATES.index('pi'), ALTS.index('enting50'), STATES.index(2014), STATES.index(2024)
    dec = lambda L, j, i0=ipi: 100 * (1 - L[..., j] / L[..., i0])
    alts = [x for x in ALTS if x != 'enting50']
    print(f'\n{"model":16s} {"piControl":>11s} {"2024 to end":>11s} {"check":>7s} {"pi->2014":>8s} {"pi->2024":>8s}   '
          + ' '.join(f'{x:>15s}' for x in alts))
    print(f'{"":16s} {"years":>11s} {"(years)":>11s} {"DIC":>7s} {"50 yr":>8s} {"50 yr":>8s}   '
          + ' '.join(f'{"d2014 d2024":>15s}' for _ in alts))
    L = np.array([x[2] for x in res])
    for (model, member, lev, meta) in res:
        row = ' '.join(f'{dec(lev[i], i14) - dec(lev[i50], i14):+7.3f} {dec(lev[i], i24) - dec(lev[i50], i24):+7.3f}'
                       for i in (ALTS.index(x) for x in alts))
        print(f'{model:16s} {meta["years"]:>11s} {meta["to_end"]:11d} {meta["check"]["dissic"]:7.1e} '
              f'{dec(lev[i50], i14):8.2f} {dec(lev[i50], i24):8.2f}   {row}', flush=True)
    sd = lambda a: a.std(ddof=1) if a.size > 1 else np.nan
    print(f'\nacross {len(res)} models (points; mean of the models, s.d. across them):')
    for lab, j, i0 in (('pi -> 2014', i14, ipi), ('pi -> 2024', i24, ipi), ('1850 -> 2024', i24, STATES.index(1850))):
        base = dec(L[:, i50], j, i0)
        print(f'  {lab:13s} 50-year spline: decline mean {base.mean():.3f} %, s.d. {sd(base):.3f}')
        for x in alts:
            dd = dec(L[:, ALTS.index(x)], j, i0) - base
            print(f'      {x:10s} change: mean {dd.mean():+.3f}, s.d. {sd(dd):.3f}, largest |change| {np.abs(dd).max():.3f} '
                  f'({res[int(np.abs(dd).argmax())][0]}); multi-model s.d. of the decline {sd(base + dd):.3f}')
    print('check: largest difference of the recomputed 50-year spline from the stored drift (area mean, per variable): '
          + ', '.join(f'{v} {max(x[3]["check"][v] for x in res):.1e}' for v in VARS))
    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / 'drift_cutoff.npz'
    np.savez(out, models=np.array([x[0] for x in res]), members=np.array([x[1] for x in res]), levels=L,
             alts=np.array(ALTS), states=np.array([str(s) for s in STATES]), window=np.array(WINDOW))
    print(f'written {out}')


if __name__ == '__main__':
    main()
