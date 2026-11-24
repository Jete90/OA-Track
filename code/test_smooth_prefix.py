#!/usr/bin/env python
"""
test_smooth_prefix.py  (how large is the end effect of the 30-year smoothing at 1850?)
=====================
calc_smooth_states.py smooths every model's drift-removed state, X(t) - a(k(t)), with a 30-year Enting spline over the
historical run continued with the scenario, with up to N_PRE = 30 control years in front of 1850 where the control run
has years before the branch point. For most models it has none (test_polar_smoothing.py: the chain equals the spline
without years in front), so the smoothed series starts in 1850 and has an end effect there.

Here, per model, from the stored unsmoothed annual means of the smoothed files (<var>_hist_raw_annual and the
continuation, drift-removed) and the control run (read from the model files, drift-removed with the stored spline):
    A  the 30-year spline without years in front (what the chain does for most models)
    B  the same with PREFIX drift-removed control years in front (the years before the branch point where the control
       run has them, otherwise its first PREFIX years): the unforced state of the model at the CO2 of 1850, which the
       historical run starts from
Printed per model and as mean, s.d. and largest absolute value over the models, for Omega_arag (annual-mean state,
mocsy as the chain, drift window mean added as test_polar_shift_models): B minus A of the common-area mean in 1850,
1860, 1870, 1880 and 1900, of the decline 1850 -> 2024 on the common area (points) and of the relative whole-ocean
shift 1850 -> 2024 (% of the decline); and the root mean square over the common-area cells of B minus A in 1850.
Only for us, no figures.
Usage (code/):  OA_RUN=u74 python -u test_smooth_prefix.py --jobs 3 2>&1 | tee ../logs/u74/test_smooth_prefix.log
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
import regional_common as rc
import test_polar_shift_models as tp
from feedback_pattern import DRIFT_WINDOW as WINDOW

PREFIX = 50                          # control years in front of 1850 in variant B
CUTOFF = 30.0                        # the cutoff of calc_smooth_states.py
YEARS = (1850, 1860, 1870, 1880, 1900, 2024)
VARS = tp.VARS


def control_years(model, member, d, n):
    """drift-removed annual means of n control years {var: (n, ny, nx)}: those before the branch point where the
    control run has them, otherwise its first n years; and which"""
    run = cb.ModelRun(model, 'piControl', member)
    try:
        years = run.years[::run.step][:run.nt // run.step]
        before = np.nonzero(years < d.by)[0]
        idx = before[-n:] if before.size >= n else np.arange(min(n, years.size))
        lab = (f'{years[idx[0]]}-{years[idx[-1]]} ' + ('(before the branch)' if before.size >= n else
                                                     f'(first years; {before.size} before the branch)'))
        jy, ix = np.arange(run.lat.size), np.arange(run.lon.size)
        lon2, lat2 = np.meshgrid(run.lon, run.lat)
        out = {v: np.full((idx.size, run.lat.size, run.lon.size), np.nan) for v in VARS}
        for j, k in enumerate(idx):
            f = run.read(int(k) * run.step, (int(k) + 1) * run.step, jy, ix, lon2, lat2)
            for v in VARS:
                out[v][j] = f[v].mean(axis=0) - d.anom[v][int(np.searchsorted(d.years, years[k]))]
    finally:
        run.close()
    return out, lab


def process(model, member, cmask, clat, clon):
    d = dt.Drift(model, member)
    with netCDF4.Dataset(css.out_path(model, member)) as f:
        lat, lon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        cont = str(f.continuation)
        n_pre = int(f.n_pre)
        yh = np.asarray(f['year_hist'][:]).astype(int)
        ys = np.asarray(f['year_scen'][:]).astype(int)
        raw = {}
        for v in VARS:
            r_h = np.ma.filled(f[f'{v}_hist_raw_annual'][:].astype('f8'), np.nan)
            if cont != 'none':
                r_s = np.ma.filled(f[f'{v}_{cont}_raw_annual'][:].astype('f8'), np.nan)
                k = ys > yh[-1]
                raw[v] = np.concatenate([r_h, r_s[k]], axis=0)
                years = np.concatenate([yh, ys[k]])
            else:
                raw[v], years = r_h, yh
    have = np.all([np.isfinite(raw[v].reshape(years.size, -1)).any(axis=1) for v in VARS], axis=0)
    last = int(np.nonzero(have)[0][-1]) + 1
    years = years[:last]
    raw = {v: raw[v][:last] for v in VARS}
    ctl, lab = control_years(model, member, d, PREFIX)

    ny, nx = lat.size, lon.size
    rows = {y: int(np.searchsorted(years, y)) for y in YEARS}
    states = {'A': {y: {} for y in YEARS}, 'B': {y: {} for y in YEARS}}
    for v in VARS:
        off = d.window_mean(v, WINDOW)
        n = years.size
        flat = raw[v].reshape(n, -1)
        pre = ctl[v].reshape(ctl[v].shape[0], -1)
        ok = np.all(np.isfinite(flat), axis=0) & np.all(np.isfinite(pre), axis=0)
        SA = dt.smoother(n, CUTOFF)[[rows[y] for y in YEARS]]
        m = pre.shape[0]
        SB = dt.smoother(n + m, CUTOFF)[[m + rows[y] for y in YEARS]]
        a = np.full((len(YEARS), flat.shape[1]), np.nan)
        b = np.full((len(YEARS), flat.shape[1]), np.nan)
        a[:, ok] = SA @ flat[:, ok]
        b[:, ok] = SB @ np.concatenate([pre[:, ok], flat[:, ok]], axis=0)
        for i, y in enumerate(YEARS):
            states['A'][y][v] = a[i].reshape(ny, nx) + off
            states['B'][y][v] = b[i].reshape(ny, nx) + off

    jy = np.abs(lat[:, None] - clat[None, :]).argmin(1)
    ix = np.abs(((lon[:, None] - clon[None, :]) + 180) % 360 - 180).argmin(1)
    m_c = cmask[np.ix_(jy, ix)]
    m_o = rc.ocean_on(lat, lon).astype(bool)
    si, po4 = co.load_nutrients(lat, lon)
    la = np.broadcast_to(lat[:, None], (ny, nx))
    w = oa_config.area_weights(lat, lon)
    om = {s: {y: tp.omega(states[s][y], si, po4, la) for y in YEARS} for s in states}
    same = np.all([np.isfinite(om[s][y]) for s in om for y in YEARS], axis=0)
    am = lambda a_, m_: float(np.sum(a_[m_ & same] * w[m_ & same]) / np.sum(w[m_ & same]))
    res = {}
    for s in om:
        L = {(mk, y): am(om[s][y], mm) for mk, mm in (('c', m_c), ('o', m_o)) for y in YEARS}
        dc = 100 * (1 - L[('c', 2024)] / L[('c', 1850)])
        do = 100 * (1 - L[('o', 2024)] / L[('o', 1850)])
        res[s] = dict(L=L, dc=dc, rel=(do - dc) / dc)
    cc = m_c & same
    rms = float(np.sqrt(np.sum(((om['B'][1850] - om['A'][1850]) ** 2)[cc] * w[cc]) / np.sum(w[cc])))
    return dict(model=model, n_pre=n_pre, lab=lab, res=res, rms=rms)


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
    print(f'{len(runs)} models; B minus A: {PREFIX} drift-removed control years in front of 1850 minus none; '
          f'30-year Enting spline; weights {oa_config.WEIGHTS}', flush=True)
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            res = pool.map(run_one, runs, chunksize=1)
    else:
        res = [run_one(x) for x in runs]
    res = [x for x in res if x is not None]
    if not res:
        raise SystemExit('no model')
    yl = [y for y in YEARS if y != 2024]
    print(f'\n{"model":16s} {"n_pre":>5s}  {"control years in front (B)":38s} '
          + ' '.join(f'{"d" + str(y):>8s}' for y in yl) + f' {"rms1850":>8s} {"d decl":>8s} {"d rel":>7s}')
    D = {y: [] for y in yl}
    dd, dr, rm = [], [], []
    for x in res:
        A, B = x['res']['A'], x['res']['B']
        for y in yl:
            D[y].append(B['L'][('c', y)] - A['L'][('c', y)])
        dd.append(B['dc'] - A['dc']); dr.append(100 * (B['rel'] - A['rel'])); rm.append(x['rms'])
        print(f'{x["model"]:16s} {x["n_pre"]:5d}  {x["lab"]:38s} ' + ' '.join(f'{D[y][-1]:+8.4f}' for y in yl)
              + f' {rm[-1]:8.4f} {dd[-1]:+8.3f} {dr[-1]:+7.3f}', flush=True)
    st = lambda v: f'mean {np.mean(v):+.4f}, s.d. {np.std(v, ddof=1):.4f}, largest |.| {np.max(np.abs(v)):.4f}'
    print(f'\nacross {len(res)} models (B minus A, common area):')
    for y in yl:
        print(f'  Omega_arag {y}: {st(D[y])}')
    print(f'  rms over the cells in 1850: mean {np.mean(rm):.4f}, largest {np.max(rm):.4f}')
    print(f'  decline 1850 -> 2024 (points): {st(dd)}')
    print(f'  relative whole-ocean shift (% of the decline): {st(dr)}')
    print(f'  models with control years before the branch point in the chain (n_pre > 0): '
          f'{sum(x["n_pre"] > 0 for x in res)} of {len(res)}')


if __name__ == '__main__':
    main()
