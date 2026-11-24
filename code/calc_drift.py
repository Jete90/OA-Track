#!/usr/bin/env python
"""
calc_drift.py
=============
piControl drift of surface T, S, DIC and ALK for every model, aligned with the historical run through its
branch point. Per cell, the annual means of the whole available piControl are smoothed with a spline in the
sense of Enting (1987) with a 50-year cutoff period (50 % response at 50 years): s(k) for every piControl
year k. The smoother keeps the mean and a linear trend exactly.

The drift correction of the chain (calc_bias_adjusted.py, calc_pi_xco2.py) is the same for both experiments:

    X'(t) = X(t) - [s(k(t)) - mean of s over W']        W' = piControl years parallel to the reference window W

with k(t) the piControl year of a piControl month, and for a historical month of year y the parallel piControl
year  branch_year + (y - first historical year). Over the window W the correction averages to zero, so the bias
(observation minus historical over W) is unchanged by it, and the mean over all years of the corrected piControl
equals the smoothed piControl at the window years. The reference point of the spline cancels.

Stored, on the model input grid (r360x180), same units as calc_mechanism.py (degC, psu, umol kg-1):

    <var>_smooth_anom(year, lat, lon)   s(k) - mean(piControl)       (the Drift class below turns this into
                                                                      the correction for any window)
    <var>_drift_<y0>_<y1>(lat, lon)     mean of s over W' - mean(piControl)   for the periods in PERIODS
                                                                      (used by constraint_refs.py for the
                                                                      alkalinity and SST changes)

Alignment: historical year Y corresponds to piControl year  branch_year + (Y - first historical year),
with branch_year from the historical file's branch_time_in_parent / parent_time_units. If the attribute
is missing or unusable, or the branch year is not within the years of the piControl file (e.g. files
relabelled to 1850-2100), the piControl file is assumed to start at the branch point (printed as a
warning). Periods partly outside the available piControl years use only the overlapping years
(printed as a warning).

Periods: 1850-1900 and the window of every reference (oa_config.PERIODS: 1980-2014, 1983-2014, 1985-2014, 1990-2014,
2005-2014).
Output: data/<run>/drift/drift_<model>_<member>.nc. Needs only the raw model files (runs before the bias step).

Usage:
    python calc_drift.py --jobs 4
    python calc_drift.py --models GFDL-ESM4
"""
from __future__ import annotations

import argparse
import oa_config   # run: data folder, mocsy constants, GFDL density (OA_RUN)
import traceback
import warnings
from multiprocessing import Pool

import netCDF4
import numpy as np

import calc_bias_adjusted as cb
import calc_omega_arag as cm

from drift_tools import OUT, VARS, PERIODS, CUTOFF, drift_path, smoother, parallel_years, window_slice, Drift  # noqa: F401

ROOT = cm.ROOT


def branch_year(model, member, hist_first_year):
    """piControl calendar year at which historical branches off, or None."""
    p = cm.input_path('thetao', model, 'historical', member)
    with netCDF4.Dataset(p) as f:
        bt = getattr(f, 'branch_time_in_parent', None)
        pu = getattr(f, 'parent_time_units', None)
        cal = getattr(f['time'], 'calendar', 'standard')
    if bt is None or pu is None:
        return None, 'no branch attributes'
    try:
        bt = float(np.atleast_1d(bt)[0])
        u = pu.strip().lower()
        if u.startswith(('years since', 'months since')):
            y, m = cm.year_month(np.array([bt]), pu, cal)
            return int(y[0]), f'branch_time_in_parent {bt} {pu}'
        d = netCDF4.num2date(bt, pu, cal)
        return int(d.year), f'branch_time_in_parent {bt} {pu}'
    except Exception as e:
        return None, f'branch attributes unusable ({e})'


def process(model, member):
    # historical first year (for the alignment)
    hp = cm.input_path('thetao', model, 'historical', member)
    with netCDF4.Dataset(hp) as f:
        t = f['time']
        hy, _ = cm.year_month(t[:1], t.units, getattr(t, 'calendar', 'standard'))
    hist0 = int(hy[0])

    run = cb.ModelRun(model, 'piControl', member)
    lat, lon = run.lat, run.lon
    lon2, lat2 = np.meshgrid(lon, lat)
    jy, ix = np.arange(lat.size), np.arange(lon.size)
    pyears = run.years[::run.step]
    by, note = branch_year(model, member, hist0)
    msg = []
    if by is None:
        by = int(pyears[0])
        msg.append(f'WARNING {note}: piControl assumed to start at the branch point ({by})')
    elif not (pyears[0] <= by <= pyears[-1]):
        msg.append(f'WARNING branch year {by} ({note}) is outside the piControl file {pyears[0]}-{pyears[-1]} '
                   f'(different year labels?): piControl assumed to start at the branch point')
        by = int(pyears[0])
    else:
        msg.append(f'branch at piControl year {by} ({note}); piControl file {pyears[0]}-{pyears[-1]}')

    # annual means of the whole piControl (float32: 4 variables x ~250 years x 1 deg is a few 100 MB)
    n = len(pyears)
    ann = {v: np.full((n, lat.size, lon.size), np.nan, 'f4') for v in VARS}
    for k in range(n):
        f = run.read(k * run.step, (k + 1) * run.step, jy, ix, lon2, lat2)
        for v in VARS:
            ann[v][k] = f[v].mean(axis=0)
    run.close()

    S = smoother(n)
    out, anom = {}, {}
    for v in VARS:
        a = ann[v].reshape(n, -1).astype('f8')
        ok = np.all(np.isfinite(a), axis=0)
        sm = np.full(a.shape, np.nan)
        sm[:, ok] = S @ a[:, ok]
        mean = np.where(ok, a.mean(axis=0) if ok.all() else np.nanmean(a, axis=0), np.nan)
        anom[v] = (sm - mean).reshape(n, lat.size, lon.size).astype('f4')     # s(k) - mean(piControl)
        for p0, p1 in PERIODS:
            q0, q1 = parallel_years(by, hist0, p0, p1)
            k0, k1 = window_slice(pyears, q0, q1)
            if (q0 < pyears[0] or q1 > pyears[-1]) and v == VARS[0]:
                msg.append(f'WARNING {p0}-{p1} -> piControl {q0}-{q1} outside the file ({pyears[0]}-{pyears[-1]}): '
                           f'the nearest available years {pyears[k0]}-{pyears[k1 - 1]} are used')
            out[f'{v}_drift_{p0}_{p1}'] = (sm[k0:k1].mean(axis=0) - mean).reshape(lat.size, lon.size)
        del ann[v]

    OUT.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(drift_path(model, member), 'w') as g:
        g.description = ('piControl drift (Enting spline, 50-year cutoff) aligned with historical via the branch point; '
                         '<var>_smooth_anom = smoothed piControl - piControl mean, per piControl year; '
                         '<var>_drift_<y0>_<y1> = smoothed parallel piControl years - piControl mean; calc_drift.py')
        g.branch_year = by
        g.hist_first_year = hist0
        g.picontrol_years = f'{pyears[0]}-{pyears[-1]}'
        g.cutoff_years = CUTOFF
        g.units = 'degC, psu, umol kg-1'
        for n_, vals in (('lat', lat), ('lon', lon)):
            g.createDimension(n_, vals.size)
            g.createVariable(n_, 'f8', (n_,))[:] = vals
        g.createDimension('year', n)
        x = g.createVariable('year', 'i4', ('year',))
        x.long_name = 'piControl year as labelled in the model file'
        x[:] = pyears
        for v in VARS:
            x = g.createVariable(f'{v}_smooth_anom', 'f4', ('year', 'lat', 'lon'), zlib=True,
                                 fill_value=np.float32(1e20))
            x.long_name = f'{v}: Enting spline of the piControl annual means minus the piControl mean'
            x[:] = np.ma.masked_invalid(anom[v])
        for k_, a in out.items():
            x = g.createVariable(k_, 'f4', ('lat', 'lon'), zlib=True, fill_value=np.float32(1e20))
            x[:] = np.ma.masked_invalid(a)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones(lon2.shape)    # native model grid, diagnostic: cell area (oa_config.WEIGHTS applies to the product grid)
    gm = lambda a: np.nansum(a * w) / w[np.isfinite(a)].sum()
    d = {v: gm(out[f'{v}_drift_1983_2014']) - gm(out[f'{v}_drift_1850_1900']) for v in VARS}
    print(f'[done] {model} {member}: ' + '; '.join(msg) + '\n       drift 1850-1900 -> 1983-2014: '
          + ', '.join(f'{v} {d[v]:+.3f}' for v in VARS), flush=True)


def run_one(model, member):
    try:
        process(model, member)
    except Exception:
        print(f'[FAILED] {model} {member}\n{traceback.format_exc()}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+')
    ap.add_argument('--jobs', type=int, default=1)
    a = ap.parse_args()
    runs = cb.find_runs(a.models)            # every model with historical and piControl inputs
    print(f'{len(runs)} runs -> {OUT}', flush=True)
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            pool.starmap(run_one, runs)
    else:
        for r in runs:
            run_one(*r)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
