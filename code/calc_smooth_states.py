#!/usr/bin/env python
"""
calc_smooth_states.py  (data product, pass 1: once for all references and all configurations)
=====================
The forced monthly state of every model for the data product: T, S, DIC and ALK of the historical run (1850-2014)
continued with each scenario (2015-2100), drift-removed and smoothed in time, on the model grid.

For every model, calendar month and cell, the series  X(t) - s(k(t))  is built (X the model field in degC, psu
and umol/kg as read by calc_bias_adjusted.ModelRun, s the Enting spline of the piControl, calc_drift.py, at the
piControl year parallel to t), from the drift-removed piControl years before the branch point where the control
file has them (attribute n_pre; most models branch at the first year of their control file, so that their series
starts in 1850 and its first years carry the end effect of the spline), through the historical run, into the
scenario. The series is
smoothed with an Enting spline (CUTOFF years, 50 % response), which removes the model's internal variability
(ENSO, decadal) and keeps the forced change. The constrained state of a year (calc_product_states.py) is built
from these smoothed states of all models; the reference enters only there, as the constants of the chain:
    adjusted state = smoothed state + mean of s over the window years W' + bias(month)
(both constants per cell, unchanged by the smoothing), with mean of s from the drift file and the bias from
calc_bias_adjusted.py. The raw state and the drift do not depend on the carbonate constants, so one pass serves
the four configurations; the drift files of the run in OA_RUN are used (identical in every run).

Output: BASE/data/smoothed/smoothed_<model>_<member>.nc, float32, chunked per year, on the model grid:
    <var>_hist(year_hist, month, lat, lon)      1850-2014 smoothed, continued with the scenario named in the
                                                attribute 'continuation' (SSP2-4.5 if the model has it)
    <var>_<ssp>(year_scen, month, lat, lon)     1970-2100 smoothed with that scenario after 2014 (its years
                                                before 2015 differ from <var>_hist only through the continuation,
                                                negligibly in 1970; the product's scenario series use <var>_hist
                                                to 1969 and <var>_<ssp> from 1970)
    <var>_hist_anom, <var>_<ssp>_anom           raw minus smoothed, monthly: the model's internal variability
                                                (calc_product_states.py adds every model's anomaly of a year to
                                                the constrained state: the sigma of internal variability)
    <var>_hist_raw_annual, <var>_<ssp>_raw_annual(year, lat, lon)   annual means of the unsmoothed series, for
                                                the diagnostics of the smoothing (removed variability, distortion
                                                of the multi-model mean)
    <var>_hist_resid_std(month, lat, lon), <var>_<ssp>_resid_std   std over the years of raw - smoothed: the
                                                internal variability removed by the smoothing
    <var>_hist_resid_rms(year_hist), <var>_<ssp>_resid_rms(year_scen)   area-weighted RMS over the valid cells
                                                of raw - smoothed per year
    continuation_diff(scenario, year_hist)      max over cells of |<var>_hist smoothed with the default
                                                continuation - smoothed with that scenario| for thetao (K) and,
                                                as continuation_diff_dissic, for dissic (umol/kg): how far back
                                                the choice of the scenario matters
    month: 12 for monthly models, 1 for annual models (attribute step)

Usage:  python calc_smooth_states.py --jobs 4            # all models with a historical run
        python calc_smooth_states.py --models GFDL-ESM4 --cutoff 30
"""
from __future__ import annotations

import argparse
import oa_config   # run: data folder, mocsy constants, GFDL density (OA_RUN)
import datetime
import traceback
import warnings
from multiprocessing import Pool
from pathlib import Path

import netCDF4
import numpy as np

import calc_bias_adjusted as cb
import calc_omega_arag as cm
import calc_product_members as pm
import drift_tools
from drift_tools import Drift

OUT = oa_config.BASE / 'data' / 'smoothed'
VARS = cb.VARS
SCENARIOS = pm.SCENARIOS
CONTINUATION = ('ssp245', 'ssp370', 'ssp126', 'ssp585')     # preference for the continuation of <var>_hist
CUTOFF = 30.0                                              # years, 50 % response of the Enting spline
N_PRE = 30                                                 # piControl years in front of 1850 (as available)
SCEN_FIRST = 1970                                          # first year stored of the scenario series
YEAR_LAST = 2100
FILL = cb.FILL
_SMOOTHERS = {}


def smoother(n, cutoff):
    key = (n, cutoff)
    if key not in _SMOOTHERS:
        _SMOOTHERS[key] = drift_tools.smoother(n, cutoff)
    return _SMOOTHERS[key]


def out_path(model, member):
    return OUT / f'smoothed_{model}_{member}.nc'


def read_run(run, drift, exp, years_keep=None):
    """the drift-removed state of a run as {var: (nyr, step, ny, nx)} float32 on the model grid, and its years;
    years_keep: keep only these years (e.g. the scenario to 2100, the piControl years before the branch)"""
    jy, ix = np.arange(run.lat.size), np.arange(run.lon.size)
    lon2, lat2 = np.meshgrid(run.lon, run.lat)
    years = run.years[::run.step][:run.nt // run.step]
    sel = np.ones(years.size, bool) if years_keep is None else np.isin(years, years_keep)
    iy = np.where(sel)[0]
    if iy.size == 0:
        return None, years[sel]
    i_first, i_last = int(iy[0]) * run.step, (int(iy[-1]) + 1) * run.step
    out = {v: np.full((iy.size, run.step) + lat2.shape, np.nan, 'f4') for v in VARS}
    for i0 in range(i_first, i_last, cb.CHUNK):
        i1 = min(i0 + cb.CHUNK, i_last)
        f = run.read(i0, i1, jy, ix, lon2, lat2)
        yrs = run.years[i0:i1]
        for v in VARS:
            a = f[v] - drift.anom[v][drift.index(exp, yrs)].astype('f8')     # X(t) - s(k(t))
            j0 = (i0 - i_first) // run.step
            out[v][j0:j0 + (i1 - i0) // run.step] = a.reshape(-1, run.step, *lat2.shape)
    return out, years[sel]


def smooth_series(blocks, cutoff):
    """blocks: list of (nyr, step, ny, nx) arrays in time order; returns the smoothed concatenation, same shape
    list, per calendar month (slot) an Enting spline over all years of the series; cells with a missing value
    anywhere in the series are NaN"""
    X = np.concatenate(blocks, axis=0)                         # (n, step, ny, nx)
    n, step = X.shape[:2]
    S = smoother(n, cutoff)
    Y = np.full(X.shape, np.nan, 'f4')
    for m in range(step):
        a = X[:, m].reshape(n, -1).astype('f8')
        ok = np.all(np.isfinite(a), axis=0)
        b = np.zeros_like(a)
        b[:, ok] = S @ a[:, ok]
        b[:, ~ok] = np.nan
        Y[:, m] = b.reshape(n, *X.shape[2:])
    lengths = [b.shape[0] for b in blocks]
    cuts = np.cumsum([0] + lengths)
    return [Y[cuts[i]:cuts[i + 1]] for i in range(len(blocks))]


def area_rms(d, lat):
    """area-weighted RMS over the valid cells of a (nyr, step, ny, nx) field, per year (native model grid: cell area;
    a diagnostic, oa_config.WEIGHTS applies to the product grid)"""
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones(d.shape[-1])
    ok = np.isfinite(d)
    num = np.nansum(np.where(ok, d.astype('f8') ** 2, 0.0) * w, axis=(1, 2, 3))
    den = np.sum(ok * w, axis=(1, 2, 3))
    return np.sqrt(num / np.where(den > 0, den, np.nan))


def process(model, member, cutoff=CUTOFF):
    tag = f'{model} {member}'
    drift = Drift(model, member)
    hist = cb.ModelRun(model, 'historical', member)
    try:
        H, years_h = read_run(hist, drift, 'historical')
        lat, lon, step = hist.lat, hist.lon, hist.step
    finally:
        hist.close()
    # piControl years before the branch point, drift-removed (the years themselves: s(k) of the piControl)
    pre_years = np.arange(max(drift.by - N_PRE, int(drift.years[0])), drift.by)
    P = None
    if pre_years.size:
        pi = cb.ModelRun(model, 'piControl', member)
        try:
            P, pre_years = read_run(pi, drift, 'piControl', pre_years)
        finally:
            pi.close()
    n_pre = 0 if P is None else pre_years.size
    # scenarios: the historical member only (calc_product_members.scenario_member, no substitute member)
    scen = {}
    notes = []
    for s in SCENARIOS:
        mem, note = pm.scenario_member(model, s, member)
        if note:
            notes.append(note)
        if mem is None:
            continue
        run = cb.ModelRun(model, s, mem)
        try:
            if run.step != step:
                notes.append(f'{s}: step {run.step} differs from the historical ({step}): not used')
                continue
            years_s = run.years[::run.step][:run.nt // run.step]
            keep = years_s[(years_s > int(years_h[-1])) & (years_s <= YEAR_LAST)]
            Sc, years_s = read_run(run, drift, s, keep)
        finally:
            run.close()
        if Sc is None or years_s.size == 0 or int(years_s[0]) != int(years_h[-1]) + 1:
            notes.append(f'{s}: years {years_s[0] if years_s.size else "-"}-{years_s[-1] if years_s.size else "-"} '
                         f'do not continue the historical ({years_h[-1]}): not used')
            continue
        scen[s] = (Sc, years_s, mem)
    cont = next((s for s in CONTINUATION if s in scen), None)
    print(f'{tag}: historical {years_h[0]}-{years_h[-1]}, {n_pre} piControl years in front, scenarios '
          f'{list(scen)}, continuation of the historical series: {cont or "none (end effect at 2014)"}', flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    path = out_path(model, member)
    g = netCDF4.Dataset(path, 'w', format='NETCDF4')
    g.title = f'Smoothed, drift-removed monthly state of {model} for the data product (pass 1)'
    g.method = __doc__.split('Output:')[0].split('\n', 3)[3].strip()
    g.model, g.member, g.step = model, member, int(step)
    g.cutoff_years, g.n_pre, g.branch_year, g.hist_first_year = float(cutoff), int(n_pre), int(drift.by), int(drift.hist0)
    g.continuation = cont or 'none'
    g.scenario_members = ', '.join(f'{s} {scen[s][2]}' for s in scen)
    g.drift_run = oa_config.RUN
    g.notes = '; '.join(notes)
    g.units = 'thetao degC, so psu, dissic and talk umol kg-1'
    g.history = f'{datetime.datetime.now():%Y-%m-%d %H:%M} calc_smooth_states.py, {oa_config.VERSION}'
    g.createDimension('lat', lat.size)
    g.createDimension('lon', lon.size)
    g.createDimension('month', step)
    g.createDimension('year_hist', years_h.size)
    x = g.createVariable('lat', 'f8', ('lat',)); x[:] = lat
    x = g.createVariable('lon', 'f8', ('lon',)); x[:] = lon
    x = g.createVariable('year_hist', 'i4', ('year_hist',)); x[:] = years_h
    years_scen = np.arange(SCEN_FIRST, YEAR_LAST + 1)
    g.createDimension('year_scen', years_scen.size)
    x = g.createVariable('year_scen', 'i4', ('year_scen',)); x[:] = years_scen
    g.createDimension('scenario', len(SCENARIOS))
    x = g.createVariable('scenario', str, ('scenario',)); x[:] = np.array(SCENARIOS, dtype=object)

    def var4(name, dim, data, long_name):
        x = g.createVariable(name, 'f4', (dim, 'month', 'lat', 'lon'), zlib=True, complevel=1,
                             chunksizes=(1, step, lat.size, lon.size), fill_value=FILL)
        x.long_name = long_name
        x[:] = np.ma.masked_invalid(data)

    def var3(name, dim, data, long_name):
        x = g.createVariable(name, 'f4', (dim, 'lat', 'lon'), zlib=True, complevel=1, fill_value=FILL)
        x.long_name = long_name
        x[:] = np.ma.masked_invalid(data)

    def diagnostics(prefix, dim, years, raw, smooth):
        d = raw.astype('f8') - smooth.astype('f8')
        var3(f'{prefix}_raw_annual', dim, raw.mean(axis=1), f'annual mean of the unsmoothed series ({prefix})')
        x = g.createVariable(f'{prefix}_resid_std', 'f4', ('month', 'lat', 'lon'), zlib=True, complevel=1, fill_value=FILL)
        x.long_name = f'std over the years of raw minus smoothed ({prefix}): internal variability removed'
        x[:] = np.ma.masked_invalid(d.std(axis=0, ddof=1) if d.shape[0] > 1 else np.full(d.shape[1:], np.nan))
        x = g.createVariable(f'{prefix}_resid_rms', 'f8', (dim,))
        x.long_name = f'area-weighted RMS over the cells of raw minus smoothed per year ({prefix})'
        x[:] = area_rms(d, lat)

    # the historical series with every continuation: the default continuation is stored as <var>_hist, the
    # others give the continuation diagnostics; each scenario's own smoothed series from SCEN_FIRST
    hist_sm = {}                                   # <var>: historical smoothed with the default continuation
    hist_by_scen = {s: {} for s in scen}           # <scenario>: {<var>: historical smoothed with that scenario}
    for v in VARS:
        base = [] if P is None else [P[v]]
        if not scen:
            hist_sm[v] = smooth_series(base + [H[v]], cutoff)[-1]
        for s, (Sc, years_s, _) in scen.items():
            sh, ss = smooth_series(base + [H[v], Sc[v]], cutoff)[-2:]
            hist_by_scen[s][v] = sh
            if s == cont:
                hist_sm[v] = sh
            # the scenario series from SCEN_FIRST: historical part (smoothed with this scenario) and scenario part
            full = np.concatenate([sh, ss], axis=0)
            yfull = np.concatenate([years_h, years_s])
            keep = (yfull >= SCEN_FIRST) & (yfull <= YEAR_LAST)
            out_s = np.full((years_scen.size, step, lat.size, lon.size), np.nan, 'f4')
            raw_s = np.full_like(out_s, np.nan)
            idx = np.searchsorted(years_scen, yfull[keep])
            out_s[idx] = full[keep]
            raw_s[idx] = np.concatenate([H[v], Sc[v]], axis=0)[keep]
            var4(f'{v}_{s}', 'year_scen', out_s, f'{v}, drift-removed, smoothed ({cutoff:.0f}-year Enting spline) '
                 f'historical continued with {s}; years {SCEN_FIRST}-{YEAR_LAST}')
            diagnostics(f'{v}_{s}', 'year_scen', years_scen, raw_s, out_s)
            var4(f'{v}_{s}_anom', 'year_scen', raw_s - out_s, f'{v}: internal variability, raw minus smoothed '
                 f'(historical continued with {s}); added to the constrained state for the sigma of internal variability')
        var4(f'{v}_hist', 'year_hist', hist_sm[v], f'{v}, drift-removed, smoothed ({cutoff:.0f}-year Enting spline) '
             f'historical {years_h[0]}-{years_h[-1]}, continued with {cont or "nothing"}')
        diagnostics(f'{v}_hist', 'year_hist', years_h, H[v], hist_sm[v])
        var4(f'{v}_hist_anom', 'year_hist', H[v] - hist_sm[v], f'{v}: internal variability, raw minus smoothed '
             '(historical); added to the constrained state for the sigma of internal variability')
    cdiff = {v: np.full((len(SCENARIOS), years_h.size), np.nan) for v in ('thetao', 'dissic')}
    for v in cdiff:
        for s in scen:
            cdiff[v][SCENARIOS.index(s)] = np.nanmax(np.abs(hist_by_scen[s][v].astype('f8') - hist_sm[v].astype('f8')),
                                                     axis=(1, 2, 3))
    for v, name in (('thetao', 'continuation_diff'), ('dissic', 'continuation_diff_dissic')):
        x = g.createVariable(name, 'f8', ('scenario', 'year_hist'), fill_value=np.nan)
        x.long_name = (f'max over cells of |smoothed historical {v} with the default continuation ({cont}) minus '
                       'with this scenario|: how far back the scenario matters (0 for the default)')
        x[:] = cdiff[v]
    g.close()
    # report: continuation effect at SCEN_FIRST - 1 and 1989, 2014; removed variability
    lines = [f'{tag}: written {path}']
    for s in scen:
        i = SCENARIOS.index(s)
        if s != cont:
            d = cdiff['thetao'][i]
            pick = lambda y: d[np.searchsorted(years_h, y)] if y in years_h else np.nan
            lines.append(f'  continuation {s} instead of {cont}: max |dT| {pick(SCEN_FIRST - 1):.1e} K in {SCEN_FIRST - 1}, '
                         f'{pick(1989):.1e} in 1989, {pick(2014):.1e} in 2014; dissic {cdiff["dissic"][i][np.searchsorted(years_h, 1989)]:.1e} umol/kg in 1989')
    with netCDF4.Dataset(path) as f:
        for v in ('thetao', 'dissic'):
            r = np.asarray(f[f'{v}_hist_resid_rms'][:])
            lines.append(f'  {v}: internal variability removed (area RMS of raw - smoothed), historical mean {np.nanmean(r):.3f}, '
                         f'max {np.nanmax(r):.3f} {"K" if v == "thetao" else "umol/kg"}')
    print('\n'.join(lines), flush=True)


def run_one(model, member, cutoff):
    try:
        process(model, member, cutoff)
        return f'{model} {member}', 'done'
    except Exception:
        print(f'[FAILED] {model} {member}\n{traceback.format_exc()}', flush=True)
        return f'{model} {member}', 'FAILED'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--cutoff', type=float, default=CUTOFF)
    a = ap.parse_args()
    runs = [(m, r, a.cutoff) for m, r in cb.find_runs(a.models)]
    print(f'{len(runs)} models with historical and piControl -> {OUT} (drift from {drift_tools.OUT})', flush=True)
    if a.jobs > 1:
        with Pool(a.jobs) as p:
            results = p.starmap(run_one, runs)
    else:
        results = [run_one(*r) for r in runs]
    print('\nsummary:')
    for tag, status in results:
        print(f'  {status:7s} {tag}')
    if any(status == 'FAILED' for _, status in results):
        raise SystemExit(1)                      # rerun_product.sh then does not mark P0 complete


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
