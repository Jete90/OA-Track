#!/usr/bin/env python
"""
calc_product_members.py
=======================
Members of the data product: for one reference product and every model, the annual means of all carbonate
system indicators of the adjusted (drift-corrected, bias-adjusted) historical run (1850-2014) and of every
scenario run the model has (ssp126, ssp245, ssp370, ssp585; 2015-2100), computed monthly with mocsy from the
adjusted state (calc_bias_adjusted.adjusted_fields, the same state as omega_arag_adj there) and then averaged:

    X~(t) = X(t) - [s(k(t)) - mean of s over W'] + bias(month of t)

for the scenario years k(t) is the piControl year parallel to the historical year via the branch point, as for
the historical run (the piControl files cover 251 years from the branch). The bias and the drift are those of
the model's historical member, and a scenario is used only with that same member (no substitute member: a model
without the scenario of its historical member does not enter that scenario; 13 Nov 2026).

Together with the preindustrial states of calc_pi_xco2.py these files are the model members from which
calc_product_ensemble.py builds the unconstrained and constrained product.

Output: data/<run>/product/members/<reference>/<model>_<member>_<experiment>.nc
    <indicator>(year, lat, lon)   annual means (NaN where any month of the year is missing)
    indicators: omega_arag, omega_calc, ph, hplus, hplus_free, pco2, fco2, co2, hco3, co3, revelle, dic, and the state
                thetao, so, talk (annual means of the adjusted state)

Usage:
    python calc_product_members.py --reference OceanSODA --jobs 4
    python calc_product_members.py --reference Jiang --models GFDL-ESM4 --experiments historical ssp585
"""
from __future__ import annotations

import argparse
import oa_config   # run: data folder, mocsy constants, GFDL density (OA_RUN)
import datetime
import re
import traceback
import warnings
from multiprocessing import Pool

import netCDF4
import numpy as np

import calc_bias_adjusted as cb
import calc_omega_arag as cm
import calc_omega_arag_obs as co
import calc_pi_xco2 as px
from drift_tools import Drift

SCENARIOS = ('ssp126', 'ssp245', 'ssp370', 'ssp585')
EXPERIMENTS = ('historical',) + SCENARIOS
STATE_VARS = ('thetao', 'so', 'talk')
OUT = oa_config.DATA / 'product' / 'members'
MEMBER_RE = re.compile(r'^r(\d+)i(\d+)p(\d+)f(\d+)$')


def out_path(model, member, exp):
    return OUT / cb.REF_NAME / f'{model}_{member}_{exp}.nc'


def scenario_member(model, exp, member_hist):
    """member of `model` for scenario `exp`: the historical member if it has a complete run of the scenario, else
    None (no substitute member, so that the historical and the scenario years come from the same run)"""
    runs = [(m, e, r) for m, e, r in cm.find_runs([exp], [model]) if m == model]
    members = sorted({r for _, _, r in runs})
    if member_hist in members:
        return member_hist, ''
    if not members:
        return None, ''
    return None, f'{exp}: historical member {member_hist} not available (members {members}): the model does not enter {exp}'


def annual_indicators(run, bias, drift, win, si, po4, jy, ix, lon2, lat2):
    """annual means of all indicators and of the state for the whole run, {name: (nyr, ny, nx)} float32"""
    ny, nx = lat2.shape
    nyr = run.nt // run.step
    names = px.INDICATOR_NAMES + list(STATE_VARS)
    out = {k: np.full((nyr, ny, nx), np.nan, 'f4') for k in names}
    for i0 in range(0, run.nt, cb.CHUNK):
        i1 = min(i0 + cb.CHUNK, run.nt)
        _, _, fa = cb.adjusted_fields(run, i0, i1, bias, drift, win, jy, ix, lon2, lat2)
        sl = slice(i0 // run.step, i1 // run.step)
        _, res = next(px.target_states(fa, si, po4, lat2, {'state': px.XCO2_MODEL}, [0.0]))
        for k in px.INDICATOR_NAMES:
            out[k][sl] = res[k].reshape(-1, run.step, ny, nx).mean(axis=1)
        for v in STATE_VARS:
            out[v][sl] = fa[v].reshape(-1, run.step, ny, nx).mean(axis=1)
        print(f'  {run.tag}: {run.years[i0]}-{run.years[i1 - 1]} done', flush=True)
    return run.years[::run.step][:nyr], out


def process(model, member, experiments):
    tag = f'{model} {member}'
    with netCDF4.Dataset(cb.REF['omega']) as f:
        olat = np.asarray(f['lat'][:], 'f8')
        olon = np.asarray(f['lon'][:], 'f8')
    with netCDF4.Dataset(cb.bias_path(model, member)) as f:
        bias = {v: np.ma.filled(f[f'bias_{v}'][:].astype('f8'), np.nan) for v in cb.VARS}
        y0, y1 = (int(x) for x in f.overlap.split('-'))
    lon2, lat2 = np.meshgrid(olon, olat)
    si, po4 = co.load_nutrients(olat, olon)
    drift = Drift(model, member)
    units = {**px.UNITS_ALL, **cb.UNITS}
    done = []
    for exp in experiments:
        if exp == 'historical':
            mem, note = member, ''
        else:
            mem, note = scenario_member(model, exp, member)
            if note:
                print(f'  {tag}: {note}', flush=True)
            if mem is None:
                continue
        run = cb.ModelRun(model, exp, mem)
        try:
            jy = co.nearest_index(run.lat, olat)
            ix = co.nearest_index(run.lon, olon, periodic=True)
            years, out = annual_indicators(run, bias, drift, (y0, y1), si, po4, jy, ix, lon2, lat2)
        finally:
            run.close()
        path = out_path(model, mem, exp)
        path.parent.mkdir(parents=True, exist_ok=True)
        g = cb.new_nc(path, olat, olon, dict(
            title=f'Annual carbonate system of the adjusted {exp} run, {model} {mem}, reference {cb.REF["label"]}',
            method=('monthly indicators with mocsy from the drift-corrected, bias-adjusted state '
                    f'(calc_bias_adjusted.adjusted_fields; bias and drift of historical member {member}, window '
                    f'{y0}-{y1}), then annual means; ' + oa_config.CONSTANTS),
            experiment=exp, model=model, member=mem, historical_member=member, note=note,
            history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} calc_product_members.py, {oa_config.VERSION}'))
        g.createDimension('year', years.size)
        x = g.createVariable('year', 'i4', ('year',))
        x[:] = years
        for k, a in out.items():
            x = g.createVariable(k, 'f4', ('year', 'lat', 'lon'), zlib=True, complevel=4, fill_value=cb.FILL)
            x.units = units[k]
            x[:] = np.ma.masked_invalid(a)
        g.close()
        w = oa_config.area_weights(olat, olon)
        om = out['omega_arag']
        v = np.all(np.isfinite(om), axis=0)
        gm = np.array([np.sum(a[v] * w[v]) / w[v].sum() for a in om])
        done.append(f'{exp} {years[0]}-{years[-1]}: Omega_arag {gm[0]:.3f} -> {gm[-1]:.3f} ({100 * v.mean():.0f} % of cells)')
    print(f'[done] {tag}: ' + '; '.join(done), flush=True)


def run_one(model, member, experiments):
    try:
        process(model, member, experiments)
        return f'{model} {member}', 'done'
    except Exception:
        print(f'[FAILED] {model} {member}\n{traceback.format_exc()}', flush=True)
        return f'{model} {member}', 'FAILED'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', default='OceanSODA', choices=list(cb.REFERENCES))
    ap.add_argument('--models', nargs='+')
    ap.add_argument('--experiments', nargs='+', default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    ap.add_argument('--jobs', type=int, default=1)
    a = ap.parse_args()
    cb.set_reference(a.reference)
    runs = [(m, r, a.experiments) for m, r in px.find_runs(a.models)]     # models with preindustrial states
    print(f'{len(runs)} models, reference {cb.REF["label"]} -> {OUT / cb.REF_NAME}', flush=True)
    if a.jobs > 1:
        with Pool(a.jobs) as p:
            results = p.starmap(run_one, runs)
    else:
        results = [run_one(*r) for r in runs]
    print('\nsummary:')
    for tag, status in results:
        print(f'  {status:7s} {tag}')


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
