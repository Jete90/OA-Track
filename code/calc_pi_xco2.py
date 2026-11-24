#!/usr/bin/env python
"""
calc_pi_xco2.py
===============
Preindustrial Omega_arag of every model for the preindustrial CO2 states used in the paper (1750 and 1850 of
AR6 with their +-1 sigma, their midpoint, the effective CO2 of Jiang et al. 2023, 280 ppm; see TARGETS in
pi_states.py), from the drift-corrected, bias-adjusted monthly piControl of calc_bias_adjusted.py (CMIP6
piControl runs at 284.317 ppm), cell by cell and month by month, with no global factors:

1. The adjusted state of every piControl month t (calc_bias_adjusted.adjusted_fields, the same state as
   omega_arag_adj there):   X~(t) = X(t) - [s(k(t)) - mean of s over W'] + bias(month of t).
2. For every cell and month, T, S and ALK of that state are kept and DIC is changed until the surface
   ocean pCO2 has changed by the atmospheric difference:

       target pCO2 = pCO2_model + (xCO2_target - 284.317) * (1 - pH2O)

   with pH2O the water vapour pressure (Weiss and Price, 1980) at 1 atm. DIC is found by Newton iteration
   using the Revelle factor from mocsy (dDIC = DIC * dpCO2 / (R * pCO2)); Omega_arag follows from the
   final state. This assumes the surface ocean follows the atmosphere in pCO2, which holds for changes
   that are slow compared with air-sea equilibration (months).
3. Omega_arag of every target is averaged over all months and years (the time mean of the monthly values,
   not the value of the mean state), per cell. The target 'pi284' (284.317 ppm, no change of DIC) is the
   adjusted piControl itself and must reproduce omega_arag_adj of calc_bias_adjusted.py (checked).

Global means use the joint mask of the bias-adjusted output, further restricted to cells valid for every
target in every month.

Output: data/omega_arag_biasadj[_<reference>]/piControl/omega_arag_biasadj_xco2_Oyr_<model>_piControl_<member>_<reference>grid.nc
    <indicator>_<target>_mean    2D time mean over all piControl months (NaN unless valid in every month) of
                                 omega_arag, omega_calc, ph (total scale), hplus ([H+] total, nmol kg-1), hplus_free ([H+] free), pco2,
                                 fco2 (uatm), co2, hco3, co3 (umol kg-1), revelle, dic (umol kg-1): the
                                 unconstrained preindustrial carbonate system of the model (data product)
    <indicator>_<target>_std     interannual standard deviation of the annual means per cell
    omega_arag_<target>_clim     monthly climatology (12, lat, lon)
    omega_arag_<target>_gm       annual global mean over the joint mask
    thetao_mean, so_mean, dissic_mean, talk_mean   time means of the adjusted state at 284.317 ppm
    thetao_clim, so_clim, dissic_clim, talk_clim   monthly climatology (12, lat, lon) of the adjusted state at
                                 284.317 ppm, over all piControl years: the model's preindustrial state for the
                                 data product (calc_product_states.py builds the constrained state from these)
    ddic_1750_mean, ddic_1750_gm      DIC change from 284.317 to 278.3 ppm (time mean field, its global mean)
    omega_arag_<target>          annual 2D fields (year, lat, lon), only with --annual-fields

Usage:
    python calc_pi_xco2.py --reference OceanSODA --jobs 4
    python calc_pi_xco2.py --models GFDL-ESM4 --annual-fields
    python calc_pi_xco2.py --explicit-years        # per-model states of every year 1750-1849 (tests only)
"""
from __future__ import annotations

import argparse
import oa_config   # run: data folder, mocsy constants, GFDL density (OA_RUN)
import datetime
import traceback
import warnings
from multiprocessing import Pool

import netCDF4
import numpy as np

import calc_bias_adjusted as cb     # model reading, bias, grid mapping, adjusted state (imports mocsy path)
import calc_omega_arag as cm
import calc_omega_arag_obs as co
from drift_tools import Drift
import mocsy
MOCSY_OPTB, MOCSY_OPTK1K2 = oa_config.OPTB, oa_config.OPTK1K2   # total boron, K1/K2 of the run (oa_config.py)

XCO2_MODEL = 284.317               # CMIP6 piControl
from pi_states import TARGETS, XCO2_SIGMA, year_targets     # the preindustrial CO2 states (see pi_states.py)
MAIN_TARGETS = {**TARGETS, 'pi284': XCO2_MODEL}  # pi284: the adjusted piControl itself (check against calc_bias_adjusted)
N_ITER_MAX = 8                     # Newton iterations at most
PCO2_TOL = 1e-4                    # uatm: iterate until max |pCO2 - target| < PCO2_TOL over the valid cells
VARS = cb.VARS


def adj_path(model, member):
    return cb.out_path('piControl', model, member)


def xco2_path(model, member):
    return cb.OUT_DIR / 'piControl' / f'omega_arag_biasadj_xco2_Oyr_{model}_piControl_{member}_{cb.REF_NAME}grid.nc'


def find_runs(models=None):
    """(model, member) of every existing bias-adjusted piControl file."""
    runs = []
    for p in sorted((cb.OUT_DIR / 'piControl').glob(f'omega_arag_biasadj_Oyr_*_piControl_*_{cb.REF_NAME}grid.nc')):
        model, member = p.name[len('omega_arag_biasadj_Oyr_'):].split('_piControl_')
        member = member.split('_')[0]
        if not models or model in models:
            runs.append((model, member))
    return runs


def mvars_cells(t, s, dic, alk, sil, phos, lat):
    """pCO2 (uatm), Omega_arag and Revelle factor; concentrations in mol kg-1."""
    n = t.size
    r = mocsy.mvars(temp=cm.floor_temperature(t, s, count=False), sal=s, alk=alk, dic=dic, sil=sil, phos=phos,
                    patm=np.ones(n), depth=np.zeros(n), lat=lat,
                    optcon='mol/kg', optt='Tpot', optp='m', optb=MOCSY_OPTB, optk1k2=MOCSY_OPTK1K2)
    return np.asarray(r[1], float), np.asarray(r[6], float), np.asarray(r[8], float)


def water_vapour(t, s):
    """Water vapour pressure in atm (Weiss and Price, 1980)."""
    tk = t + 273.15
    return np.exp(24.4543 - 67.4509 * (100.0 / tk) - 4.8489 * np.log(tk / 100.0) - 0.000544 * s)


def valid_omega(om, ph=None):
    """a solved carbonate system (calc_omega_arag.valid_carbonate when the pH is given)"""
    if ph is not None:
        return cm.valid_carbonate(ph, om)
    return np.isfinite(om) & (om > 0) & (om < co.OMEGA_MAX)


# indicators kept from every mocsy call (index in the mvars output, unit written, scale applied to the output)
INDICATORS = {
    'omega_arag': (6, '1', 1.0), 'omega_calc': (7, '1', 1.0), 'ph': (0, 'total scale', 1.0),
    'pco2': (1, 'uatm', 1.0), 'fco2': (2, 'uatm', 1.0), 'co2': (3, 'umol kg-1', 1e6), 'hco3': (4, 'umol kg-1', 1e6),
    'co3': (5, 'umol kg-1', 1e6), 'revelle': (8, '1', 1.0),
}
# hplus: [H+] on the total scale, hplus_free: [H+] on the free scale (nmol kg-1), dic: umol kg-1
INDICATOR_NAMES = list(INDICATORS) + ['hplus', 'hplus_free', 'dic']
UNITS_ALL = {**{ind: u for ind, (_, u, _) in INDICATORS.items()},
             'hplus': 'nmol kg-1', 'hplus_free': 'nmol kg-1', 'dic': 'umol kg-1'}


def total_to_free(t, s):
    """[H+]free / [H+]total = 1 / (1 + ST/KS): total sulfate ST (Morris and Riley 1966) and KS on the free scale
    (Dickson 1990), the constants mocsy uses for the total scale; t in degC (surface: potential = in situ), s in
    psu. Zero pressure (surface)."""
    tk = t + 273.15
    ion = 19.924 * s / (1000.0 - 1.005 * s)                                   # ionic strength (mol kg-H2O)
    ln_ks = (-4276.1 / tk + 141.328 - 23.093 * np.log(tk)
             + (-13856.0 / tk + 324.57 - 47.986 * np.log(tk)) * np.sqrt(ion)
             + (35474.0 / tk - 771.54 + 114.723 * np.log(tk)) * ion
             - 2698.0 / tk * ion ** 1.5 + 1776.0 / tk * ion ** 2
             + np.log(1.0 - 0.001005 * s))                                    # mol kg-H2O -> mol kg-SW
    st = 0.14 / 96.062 * s / 1.80655                                          # mol kg-SW
    return 1.0 / (1.0 + st / np.exp(ln_ks))


def mvars_all(t, s, dic, alk, sil, phos, lat):
    """all outputs of mocsy.mvars (list of arrays), concentrations in mol kg-1; t floored at the freezing point"""
    n = t.size
    return mocsy.mvars(temp=cm.floor_temperature(t, s, count=False), sal=s, alk=alk, dic=dic, sil=sil, phos=phos,
                       patm=np.ones(n), depth=np.zeros(n), lat=lat,
                       optcon='mol/kg', optt='Tpot', optp='m', optb=MOCSY_OPTB, optk1k2=MOCSY_OPTK1K2)


def target_states(fields, si, po4, lat2, targets, resid, iters=None):
    """For the (n, ny, nx) state `fields`, one target at a time (memory): yields (tag, {indicator: (n, ny, nx)
    field}) with the indicators of INDICATOR_NAMES at the target CO2 (a target at XCO2_MODEL is the state itself,
    no iteration). resid: one-element list, updated with max |pCO2 - target| over all targets; iters: dict,
    counts of the Newton iterations used. Targets are solved in the order of their CO2, each Newton iteration
    starting from the nearer of the piControl state and the previously solved target (the explicit years lie
    within 0.1 ppm of each other), until max |pCO2 - target| < PCO2_TOL."""
    t, s, dic, alk = (fields[v] for v in VARS)
    ok = (np.isfinite(t) & np.isfinite(s) & np.isfinite(dic) & np.isfinite(alk)
          & (s > 0) & (dic > 0) & (alk > 0))
    ok &= np.isfinite(si) & np.isfinite(po4)
    if not ok.any():                                     # nothing to compute (mocsy cannot take empty arrays)
        for tag, _ in sorted(targets.items(), key=lambda kv: kv[1]):
            yield tag, {name: np.full(t.shape, np.nan) for name in INDICATOR_NAMES}
        return
    tt, ss = t[ok], s[ok]
    a, d0 = alk[ok] * 1e-6, dic[ok] * 1e-6
    sil = np.broadcast_to(si * 1e-6, t.shape)[ok]
    phos = np.broadcast_to(po4 * 1e-6, t.shape)[ok]
    la = np.broadcast_to(lat2, t.shape)[ok]

    free_fac = total_to_free(tt, ss)
    out0 = mvars_all(tt, ss, d0, a, sil, phos, la)
    p0, om0, r0 = (np.asarray(out0[i], float) for i in (1, 6, 8))
    dry = 1.0 - water_vapour(tt, ss)
    prev = None                                          # (xCO2, dic, pCO2, Revelle) of the last solved target
    for tag, x in sorted(targets.items(), key=lambda kv: kv[1]):
        if x == XCO2_MODEL:
            d, out = d0, out0
        else:
            target = p0 + (x - XCO2_MODEL) * dry
            if prev is not None and abs(prev[0] - x) < abs(XCO2_MODEL - x):
                d, p, r = prev[1].copy(), prev[2], prev[3]
            else:
                d, p, r = d0.copy(), p0, r0
            for it in range(1, N_ITER_MAX + 1):
                with np.errstate(divide='ignore', invalid='ignore'):
                    d = d + (target - p) * d / (r * p)
                out = mvars_all(tt, ss, d, a, sil, phos, la)
                p, r = np.asarray(out[1], float), np.asarray(out[8], float)
                err = np.abs(p - target)[valid_omega(np.asarray(out[6], float), out[0]) & np.isfinite(p)]
                if err.size == 0 or float(err.max()) < PCO2_TOL:
                    break
            if iters is not None:
                iters[it] = iters.get(it, 0) + 1
            prev = (x, d, p, r)
            good = valid_omega(np.asarray(out[6], float), out[0]) & np.isfinite(p)
            resid[0] = max(resid[0], float(np.nanmax(np.abs(p - target)[good])) if good.any() else np.nan)
        om, p = np.asarray(out[6], float), np.asarray(out[1], float)
        good = valid_omega(om, out[0]) & np.isfinite(p)
        res = {}
        for name, (i, _, scale) in INDICATORS.items():
            f = np.full(t.shape, np.nan)
            f[ok] = np.where(good, np.asarray(out[i], float) * scale, np.nan)
            res[name] = f
        f = np.full(t.shape, np.nan)
        f[ok] = np.where(good, 10.0 ** (-np.asarray(out[0], float)) * 1e9, np.nan)     # [H+] nmol kg-1
        res['hplus'] = f
        f = np.full(t.shape, np.nan)
        f[ok] = np.where(good, 10.0 ** (-np.asarray(out[0], float)) * 1e9 * free_fac, np.nan)   # [H+] free
        res['hplus_free'] = f
        f = np.full(t.shape, np.nan)
        f[ok] = np.where(good, d * 1e6, np.nan)
        res['dic'] = f
        yield tag, res


def process(model, member, annual_fields=False, year_targets=None):
    YEAR_TARGETS = dict(year_targets or {})        # explicit years 1750-1849 {'y1750': xCO2, ...}
    tag_run = f'{model} {member}'
    with netCDF4.Dataset(adj_path(model, member)) as f:
        mask = np.asarray(f['mask'][:]).astype(bool)
        olat = np.asarray(f['lat'][:], 'f8')
        olon = np.asarray(f['lon'][:], 'f8')
        adj_years = np.asarray(f['year'][:])
        adj_field = np.ma.filled(f['omega_arag_adj'][:].astype('f8'), np.nan)
        pi_adj = float(np.asarray(f['omega_arag_adj_gm'][:]).mean())
    with netCDF4.Dataset(cb.bias_path(model, member)) as f:
        bias = {v: np.ma.filled(f[f'bias_{v}'][:].astype('f8'), np.nan) for v in VARS}
        y0, y1 = (int(x) for x in f.overlap.split('-'))
    lon2, lat2 = np.meshgrid(olon, olat)
    si, po4 = co.load_nutrients(olat, olon)
    drift = Drift(model, member)

    run = cb.ModelRun(model, 'piControl', member)
    jy = co.nearest_index(run.lat, olat)
    ix = co.nearest_index(run.lon, olon, periodic=True)
    ny, nx = lat2.shape
    nyr = run.nt // run.step
    if not np.array_equal(run.years[::run.step][:nyr], adj_years):
        raise ValueError(f'{tag_run}: piControl years differ from the bias-adjusted file')
    # annual means of Omega_arag per target (global-mean series, interannual std, optional annual fields), time
    # sums of every indicator for the means, monthly sums of Omega_arag for the climatologies (NaN propagates:
    # a cell needs every month)
    ALL = {**MAIN_TARGETS, **YEAR_TARGETS}
    if len(ALL) != len(MAIN_TARGETS) + len(YEAR_TARGETS):
        raise ValueError('target names of the years collide with the named targets')
    ann = {k: np.full((nyr, ny, nx), np.nan, 'f4') for k in MAIN_TARGETS}
    tsum = {k: {ind: np.zeros((ny, nx)) for ind in INDICATOR_NAMES} for k in ALL}
    # interannual std of the annual means of every indicator: sums of the annual means and of their squares,
    # shifted by the first year (no cancellation), without keeping the annual arrays
    aoff = {k: {ind: None for ind in INDICATOR_NAMES} for k in MAIN_TARGETS}
    asum = {k: {ind: np.zeros((ny, nx)) for ind in INDICATOR_NAMES} for k in MAIN_TARGETS}
    asq = {k: {ind: np.zeros((ny, nx)) for ind in INDICATOR_NAMES} for k in MAIN_TARGETS}
    csum = {k: np.zeros((12, ny, nx)) for k in MAIN_TARGETS}
    iters = {}
    ssum = {v: np.zeros((ny, nx)) for v in VARS}                 # time sums of the adjusted state
    sclim = {v: np.zeros((12, ny, nx)) for v in VARS}            # monthly sums of the adjusted state
    ccount = np.zeros(12)
    resid = [0.0]
    try:
        for i0 in range(0, run.nt, cb.CHUNK):
            i1 = min(i0 + cb.CHUNK, run.nt)
            _, _, fa = cb.adjusted_fields(run, i0, i1, bias, drift, (y0, y1), jy, ix, lon2, lat2)
            slot = np.arange(i0, i1) % run.step
            sl = slice(i0 // run.step, i1 // run.step)
            for k, res in target_states(fa, si, po4, lat2, ALL, resid, iters):
                for ind in INDICATOR_NAMES:
                    tsum[k][ind] += res[ind].sum(axis=0)
                if k not in MAIN_TARGETS:                      # explicit years: time means only
                    continue
                om = res['omega_arag']
                ann[k][sl] = om.reshape(-1, run.step, ny, nx).mean(axis=1)
                for ind in INDICATOR_NAMES:
                    am = res[ind].reshape(-1, run.step, ny, nx).mean(axis=1)      # annual means of the chunk
                    if aoff[k][ind] is None:
                        aoff[k][ind] = am[0].copy()
                    am = am - aoff[k][ind]
                    asum[k][ind] += am.sum(axis=0)
                    asq[k][ind] += (am ** 2).sum(axis=0)
                for m_ in range(run.step):
                    csum[k][m_] += om[slot == m_].sum(axis=0)
            for v in VARS:
                ssum[v] += fa[v].sum(axis=0)
                for m_ in range(run.step):
                    sclim[v][m_] += fa[v][slot == m_].sum(axis=0)
            for m_ in range(run.step):
                ccount[m_] += np.sum(slot == m_)
            print(f'  {tag_run}: {run.years[i0]}-{run.years[i1 - 1]} done', flush=True)
    finally:
        run.close()
    nt = float(run.nt)
    resid = resid[0]
    if run.step == 1:                                   # annual model: the 'climatology' is the time mean
        csum = {k: np.repeat(tsum[k]['omega_arag'][None], 12, axis=0) for k in MAIN_TARGETS}
        sclim = {v: np.repeat(ssum[v][None], 12, axis=0) for v in VARS}
        ccount = np.full(12, nt)

    m = mask.copy()
    for k in MAIN_TARGETS:
        m &= np.all(np.isfinite(ann[k]), axis=0)
    area = oa_config.area_weights(olat, olon)
    gm = {k: cb.masked_mean(ann[k], olat, olon, m) for k in MAIN_TARGETS}
    ddic = {k: (tsum[k]['dic'] - tsum['pi284']['dic']) / nt for k in MAIN_TARGETS}
    gm_ddic = cb.masked_mean(ddic['1750'], olat, olon, m)
    check = np.nanmax(np.abs(gm['pi284'] - cb.masked_mean(adj_field, olat, olon, m)))

    path = xco2_path(model, member)
    g = cb.new_nc(path, olat, olon, dict(
        title=f'Preindustrial Omega_arag of the drift-corrected, bias-adjusted piControl at other xCO2, {model}',
        method=(f'adjusted monthly piControl of calc_bias_adjusted.py ({cb.REF["label"]}, window {y0}-{y1}, drift '
                f'file {drift.tag}); DIC changed at fixed T, S, ALK so that surface pCO2 changes by '
                f'(xCO2 - {XCO2_MODEL}) * (1 - pH2O), Newton iteration with the mocsy Revelle factor; Omega_arag '
                f'monthly, then averaged; ' + oa_config.CONSTANTS),
        targets=', '.join(f'{k}={v}' for k, v in MAIN_TARGETS.items()),
        year_targets=(f'y{min(YEAR_TARGETS)[1:]}-y{max(YEAR_TARGETS)[1:]}: the years at their CO2 (CMIP6 historical '
                      'concentrations), time means only; xCO2 in the attribute of each variable')
                     if YEAR_TARGETS else 'none',
        newton=f'warm-started Newton iteration until max |pCO2 - target| < {PCO2_TOL} uatm (at most {N_ITER_MAX})',
        joint_mask=f'{100 * (area * m).sum() / (area * mask).sum():.2f} % of the bias-adjusted joint mask',
        pi284_check=f'max |gm(pi284) - gm(omega_arag_adj)| over years = {check:.2e}',
        history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} calc_pi_xco2.py, {oa_config.VERSION}'))
    g.createDimension('year', nyr)
    g.createDimension('month', 12)
    x = g.createVariable('year', 'i4', ('year',))
    x[:] = adj_years
    x = g.createVariable('mask', 'i1', ('lat', 'lon'))
    x.long_name = 'joint mask used for the global means'
    x[:] = m.astype('i1')
    for k in MAIN_TARGETS:
        x = g.createVariable(f'omega_arag_{k}_gm', 'f8', ('year',))
        x.xco2 = MAIN_TARGETS[k]
        x[:] = gm[k]
    x = g.createVariable('ddic_1750_gm', 'f8')
    x.units = 'umol kg-1'
    x.long_name = 'global mean (joint mask) of the time-mean DIC change from 284.317 to 278.3 ppm'
    x[...] = gm_ddic
    UNITS = UNITS_ALL
    for k in ALL:
        valid = np.all(np.isfinite(ann[k]), axis=0) if k in MAIN_TARGETS else np.isfinite(tsum[k]['omega_arag'])
        for ind in INDICATOR_NAMES:
            x = g.createVariable(f'{ind}_{k}_mean', 'f4', ('lat', 'lon'), zlib=True, fill_value=cb.FILL)
            x.long_name = (f'time mean of the monthly {ind} of the state at {ALL[k]} ppm over all piControl '
                           f'months, NaN where any month is missing')
            x.units = UNITS[ind]
            x.xco2 = ALL[k]
            x[:] = np.ma.masked_invalid(np.where(valid, tsum[k][ind] / nt, np.nan))
        if k not in MAIN_TARGETS:
            continue
        for ind in INDICATOR_NAMES:
            var = (asq[k][ind] - asum[k][ind] ** 2 / nyr) / (nyr - 1) if nyr > 1 else np.full((ny, nx), np.nan)
            x = g.createVariable(f'{ind}_{k}_std', 'f4', ('lat', 'lon'), zlib=True, fill_value=cb.FILL)
            x.long_name = (f'standard deviation of the annual mean {ind} over the piControl years at '
                           f'{ALL[k]} ppm (interannual variability)')
            x.units = UNITS[ind]
            x.xco2 = ALL[k]
            x[:] = np.ma.masked_invalid(np.where(valid, np.sqrt(np.maximum(var, 0.0)), np.nan))
        x = g.createVariable(f'omega_arag_{k}_clim', 'f4', ('month', 'lat', 'lon'), zlib=True, fill_value=cb.FILL)
        x.long_name = 'monthly climatology of Omega_arag over all piControl years'
        x.units = '1'
        x.xco2 = ALL[k]
        x[:] = np.ma.masked_invalid(np.where(valid[None], csum[k] / ccount[:, None, None], np.nan))
        if annual_fields:
            x = g.createVariable(f'omega_arag_{k}', 'f4', ('year', 'lat', 'lon'), zlib=True, fill_value=cb.FILL)
            x.units = '1'
            x[:] = np.ma.masked_invalid(ann[k])
    valid = np.all(np.isfinite(ann['1750']), axis=0)
    x = g.createVariable('ddic_1750_mean', 'f4', ('lat', 'lon'), zlib=True, fill_value=cb.FILL)
    x.long_name = 'time mean DIC change from 284.317 to 278.3 ppm'
    x.units = 'umol kg-1'
    x[:] = np.ma.masked_invalid(np.where(valid, ddic['1750'], np.nan))
    for v in VARS:
        x = g.createVariable(f'{v}_mean', 'f4', ('lat', 'lon'), zlib=True, fill_value=cb.FILL)
        x.long_name = f'time mean {v} of the adjusted piControl (284.317 ppm)'
        x.units = cb.UNITS[v]
        x[:] = np.ma.masked_invalid(np.where(valid, ssum[v] / nt, np.nan))
        x = g.createVariable(f'{v}_clim', 'f4', ('month', 'lat', 'lon'), zlib=True, fill_value=cb.FILL)
        x.long_name = (f'monthly climatology of {v} of the drift-corrected, bias-adjusted piControl (284.317 ppm) '
                       'over all piControl years; the preindustrial state of the model for the data product')
        x.units = cb.UNITS[v]
        x[:] = np.ma.masked_invalid(np.where(valid[None], sclim[v] / ccount[:, None, None], np.nan))
    g.close()

    with netCDF4.Dataset(cb.REF['omega']) as f:
        oyr = np.asarray(f['year'][:])
        obs = cb.masked_mean(np.ma.filled(f['omega_arag'][:].astype('f8'), np.nan), olat, olon, m)
    lines = [f'\n{tag_run}: max |pCO2 - target| {resid:.2e} uatm, mean dDIC(1750) {gm_ddic:+.2f} umol/kg, '
             f'joint mask {100 * (area * m).sum() / (area * mask).sum():.2f} % of the bias-adjusted mask',
             f'  pi284 check: max |gm(pi284) - gm(omega_arag_adj)| = {check:.2e} '
             f'(calc_bias_adjusted joint mask: {pi_adj:.3f})',
             '  Newton iterations per target and chunk: ' + ', '.join(f'{n}: {c}' for n, c in sorted(iters.items()))]
    if YEAR_TARGETS:
        w = area * m
        ys = sorted(YEAR_TARGETS)
        om_y = [np.nansum(np.where(m, tsum[k]['omega_arag'] / nt, 0) * w) / w.sum() for k in ys]
        lines.append(f'  explicit years: {ys[0]} {YEAR_TARGETS[ys[0]]:.2f} ppm {om_y[0]:.4f}, '
                     f'{ys[-1]} {YEAR_TARGETS[ys[-1]]:.2f} ppm {om_y[-1]:.4f} (time-mean Omega_arag, joint mask)')
    for k in MAIN_TARGETS:
        pi = gm[k].mean()
        ch = '  '.join(f'{y} {100 * (obs[oyr == y][0] / pi - 1):+.1f} %' for y in (2010, 2014, 2024)
                       if np.any(oyr == y))
        lines.append(f'  {k:10s} {MAIN_TARGETS[k]:6.1f} ppm: {pi:.3f}, {cb.REF_NAME} {ch}')
    print('\n'.join(lines), flush=True)


def run_one(model, member, annual_fields=False, year_targets=None):
    try:
        process(model, member, annual_fields, year_targets)
        return f'{model} {member}', 'done'
    except Exception:
        print(f'[FAILED] {model} {member}\n{traceback.format_exc()}', flush=True)
        return f'{model} {member}', 'FAILED'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', default='OceanSODA', choices=list(cb.REFERENCES))
    ap.add_argument('--models', nargs='+', help='default: all models')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--annual-fields', action='store_true',
                    help='also write the annual 2D fields of every target (large files)')
    ap.add_argument('--explicit-years', action='store_true',
                    help='also compute per-model states of every year 1750-1849 at its CO2 (tests only; the product '
                         'computes the years on the constrained state)')
    a = ap.parse_args()
    cb.set_reference(a.reference)
    yt = year_targets() if a.explicit_years else {}                 # needs the CMIP6 CO2 file in inputs/co2/
    if yt:
        print(f'explicit years: {len(yt)}, CO2 {min(yt.values()):.2f}-{max(yt.values()):.2f} ppm', flush=True)
    runs = [(m, r, a.annual_fields, yt) for m, r in find_runs(a.models)]
    print(f'{len(runs)} runs with bias-adjusted piControl -> {cb.OUT_DIR / "piControl"}', flush=True)
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
