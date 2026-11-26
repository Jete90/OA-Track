#!/usr/bin/env python
"""
make_oatrack_product.py  (OA-Track as published)
=======================
Writes OA-Track as defined in oatrack_names.py (every file, variable, dimension, long name and description from there),
from the complete output of the chain (make_oatrack.py, data/<run>/oatrack, unquantized):
    OA-Track_<q>_global_mean.nc (.csv)       global means 1750-2100                    (--part global)
    OA-Track_<q>_maps_obs_based_products.nc  maps of the obs-based products 1980-2025  (--part obs_based_products)
    OA-Track_<q>_maps_esms.nc                maps of the ESMs 1750-2100                (--part esms)

Global means, for each fCO2-product r and their mean, on the common area and on the area of the product (the global
series of OA-Track_<r>.nc), Omega_arag also on the whole ocean:
    values        preindustrial (state), obs_annual, obs_annual_forced (year_obs), esms_annual_forced (scenario: the
                  historical series of the 18 ESMs and the four SSPs); with the alkalinity adjustment
    _sigma        every term in quadrature (named as in oatrack_names.TERMS): sigma_delta, the uncertainty of delta,
                  the change that the bias-adjusted and constrained ESMs give from the reference period of the
                  fCO2-products to the state or year (with CO2; the ESM years and the states); sigma_mapping, the
                  mapping error with the same error in all months of a year; sigma_constants, the equilibrium
                  constants (not for the Revelle factor, mocsy gives none); sigma_alkalinity, the alkalinity
                  adjustment taken in full (|a|); for obs_annual_forced sigma_fit, the estimate of the forced state
                  (fit_sigma.py: AR(1) error of the quadratic fit and half the range of the alternative estimators, as
                  Extended Data Table 2); for the mean sigma_spread, the spread between the products; on the whole
                  ocean sigma_whole_ocean
    _alkalinity_adjustment  the file with minus the file without the adjustment (obs_annual_forced: the adjustment
                  evaluated at the fitted state of each year, as make_oatrack.py for 2024)
    _change       from the preindustrial state of the same product and area (state 1750 or 1850), relative (%), absolute
                  for pH, T and S; obs_annual_forced with the exact 1 sigma (the model side is only the preindustrial
                  state; sigma_mapping and sigma_constants cancel), the ESM years with an upper limit (sigma_delta of
                  both years in quadrature)
    mean          the mean of the four fCO2-products (obs_annual: of those present in the year); its 1 sigma with the
                  root mean square of sigma_delta and sigma_fit of the products, the mean of sigma_mapping and
                  sigma_constants, and sigma_spread
    whole ocean   (Omega_arag) the common area extended as in the paper (structural_adjustments.npz): the relative
                  change on the common area without the adjustment times (1 + r), r = polar_shift / the mean decline
                  1750-2024 (constant in time: test_polar_shift_models.py), from the 1750 level of the common area plus
                  the level difference of 1750; the alkalinity adjustment on the whole ocean equal to the paper in 1750
                  and 2024 and in between proportional to the change of the adjustment on the common area; 1 sigma of
                  the extension scaled with the change (levels: from the 1750 to the 2024 value). The 1750 level and
                  the change 1750-2024 equal those of the paper; the 2024 level of the paper (the common-area value plus
                  the difference whole ocean minus common area of the filled maps, which corresponds to an extension of
                  0.16 instead of 0.20 percentage points) is 0.002 higher (see the README of OA-Track)

Maps, each fCO2-product on its area (the cells shared with all 18 ESMs and covered in every year) and their mean, the
mean filled to the whole ocean where no fCO2-product covers the cell (fill_stack: the rule of fill_tools.py, as in the
paper):
    preindustrial  (both files) the states of 1750 and 1850; _sigma: sigma_delta, sigma_mapping, sigma_constants,
                  sigma_alkalinity
    obs_annual, obs_annual_forced  (obs-based products) as the global means, in every cell: the annual values
                  (sigma_mapping, sigma_constants, sigma_alkalinity) and the forced state (the fit of every cell with
                  sigma_fit of fit_sigma.py, the adjustment at the fitted state of the cell)
    esms_annual_forced  (ESMs, the mean only) the ESM years, historical and the four SSPs; _sigma: sigma_delta,
                  sigma_mapping, sigma_constants, sigma_alkalinity
    _change       from the preindustrial state of the same product; obs-based products: sigma_delta of the state, sigma_fit
                  and sigma_alkalinity; ESMs: sigma_delta of both years in quadrature (_sigma_upper_limit) and
                  sigma_alkalinity
    _sigma_alkalinity  sqrt(a^2 + s^2), a the adjustment and s its spread over the four single-model patterns, each
                  with its own m(t), recomputed exactly as make_oatrack.py computes the adjustment (pattern_terms,
                  esm_patterns; saved in <out>/work); for a change, the spread of the change
    mean          the mean of the fCO2-products covering the cell (and present in the year); 1 sigma with the root mean
                  square of sigma_delta and sigma_fit, the means of sigma_mapping, sigma_constants and sigma_alkalinity,
                  and sigma_spread (ddof 1); a change: the mean of the products' changes
    sigma_flag    quantities that depend on temperature: the 1 sigma of the levels left out where the mapping error of
                  temperature of a product exceeds 1 degC (oatrack_names.py)
Checks printed: the mean of 2024 against the 2024 trend values of make_oatrack.py, the whole ocean against the paper
(the figures notebook: the common-area means plus abs_ocean - abs_common and the alkalinity of abs_ocean, the decline
plus polar_shift and alk_cell_ocean of structural_adjustments.npz), the recomputed adjustment against the files (the
alkalinity increment, the central pattern and its 1 sigma), and the mean maps against OA-Track_all[_noalk].nc.

Needs: data/<run>/oatrack/OA-Track_<product>[_noalk].nc and OA-Track_all[_noalk].nc, data/<run>/structural/
structural_adjustments.npz (for the maps also structural_adjustments_<key>.npz and feedback_pattern.npz of the
patterns), the NOAA record (co2_annmean_gl.txt) and mocsy (make_product_alkalinity.Chem).
Usage (code/):  OA_RUN=u74 OMP_NUM_THREADS=1 python -u make_oatrack_product.py --part global --jobs 4
                OA_RUN=u74 OMP_NUM_THREADS=1 python -u make_oatrack_product.py --part obs_based_products --jobs 4
                OA_RUN=u74 OMP_NUM_THREADS=1 python -u make_oatrack_product.py --part esms --jobs 5
                (--quantities: some of them; --skip-patterns: the maps with the patterns already in <out>/work)
"""
import argparse
import csv
import datetime
import multiprocessing as mp_
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np

import fill_tools
import fit_sigma
import make_oatrack_simple as ms          # hat_fit, obs_years, noaa_co2, rd, DIGITS, MODE
import make_product_alkalinity as ma
import oa_config
import oatrack_names as N
import regional_common as rc
import state_constraint as sc

SRC = oa_config.DATA / 'oatrack'
OUT = oa_config.DATA / 'oatrack_product'
REFS = N.PRODUCTS[1:]                     # the four fCO2-products, in the order of the dimension product
SSPS = N.SCENARIOS[1:]
EVAL_YEAR = 2024
YEARS = np.arange(N.YEARS_GLOBAL[0], N.YEARS_GLOBAL[1] + 1)
STATES = ('1750', '1850')
VERSION = 'OA-Track version 1.0'
SOURCE = (f'{VERSION} (Terhaar, Xue and Hollitzer: Precise tracking of ocean acidification as it nears its planetary '
          'boundary; data https://doi.org/10.17882/121840; code https://doi.org/10.5281/zenodo.24493310)')
rd = ms.rd


def rms(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.sqrt(np.nanmean(np.square(a), axis))


def std(a, axis=0):
    n = np.sum(np.isfinite(a), axis)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.where(n > 1, np.nanstd(a, axis, ddof=1), np.nan)


def mean(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(a, axis)


def on_years(years, a):
    """a series on `years` placed on YEARS (1750-2100)"""
    x = np.full(YEARS.size, np.nan)
    x[np.searchsorted(YEARS, np.asarray(years).astype(int))] = a
    return x


# ------------------------------------------------------------------------------------------------ the forced state
def adjustment_at_fit(r):
    """the alkalinity adjustment of fCO2-product r in every year up to EVAL_YEAR at the fitted (forced) state of the
    year, as make_oatrack.py computes it for the 2024 trend value: the quadratic fits of T, S, TA and DIC without the
    adjustment, raised by the alkalinity increase of the year (talk_obs with minus without); returns the years, the maps
    {q: (year, lat, lon)} and their area means on the common area and on the area of the product"""
    t0 = time.time()
    with netCDF4.Dataset(SRC / f'OA-Track_{r}.nc') as f, netCDF4.Dataset(SRC / f'OA-Track_{r}_noalk.nc') as g:
        lat, lon = rd(f, 'lat'), rd(f, 'lon')
        yy, sel = ms.obs_years(f)
        masks = {'common': rd(f, 'mask_common') > 0, 'product': rd(f, 'mask') > 0}
        state = {q: rd(g, f'{q}_obs')[sel] for q in ('thetao', 'so', 'talk', 'dic')}
        dta = (rd(f, 'talk_obs') - rd(g, 'talk_obs'))[sel]
    ok = masks['product'] & np.all([np.all(np.isfinite(a), axis=0) for a in state.values()], axis=0)
    fit = {q: ms.hat_fit(yy, a, ok) for q, a in state.items()}
    chem = ma.Chem(lat, lon)
    D = {q: np.full(dta.shape, np.nan) for q in ma.DELTA_Q}
    for j in range(yy.size):
        d = chem.deltas(fit['thetao'][j], fit['so'][j], fit['talk'][j], fit['dic'][j], dta[j])
        for q in ma.DELTA_Q:
            D[q][j] = d[q]
    gm = {area: {q: sc.area_mean(D[q], lat, lon, m) for q in ma.DELTA_Q} for area, m in masks.items()}
    return r, yy, {q: a.astype('f4') for q, a in D.items()}, gm, time.time() - t0


ADJ = {}                                  # r -> (years, maps, area means), set in main before the pools
FS = {}                                   # r -> fit_sigma.FitSigma of its years, set in main before the pools


# ------------------------------------------------------------------------------------------------ global means
class Terms:
    """a value and its 1 sigma terms on one time axis: v, delta, c2, k, a, fit (missing terms: None)"""

    def __init__(self, v, delta=None, c2=None, k=None, a=None, fit=None):
        self.v, self.delta, self.c2, self.k, self.a, self.fit = v, delta, c2, k, a, fit

    def sigma(self, spread=None):
        parts = [p for p in (self.delta, spread, self.c2, self.k, self.a, self.fit) if p is not None]
        return np.sqrt(np.sum([np.square(np.nan_to_num(p)) for p in parts], 0)) * np.where(np.isfinite(self.v), 1, np.nan)


def product_series(q, r, area):
    """the global series of fCO2-product r on `area` ('common' or 'product'): {block: Terms} on YEARS, the states, the
    ssps; the adjustment as the file with minus the file without it"""
    sfx = '_common' if area == 'common' else ''
    alk = q in N.ALK
    const = q not in N.NO_CONSTANTS
    with netCDF4.Dataset(SRC / f'OA-Track_{r}.nc') as f, netCDF4.Dataset(SRC / f'OA-Track_{r}_noalk.nc') as g:
        b = f'{q}_gm{sfx}'
        yh = np.asarray(f['year'][:]).astype(int)
        ys = np.asarray(f['year_ssp'][:]).astype(int)
        yo = np.asarray(f['year_obs'][:]).astype(int)
        out = {}
        a = rd(f, f'{b}_global_state') - rd(g, f'{b}_global_state') if alk else None
        out['preindustrial'] = Terms(rd(f, f'{b}_global_state'), rd(f, f'{b}_global_state_sigma'),
                                     rd(f, f'{b}_sigma_mapping_c2_state'),
                                     rd(f, f'{b}_sigma_constants_state') if const else None, np.abs(a) if alk else None)
        out['preindustrial'].a_signed = a
        for scen, pre, years in [('historical', '', yh)] + [(s, f'_{s}', ys) for s in SSPS]:
            v = rd(f, f'{b}{pre}_global')
            a = v - rd(g, f'{b}{pre}_global') if alk else None
            t = Terms(on_years(years, v), on_years(years, rd(f, f'{b}{pre}_global_sigma')),
                      on_years(years, rd(f, f'{b}{pre}_sigma_mapping_c2')),
                      on_years(years, rd(f, f'{b}{pre}_sigma_constants')) if const else None,
                      on_years(years, np.abs(a)) if alk else None)
            t.a_signed = on_years(years, a) if alk else None
            out[('esms', scen)] = t
        v = rd(f, f'{b}_obs')
        a = v - rd(g, f'{b}_obs') if alk else None
        t = Terms(on_years(yo, v), None, on_years(yo, rd(f, f'{b}_obs_sigma_mapping_c2')),
                  on_years(yo, rd(f, f'{b}_obs_sigma_constants')) if const else None, on_years(yo, np.abs(a)) if alk else None)
        t.a_signed = on_years(yo, a) if alk else None
        out['obs_annual'] = t
        # the forced state: the fit of the values without the adjustment up to EVAL_YEAR, plus the adjustment at the
        # fitted state; its 1 sigma from the annual values (as Extended Data Table 2); mapping and constants of the year
        yy, sel = ms.obs_years(f)
        y0 = rd(g, f'{b}_obs')[sel]
        fit = ms.hat_fit(yy, y0, np.isfinite(y0).all())
        af = np.asarray(ADJ[r][2][area][q], 'f8') if alk else np.zeros(yy.size)
        assert not alk or np.array_equal(ADJ[r][0], yy), f'{r}: years of the adjustment'
        sfit = FS[r](y0[:, None])[:, 0]
        c2 = rd(f, f'{b}_obs_sigma_mapping_c2')[sel]
        k = rd(f, f'{b}_obs_sigma_constants')[sel] if const else None
        t = Terms(on_years(yy, fit + af), None, on_years(yy, c2), on_years(yy, k) if const else None,
                  on_years(yy, np.abs(af)) if alk else None, on_years(yy, sfit))
        t.a_signed = on_years(yy, af) if alk else None
        out['obs_annual_forced'] = t
        out['check_2024'] = float(f[f'{b}_2024_trend'][...])
        out['xco2'] = dict(historical=on_years(yh, rd(f, 'xco2')), **{s: on_years(ys, rd(f, f'xco2_{s}')) for s in SSPS},
                           obs=on_years(yo, rd(f, 'xco2_obs')), preindustrial=rd(f, 'state_xco2'))
    return out


def change(q, x1, d1, a1, x2, d2, a2, extra=()):
    """change from x1 (preindustrial, 1 sigma of the constraint d1, adjustment a1) to x2 (d2, a2: None for the years
    of the fCO2-products, where only the preindustrial side carries the constraint): (change, 1 sigma, part of the
    adjustment); relative (%) except for pH, T and S; extra: further 1 sigma terms of x2 (the fit), in its units"""
    a1 = 0.0 if a1 is None else a1
    a2 = 0.0 if a2 is None else a2
    if q in N.ABSOLUTE_CHANGE:
        c, c0 = x2 - x1, (x2 - a2) - (x1 - a1)
        parts = [d1] + ([d2] if d2 is not None else []) + list(extra)
    else:
        c, c0 = 100 * (x2 / x1 - 1), 100 * ((x2 - a2) / (x1 - a1) - 1)
        parts = [100 * x2 * d1 / x1 ** 2] + ([100 * d2 / x1] if d2 is not None else []) + [100 * e / x1 for e in extra]
    alk = c - c0
    parts.append(alk)
    return c, np.sqrt(sum(np.square(p) for p in parts)), alk


def global_quantity(q):
    """every variable of OA-Track_<q>_global_mean.nc: {name: array}, and the lines to print"""
    t0 = time.time()
    areas = ['common', 'product'] + (['whole'] if q in N.WHOLE_OCEAN else [])
    P = {r: {area: product_series(q, r, area) for area in ('common', 'product')} for r in REFS}
    nA, nP, nS, nY = len(areas), len(N.PRODUCTS), len(N.SCENARIOS), YEARS.size
    alk = q in N.ALK
    shape = dict(pre=(nA, nP, 2), obs=(nA, nP, nY), fit=(nA, nP, nY), esms=(nA, nP, nS, nY))
    out_names = {'pre': f'{q}_preindustrial_gm', 'obs': f'{q}_obs_annual_gm', 'fit': f'{q}_obs_annual_forced_gm',
                 'esms': f'{q}_esms_annual_forced_gm'}
    val = {k: np.full(shape[k], np.nan) for k in out_names}
    sig = {k: np.full(shape[k], np.nan) for k in out_names}
    adj = {k: np.full(shape[k], np.nan) for k in out_names} if alk else {}
    fch = np.full((nA, nP, 2, nY), np.nan)
    fch_s, fch_a = fch.copy(), fch.copy()
    ech = np.full((nA, nP, 2, nS, nY), np.nan)
    ech_s, ech_a = ech.copy(), ech.copy()
    info = []
    # every fCO2-product and their mean, on the common area and on the area of the product
    for ia, area in enumerate(('common', 'product')):
        T = {r: P[r][area] for r in REFS}
        keys = [('pre', 'preindustrial'), ('obs', 'obs_annual'), ('fit', 'obs_annual_forced')] + [
            ('esms', ('esms', s)) for s in N.SCENARIOS]
        for k, key in keys:
            for ip, r in enumerate(REFS, start=1):
                t = T[r][key]
                idx = (ia, ip) + ((N.SCENARIOS.index(key[1]),) if k == 'esms' else ())
                val[k][idx], sig[k][idx] = t.v, t.sigma()
                if alk:
                    adj[k][idx] = t.a_signed
            if area == 'product':
                continue                                            # the mean has no area of its own
            # the mean of the fCO2-products: rms of the constraint and fit terms, mean of mapping, constants and
            # adjustment, and their spread
            stack = lambda attr: np.array([getattr(T[r][key], attr) for r in REFS], 'f8') if getattr(
                T[REFS[0]][key], attr) is not None else None
            v = stack('v')
            m = Terms(mean(v), rms(stack('delta')) if stack('delta') is not None else None,
                      mean(stack('c2')), mean(stack('k')) if stack('k') is not None else None,
                      None, rms(stack('fit')) if stack('fit') is not None else None)
            if alk:
                a_m = mean(np.array([T[r][key].a_signed for r in REFS]))
                m.a = np.abs(a_m)
            idx = (ia, 0) + ((N.SCENARIOS.index(key[1]),) if k == 'esms' else ())
            val[k][idx], sig[k][idx] = m.v, m.sigma(std(v))
            if alk:
                adj[k][idx] = a_m
        # changes from the preindustrial state of the same product and area
        for ip, r in enumerate(REFS, start=1):
            pre = T[r]['preindustrial']
            for js in range(2):
                x1, d1, a1 = pre.v[js], pre.delta[js], pre.a_signed[js] if alk else None
                t = T[r]['obs_annual_forced']
                fch[ia, ip, js], fch_s[ia, ip, js], fch_a[ia, ip, js] = change(q, x1, d1, a1, t.v, None, t.a_signed, (t.fit,))
                for js_, s in enumerate(N.SCENARIOS):
                    t = T[r][('esms', s)]
                    ech[ia, ip, js, js_], ech_s[ia, ip, js, js_], ech_a[ia, ip, js, js_] = change(
                        q, x1, d1, a1, t.v, t.delta, t.a_signed)
        if area == 'common':
            # the mean: mean of the changes, rms of the products' terms without their adjustment part, the spread,
            # and the adjustment part of the mean change
            for arr, s_, a_, d2 in ((fch, fch_s, fch_a, None), (ech, ech_s, ech_a, 'esms')):
                c = arr[ia, 1:]
                a_m = mean(a_[ia, 1:])
                no_alk = np.sqrt(np.maximum(np.square(s_[ia, 1:]) - np.square(np.nan_to_num(a_[ia, 1:])), 0.0))
                arr[ia, 0] = mean(c)
                s_[ia, 0] = np.sqrt(rms(no_alk) ** 2 + np.nan_to_num(std(c)) ** 2 + np.nan_to_num(a_m) ** 2) * np.where(
                    np.isfinite(arr[ia, 0]), 1, np.nan)
                a_[ia, 0] = a_m
        # check: the mean of 2024 against the mean of the 2024 trend values of make_oatrack.py
        if area == 'common':
            i24 = list(YEARS).index(EVAL_YEAR)
            ref = np.mean([P[r]['common']['check_2024'] for r in REFS])
            info.append(f'  {q}: obs_annual_forced 2024 common area mean {val["fit"][ia, 0, i24]:.6g}, 2024 trend of '
                        f'make_oatrack.py {ref:.6g}, difference {val["fit"][ia, 0, i24] - ref:+.1e}')
    if 'whole' in areas:
        whole_ocean(q, val, sig, adj, fch, fch_s, fch_a, ech, ech_s, ech_a, P, info)
    V = {}
    for k, n in out_names.items():
        V[n], V[n + '_sigma'] = val[k], sig[k]
        if alk:
            V[n + '_alkalinity_adjustment'] = adj[k]
    n = f'{q}_obs_annual_forced_gm_change'
    V[n], V[n + '_sigma'] = fch, fch_s
    n2 = f'{q}_esms_annual_forced_gm_change'
    V[n2], V[n2 + '_sigma_upper_limit'] = ech, ech_s
    if alk:
        V[n + '_alkalinity_adjustment'], V[n2 + '_alkalinity_adjustment'] = fch_a, ech_a
    x = P[REFS[0]]['common']['xco2']
    V['xco2_esms_annual'] = np.array([x[s] for s in N.SCENARIOS])
    V['xco2_obs_annual'] = x['obs']
    V['xco2_preindustrial'] = x['preindustrial']
    info.append(f'  {q}: global means, {time.time() - t0:.0f} s')
    return q, V, info


SA = {}                                   # structural_adjustments.npz, set in main


def whole_ocean(q, val, sig, adj, fch, fch_s, fch_a, ech, ech_s, ech_a, P, info):
    """the whole ocean (index 2 of area) for every product and the mean, from the common area (index 0), as the paper"""
    alk = q in N.ALK
    i24 = list(YEARS).index(EVAL_YEAR)
    ac, ao = SA['abs_common'], SA['abs_ocean']               # 1750, 2024, boundary; with alkalinity: 3, 4, 5
    pol_d50, pol_d24 = float(ao[0] - ac[0]), float(ao[1] - ac[1])
    alk_d50, alk_d24 = float(ao[3] - ao[0]), float(ao[4] - ao[1])
    s50, s24, _ = (float(v) for v in SA['polar_level_sigma'])
    pol_shift, pol_sig = float(SA['polar_shift']), float(SA['polar_sigma'])
    alk_shift = float(SA['alk_cell_ocean'])
    c, w = 0, 2
    # without the adjustment, on the common area
    a0 = lambda arr: np.nan_to_num(arr) if alk else 0.0
    pre0 = val['pre'][c] - a0(adj['pre'][c]) if alk else val['pre'][c]          # (product, state)
    fit0 = val['fit'][c] - a0(adj['fit'][c]) if alk else val['fit'][c]          # (product, year)
    esm0 = val['esms'][c] - a0(adj['esms'][c]) if alk else val['esms'][c]       # (product, scenario, year)
    dec_m = float(np.mean(100 * (1 - fit0[1:, i24] / pre0[1:, 0])))             # the mean decline 1750-2024, as the paper
    r_w = pol_shift / dec_m
    # the adjustment on the whole ocean: equal to the paper in 1750 and 2024 (abs_ocean), in between and after
    # proportional to the change of the adjustment on the common area (the mean of the products)
    if alk:
        ac50, ac24 = adj['pre'][c, 0, 0], adj['fit'][c, 0, i24]
        slope = (alk_d24 - alk_d50) / (ac24 - ac50)
    lev1750 = pre0[:, 0] + pol_d50                                             # (product,)

    def whole(x0, a_c):
        """level without the adjustment on the whole ocean from the common-area level x0, plus the adjustment"""
        rel = x0 / pre0[:, 0].reshape((-1,) + (1,) * (x0.ndim - 1)) - 1          # relative change on the common area
        lev = lev1750.reshape((-1,) + (1,) * (x0.ndim - 1)) * (1 + rel * (1 + r_w))
        return lev, rel, (alk_d50 + slope * (a_c - ac50) if alk else 0.0)
    for k, x0 in (('pre', pre0), ('fit', fit0), ('esms', esm0)):
        lev, rel, a_w = whole(x0, adj[k][c] if alk else None)
        val[k][w] = lev + a_w
        rel24 = 1 - fit0[0, i24] / pre0[0, 0]
        s_lev = s50 + (s24 - s50) * (-rel) / rel24                              # level 1 sigma, from 1750 to 2024
        s_c2 = np.square(sig[k][c]) - (np.square(np.nan_to_num(adj[k][c])) if alk else 0.0)
        sig[k][w] = np.sqrt(np.maximum(s_c2, 0) + np.square(a_w if alk else 0.0) + np.square(s_lev))
        if alk:
            adj[k][w] = a_w
    # changes on the whole ocean from its own levels; 1 sigma: the common-area terms without the adjustment, scaled
    # with (1 + r), the extension (polar_sigma scaled with the change) and the adjustment part
    for arr, s_, a_, k in ((fch, fch_s, fch_a, 'fit'), (ech, ech_s, ech_a, 'esms')):
        for js in range(2):
            x1, x1a = val['pre'][w][:, js], (adj['pre'][w][:, js] if alk else 0.0)
            sh = (-1,) + (1,) * (val[k][w].ndim - 1)
            x2, x2a = val[k][w], (adj[k][w] if alk else 0.0)
            if q in N.ABSOLUTE_CHANGE:
                cc, c0 = x2 - x1.reshape(sh), (x2 - x2a) - (x1 - x1a).reshape(sh)
            else:
                cc, c0 = 100 * (x2 / x1.reshape(sh) - 1), 100 * ((x2 - x2a) / (x1 - x1a).reshape(sh) - 1)
            no_alk = np.sqrt(np.maximum(np.square(s_[c, :, js]) - np.square(np.nan_to_num(a_[c, :, js])), 0.0))
            ext = pol_sig * np.abs(c0) / dec_m
            arr[w, :, js] = cc
            s_[w, :, js] = np.sqrt(np.square(no_alk * (1 + r_w)) + np.square(ext) + np.square(cc - c0))
            a_[w, :, js] = cc - c0
    om50 = val['pre'][w][0, 0]
    om24 = val['fit'][w][0, i24]
    lev24 = om24 - (adj['fit'][w][0, i24] if alk else 0.0)
    d24 = fch[w, 0, 0, i24]
    # the paper (figures notebook): OM50_G = yc_m + ALK_D50 + POL_D50, OM24_G = now_m + ALK_D24 + POL_D24,
    # DEC_G = dec_m + POL_SHIFT + ALK_SHIFT (yc_m, now_m: the means of the products on the common area, without the adjustment)
    p50 = float(np.mean(pre0[1:, 0])) + pol_d50 + (alk_d50 if alk else 0.0)
    p24n = float(np.mean(fit0[1:, i24])) + pol_d24
    p24 = p24n + (alk_d24 if alk else 0.0)
    info.append(f'  {q}: whole ocean (r = {r_w:.4f}, alkalinity slope {slope if alk else 0:.4f}): 1750 {om50:.4f} (paper {p50:.4f}), '
                f'change 1750-2024 {d24:+.3f} % (paper {-(dec_m + pol_shift + alk_shift):+.3f} %), alkalinity 1750 '
                f'{adj["pre"][w][0, 0] if alk else 0:+.4f} (paper {alk_d50:+.4f}); 2024 {om24:.4f}, without the adjustment '
                f'{lev24:.4f} (paper {p24:.4f} and {p24n:.4f}: the map difference of 2024, extension '
                f'{float(SA["polar_products_shift_constrained"]):.2f} instead of {pol_shift:.2f} points, so that level and '
                'decline of the paper differ; README)')


# ------------------------------------------------------------------------------------------------ maps (both parts)
WORK = OUT / 'work'
FILL = {}                                 # have, need, region, w of the filling of the mean, set in maps_setup
TFLAG = {}                                # 'cells': the cells of sigma_flag, set in maps_setup
CHECK_Q = ('omega_arag', 'ph', 'dic')      # quantities of the checks of the recomputed adjustment


def maps_setup():
    """the cells of sigma_flag, and the filling of the mean: the cells covered by at least one fCO2-product (have) and
    the other cells of the ocean (need)"""
    TFLAG['cells'], n = ms.temperature_flag()
    print('sigma_flag: ' + ', '.join(f'{r} {k}' for r, k in n.items()) + f' cells; {int(TFLAG["cells"].sum())} in all',
          flush=True)
    union = None
    for r in REFS:
        with netCDF4.Dataset(SRC / f'OA-Track_{r}.nc') as f:
            m = rd(f, 'mask') > 0
            lat, lon = rd(f, 'lat'), rd(f, 'lon')
        union = m if union is None else union | m
    ocean = rc.ocean_on(lat, lon).astype(bool)
    FILL.update(have=union, need=ocean & ~union, region=fill_tools.regions(lat, union.shape),
                w=oa_config.area_weights(lat, lon))
    print(f'filled: {int(FILL["need"].sum())} cells of the ocean without an fCO2-product, '
          f'{100 * FILL["w"][FILL["need"]].sum() / FILL["w"][ocean].sum():.1f} % of the ocean', flush=True)


class Check:
    """the largest difference of a field from its reference, relative to the largest rms of the reference (over the
    cells; the largest over all calls of the same name and key)"""

    def __init__(self):
        self.e, self.s = {}, {}

    def add(self, name, x, ref, key=()):
        k = (name,) + tuple(key)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            d = np.abs(np.asarray(x, 'f8') - np.asarray(ref, 'f8'))
            e = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
            s = float(np.sqrt(np.nanmean(np.square(ref)))) if np.isfinite(ref).any() else 0.0
        self.e[k], self.s[k] = max(self.e.get(k, 0.0), e), max(self.s.get(k, 0.0), s)

    def ratio(self, name):
        r = [self.e[k] / self.s[k] if self.s[k] > 0 else (0.0 if self.e[k] == 0 else np.inf) for k in self.e
             if k[0] == name]
        return max(r) if r else 0.0


def alk_setup(r):
    """the alkalinity adjustment of fCO2-product r as make_oatrack.py computes it, for its recomputation with the
    central pattern and the four single-model patterns (make_product_alkalinity.PATTERN_ALTERNATIVES, each with its own
    m(t); Chem.deltas_multi: the change at constant pCO2 of the state without the adjustment): the patterns on the area
    of the product, their Params, the preindustrial states with (st) and without (st0) the adjustment, and the
    adjustment of every pattern at the two states (Dst; m of the states from the common-area Omega_arag,
    make_product_alkalinity.m_states). Checks: the central adjustment of the states against the file (chk_a) and its
    1 sigma sqrt(a^2 + s^2), s the spread of the four patterns, against _state_sigma_alkalinity of the file (chk_s;
    compared as stored: the spread recovered from the file, sqrt(sigma^2 - a^2), loses its precision in float32 where
    s is much smaller than a); both the largest difference in any cell relative to the rms of the adjustment of the
    file (the adjustment crosses zero with the pattern, so that a cell-by-cell ratio says nothing there)"""
    with netCDF4.Dataset(SRC / f'OA-Track_{r}.nc') as f, netCDF4.Dataset(SRC / f'OA-Track_{r}_noalk.nc') as g:
        lat, lon = rd(f, 'lat'), rd(f, 'lon')
        own, common = rd(f, 'mask') > 0, rd(f, 'mask_common') > 0
        yh = np.asarray(f['year'][:]).astype(int)
        st0 = {q: rd(g, f'{q}_state') for q in list(ma.DELTA_Q) + ['thetao', 'so']}
        st = {q: rd(f, f'{q}_state') for q in ma.DELTA_Q}
        sa_st = {q: rd(f, f'{q}_state_sigma_alkalinity') for q in ma.DELTA_Q}
        am_h = sc.area_mean(rd(g, 'omega_arag'), lat, lon, common & own)
        om_st = sc.area_mean(st0['omega_arag'], lat, lon, common & own)
    a_st = {q: st[q] - st0[q] for q in ma.DELTA_Q}
    keys = ma.PATTERN_ALTERNATIVES
    P, Pk = ma.Params(), [ma.Params(k) for k in keys]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)          # zonal means of latitudes without ocean
        p0 = np.where(own, ma.load_pattern(lat, lon)[0], np.nan)
        pk = [np.where(own, ma.load_pattern(lat, lon, k)[0], np.nan) for k in keys]
    m_st = [ma.m_states(om_st, yh, am_h, Q.on(Q.m_model[r], yh)) for Q in [P] + Pk]
    chem = ma.Chem(lat, lon)
    nk = len(keys)
    Dst = [chem.deltas_multi(st0['thetao'][j], st0['so'][j], st0['talk'][j], st0['dic'][j],
                             [p0 * m_st[0][j]] + [pk[i] * m_st[1 + i][j] for i in range(nk)]) for j in range(2)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        scale = {(j, q): float(np.sqrt(np.nanmean(np.square(a_st[q][j])))) for j in range(2) for q in CHECK_Q}
        chk_a = max(float(np.nanmax(np.abs(Dst[j][0][q] - a_st[q][j]))) / scale[j, q] for j in range(2) for q in CHECK_Q)
        chk_s = max(float(np.nanmax(np.abs(np.hypot(a_st[q][j], np.std([Dst[j][1 + i][q] for i in range(nk)], 0, ddof=1))
                                           - sa_st[q][j])))
                    / scale[j, q] for j in range(2) for q in CHECK_Q)
    return SimpleNamespace(lat=lat, lon=lon, own=own, st0=st0, st=st, p0=p0, pk=pk, P=P, Pk=Pk, nk=nk, chem=chem,
                           Dst=Dst, chk_a=chk_a, chk_s=chk_s)


def _shift3(a, dj, di):
    """fill_tools._shift for a stack (k, ny, nx)"""
    r = np.roll(a, -di, axis=-1)
    if dj == 0:
        return r
    out = np.zeros_like(r)
    if dj > 0:
        out[:, :-dj] = r[:, dj:]
    else:
        out[:, -dj:] = r[:, :dj]
    return out


def fill_stack(a, have, need, region, w):
    """fill_tools.fill_to for a stack (..., ny, nx) of fields at once: the cells need & ~have filled from the finite
    cells of have around them (5 x 5, the 8 neighbours with twice the weight, in passes from the edges inward), the
    cells no pass reaches with the area mean of the filled cells of their region (as fill_tools.regional_fill)"""
    A = np.array(a, 'f8')
    sh = A.shape
    A = A.reshape((-1,) + have.shape)
    known = have[None] & np.isfinite(A)
    todo = np.broadcast_to(need & ~have, A.shape).copy()
    rings = fill_tools.FILL_RINGS
    while todo.any():
        den = np.zeros(A.shape)
        num = np.zeros(A.shape)
        z = np.where(known, A, 0.0)
        for dj in range(-2, 3):
            for di in range(-2, 3):
                d = max(abs(dj), abs(di))
                if d == 0:
                    continue
                den += rings[d] * _shift3(known, dj, di)
                num += rings[d] * _shift3(z, dj, di)
        new = todo & (den > 0)
        if not new.any():
            break
        A[new] = num[new] / den[new]
        known |= new
        todo &= ~new
    if todo.any():
        filled = (need & ~have)[None] & ~todo & np.isfinite(A)
        for g_ in ('Arctic', 'Southern', 'other'):
            rg = region == g_
            for i in range(A.shape[0]):
                h, m = filled[i] & rg, todo[i] & rg
                if h.any() and m.any():
                    A[i][m] = np.sum(A[i][h] * w[h]) / w[h].sum()
    return A.reshape(sh)


def mean_block(B, first, c2, k, alk):
    """index 0 of the block B {v, s, a, sa} (product, ..., lat, lon): the mean of the fCO2-products covering the cell;
    its 1 sigma with the root mean square of `first` (sigma_delta, or the fit of the forced state; None: no such term), the
    spread (ddof 1), and the means of the mapping (c2), constants (k) and alkalinity terms"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        v = B['v'][1:]
        n = np.sum(np.isfinite(v), 0)
        sp = np.where(n > 1, np.nanstd(v, 0, ddof=1), 0.0)
        B['v'][0] = np.nanmean(v, 0)
        f_ = np.sqrt(np.nanmean(np.square(first[1:]), 0)) if first is not None else 0.0
        sa_m = np.nanmean(B['sa'][1:], 0) if alk else 0.0
        B['s'][0] = np.sqrt(np.square(f_) + sp ** 2 + np.nanmean(c2[1:], 0) ** 2 + np.nanmean(k[1:], 0) ** 2
                            + np.square(sa_m)) * np.where(np.isfinite(B['v'][0]), 1, np.nan)
        if alk:
            B['a'][0], B['sa'][0] = np.nanmean(B['a'][1:], 0), sa_m


def change_mean(c, noalk, a, sa_m):
    """the mean change of the fCO2-products covering the cell (c: product first), its 1 sigma with the root mean square
    of the products' terms without the alkalinity part (noalk), the spread (ddof 1) and the alkalinity part of the mean
    sa_m, and the mean alkalinity part of the change (a; None: none)"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        n = np.sum(np.isfinite(c), 0)
        sp = np.where(n > 1, np.nanstd(c, 0, ddof=1), 0.0)
        v = np.nanmean(c, 0)
        s = np.sqrt(np.nanmean(np.square(noalk), 0) + sp ** 2 + np.square(sa_m)) * np.where(np.isfinite(v), 1, np.nan)
        return v, s, (np.nanmean(a, 0) if a is not None else None)


def flag_and_fill(q, blocks, flagged):
    """the 1 sigma of the blocks in `flagged` left out in the cells of sigma_flag (quantities of N.T_DEPENDENT; known
    limitation of version 1.0), the mean (index 0) of every block {v, s, a, sa} filled to the whole ocean (FILL), and
    the 1 sigma of the filled means of `flagged` left out again in the cells of sigma_flag"""
    alk, flag = q in N.ALK, q in N.T_DEPENDENT
    if flag:
        for B in flagged:
            B['s'][..., TFLAG['cells']] = np.nan
    have, need, region, w = FILL['have'], FILL['need'], FILL['region'], FILL['w']
    for B in blocks:
        for key in B:
            if alk or key in ('v', 's'):
                B[key][0] = fill_stack(B[key][0], have, need, region, w).astype('f4')
    if flag:
        for B in flagged:
            B['s'][0][..., TFLAG['cells']] = np.nan


def preindustrial_block(q):
    """the preindustrial states of every fCO2-product and their mean (index 0; not yet flagged and filled): the grid,
    the common area, the areas of the products (index 0: their union), the block {v, s, a, sa} (product, state, lat,
    lon), and the sigma_delta, mapping and constants terms of the products {d, c2, k}"""
    alk, const = q in N.ALK, q not in N.NO_CONSTANTS
    with netCDF4.Dataset(SRC / f'OA-Track_{REFS[0]}.nc') as f:
        lat, lon = rd(f, 'lat'), rd(f, 'lon')
        common = rd(f, 'mask_common') > 0
    ny, nx, nP = lat.size, lon.size, len(N.PRODUCTS)
    pre = {k: np.full((nP, 2, ny, nx), np.nan, 'f4') for k in ('v', 's', 'a', 'sa')}
    pd = {k: np.full((nP, 2, ny, nx), np.nan, 'f4') for k in ('d', 'c2', 'k')}
    own_masks = np.zeros((nP, ny, nx), bool)
    for ip, r in enumerate(REFS, start=1):
        with netCDF4.Dataset(SRC / f'OA-Track_{r}.nc') as f, netCDF4.Dataset(SRC / f'OA-Track_{r}_noalk.nc') as g:
            own_masks[ip] = rd(f, 'mask') > 0
            v = rd(f, f'{q}_state')
            d, c2 = rd(f, f'{q}_state_sigma'), rd(f, f'{q}_state_sigma_mapping_c2')
            k = rd(f, f'{q}_state_sigma_constants') if const else 0.0
            sa = rd(f, f'{q}_state_sigma_alkalinity') if alk else 0.0
            pre['v'][ip] = v
            pre['s'][ip] = np.sqrt(d ** 2 + c2 ** 2 + np.square(k) + np.square(sa))
            pd['d'][ip], pd['c2'][ip], pd['k'][ip] = d, c2, k
            if alk:
                pre['a'][ip], pre['sa'][ip] = v - rd(g, f'{q}_state'), sa
    own_masks[0] = own_masks[1:].any(0)
    assert np.array_equal(FILL['have'], own_masks[0]), 'the area of the mean differs from the filling'
    mean_block(pre, pd['d'], pd['c2'], pd['k'], alk)
    return lat, lon, common, own_masks, pre, pd


# ------------------------------------------------------------------------------------------------ maps of the obs-based products
def pattern_terms(r):
    """the alkalinity adjustment of fCO2-product r with the central pattern and with the four single-model patterns
    (alk_setup) at the fitted (forced) state of every year to 2024 (central: the increase of the year in the file), and
    the change from the two preindustrial states. Saved per quantity in WORK: a_fit (year): the central adjustment at
    the fitted state; s_fit (year): the spread (ddof 1) of the four patterns; s_change (state, year): the spread of the
    change from the preindustrial state over the four patterns. Checks: those of alk_setup, and the increment of the
    alkalinity in the file against pattern x m"""
    t0 = time.time()
    A = alk_setup(r)
    qs = list(ma.DELTA_Q) + ['thetao', 'so']
    with netCDF4.Dataset(SRC / f'OA-Track_{r}.nc') as f, netCDF4.Dataset(SRC / f'OA-Track_{r}_noalk.nc') as g:
        yy, sel = ms.obs_years(f)
        obs0 = {q: rd(g, f'{q}_obs')[sel] for q in qs}
        dta = (rd(f, 'talk_obs') - rd(g, 'talk_obs'))[sel]
    own, st0, pk, nk, chem = A.own, A.st0, A.pk, A.nk, A.chem
    ok = own & np.all([np.all(np.isfinite(obs0[q]), axis=0) for q in ('thetao', 'so', 'talk', 'dic')], axis=0)
    fit0 = {q: ms.hat_fit(yy, obs0[q], ok) for q in qs}
    m0 = A.P.on(A.P.m_product[r], yy)
    mk = [Q.on(Q.m_product[r], yy) for Q in A.Pk]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        d_inc = float(np.nanmax(np.abs(dta - A.p0[None] * m0[:, None, None])))
    ny, nx = A.lat.size, A.lon.size
    a_fit = {q: np.full((yy.size, ny, nx), np.nan, 'f4') for q in ma.DELTA_Q}
    s_fit = {q: np.full((yy.size, ny, nx), np.nan, 'f4') for q in ma.DELTA_Q}
    s_ch = {q: np.full((2, yy.size, ny, nx), np.nan, 'f4') for q in ma.DELTA_Q}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for j in range(yy.size):
            Ds = chem.deltas_multi(fit0['thetao'][j], fit0['so'][j], fit0['talk'][j], fit0['dic'][j],
                                   [dta[j]] + [pk[i] * mk[i][j] for i in range(nk)])
            for q in ma.DELTA_Q:
                a_fit[q][j] = Ds[0][q]
                alt = np.array([Ds[1 + i][q] for i in range(nk)])
                s_fit[q][j] = np.std(alt, 0, ddof=1)
                for js in range(2):
                    alt1 = np.array([A.Dst[js][1 + i][q] for i in range(nk)])
                    x1, x2 = st0[q][js], fit0[q][j]
                    ck = (x2 + alt) - (x1 + alt1) if q in N.ABSOLUTE_CHANGE else 100 * ((x2 + alt) / (x1 + alt1) - 1)
                    s_ch[q][js, j] = np.std(ck, 0, ddof=1)
    WORK.mkdir(parents=True, exist_ok=True)
    for q in ma.DELTA_Q:
        for nm, arr in (('a_fit', a_fit[q]), ('s_fit', s_fit[q]), ('s_change', s_ch[q])):
            np.save(WORK / f'{r}_{q}_{nm}.npy', arr)
    np.save(WORK / f'{r}_years.npy', yy)
    return r, (f'{r:10s} alkalinity patterns at the forced state {yy[0]}-{yy[-1]} and the two states, {time.time() - t0:.0f} s; '
               f'checks: increment of the file against pattern x m {d_inc:.1e} umol/kg, central adjustment of the states '
               f'against the file {A.chk_a:.1e}, its 1 sigma with the spread of the patterns against the file '
               f'{A.chk_s:.1e} (largest difference relative to the rms of the adjustment, {", ".join(CHECK_Q)})')


def obs_maps_quantity(q, out):
    """OA-Track_<q>_maps_obs_based_products.nc: every fCO2-product on its area and their mean, filled to the whole ocean"""
    t0 = time.time()
    alk, const = q in N.ALK, q not in N.NO_CONSTANTS
    yo = np.arange(N.YEARS_OBS[0], N.YEARS_OBS[1] + 1)
    lat, lon, common, own_masks, pre, pd = preindustrial_block(q)
    ny, nx, nP = lat.size, lon.size, len(N.PRODUCTS)
    blk = lambda n_t: {k: np.full((nP, n_t, ny, nx), np.nan, 'f4') for k in ('v', 's', 'a', 'sa')}
    obs, fit = blk(yo.size), blk(yo.size)
    ch = {k: np.full((nP, 2, yo.size, ny, nx), np.nan, 'f4') for k in ('v', 's', 'a', 'sa')}
    parts = {k: np.full((nP, yo.size, ny, nx), np.nan, 'f4') for k in ('obs_c2', 'obs_k', 'fit_c2', 'fit_k', 'fit_f')}
    parts['ch_noalk'] = np.full((nP, 2, yo.size, ny, nx), np.nan, 'f4')
    info = []
    for ip, r in enumerate(REFS, start=1):
        with netCDF4.Dataset(SRC / f'OA-Track_{r}.nc') as f, netCDF4.Dataset(SRC / f'OA-Track_{r}_noalk.nc') as g:
            own = own_masks[ip]
            yr = np.asarray(f['year_obs'][:]).astype(int)
            io = np.searchsorted(yo, yr)
            # the annual values
            v = rd(f, f'{q}_obs')
            c2 = rd(f, f'{q}_obs_sigma_mapping_c2')
            k = rd(f, f'{q}_obs_sigma_constants') if const else np.zeros(v.shape)
            sa = rd(f, f'{q}_obs_sigma_alkalinity') if alk else np.zeros(v.shape)
            obs['v'][ip, io] = v
            obs['s'][ip, io] = np.sqrt(c2 ** 2 + k ** 2 + sa ** 2)
            parts['obs_c2'][ip, io], parts['obs_k'][ip, io] = c2, k
            if alk:
                obs['a'][ip, io], obs['sa'][ip, io] = v - rd(g, f'{q}_obs'), sa
            # the forced state: the fit of the values without the adjustment, plus the adjustment at the fitted state
            yy, sel = ms.obs_years(f)
            y0 = rd(g, f'{q}_obs')[sel]
            ok = own & np.all(np.isfinite(y0), axis=0)
            f0 = ms.hat_fit(yy, y0, ok)
            sf = np.where(ok[None], FS[r](np.where(ok[None], y0, np.nan)), np.nan)
            if alk:
                assert np.array_equal(np.load(WORK / f'{r}_years.npy'), yy), f'{r}: years of the patterns'
                af = np.load(WORK / f'{r}_{q}_a_fit.npy').astype('f8')
                sfa = np.sqrt(af ** 2 + np.load(WORK / f'{r}_{q}_s_fit.npy').astype('f8') ** 2)
            else:
                af, sfa = np.zeros(f0.shape), np.zeros(f0.shape)
            iy = np.searchsorted(yo, yy)
            c2, k = c2[sel], k[sel]
            fit['v'][ip, iy] = f0 + af
            fit['s'][ip, iy] = np.sqrt(c2 ** 2 + k ** 2 + sf ** 2 + sfa ** 2)
            parts['fit_c2'][ip, iy], parts['fit_k'][ip, iy], parts['fit_f'][ip, iy] = c2, k, sf
            if alk:
                fit['a'][ip, iy], fit['sa'][ip, iy] = af, sfa
            # the change from the preindustrial state: only its constraint, the fit and the adjustment
            s_chg = np.load(WORK / f'{r}_{q}_s_change.npy').astype('f8') if alk else None
            for js in range(2):
                x1, d1 = pre['v'][ip, js].astype('f8'), pd['d'][ip, js].astype('f8')
                a1 = pre['a'][ip, js].astype('f8') if alk else 0.0
                x2 = f0 + af
                if q in N.ABSOLUTE_CHANGE:
                    c, c0 = x2 - x1, f0 - (x1 - a1)
                    t1, tf = d1, sf
                else:
                    c, c0 = 100 * (x2 / x1 - 1), 100 * (f0 / (x1 - a1) - 1)
                    t1, tf = 100 * x2 * d1 / x1 ** 2, 100 * sf / x1
                s_alk = np.sqrt((c - c0) ** 2 + (s_chg[js] ** 2 if alk else 0.0))
                ch['v'][ip, js, iy] = c
                ch['s'][ip, js, iy] = np.sqrt(t1 ** 2 + tf ** 2 + s_alk ** 2)
                parts['ch_noalk'][ip, js, iy] = np.sqrt(t1 ** 2 + tf ** 2)
                if alk:
                    ch['a'][ip, js, iy], ch['sa'][ip, js, iy] = c - c0, s_alk
    # the mean of the fCO2-products covering the cell (and present in the year)
    mean_block(obs, None, parts['obs_c2'], parts['obs_k'], alk)
    mean_block(fit, parts['fit_f'], parts['fit_c2'], parts['fit_k'], alk)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sa_m = np.nanmean(ch['sa'][1:], 0) if alk else 0.0
    ch['v'][0], ch['s'][0], a_m = change_mean(ch['v'][1:], parts['ch_noalk'][1:], ch['a'][1:] if alk else None, sa_m)
    if alk:
        ch['a'][0], ch['sa'][0] = a_m, sa_m
    # checks against the combined file of make_oatrack.py (the mean of the products covering the cell)
    with netCDF4.Dataset(SRC / 'OA-Track_all.nc') as A:
        rel = lambda x, y: float(np.nanmax(np.abs(x - y) / np.maximum(np.abs(y), 1e-12)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ya = np.asarray(A['year_obs'][:]).astype(int)
            info.append(f'  {q}: mean against OA-Track_all.nc: preindustrial {rel(pre["v"][0], rd(A, f"{q}_state")):.1e}, '
                        f'annual values {rel(obs["v"][0][np.searchsorted(yo, ya)], rd(A, f"{q}_obs")):.1e}, forced 2024 '
                        f'{rel(fit["v"][0][list(yo).index(EVAL_YEAR)], rd(A, f"{q}_2024_trend")):.1e} (relative)')
    # the 1 sigma left out in the cells of sigma_flag, the mean filled to the whole ocean
    flag_and_fill(q, (pre, obs, fit, ch), (pre,) + ((obs, fit) if q == 'thetao' else ()))
    V = {}
    for B, n in ((pre, f'{q}_preindustrial'), (obs, f'{q}_obs_annual'), (fit, f'{q}_obs_annual_forced'),
                 (ch, f'{q}_obs_annual_forced_change')):
        V[n], V[n + '_sigma'] = B['v'], B['s']
        if alk:
            V[n + '_alkalinity_adjustment'], V[n + '_sigma_alkalinity'] = B['a'], B['sa']
    V['cell_weight'] = FILL['w']
    V['mask_common_area'] = common.astype('i1')
    V['mask_product_area'] = own_masks.astype('i1')
    V['filled_flag'] = (FILL['need'] & ~FILL['have']).astype('i1')
    if q in N.T_DEPENDENT:
        V['sigma_flag'] = TFLAG['cells'].astype('i1')
    with netCDF4.Dataset(SRC / f'OA-Track_{REFS[0]}.nc') as f:
        noaa = ms.noaa_co2()
        V['xco2_obs_annual'] = np.array([noaa.get(int(y), np.nan) for y in yo])
        V['xco2_preindustrial'] = rd(f, 'state_xco2')
    F = [F for F in N.files() if F.name == f'OA-Track_{q}_maps_obs_based_products.nc'][0]
    coords = dict(lat=lat, lon=lon, product=N.PRODUCTS, state=list(STATES), year=yo)
    write(F, V, out / F.name, global_attrs(q, F), coords)
    info.append(f'  {q}: maps of the obs-based products written, {time.time() - t0:.0f} s')
    return q, info


def obs_maps_args(args):
    return obs_maps_quantity(*args)


# ------------------------------------------------------------------------------------------------ maps of the ESMs
def scenario_years(f, s):
    """the suffix of the variables of scenario s in a file of make_oatrack.py, and its years"""
    if s == 'historical':
        return '', np.asarray(f['year'][:]).astype(int)
    return f'_{s}', np.asarray(f['year_ssp'][:]).astype(int)


def small_cache(x, ny, nx):
    """a variable read map by map, with a chunk cache of two maps (see the chunk caches of make_oatrack.py)"""
    x.set_var_chunk_cache(size=2 * ny * nx * 4 + 4096, nelems=7, preemption=1.0)
    return x


def esm_patterns(s):
    """the alkalinity part of the 1 sigma of the change from the preindustrial states to the ESM years of scenario s
    (historical or an SSP), for the mean of the fCO2-products. In every fCO2-product, cell and year: the change c with
    and c0 without the adjustment (from the files), and the spread s (ddof 1) of the change over the four single-model
    patterns, the adjustment of each recomputed exactly as make_oatrack.py computes it (alk_setup; at the constrained
    state of the year without the adjustment; m(t) of the model formula, for an SSP of the scenario, as
    make_oatrack.alk_series); per product sqrt((c - c0)^2 + s^2), then the mean over the products covering the cell.
    Saved per quantity in WORK: esms_<s>_<q>_sa_change.npy (state, year, lat, lon) and the years. Checks: the increment
    of the alkalinity in the file against pattern x m, the central adjustment recomputed against the file, and its
    1 sigma sqrt(a^2 + spread^2) against _sigma_alkalinity of the file (relative to the largest rms of the adjustment
    over the years, as the adjustment is small in the early years)"""
    t0 = time.time()
    A = {r: alk_setup(r) for r in REFS}
    a0 = A[REFS[0]]
    ny, nx, nk = a0.lat.size, a0.lon.size, a0.nk
    qs = list(ma.DELTA_Q) + ['thetao', 'so']
    D = {}
    try:
        for r in REFS:
            D[r] = (netCDF4.Dataset(SRC / f'OA-Track_{r}.nc'), netCDF4.Dataset(SRC / f'OA-Track_{r}_noalk.nc'))
        tag, years = scenario_years(D[REFS[0]][0], s)
        X, m = {}, {}
        for r, (f, g) in D.items():
            assert np.array_equal(scenario_years(f, s)[1], years), f'{r} {s}: years'
            X[r] = dict(alk={q: small_cache(f[f'{q}{tag}'], ny, nx) for q in ma.DELTA_Q},
                        noalk={q: small_cache(g[f'{q}{tag}'], ny, nx) for q in qs},
                        sa={q: small_cache(f[f'{q}{tag}_sigma_alkalinity'], ny, nx) for q in CHECK_Q})
            m[r] = [Q.on(Q.m_model[r], years) if s == 'historical' else
                    (Q.on(Q.m_s[s][r], years) if s in Q.m_s else np.zeros(years.size)) for Q in [A[r].P] + A[r].Pk]
            assert all(np.all(np.isfinite(x)) for x in m[r]), f'{r} {s}: m(t) missing'
        WORK.mkdir(parents=True, exist_ok=True)
        out = {q: np.lib.format.open_memmap(WORK / f'esms_{s}_{q}_sa_change.npy', mode='w+', dtype='f4',
                                            shape=(2, years.size, ny, nx)) for q in ma.DELTA_Q}
        get = lambda x, j: np.ma.filled(x[j].astype('f8'), np.nan)
        chk = Check()
        inc = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for j in range(years.size):
                tot = {q: np.zeros((2, ny, nx)) for q in ma.DELTA_Q}
                cnt = {q: np.zeros((2, ny, nx), int) for q in ma.DELTA_Q}
                for r in REFS:
                    a_, x = A[r], X[r]
                    g2 = {q: get(x['noalk'][q], j) for q in qs}
                    x2 = {q: get(x['alk'][q], j) for q in ma.DELTA_Q}
                    Ds = a_.chem.deltas_multi(g2['thetao'], g2['so'], g2['talk'], g2['dic'],
                                              [a_.p0 * m[r][0][j]] + [a_.pk[i] * m[r][1 + i][j] for i in range(nk)])
                    d = np.abs((x2['talk'] - g2['talk']) - a_.p0 * m[r][0][j])
                    inc = max(inc, float(np.nanmax(d)) if np.isfinite(d).any() else 0.0)
                    for q in CHECK_Q:
                        a2 = x2[q] - g2[q]
                        chk.add('central', Ds[0][q], a2, (r, q))
                        sp = np.std([Ds[1 + i][q] for i in range(nk)], 0, ddof=1)
                        chk.add('sigma', np.hypot(a2, sp), get(x['sa'][q], j), (r, q))
                    for q in ma.DELTA_Q:
                        alt = np.array([Ds[1 + i][q] for i in range(nk)])
                        for js in range(2):
                            x1, g1 = a_.st[q][js], a_.st0[q][js]
                            alt1 = np.array([a_.Dst[js][1 + i][q] for i in range(nk)])
                            if q in N.ABSOLUTE_CHANGE:
                                c, c0 = x2[q] - x1, g2[q] - g1
                                ck = (g2[q] + alt) - (g1 + alt1)
                            else:
                                c, c0 = 100 * (x2[q] / x1 - 1), 100 * (g2[q] / g1 - 1)
                                ck = 100 * ((g2[q] + alt) / (g1 + alt1) - 1)
                            sa = np.sqrt((c - c0) ** 2 + np.std(ck, 0, ddof=1) ** 2)
                            ok = np.isfinite(sa)
                            tot[q][js][ok] += sa[ok]
                            cnt[q][js] += ok
                for q in ma.DELTA_Q:
                    out[q][:, j] = np.where(cnt[q] > 0, tot[q] / np.maximum(cnt[q], 1), np.nan).astype('f4')
        for q in ma.DELTA_Q:
            out[q].flush()
        del out
        np.save(WORK / f'esms_{s}_years.npy', years)
    finally:
        for f, g in D.values():
            f.close()
            g.close()
    return s, (f'{s:10s} alkalinity patterns of the change to the ESM years {years[0]}-{years[-1]}, '
               f'{time.time() - t0:.0f} s; checks: increment of the file against pattern x m {inc:.1e} umol/kg, central '
               f'adjustment against the file {chk.ratio("central"):.1e}, its 1 sigma with the spread of the patterns '
               f'against the file {chk.ratio("sigma"):.1e} (largest difference relative to the largest rms over the years), '
               f'at the states {max(a_.chk_a for a_ in A.values()):.1e} and {max(a_.chk_s for a_ in A.values()):.1e} '
               f'({", ".join(CHECK_Q)})')


def esm_maps_quantity(q, out):
    """OA-Track_<q>_maps_esms.nc: the preindustrial states of every fCO2-product and their mean (as in the maps of the
    obs-based products), and for the mean of the fCO2-products the ESM years (historical and the four SSPs) with their
    change from the preindustrial states, filled to the whole ocean; read and written ms.BLOCK years at a time. The
    change of each product from its own preindustrial state: sigma_delta of both years in quadrature (an upper limit,
    as the errors of delta in the two years are correlated) and its alkalinity part (esm_patterns); the mean: the mean of the products'
    changes with the root mean square of their terms, the spread and the mean alkalinity part. Checks: the mean against
    OA-Track_all[_noalk].nc of make_oatrack.py (value, adjustment, its 1 sigma)"""
    t0 = time.time()
    alk, const = q in N.ALK, q not in N.NO_CONSTANTS
    lat, lon, common, own_masks, pre, pd = preindustrial_block(q)
    ny, nx, nP = lat.size, lon.size, len(N.PRODUCTS)
    flag_and_fill(q, (pre,), (pre,))
    F = [F for F in N.files() if F.name == f'OA-Track_{q}_maps_esms.nc'][0]
    coords = dict(lat=lat, lon=lon, product=N.PRODUCTS, scenario=N.SCENARIOS, state=list(STATES), year=YEARS)
    n_pre, n_esm, n_ch = f'{q}_preindustrial', f'{q}_esms_annual_forced', f'{q}_esms_annual_forced_change'
    terms_alk = ['_alkalinity_adjustment', '_sigma_alkalinity'] if alk else []
    terms, terms_ch = ['', '_sigma'] + terms_alk, ['', '_sigma_upper_limit'] + terms_alk
    key = {'': 'v', '_sigma': 's', '_sigma_upper_limit': 's', '_alkalinity_adjustment': 'a', '_sigma_alkalinity': 'sa'}
    dtypes = {n + t: 'f4' for n in (n_pre, n_esm) for t in terms}
    dtypes.update({n_ch + t: 'f4' for t in terms_ch})
    dtypes.update(cell_weight='f4', mask_common_area='i1', filled_flag='i1', constrained_flag='i1', xco2_esms_annual='f8',
                  xco2_preindustrial='f8')
    if q in N.T_DEPENDENT:
        dtypes['sigma_flag'] = 'i1'
    info, chk = [], Check()
    D = {}
    try:
        for r in REFS:
            D[r] = (netCDF4.Dataset(SRC / f'OA-Track_{r}.nc'), netCDF4.Dataset(SRC / f'OA-Track_{r}_noalk.nc'))
        D['all'] = (netCDF4.Dataset(SRC / 'OA-Track_all.nc'), netCDF4.Dataset(SRC / 'OA-Track_all_noalk.nc'))
        f0 = D[REFS[0]][0]
        with Writer(F, out / F.name, global_attrs(q, F), coords, dtypes) as W:
            for t in terms:
                W.put(n_pre + t, pre[key[t]])
            W.put('cell_weight', FILL['w'])
            W.put('mask_common_area', common.astype('i1'))
            W.put('filled_flag', (FILL['need'] & ~FILL['have']).astype('i1'))
            # constrained_flag of the mean: 2 where one of the fCO2-products covering the cell has no cell-by-cell
            # constraint there, 1 where all have it, 0 where no product covers the cell
            cf = np.zeros((ny, nx), 'i1')
            for ip, r in enumerate(REFS, start=1):
                fl, own = rd(D[r][0], 'constrained_flag'), own_masks[ip]
                assert np.all(np.isin(fl[own], (1, 2))), f'{r}: constrained_flag of its area not 1 or 2'
                cf[own] = np.maximum(cf[own], fl[own].astype('i1'))
            W.put('constrained_flag', cf)
            if q in N.T_DEPENDENT:
                W.put('sigma_flag', TFLAG['cells'].astype('i1'))
            W.put('xco2_esms_annual', np.array([on_years(scenario_years(f0, s)[1], rd(f0, 'xco2' if s == 'historical'
                                                                                       else f'xco2_{s}'))
                                                for s in N.SCENARIOS]))
            W.put('xco2_preindustrial', rd(f0, 'state_xco2'))
            for iscen, s in enumerate(N.SCENARIOS):
                tag, years = scenario_years(f0, s)
                iy = np.searchsorted(YEARS, years)
                assert np.array_equal(YEARS[iy], years) and np.all(np.diff(iy) == 1), f'{s}: years'
                for r in REFS:
                    assert np.array_equal(scenario_years(D[r][0], s)[1], years), f'{r} {s}: years'
                if alk:
                    assert np.array_equal(np.load(WORK / f'esms_{s}_years.npy'), years), f'{s}: years of the patterns'
                    sa_ch = np.load(WORK / f'esms_{s}_{q}_sa_change.npy', mmap_mode='r')
                for j0 in range(0, years.size, ms.BLOCK):
                    j1 = min(j0 + ms.BLOCK, years.size)
                    sl, nt = slice(j0, j1), j1 - j0
                    E = {k: np.full((nP, nt, ny, nx), np.nan, 'f4') for k in ('v', 's', 'a', 'sa')}
                    T = {k: np.full((nP, nt, ny, nx), np.nan, 'f4') for k in ('d', 'c2', 'k')}
                    C = {k: np.full((nP, 2, nt, ny, nx), np.nan, 'f4') for k in ('v', 's', 'a', 'sa')}
                    c_noalk = np.full((nP, 2, nt, ny, nx), np.nan, 'f4')
                    for ip, r in enumerate(REFS, start=1):
                        f, g = D[r]
                        x2, d2 = rd(f, f'{q}{tag}', sl), rd(f, f'{q}{tag}_sigma', sl)
                        E['v'][ip] = x2
                        T['d'][ip], T['c2'][ip] = d2, rd(f, f'{q}{tag}_sigma_mapping_c2', sl)
                        T['k'][ip] = rd(f, f'{q}{tag}_sigma_constants', sl) if const else 0.0
                        a2 = x2 - rd(g, f'{q}{tag}', sl) if alk else 0.0
                        if alk:
                            E['a'][ip], E['sa'][ip] = a2, rd(f, f'{q}{tag}_sigma_alkalinity', sl)
                        for js in range(2):
                            x1, d1 = pre['v'][ip, js].astype('f8'), pd['d'][ip, js].astype('f8')
                            a1 = pre['a'][ip, js].astype('f8') if alk else 0.0
                            if q in N.ABSOLUTE_CHANGE:
                                c, c0, t1, t2 = x2 - x1, (x2 - a2) - (x1 - a1), d1, d2
                            else:
                                c, c0 = 100 * (x2 / x1 - 1), 100 * ((x2 - a2) / (x1 - a1) - 1)
                                t1, t2 = 100 * x2 * d1 / x1 ** 2, 100 * d2 / x1
                            C['v'][ip, js] = c
                            c_noalk[ip, js] = np.sqrt(t1 ** 2 + t2 ** 2)
                            if alk:
                                C['a'][ip, js] = c - c0
                    mean_block(E, T['d'], T['c2'], T['k'], alk)
                    sa_m = np.asarray(sa_ch[:, sl], 'f8') if alk else 0.0
                    C['v'][0], C['s'][0], a_m = change_mean(C['v'][1:], c_noalk[1:], C['a'][1:] if alk else None, sa_m)
                    if alk:
                        C['a'][0], C['sa'][0] = a_m, sa_m
                    # checks against the combined files of make_oatrack.py (the mean of the products covering the cell)
                    fa, ga = D['all']
                    ref = rd(fa, f'{q}{tag}', sl)
                    chk.add('value', E['v'][0], ref)
                    if alk:
                        chk.add('adjustment', E['a'][0], ref - rd(ga, f'{q}{tag}', sl))
                        chk.add('sigma_alkalinity', E['sa'][0], rd(fa, f'{q}{tag}_sigma_alkalinity', sl))
                    flag_and_fill(q, (E, C), (E,))
                    it = (iscen, slice(int(iy[j0]), int(iy[j1 - 1]) + 1))
                    for t in terms:
                        W.put(n_esm + t, E[key[t]][0], it)
                    for t in terms_ch:
                        W.put(n_ch + t, C[key[t]][0], (slice(None),) + it)
    finally:
        for f, g in D.values():
            f.close()
            g.close()
    info.append(f'  {q}: mean of the ESM years against OA-Track_all.nc: value {chk.ratio("value"):.1e}'
                + (f', alkalinity adjustment {chk.ratio("adjustment"):.1e}, its 1 sigma {chk.ratio("sigma_alkalinity"):.1e}'
                   if alk else '') + ' (largest difference relative to the rms of the reference)')
    info.append(f'  {q}: maps of the ESMs written, {time.time() - t0:.0f} s')
    return q, info


def esm_maps_args(args):
    return esm_maps_quantity(*args)


# ------------------------------------------------------------------------------------------------ writing
class Writer:
    """the file F of oatrack_names.py, written in parts: every row created at the start (dtypes: {name: dtype}; maps of
    float values as float32, quantized and compressed), the data then given with put (whole, or the part idx); on
    closing, every row must have been written, and the file is renamed from <name>.part"""

    def __init__(self, F, path, attrs, coords, dtypes):
        rows = {r['name']: r for r in F.rows}
        missing = [n for n in rows if n not in coords and n not in dtypes]
        extra = [n for n in dtypes if n not in rows]
        assert not missing and not extra, f'{F.name}: missing {missing}, not defined {extra}'
        self.path, self.part = path, path.with_name(path.name + '.part')
        self.g = g = netCDF4.Dataset(self.part, 'w', format='NETCDF4')
        g.setncatts(attrs)
        for name, vals in coords.items():
            g.createDimension(name, len(vals))
            x = g.createVariable(name, str if isinstance(vals[0], str) else 'f8' if name in ('lat', 'lon') else 'i4', (name,))
            x[:] = np.array(vals, dtype=object) if isinstance(vals[0], str) else np.asarray(vals)
            r = rows[name]
            x.long_name, x.description = r['long_name'], r['description']
            if r['units']:
                x.units = r['units']
        self.var = {}
        for name, r in rows.items():
            if name in coords:
                continue
            dims = tuple(r['dimlist'])
            dt = np.dtype(dtypes[name])
            ismap = dims[-2:] == ('lat', 'lon') and dt.kind == 'f'
            if ismap:
                chunks = tuple(1 if d not in ('lat', 'lon') else len(coords[d]) for d in dims)
                kind = 'sigma' if ('_sigma' in name or name.endswith('_adjustment')) else 'value'
                x = g.createVariable(name, 'f4', dims, zlib=True, complevel=4, shuffle=True, fill_value=np.float32(1e20),
                                     chunksizes=chunks, significant_digits=ms.DIGITS[kind != 'value'], quantize_mode=ms.MODE)
            else:
                x = g.createVariable(name, 'f8' if dt.kind == 'f' else dt, dims, fill_value=np.nan if dt.kind == 'f' else None)
            x.long_name, x.description = r['long_name'], r['description']
            if r['units']:
                x.units = r['units']
            self.var[name] = (x, ismap, tuple(len(coords[d]) for d in dims))
        self.todo = set(self.var)

    def put(self, name, a, idx=None):
        x, ismap, shape = self.var[name]
        a = np.asarray(a)
        if idx is None:
            assert a.shape == shape, f'{name}: shape {a.shape}, defined {shape}'
            idx = slice(None)
        x[idx] = np.ma.masked_invalid(a.astype('f4')) if ismap else a
        self.todo.discard(name)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        self.g.close()
        if kind is None:
            assert not self.todo, f'{self.path.name}: not written {sorted(self.todo)}'
            self.part.rename(self.path)


def write(F, V, path, attrs, coords):
    """the file F of oatrack_names.py with the data V: every row written, nothing else"""
    with Writer(F, path, attrs, coords, {n: np.asarray(a).dtype for n, a in V.items()}) as W:
        for n, a in V.items():
            W.put(n, a)


def write_csv(F, V, path, coords):
    """long format: one row per value; the 1 sigma and the adjustment in their own columns"""
    cols = N.csv_columns(F)
    q = F.q
    blocks = [n for n in (f'{q}_preindustrial_gm', f'{q}_obs_annual_gm', f'{q}_obs_annual_forced_gm',
                          f'{q}_obs_annual_forced_gm_change', f'{q}_esms_annual_forced_gm', f'{q}_esms_annual_forced_gm_change')]
    dims_of = {r['name']: r['dimlist'] for r in F.rows}
    with open(path, 'w', newline='') as fh:
        fh.write(f'# {SOURCE}\n# {F.title}; units: {N.UNITS[q]} (changes: '
                 f'{N.UNITS[q] if q in N.ABSOLUTE_CHANGE else "%"}); see OA-Track_{q}_global_mean.nc for the descriptions\n')
        w = csv.writer(fh)
        w.writerow(cols)
        for n in blocks:
            dims = dims_of[n]
            a = V[n]
            s = V.get(n + '_sigma')
            sul = V.get(n + '_sigma_upper_limit')
            ad = V.get(n + '_alkalinity_adjustment')
            for idx in np.ndindex(a.shape):
                if not np.isfinite(a[idx]):
                    continue
                key = dict(zip(dims, idx))
                row = dict(variable=n.replace(f'{q}_', '', 1), area=coords['area'][key['area']],
                           product=coords['product'][key['product']],
                           scenario=coords['scenario'][key['scenario']] if 'scenario' in key else '',
                           state=coords['state'][key['state']] if 'state' in key else '',
                           year=int(coords['year'][key['year']]) if 'year' in key else '',
                           value=f'{a[idx]:.6g}', sigma=f'{s[idx]:.4g}' if s is not None and np.isfinite(s[idx]) else '',
                           sigma_upper_limit=f'{sul[idx]:.4g}' if sul is not None and np.isfinite(sul[idx]) else '',
                           alkalinity_adjustment=f'{ad[idx]:.4g}' if ad is not None and np.isfinite(ad[idx]) else '')
                w.writerow([row[c] for c in cols])


def global_attrs(q, F):
    with netCDF4.Dataset(SRC / f'OA-Track_{REFS[0]}.nc') as f:
        constants, weights = f.constants, f.area_weights
    return dict(title=f'OA-Track: {F.title}', source=SOURCE, Conventions='CF-1.8', license='CC BY 4.0',
                constants=constants, area_weights=weights,
                alkalinity_adjustment='included in every value (the best estimate of the paper); _alkalinity_adjustment '
                                      'gives it, so that it can be removed',
                history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} make_oatrack_product.py from the complete output of '
                        'make_oatrack.py')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--part', choices=('global', 'obs_based_products', 'esms'), required=True)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--quantities', nargs='+')
    ap.add_argument('--skip-patterns', action='store_true', help='the maps: the alkalinity patterns are already in <out>/work')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    global WORK
    WORK = out / 'work'
    qs = a.quantities or N.QUANTITIES
    files = {F.name: F for F in N.files()}
    if a.part in ('global', 'obs_based_products'):
        noaa = ms.noaa_co2()
        for r in REFS:
            with netCDF4.Dataset(SRC / f'OA-Track_{r}.nc') as f:
                yy, _ = ms.obs_years(f)
            FS[r] = fit_sigma.FitSigma(yy, [noaa[int(y)] for y in yy])
            print(f'{r:10s} fit uncertainty tabulated for {yy[0]}-{yy[-1]}', flush=True)
    if a.part == 'global':
        SA.update({k: v for k, v in np.load(oa_config.DATA / 'structural' / 'structural_adjustments.npz').items()})
        assert list(SA['products']) == REFS, f'structural_adjustments.npz for {list(SA["products"])}'
        with mp_.get_context('fork').Pool(min(a.jobs, len(REFS))) as pool:
            for r, yy, D, gm, dt in pool.imap_unordered(adjustment_at_fit, REFS):
                ADJ[r] = (yy, D, gm)
                print(f'{r:10s} alkalinity adjustment at the forced state {yy[0]}-{yy[-1]}, {dt:.0f} s', flush=True)
        with mp_.get_context('fork').Pool(a.jobs) as pool:
            for q, V, info in pool.imap_unordered(global_quantity, qs):
                F = files[f'OA-Track_{q}_global_mean.nc']
                areas = ['common_area', 'product_area'] + (['whole_ocean'] if q in N.WHOLE_OCEAN else [])
                coords = dict(area=areas, product=N.PRODUCTS, scenario=N.SCENARIOS, state=list(STATES), year=YEARS)
                write(F, V, out / F.name, global_attrs(q, F), coords)
                write_csv(F, V, out / F.name.replace('.nc', '.csv'), coords)
                for line in info:
                    print(line, flush=True)
        print(f'global means written to {out}', flush=True)
        return
    maps_setup()
    if a.part == 'obs_based_products':
        patterns, tasks, quantity, what = pattern_terms, REFS, obs_maps_args, 'maps of the obs-based products'
    else:
        patterns, tasks, quantity, what = esm_patterns, N.SCENARIOS, esm_maps_args, 'maps of the ESMs'
    if not a.skip_patterns:
        with mp_.get_context('fork').Pool(min(a.jobs, len(tasks))) as pool:
            for _, line in pool.imap_unordered(patterns, tasks):
                print(line, flush=True)
    with mp_.get_context('fork').Pool(a.jobs) as pool:
        for q, info in pool.imap_unordered(quantity, [(q, out) for q in qs]):
            for line in info:
                print(line, flush=True)
    print(f'{what} written to {out}', flush=True)


if __name__ == '__main__':
    main()
