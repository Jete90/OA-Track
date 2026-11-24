#!/usr/bin/env python
"""
structural_adjustments.py
=========================
The two adjustments of the decline 1750-2024 beyond the analysis on the common area (Extended Data Table 3,
Extended Data Fig. 5, Extended Data Table 1), for each of the four products, written to
data/<run>/structural/structural_adjustments.npz.

1  Alkalinity (calcification feedback). Observed: surface TA rises relative to its salinity-based prediction by
   0.063 +- 0.023 umol/kg per year over 1991-2021 (Barrett et al. 2025: 0.072 robust, 0.054-0.056 least squares
   regression; mean of the two 0.063, their 1 sigma). The products derive TA from S and T and miss it; 15 of the
   18 models do not simulate it. The missing TA m(t) is one continuous series per product (and per scenario):
     real rise R(t): the common-area mean nTA of the four models with Omega-dependent calcification (GFDL-CM4,
       GFDL-ESM4, UKESM1-0-LL from 1850, GFDL-ESM2M-COBALTv2 from 1860; feedback_pattern.npz, as anomalies to the
       model's own piControl (the forced change), each series smoothed with the Enting spline of the chain over its
       whole length, mean of the four, GFDL-ESM2M-COBALTv2 before 1860 extended with the change of the
       CMIP6 mean), to 2021 with the SSP2-4.5
       and RCP2.6 continuations, scaled by one factor so that its linear trend over 1991-2021 equals the observed
       common-area rate; before 1850 (no model with the feedback) extended in proportion to the decline of
       Omega_arag, 1750 to 1850 against 1850 to the window centre; after 2021 per scenario with the CMIP6 feedback
       models that have it, continued from R(2021) (same factor)
     content C(t): what the product already holds of this rise, the feedback models' share of the multi-model
       mean (sum of their anomalies over the number of models of the series: 18 for the best estimate, the models
       of the scenario for the scenario series); the products hold none (their TA has no trend)
     m(t) = [R(t) - R(t_c)] - [C(t) - C(t_c)] in model years and scenarios, R(t) - R(t_c) in product years, with
       t_c the centre of the product's window, where its TA is taken as right
   Chemistry (since 16 Nov 2026 direct, no linear sensitivity): in every cell of the two states of each product, the
   constrained 1750 state (states_<ref>_pi.nc) and the 2024 state (obs_<ref>.nc), TA is raised by the adjustment of
   the cell, pattern x m(1750) and pattern x m(2024), DIC is solved by Newton's method to the same pCO2 and Omega_arag
   recomputed with mocsy (make_product_alkalinity.Chem.deltas, the function of OA-Track, the constants of the chain).
   The pattern is that of the feedback in the models (feedback_pattern.npz from feedback_pattern.py: regression slopes
   of drift-corrected nTA on its common-area mean in the four models, smoothed over 11 x 11 cells, negative values
   kept, mean 1 over the common area). Only where S >= 20 (the range of nTA); other cells are filled from the cells
   around them (fill_neighbours, as the whole ocean). Shift of the decline: the decline of the area means with the
   changed states minus that without them.
   The observed rate is the mean over the measurements of Barrett et al. (one regression through all of them), not an
   area mean over the common area: the common-area rate is 0.063 / p, p the mean of the pattern over their
   measurements (barrett_sampling.npz from barrett_sampling.py: GLODAPv2.2023 surface measurements of their cruises
   and selection, measurements per cell as weights).
   Uncertainty: 100 % of the shift (the rise has only been measured since 1991 and the contribution of reduced
   calcification has not been quantified, Barrett et al. section 4.3; the models do not show the model piece).

2  Whole ocean (also: absolute area means of 1750, 2024 and the boundary on the common area and the ocean, without
   and with the alkalinity adjustment cell by cell; the fraction of the ocean beyond a 20 % decline in 2024, per
   product and cell, then the mean of the four, on the common area and on every cell any product covers).
   The decline over the ocean of the paper (regional_common.ocean_on) minus that over the common area, from three
   estimates that rest on the same kind of 1750 state (the multi-model mean of the bias-adjusted ESMs, without the
   emergent constraint, since 20 Nov 2026) and are all filled to the same ocean (fill_tools.fill_neighbours: 5 x 5,
   the direct neighbours twice the weight of the outer ring, inward in passes; unreachable cells the mean of their
   region, Arctic north of 60 N, Southern Ocean south of 55 S, other; test_whole_ocean_coverage.py is the test):
     products: cell by cell, every cell the mean of the products that cover it (2024: quadratic fit of the product
       years, as the chain; 1750: the unconstrained state), the cells covered by none filled
     models: each of the ESMs on its own states (polar_models.npz of test_polar_shift_models.py: area means of
       Omega_arag of the smoothed, drift-corrected states of the data product in 2024 and in 1750, the 1850 state at
       278.3 ppm as the 1750 states of the chain, each model filled to the ocean of the paper): the shift relative to
       its decline from 1750, scaled to the decline of the paper, and the relative gap of the level, whole ocean /
       common area - 1, in 1750 and 2024, times the products' common-area means
     Jiang et al.: their 2005-2014 state and the unconstrained 1750 state on their baseline (states_Jiang_pi.nc),
       filled; the shift relative to their decline 1750 -> 2005-2014, scaled to the decline of the paper
   Central value of the shift: the mean of the three. Not fully independent, as all rest on the simulated change;
   the products weakly observed outside the common area, one product in most of the Arctic and a filled share; the
   models not bias-adjusted. Uncertainty: three times the standard deviation across the models (POLAR_SIGMA_FACTOR),
   of the shift (decline) and of the level gap (Omega_arag 1750 and 2024, and the distance to the boundary, from the
   gaps of both years of each model): the models share simple representations of what matters most there (gas
   exchange through sea ice, CO2 under the ice, constant or climatological river inputs of carbon and alkalinity).
   The levels stay those of the products with the constrained 1750 state (abs_ocean, the numbers of the paper); the
   products' shift with the constrained state is kept as polar_products_shift_constrained (the estimate before 20 Nov
   2026: products 0.16, models on their own cells 0.22, Jiang et al. constrained 0.21 -> 0.20 +- 0.18).

Usage (code/):  OA_RUN=u74 python -u structural_adjustments.py [--polar ../data/u74/structural/polar_models.npz]
                                                            [--pattern pattern_esm2m_cmb | uniform | ...]
  Run feedback_pattern.py, barrett_sampling.py and test_polar_shift_models.py first (the pattern and the series of
  the feedback models, the sampling of the observations, the models' whole ocean; polar_models.npz with levels_filled,
  20 Nov 2026 or later).
"""
import argparse
import datetime
import pickle
import warnings
from pathlib import Path

import netCDF4
import numpy as np

import oa_config
import calc_pi_xco2 as cpx
import drift_tools as dt
import make_product_alkalinity as mpa   # Chem.deltas: TA up at constant pCO2, as OA-Track
import regional_common as rc
from feedback_pattern import S_RANGE   # the salinity range of nTA (the pattern and the adjustment are defined for it)
from fill_tools import fill_neighbours, regional_fill, regions, fill_to   # noqa: F401  (the filling of cells without a value,
#   moved to fill_tools.py on 20 Nov 2026 so that the ESMs are filled with the same rule; names kept importable from here)

OUT = oa_config.DATA / 'product'
DEST = oa_config.DATA / 'structural'
PRODUCTS = ('OceanSODA', 'UExP', 'JMA', 'CMEMS')
WINDOW = {q: oa_config.WINDOWS[q] for q in PRODUCTS}      # the bias windows (centre: t_c of the alkalinity add-on)
YEAR = 2024
BARRETT = (0.063, 0.023)                 # umol/kg per year, 1 sigma (mean of robust and least-squares regression)
OBS_PERIOD = (1991, 2021)                # the years of the measurements of Barrett et al.
SCENARIOS = ('ssp126', 'ssp245', 'ssp370', 'ssp585')
CONTINUATION = 'ssp245'                  # continues the historical runs to 2021 (the chain's continuation)
SERIES_CUTOFF = 30.0                     # years: the Enting spline of the model series (calc_smooth_states.CUTOFF)
FIRST_MODEL_YEAR = 1850                  # the extension in proportion to Omega_arag covers 1750 to 1849
N_MODELS = 18                            # models of the best estimate (the content of the multi-model mean)
YEARS_ALL = np.arange(1750, 2101)
FEEDBACK = ('GFDL-CM4', 'GFDL-ESM4', 'UKESM1-0-LL')   # Omega-dependent calcification (constraint_refs.ALK_MODELS)
POLAR_SIGMA_FACTOR = 3.0                 # whole ocean: 1 sigma = 3 x the spread of the ESMs (processes they lack)
PATTERN_KEY = 'pattern'           # in feedback_pattern.npz: four feedback models (3 CMIP6, GFDL-ESM2M-COBALTv2)


# ------------------------------------------------------------------------------------------------ helpers
def onto(lat, lon, la, lo, a):
    jy = np.abs(lat[:, None] - la[None, :]).argmin(1)
    ix = np.abs(((lon[:, None] - lo[None, :]) + 180) % 360 - 180).argmin(1)
    return a[..., jy, :][..., ix]


def amean(a, m, w):
    k = m & np.isfinite(a)
    return float(np.sum(a[k] * w[k]) / np.sum(w[k]))


# ------------------------------------------------------------------------------------------------ states
def product_states(ref, lat, lon):
    """per cell on the obs grid: constrained 1750 T, S, TA, DIC (states_pi) and the 2024 fit of T, S, TA, DIC and
    Omega_arag of the product (quadratic fit of the product years up to 2024)"""
    with netCDF4.Dataset(OUT / f'states_{ref}_pi.nc') as f:
        names = [str(x) for x in f['state'][:]]
        k = next(j for j, nm in enumerate(names) if '1750' in nm)
        la, lo = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        P = {q: onto(lat, lon, la, lo, np.ma.filled(f[f'{q}_state'][k].astype('f8'), np.nan))
             for q in ('thetao', 'so', 'talk', 'dic', 'omega_arag')}
        # the unconstrained 1750 state (multi-model mean of the bias-adjusted ESMs), for the whole-ocean shift
        P['omega_arag_unconstrained'] = onto(lat, lon, la, lo, np.ma.filled(f['omega_arag_unconstrained_state'][k].astype('f8'), np.nan))
    with netCDF4.Dataset(OUT / f'obs_{ref}.nc') as f:
        yo = np.asarray(f['year'][:]).astype(int)
        la, lo = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        raw = {q: onto(lat, lon, la, lo, np.ma.filled(f[q][:].astype('f8'), np.nan))
               for q in ('thetao', 'so', 'talk', 'dic', 'omega_arag')}
    sel = yo <= YEAR
    x = (yo[sel] - YEAR).astype('f8')
    X = np.column_stack([np.ones(x.size), x, x ** 2])
    N = {}
    for q, a in raw.items():
        a = a[sel]; ok = np.all(np.isfinite(a), axis=0)
        out = np.full(ok.shape, np.nan)
        b, *_ = np.linalg.lstsq(X, a[:, ok], rcond=None)
        out[ok] = b[0]
        N[q] = out
    series = {'year': yo, 'omega': raw['omega_arag']}
    return P, N, series


# ------------------------------------------------------------------------------------------------ 1 alkalinity
def common_omega_series(ref, lat, lon, cmask, w):
    """common-area mean Omega_arag of the constrained model years, 1750-2014 (states_<ref>_pi.nc, states_<ref>_hist.nc)"""
    years, vals = [], []
    for part in ('pi', 'hist'):
        with netCDF4.Dataset(OUT / f'states_{ref}_{part}.nc') as f:
            la, lo = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
            ys = np.asarray(f['year'][:]).astype(int)
            v = f['omega_arag']
            for k, y_ in enumerate(ys):
                a = onto(lat, lon, la, lo, np.ma.filled(v[k].astype('f8'), np.nan))
                years.append(int(y_)); vals.append(amean(a, cmask, w))
    return np.array(years), np.array(vals)


def scenario_models(ref, s):
    """the models of the scenario series of the product (states_<ref>_<ssp>.nc), [] if the product has none"""
    p = OUT / f'states_{ref}_{s}.nc'
    if not p.exists():
        return []
    with netCDF4.Dataset(p) as f:
        return [m.split()[0] for m in str(f.models).split(',')] if hasattr(f, 'models') else []


def smooth(y, a, cutoff=SERIES_CUTOFF):
    """Enting spline of a series over its whole length (the smoothing of the chain), so that the interannual
    variability of a model does not enter m(t)"""
    return dt.smoother(y.size, cutoff) @ np.asarray(a, 'f8')


def model_series():
    """the feedback models' common-area nTA series of feedback_pattern.npz as anomalies to their own piControl
    (series minus the control level stored there: the forced change), each smoothed over its whole length
    (historical + continuation, or historical + scenario, SERIES_CUTOFF): hist[model] on years 1850-2021
    (historical, continued with CONTINUATION), full[model][ssp] on 1850-2100 (historical + scenario smoothed together),
    scen[model][ssp] its scenario years 2015-2100, esm2m on its years (coupled ensemble mean, historical to 2005 then
    RCP2.6). A scenario run that ends before 2100 (UKESM1-0-LL and MPI-ESM1-2-HR end in 2099) is carried to 2100 by
    linear extrapolation of the smoothed series over its last decade, so that the set of models and the content are
    the same in every scenario year."""
    f = DEST / 'feedback_pattern.npz'
    assert f.exists(), f'{f} missing: run feedback_pattern.py first'
    z = np.load(f)
    assert 'years_historical' in z.files and 'control_esm2m' in z.files, \
        f'{f}: no series or control levels (feedback_pattern.py before 7 November 2026): run it again'
    yh = np.asarray(z['years_historical']).astype(int)
    hist, scen, full = {}, {}, {}
    for m in FEEDBACK:
        a = np.asarray(z[f'series_{m}_historical'], 'f8') - float(z[f'control_{m}'])
        full[m] = {}
        for s_ in SCENARIOS:
            if f'series_{m}_{s_}' not in z.files:
                continue
            ys_ = np.asarray(z[f'years_{m}_{s_}']).astype(int)
            yy = np.concatenate([yh, ys_])
            sm = smooth(yy, np.concatenate([a, np.asarray(z[f'series_{m}_{s_}'], 'f8') - float(z[f'control_{m}'])]))
            assert yy[-1] >= YEARS_ALL[-1] - 5, f'{m} {s_}: the scenario series ends in {yy[-1]}'
            if yy[-1] < YEARS_ALL[-1]:
                tail = yy >= yy[-1] - 9
                b = np.polyfit(yy[tail], sm[tail], 1)
                add = np.arange(yy[-1] + 1, YEARS_ALL[-1] + 1)
                print(f'  {m} {s_}: series ends in {yy[-1]}, {add[0]}-{add[-1]} extrapolated from the trend of '
                      f'{yy[tail][0]}-{yy[-1]} ({b[0]:+.3f} umol/kg per year)', flush=True)
                yy, sm = np.concatenate([yy, add]), np.concatenate([sm, np.polyval(b, add)])
            full[m][s_] = (yy, sm)                                # the whole smoothed series, 1850-2100
        assert CONTINUATION in full[m], f'{m}: no {CONTINUATION} series to continue the historical to {OBS_PERIOD[1]}'
        yy, sm = full[m][CONTINUATION]
        keep = yy <= OBS_PERIOD[1]
        hist[m] = (yy[keep], sm[keep])
        scen[m] = {s_: (yy_[yy_ > yh[-1]], a_[yy_ > yh[-1]]) for s_, (yy_, a_) in full[m].items()}
    ye = np.asarray(z['years_esm2m']).astype(int)
    e = smooth(ye, np.asarray(z['series_esm2m_coupled'], 'f8') - float(z['control_esm2m']))
    return hist, scen, full, (ye, e), str(z['created'])


def on_years(years, y_src, a_src):
    """series a_src (on y_src) placed on years, NaN elsewhere"""
    out = np.full(len(years), np.nan)
    k = np.searchsorted(years, y_src)
    ok = (k < len(years)) & (np.asarray(years)[np.minimum(k, len(years) - 1)] == y_src)
    out[k[ok]] = a_src[ok]
    return out


def feedback_pattern(cmask, lat, lon, w, key):
    """spatial pattern of the calcification feedback in surface TA (mean 1 over the common area), from
    feedback_pattern.py (key: 'pattern', the four feedback models; or one of the sensitivity patterns there), or
    'uniform' (1 everywhere, for reference)"""
    ocean = rc.ocean_on(lat, lon).astype(bool)
    if key == 'uniform':
        pat, src = np.where(ocean, 1.0, np.nan), 'uniform'
    else:
        f = DEST / 'feedback_pattern.npz'
        assert f.exists(), f'{f} missing: run feedback_pattern.py first'
        z = np.load(f)
        assert np.allclose(z['lat'], lat) and np.allclose(z['lon'], lon), f'{f}: grid differs from the common area'
        assert set(FEEDBACK) <= {str(m) for m in z['models']}, f'{f}: models {list(z["models"])}'
        assert key in z.files, f'{f}: no {key} (has {[k for k in z.files if k.startswith("pattern")]})'
        pat, src = np.asarray(z[key], 'f8'), f'{f.name}: {key}, {z["created"]}'
    la2 = np.broadcast_to(lat[:, None], pat.shape)
    bands = ((-90, -60), (-60, -30), (-30, 30), (30, 60), (60, 90))
    print(f'  feedback pattern ({src}): common-area mean {amean(pat, cmask, w):.2f}; by latitude: '
          + ', '.join(f'{a_}..{b_}: {amean(pat, ocean & (la2 >= a_) & (la2 < b_), w):.2f}' for a_, b_ in bands), flush=True)
    return pat


def barrett_rate(pattern, lat, lon):
    """the common-area rate that matches the observed rate over the measurements of Barrett et al.: BARRETT / p, p
    the mean of the pattern over their measurements (each with the same weight, as in their regression)"""
    f = DEST / 'barrett_sampling.npz'
    assert f.exists(), f'{f} missing: run barrett_sampling.py first'
    z = np.load(f)
    assert np.allclose(z['lat'], lat) and np.allclose(z['lon'], lon), f'{f}: grid differs from the common area'
    n = np.asarray(z['n_points'], 'f8')
    ok = (n > 0) & np.isfinite(pattern)
    p = float(np.sum(n[ok] * pattern[ok]) / np.sum(n[ok]))
    p_cells = float(np.mean(pattern[ok]))
    p_all = ''
    for k_ in sorted(k for k in z.files if k.startswith('n_points_')):
        na = np.asarray(z[k_], 'f8'); oka = (na > 0) & np.isfinite(pattern)
        p_all += f'; {k_[9:]} {float(np.sum(na[oka] * pattern[oka]) / np.sum(na[oka])):.3f}'
    sel_ = str(z['selection']) if 'selection' in z.files else ''
    print(f'  sampling of Barrett et al. ({sel_}: {int(n.sum())} measurements in {int((n > 0).sum())} cells; '
          f'{100 * (1 - n[ok].sum() / n.sum()):.1f} % outside the ocean of the pattern): pattern mean over the measurements '
          f'p = {p:.3f} (checks: each sampled cell once {p_cells:.3f}{p_all}); common-area rate {BARRETT[0]} / {p:.3f} = '
          f'{BARRETT[0] / p:.4f} umol/kg per year', flush=True)
    return BARRETT[0] / p, p, p_cells


def real_rise(hist, esm2m, rate_c):
    """the real rise G(t) of the common-area nTA, 1850-2021, from the feedback models: mean of the anomalies of the four
    models (CMIP6 from 1850, GFDL-ESM2M-COBALTv2 from 1860, before 1860 extended with the change of the CMIP6 mean), and
    the factor that scales its 1991-2021 trend to the observed common-area rate"""
    years = np.arange(FIRST_MODEL_YEAR, OBS_PERIOD[1] + 1)
    cols = [on_years(years, *hist[m]) for m in FEEDBACK] + [on_years(years, *esm2m)]
    A = np.array(cols)
    n = np.isfinite(A).sum(axis=0)
    assert np.all(n >= len(FEEDBACK)), f'years without the CMIP6 feedback models: {years[n < len(FEEDBACK)]}'
    assert np.all(np.isfinite(A[:len(FEEDBACK)])), 'a CMIP6 feedback model has gaps in 1850-2021'
    # GFDL-ESM2M-COBALTv2 starts in 1860: before, its first value plus the change of the CMIP6 mean, so that every year is
    # the mean of the same four models (the mean of 3 models before 1860 and of 4 after put a step into G at 1860, visible
    # in m(t) of Extended Data Fig. 4b; 17 Nov 2026)
    k = np.isfinite(A[-1])
    f0 = int(np.argmax(k))
    assert k[f0:].all(), 'GFDL-ESM2M-COBALTv2 has gaps after its first year'
    g3 = A[:len(FEEDBACK)].mean(axis=0)
    if f0 > 0:
        print(f'  GFDL-ESM2M-COBALTv2 from {years[f0]}: before, its {years[f0]} value plus the change of the CMIP6 mean '
              f'(step of the former 3-to-4-model mean at {years[f0]}: {(A[-1, f0] - g3[f0]) / A.shape[0]:+.3f} umol/kg)', flush=True)
    A[-1, :f0] = A[-1, f0] + g3[:f0] - g3[f0]
    G = A.mean(axis=0)                              # the four models in every year; anomalies to the controls
    it = (years >= OBS_PERIOD[0]) & (years <= OBS_PERIOD[1])
    trend = float(np.polyfit(years[it], G[it], 1)[0])
    per_model = {m: float(np.polyfit(years[it], c[it], 1)[0]) for m, c in zip(FEEDBACK + ('GFDL-ESM2M-COBALTv2',), cols)}
    scale = rate_c / trend
    print(f'  feedback models (anomalies to their controls): G(1850) {G[0]:+.2f}, G(1860) {float(np.interp(1860, years, G)):+.2f} umol/kg; '
          f'trend of the common-area nTA {OBS_PERIOD[0]}-{OBS_PERIOD[1]}: '
          + ', '.join(f'{m} {v:+.3f}' for m, v in per_model.items())
          + f'; mean series {trend:+.3f} umol/kg per year; observed common-area rate {rate_c:.4f}: scale {scale:.3f}', flush=True)
    return years, G, scale, trend, per_model


def extend_before(years_g, G, om_years, om, tc):
    """G on 1750-1849 in proportion to the decline of Omega_arag of the constrained model years (1750 -> 1850
    against 1850 -> t_c), and G(1750)"""
    y0 = FIRST_MODEL_YEAR
    o = np.interp(np.arange(1750, y0 + 1), om_years, om)          # Omega 1750..1850 of the constrained model years
    om_tc = float(np.interp(tc, om_years, om))
    g1850 = float(np.interp(y0, years_g, G))
    gtc = float(np.interp(tc, years_g, G))
    c = (gtc - g1850) * (o[0] - o[-1]) / (o[-1] - om_tc)             # rise 1750 -> 1850 in units of G
    g1750 = g1850 - c
    ext = g1750 + c * (o[0] - o[:-1]) / (o[0] - o[-1])               # 1750..1849
    return np.arange(1750, y0), ext, g1750, c


def alkalinity(pk, lat, lon, cmask, w, states, key=PATTERN_KEY):
    print('\n===== 1  alkalinity (calcification feedback)')
    pattern = feedback_pattern(cmask, lat, lon, w, key)
    rate_c, p_s, p_cells = barrett_rate(pattern, lat, lon)
    hist, scen, full, esm2m, created = model_series()
    years_g, G, scale, trend_g, trend_models = real_rise(hist, esm2m, rate_c)
    chem = mpa.Chem(lat, lon)
    ocean = rc.ocean_on(lat, lon).astype(bool)
    res = {}
    for ref in PRODUCTS:
        P, N, _ = states[ref]
        r = pk['results_common'][ref]; dA = np.asarray(r['dA'], 'f8')
        dA_all = float(dA.mean())
        w0, w1 = WINDOW[ref]; tc = 0.5 * (w0 + w1)
        valid = (ocean & np.isfinite(P['talk']) & np.isfinite(N['talk']) & np.isfinite(P['dic']) & np.isfinite(N['dic'])
                 & np.isfinite(P['thetao']) & np.isfinite(N['thetao']) & np.isfinite(P['so']) & np.isfinite(N['so']))
        # the adjustment is defined for nTA, so only where the salinity normalization is meaningful (S_RANGE, as the
        # pattern); other cells take the regional mean in whole_ocean
        fresh = valid & ~((P['so'] >= S_RANGE[0]) & (N['so'] >= S_RANGE[0]))
        valid &= ~fresh
        # ---- the real rise R(t), 1750-2100 per scenario (umol/kg since 1750), and the models' content C(t)
        om_years, om_model = common_omega_series(ref, lat, lon, cmask, w)
        y_pre, g_pre, g1750, c_pre = extend_before(years_g, G, om_years, om_model, tc)
        Y = YEARS_ALL
        R = {}                                                          # per scenario, on Y (1750-2100)
        base_R = np.full(Y.size, np.nan)
        base_R[Y < FIRST_MODEL_YEAR] = scale * (g_pre - g1750)
        mid = (Y >= FIRST_MODEL_YEAR) & (Y <= OBS_PERIOD[1])
        base_R[mid] = scale * (np.interp(Y[mid], years_g, G) - g1750)
        R_last = float(base_R[Y == OBS_PERIOD[1]][0])
        for s_ in SCENARIOS:
            mods = [mm for mm in FEEDBACK if s_ in scen[mm]]
            if not mods:
                continue
            Rs = base_R.copy()
            cols = np.array([on_years(Y, *scen[mm][s_]) for mm in mods])
            gs = np.nanmean(cols, axis=0)
            k21 = Y == OBS_PERIOD[1]
            after = Y > OBS_PERIOD[1]
            Rs[after] = R_last + scale * (gs[after] - gs[k21][0])
            R[s_] = (Rs, mods)
        R_best = R[CONTINUATION][0].copy()                               # 2022-2024 of the best estimate: the continuation
        R_best[Y > YEAR] = np.nan
        # content: the feedback models' share of the multi-model mean (anomalies on Y; before 1850 as in 1850). The best
        # estimate holds the historical years (smoothed with the continuation); a scenario holds the whole series
        # smoothed with that scenario, so that its content has no step at 2014/2015
        def content(mod_list, n_mod, s_=None):
            C = np.zeros(Y.size)
            for mm in mod_list:
                if s_ is None:
                    yh_, ah_ = hist[mm]
                    a = on_years(Y, yh_[yh_ <= 2014], ah_[yh_ <= 2014])
                else:
                    a = on_years(Y, *full[mm][s_])
                a[Y < 1850] = a[Y == 1850][0]
                C += a
            return C / n_mod
        C_best = content(FEEDBACK, N_MODELS)
        # m(t): the model formula (real rise minus the models' content) and the product formula (real rise alone,
        # the continuation beyond 2021), the best estimate the model formula before the first product year
        R_tc = float(np.interp(tc, Y, R_best))
        C_tc = float(np.interp(tc, Y, C_best))
        m_model = (R_best - R_tc) - (C_best - C_tc)                     # NaN after 2014 (no model years there)
        m_product = R[CONTINUATION][0] - R_tc                           # all years, the continuation after 2021
        m_best = np.where(Y < w0, m_model, m_product)
        m_best[Y > YEAR + 1] = np.nan                                   # UExP and CMEMS run to 2025
        m_scen, mods_scen, n_scen, C_scen = {}, {}, {}, {}
        for s_, (Rs, mods) in R.items():
            sm = scenario_models(ref, s_)
            n_s = len(sm) if sm else N_MODELS
            mods_in = [mm for mm in mods if (not sm or mm in sm)]
            Cs = content(mods_in, n_s, s_)
            Cs_tc = float(np.interp(tc, Y, Cs))
            m_scen[s_] = (Rs - R_tc) - (Cs - Cs_tc)
            mods_scen[s_], n_scen[s_], C_scen[s_] = mods, n_s, Cs
        miss_model = float(-m_best[Y == 1750][0])
        miss_prod = float(m_best[Y == YEAR][0])
        # ---- the change of Omega_arag in every cell, recomputed directly (Chem.deltas, as OA-Track): TA of the 1750
        # state + pattern x m(1750) (lower), TA of the 2024 state + pattern x m(2024) (higher), each at constant pCO2
        dmap = {}
        for k_, X, m_ in (('1750', P, -miss_model), ('2024', N, miss_prod)):
            to_u = 1.0 if np.nanmedian(X['talk'][valid]) > 10 else 1e6     # Chem.deltas takes umol/kg
            dta = np.where(valid, pattern * m_, np.nan)
            dmap[k_] = chem.deltas(X['thetao'], X['so'], X['talk'] * to_u, X['dic'] * to_u, dta)['omega_arag']
        m = cmask & valid & np.isfinite(dmap['1750']) & np.isfinite(dmap['2024']) & np.isfinite(P['omega_arag']) \
            & np.isfinite(N['omega_arag'])
        W = w[m]
        am = lambda a_: float(np.sum(a_[m] * W) / W.sum())
        O50, O24 = am(P['omega_arag']), am(N['omega_arag'])
        dO50, dO24 = am(dmap['1750']), am(dmap['2024'])                 # signed changes of the two states
        shift = 100 * (1 - (O24 + dO24) / (O50 + dO50)) - 100 * (1 - O24 / O50)
        pct = lambda a_, k_: np.nanpercentile(a_[k_ & np.isfinite(a_)], [1, 50, 99]) if (k_ & np.isfinite(a_)).any() else [np.nan] * 3
        out_ = ocean & ~cmask
        print(f'  {ref:9s} change of Omega_arag (1 / 50 / 99 %), 1750: common ' + ' / '.join(f'{v_:+.4f}' for v_ in pct(dmap['1750'], cmask))
              + ', outside ' + ' / '.join(f'{v_:+.4f}' for v_ in pct(dmap['1750'], out_))
              + '; 2024: common ' + ' / '.join(f'{v_:+.4f}' for v_ in pct(dmap['2024'], cmask))
              + ', outside ' + ' / '.join(f'{v_:+.4f}' for v_ in pct(dmap['2024'], out_))
              + f'; common-area cells without a value {int((cmask & ~m).sum())}; S < {S_RANGE[0]:g}: '
              f'{100 * (w * fresh).sum() / (w * ocean).sum():.2f} % of the ocean', flush=True)
        res[ref] = dict(shift=shift, sigma=abs(shift), miss_model=miss_model, dO50=dO50, dO24=dO24,
                        O50=O50, O24=O24, dO50_map=dmap['1750'], dO24_map=dmap['2024'],
                        miss_prod=miss_prod, dA_all=dA_all, k_shape=scale, window_centre=tc, rate_c=rate_c,
                        p_sampling=p_s, p_cells=p_cells, m_best=m_best, m_model=m_model, m_product=m_product, m_scen=m_scen,
                        R_best=R_best, C_best=C_best, R_tc=R_tc, R_scen={s_: R[s_][0] for s_ in R}, C_scen=C_scen,
                        rise_pre1850=scale * c_pre,
                        mods_scen=mods_scen, n_scen=n_scen)
        s2100 = ', '.join(f'{s_} {m_scen[s_][Y == 2100][0]:+.2f} ({len(mods_scen[s_])} models of {n_scen[s_]})' for s_ in m_scen)
        print(f'  {ref:9s} real rise 1750 -> '
              f'{tc:.1f} {R_tc:+.2f} umol/kg (1750 -> 1850 {scale * c_pre:+.2f} from Omega), models\' content {C_tc - C_best[0]:+.2f}, '
              f'missing 1750 {-miss_model:+.2f}; m(2024) {miss_prod:+.2f}; m at '
              f'{w0 - 1} {m_best[Y == w0 - 1][0]:+.3f} -> {w0} {m_best[Y == w0][0]:+.3f}; shift {shift:+.3f} points '
              f'(1750 alone {100 * (1 - O24 / (O50 + dO50)) - 100 * (1 - O24 / O50):+.3f}, 2024 alone '
              f'{100 * (1 - (O24 + dO24) / O50) - 100 * (1 - O24 / O50):+.3f}); Omega_arag 1750 '
              f'{O50:.3f} {dO50:+.4f}, 2024 {O24:.3f} {dO24:+.4f}; m(2100) {s2100} umol/kg', flush=True)
    v = np.array([res[q]['shift'] for q in PRODUCTS])
    print(f'  mean of the four products: shift {v.mean():+.3f}, sigma {abs(v.mean()):.3f} points (100 %)')
    res['_pattern'] = pattern
    res['_series'] = dict(years_g=years_g, G=G, scale=scale, trend=trend_g, trend_models=trend_models, created=created)
    return res


# ------------------------------------------------------------------------------------------------ 2 whole ocean
def whole_ocean(lat, lon, cmask, w, states, polar, alk, pattern, dm):
    print('\n===== 2  whole ocean')
    ocean = rc.ocean_on(lat, lon).astype(bool)
    outside = ocean & ~cmask
    la2 = np.broadcast_to(lat[:, None], ocean.shape)
    region = np.where(la2 > 60, 'Arctic', np.where(la2 < -55, 'Southern', 'other'))
    Ps = np.array([states[q][0]['omega_arag'] for q in PRODUCTS]); Ns = np.array([states[q][1]['omega_arag'] for q in PRODUCTS])
    good = np.isfinite(Ps) & np.isfinite(Ns)
    Ps, Ns = np.where(good, Ps, np.nan), np.where(good, Ns, np.nan)
    with np.errstate(invalid='ignore'):
        P, N = np.nanmean(Ps, axis=0), np.nanmean(Ns, axis=0)
    # ocean cells no product covers in every year (outside the common area): filled from the covered cells around
    # them (fill_neighbours); cells it cannot reach take the mean of their region (the previous fill, also printed)
    cov = np.isfinite(P) & np.isfinite(N)
    (P, N), step = fill_neighbours([P, N], cov & ocean, outside)
    rest = step == -1
    P, N = regional_fill([P, N], outside & ~rest & np.isfinite(P), rest & outside, region, w)
    P_r, N_r = regional_fill([np.where(cov, P, np.nan), np.where(cov, N, np.nan)], outside & cov, outside, region, w)
    for g in ('Arctic', 'Southern', 'other'):
        mg = outside & (region == g)
        print(f'  fill, {g:8s}: {100 * w[mg & ~cov].sum() / w[ocean].sum():4.2f} % of the ocean filled ({100 * w[mg & ~cov].sum() / w[mg].sum():4.1f} % '
              f'of the region outside the common area), passes {int(step[mg].max()) if (step[mg] > 0).any() else 0}, '
              f'unreachable {100 * w[mg & rest].sum() / w[ocean].sum():.3f} % (regional mean)', flush=True)
    dec = lambda m: 100 * (1 - amean(N, m & np.isfinite(P), w) / amean(P, m & np.isfinite(N), w))
    d_c, d_o = dec(cmask), dec(ocean)
    out = dict(products_common=d_c, products_ocean=d_o, products_shift_constrained=d_o - d_c,
               fill_share=100 * w[outside & ~cov].sum() / w[ocean].sum())
    print(f'  products cell by cell (constrained 1750 state, the levels of the paper): common {d_c:.2f} %, ocean '
          f'{d_o:.2f} %, shift {d_o - d_c:+.3f} points')
    # the shift of the decline from the products WITHOUT the constraint (since 20 Nov 2026): the 1750 state the
    # multi-model mean of the bias-adjusted ESMs in each cell, as the ESMs and Jiang et al. below, so that the three
    # estimates of the shift rest on the same 1750 states and the same filled ocean (test_whole_ocean_coverage.py)
    PUs = np.array([states[q][0]['omega_arag_unconstrained'] for q in PRODUCTS])
    goodu = np.isfinite(PUs) & np.isfinite(Ns)
    with np.errstate(invalid='ignore'):
        PU = np.nanmean(np.where(goodu, PUs, np.nan), axis=0)
        NU = np.nanmean(np.where(goodu, Ns, np.nan), axis=0)
    covu = np.isfinite(PU) & np.isfinite(NU)
    (PU, NU), _ = fill_to([PU, NU], covu & ocean, outside & ~covu, region, w)
    decu = lambda m: 100 * (1 - amean(NU, m & np.isfinite(PU), w) / amean(PU, m & np.isfinite(NU), w))
    du_c, du_o = decu(cmask), decu(ocean)
    out.update(products_shift=du_o - du_c, products_common_unconstrained=du_c, products_ocean_unconstrained=du_o)
    print(f'  products cell by cell, unconstrained 1750 state (the estimate of the shift): common {du_c:.2f} %, ocean '
          f'{du_o:.2f} %, shift {du_o - du_c:+.3f} points (constrained {d_o - d_c:+.3f})')
    d_or = 100 * (1 - amean(N_r, ocean, w) / amean(P_r, ocean, w))
    print(f'    with the previous regional fill: ocean {d_or:.2f} %, shift {d_or - d_c:+.3f} points; level gap ocean / common '
          f'- 1, 1750 and 2024: neighbours {100 * (amean(P, ocean, w) / amean(P, cmask, w) - 1):+.2f} %, '
          f'{100 * (amean(N, ocean, w) / amean(N, cmask, w) - 1):+.2f} %; regional {100 * (amean(P_r, ocean, w) / amean(P_r, cmask, w) - 1):+.2f} %, '
          f'{100 * (amean(N_r, ocean, w) / amean(N_r, cmask, w) - 1):+.2f} %', flush=True)
    # alkalinity adjustment cell by cell: every product's own change of Omega_arag in each of its cells, recomputed
    # directly (1750 side: lower; 2024 side: higher; signed); maps of the mean of the products, filled like the states
    dPs = np.array([np.where(good[i], alk[q]['dO50_map'], np.nan) for i, q in enumerate(PRODUCTS)])
    dNs = np.array([np.where(good[i], alk[q]['dO24_map'], np.nan) for i, q in enumerate(PRODUCTS)])
    with np.errstate(invalid='ignore'):
        dP, dN = np.nanmean(dPs, axis=0), np.nanmean(dNs, axis=0)
    # filled like the states: every ocean cell without a value (outside the common area as above; in the common area
    # the cells without a change, salinity below S_RANGE or no convergence), from the cells around it
    hv = np.isfinite(dP) & np.isfinite(dN)
    n_fill = int((cmask & ~hv).sum())
    (dP, dN), step_a = fill_neighbours([dP, dN], hv & ocean, ocean)
    rest_a = step_a == -1
    dP, dN = regional_fill([dP, dN], ocean & ~rest_a & np.isfinite(dP), rest_a & ocean, region, w)
    print(f'  alkalinity maps: {n_fill} common-area cells without a change; all cells without a value filled from '
          f'their neighbours ({int(rest_a.sum())} unreachable cells: regional mean)')
    Pa, Na = P + dP, N + dN
    deca = lambda m: 100 * (1 - amean(Na, m & np.isfinite(Pa), w) / amean(Pa, m & np.isfinite(Na), w))
    # where the alkalinity effect on the whole-ocean decline comes from: the adjustment applied in one region only
    covered = np.any(good, axis=0)
    print('  alkalinity effect on the whole-ocean decline by region (area % of the ocean, mean pattern, mean of the '
          'products\' change of Omega_arag 2024 dN, effect in points):')
    tot = 0.0
    for lab, R in (('common area', cmask), ('outside, Arctic > 60N, covered', outside & (region == 'Arctic') & covered),
                   ('outside, Arctic > 60N, filled', outside & (region == 'Arctic') & ~covered),
                   ('outside, Southern < 55S, covered', outside & (region == 'Southern') & covered),
                   ('outside, Southern < 55S, filled', outside & (region == 'Southern') & ~covered),
                   ('outside, other, covered', outside & (region == 'other') & covered),
                   ('outside, other, filled', outside & (region == 'other') & ~covered)):
        if not R.any():
            continue
        PrR, NrR = P + np.where(R, dP, 0.0), N + np.where(R, dN, 0.0)
        eff = 100 * (1 - amean(NrR, ocean & np.isfinite(PrR), w) / amean(PrR, ocean & np.isfinite(NrR), w)) - d_o
        tot += eff
        print(f'    {lab:34s} {100 * (w * R).sum() / (w * ocean).sum():5.1f} %  pattern {amean(pattern, R, w):+.2f}  '
              f'dN {amean(dN, R, w):+.4f}  effect {eff:+.3f}')
    print(f'    {"sum of the regions":34s} effect {tot:+.3f} (all together {deca(ocean) - d_o:+.3f})')
    out.update(alk_cell_common=deca(cmask) - d_c, alk_cell_ocean=deca(ocean) - d_o, polar_shift_alk=deca(ocean) - deca(cmask))
    print(f'  checks, cell by cell: alkalinity on the common area {deca(cmask) - d_c:+.3f} points (area means: '
          f'{np.mean([alk[q]["shift"] for q in PRODUCTS]):+.3f}), on the whole ocean {deca(ocean) - d_o:+.3f}; whole-ocean '
          f'shift with the alkalinity adjustment {deca(ocean) - deca(cmask):+.3f} (without {d_o - d_c:+.3f})')
    for lab, m in (('common', cmask), ('ocean', ocean)):
        p50, n24 = amean(P, m & np.isfinite(N), w), amean(N, m & np.isfinite(P), w)
        pa, na = amean(Pa, m & np.isfinite(Na), w), amean(Na, m & np.isfinite(Pa), w)
        out[f'abs_{lab}'] = np.array([p50, n24, 0.8 * p50, pa, na, 0.8 * pa])
        print(f'  {lab:6s}: 1750 {p50:.3f}, 2024 {n24:.3f}, boundary {0.8 * p50:.3f}; with alkalinity: 1750 {pa:.3f}, '
              f'2024 {na:.3f}, boundary {0.8 * pa:.3f}')
    # fraction of the ocean beyond a 20 % decline in 2024, as the notebook (Fig. 3 cell): per product first (its 2024
    # fit against 0.8 x its constrained 1750 state in every cell), then the mean of the four; on the common area and
    # on the largest coverage, every cell any product covers (where a product has no value: the mean of the others)
    union = ocean & np.any(good, axis=0)
    fr = {}
    for i, q in enumerate(PRODUCTS):
        oth = [j for j in range(len(PRODUCTS)) if j != i]
        with np.errstate(invalid='ignore'):
            fill = lambda A: np.where(good[i], A[i], np.nanmean(A[oth], axis=0))
            Pq, Nq, dPq, dNq = fill(Ps), fill(Ns), fill(dPs), fill(dNs)
            dq = 100 * (1 - Nq / Pq)
            dqa = 100 * (1 - (Nq + dNq) / (Pq + dPq))
        for lab, m in (('common', cmask), ('union', union)):
            mm = m & np.isfinite(dq) & np.isfinite(dqa)
            fr[(q, lab)] = (100 * (w * (mm & (dq > 20))).sum() / (w * mm).sum(),
                            100 * (w * (mm & (dqa > 20))).sum() / (w * mm).sum(), 100 * (w * mm).sum() / (w * ocean).sum())
        print(f'    {q:9s} beyond -20 %: common {fr[(q, "common")][0]:.1f} % (with alkalinity {fr[(q, "common")][1]:.1f} %), '
              f'all covered cells ({fr[(q, "union")][2]:.1f} % of the ocean) {fr[(q, "union")][0]:.1f} % ({fr[(q, "union")][1]:.1f} %)')
    for lab in ('common', 'union'):
        v = np.array([fr[(q, lab)] for q in PRODUCTS])
        out[f'frac20_products_{lab}'] = v.mean(axis=0)
        print(f'  beyond -20 % in 2024, mean of the products ({"common area" if lab == "common" else "all covered cells"}, '
              f'{v[:, 2].mean():.1f} % of the ocean): {v[:, 0].mean():.1f} %, with alkalinity {v[:, 1].mean():.1f} %')
    for g in ('Arctic', 'Southern', 'other'):
        out[f'decline_outside_{g}'] = dec(outside & (region == g))
    # models (polar_models.npz of test_polar_shift_models.py): area means of the common area and the ocean per model
    assert Path(polar).exists(), f'{polar} missing: run test_polar_shift_models.py first'
    z = np.load(polar)
    names = [str(m) for m in z['models']]
    expected = {m for m, _ in cpx.find_runs(None)}
    if set(names) != expected:
        print(f'  WARNING models in {Path(polar).name}: missing {sorted(expected - set(names))}, extra {sorted(set(names) - expected)}')
    assert 'levels_filled' in z.files, f'{polar}: no filled levels (test_polar_shift_models.py before 20 Nov 2026): run it again'
    L, yrs = np.asarray(z['levels_filled'], 'f8'), [int(y) for y in z['years']]   # each model filled to the ocean of the paper
    Lo = np.asarray(z['levels'], 'f8')                                           # each model on its own cells (comparison)
    assert 1750 in yrs, f'{polar}: no 1750 level (test_polar_shift_models.py before 16 Nov 2026): run it again'
    i0, i2 = yrs.index(1750), yrs.index(YEAR)
    c50, o50, c24, o24 = L[:, 0, i0], L[:, 1, i0], L[:, 0, i2], L[:, 1, i2]
    rel_own = (100 * (1 - Lo[:, 1, i2] / Lo[:, 1, i0]) - 100 * (1 - Lo[:, 0, i2] / Lo[:, 0, i0])) / (100 * (1 - Lo[:, 0, i2] / Lo[:, 0, i0]))
    dcm, dom = 100 * (1 - c24 / c50), 100 * (1 - o24 / o50)
    rel = (dom - dcm) / dcm                                   # shift relative to the decline
    g50, g24 = o50 / c50 - 1, o24 / c24 - 1                   # relative gap of the level, ocean / common - 1
    pc50, pc24 = out['abs_common'][:2]
    po50, po24 = out['abs_ocean'][:2]
    F, sd = POLAR_SIGMA_FACTOR, (lambda a: float(np.std(a, ddof=1)))
    level_sigma = np.array([F * sd(g50) * pc50, F * sd(g24) * pc24, F * sd(pc24 * g24 - 0.8 * pc50 * g50)])
    out.update(models_n=len(names), models_rel_mean=float(rel.mean()), models_rel_sd=sd(rel), models=names,
               models_rel=rel, models_gap=np.column_stack([g50, g24]), models_continuation=np.asarray(z['continuation']),
               products_gap=np.array([po50 / pc50 - 1, po24 / pc24 - 1]), level_sigma=level_sigma,
               models_source=f'{Path(polar).name}: {z["source"]}, {z["created"]}')
    out['models_rel_own_cells'] = rel_own
    print(f'  models ({len(names)}; {z["source"]}), each filled to the ocean of the paper:')
    print(f'    shift 1750 -> {YEAR} {100 * rel.mean():+.2f} % of the decline (sd {100 * sd(rel):.2f} %), on the decline of '
          f'the paper {rel.mean() * dm:+.3f} (sd {sd(rel) * dm:.3f}) points; products {du_o - du_c:+.3f}; on the models\' own '
          f'cells (the former estimate) {rel_own.mean() * dm:+.3f} (sd {sd(rel_own) * dm:.3f})')
    print(f'    level gap ocean / common - 1: 1750 {100 * g50.mean():+.2f} % (sd {100 * sd(g50):.2f} %), {YEAR} '
          f'{100 * g24.mean():+.2f} % (sd {100 * sd(g24):.2f} %), correlation {np.corrcoef(g50, g24)[0, 1]:+.2f}; products '
          f'1750 {100 * (po50 / pc50 - 1):+.2f} %, {YEAR} {100 * (po24 / pc24 - 1):+.2f} %')
    print(f'    1 sigma (x {F:g}) on the products\' common-area means {pc50:.3f} (1750) and {pc24:.3f} ({YEAR}): Omega_arag '
          f'1750 {level_sigma[0]:.3f}, {YEAR} {level_sigma[1]:.3f}, distance to the boundary {level_sigma[2]:.3f}; decline '
          f'{F * sd(rel) * dm:.3f} points')
    return out


def jiang_shift(lat, lon, cmask, w):
    """the whole-ocean shift of Jiang et al.: their 2005-2014 state (obs_Jiang.nc) and the unconstrained 1750 state on
    their baseline (states_Jiang_pi.nc: the multi-model mean of the 18 bias-adjusted ESMs, not constrained, as the ESMs
    and the products above), both filled to the ocean of the paper; the decline 1750 -> 2005-2014 over the ocean minus
    that on the common area, relative to the latter (since 20 Nov 2026; before: the constrained chain on their own
    cells, constraint_results_jiang.pkl, printed for comparison)"""
    ocean = rc.ocean_on(lat, lon).astype(bool)
    region = regions(lat, ocean.shape)
    with netCDF4.Dataset(OUT / 'states_Jiang_pi.nc') as f:
        names = [str(x) for x in f['state'][:]]
        k = next(j for j, nm in enumerate(names) if '1750' in nm)
        la, lo = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        PJ = onto(lat, lon, la, lo, np.ma.filled(f['omega_arag_unconstrained_state'][k].astype('f8'), np.nan))
    with netCDF4.Dataset(OUT / 'obs_Jiang.nc') as f:
        yj = [int(y) for y in np.asarray(f['year'][:])]
        la, lo = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        NJ = onto(lat, lon, la, lo, np.ma.filled(f['omega_arag'][:].astype('f8'), np.nan))
        NJ = NJ[0] if NJ.ndim == 3 else NJ
    ok = np.isfinite(PJ) & np.isfinite(NJ)
    PJ, NJ = np.where(ok, PJ, np.nan), np.where(ok, NJ, np.nan)
    (PJ, NJ), share = fill_to([PJ, NJ], ok & ocean, ocean & ~ok, region, w)
    dec = lambda m: 100 * (1 - amean(NJ, m & np.isfinite(PJ), w) / amean(PJ, m & np.isfinite(NJ), w))
    own, com = dec(ocean), dec(cmask)
    rel = (own - com) / com
    print(f'  Jiang et al. (years {yj}, unconstrained 1750 state, filled to the ocean of the paper; {100 * w[ok & ocean].sum() / w[ocean].sum():.1f} % '
          f'of it covered, {100 * share:.2f} % of the rest by the regional mean): decline 1750 -> 2005-2014 {own:.2f} % over the '
          f'ocean, {com:.2f} % on the common area: {own - com:+.3f} points, {100 * rel:+.2f} % of the decline')
    path = oa_config.DATA / 'constraint_results_jiang.pkl'
    if path.exists():
        ct = pickle.load(open(path, 'rb'))['coverage_test']
        print(f'    (the former estimate, the constrained chain on their own cells: {float(ct["own"]) - float(ct["common"]):+.3f} points, '
              f'{100 * (float(ct["own"]) - float(ct["common"])) / float(ct["common"]):+.2f} % of the decline)')
    return dict(own=own, common=com, rel=rel, covered=100 * w[ok & ocean].sum() / w[ocean].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--polar', default=str(DEST / 'polar_models.npz'), help='output of test_polar_shift_models.py')
    ap.add_argument('--pattern', default=PATTERN_KEY,
                    help='pattern of the alkalinity adjustment: a pattern_* key of feedback_pattern.npz or uniform; other '
                         'than the default, the output goes to structural_adjustments_<pattern>.npz (sensitivity)')
    a = ap.parse_args()
    warnings.filterwarnings('ignore', message='Mean of empty slice')     # cells no product covers
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    r0 = pk['results_common']['UExP']
    lat, lon = np.asarray(r0['lat'], 'f8'), np.asarray(r0['lon'], 'f8')
    cmask = np.asarray(r0['mask'], bool)
    w = oa_config.area_weights(lat, lon)
    states = {q: product_states(q, lat, lon) for q in PRODUCTS}
    alk = alkalinity(pk, lat, lon, cmask, w, states, a.pattern)
    pattern = alk.pop('_pattern')
    series = alk.pop('_series')
    # whole-ocean shift in points of the decline of the paper: the mean of the products and the models; sigma: the
    # spread of the models times POLAR_SIGMA_FACTOR
    d_paper = {q: float(100 * (1 - pk['results_common'][q]['now'] / pk['results_common'][q]['constrained']['dT + CUC'][0]))
               for q in PRODUCTS}
    dm = float(np.mean(list(d_paper.values())))
    wo = whole_ocean(lat, lon, cmask, w, states, a.polar, alk, pattern, dm)
    sigma_polar = POLAR_SIGMA_FACTOR * wo['models_rel_sd'] * dm
    shift_models = wo['models_rel_mean'] * dm
    jr = jiang_shift(lat, lon, cmask, w)
    shift_jiang = jr['rel'] * dm
    shift_polar = float(np.mean([wo['products_shift'], shift_models, shift_jiang]))
    print(f'\n===== summary (decline of the paper, mean of the four products {dm:.2f} %)')
    a_shift = float(np.mean([alk[q]['shift'] for q in PRODUCTS]))
    print(f'  alkalinity  {a_shift:+.3f} +- {abs(a_shift):.3f} points')
    print(f'  whole ocean {shift_polar:+.3f} +- {sigma_polar:.3f} points (mean of the products {wo["products_shift"]:+.3f} '
          f'the models {shift_models:+.3f} and Jiang et al. {shift_jiang:+.3f}; sd of the models x {POLAR_SIGMA_FACTOR:g}); level 1 sigma Omega_arag 1750 '
          f'{wo["level_sigma"][0]:.3f}, {YEAR} {wo["level_sigma"][1]:.3f}, distance {wo["level_sigma"][2]:.3f}')
    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / ('structural_adjustments.npz' if a.pattern == PATTERN_KEY else f'structural_adjustments_{a.pattern}.npz')
    np.savez(out, products=np.array(PRODUCTS), area_weights=oa_config.WEIGHTS_TEXT,
             alk_shift=np.array([alk[q]['shift'] for q in PRODUCTS]), alk_sigma=np.array([alk[q]['sigma'] for q in PRODUCTS]),
             alk_miss_model=np.array([alk[q]['miss_model'] for q in PRODUCTS]), alk_miss_prod=np.array([alk[q]['miss_prod'] for q in PRODUCTS]),
             alk_k_shape=alk[PRODUCTS[0]]['k_shape'], barrett=np.array(BARRETT),   # k_shape: the scale of the series
             alk_window_centre=np.array([alk[q]['window_centre'] for q in PRODUCTS]),
             alk_O50=np.array([alk[q]['O50'] for q in PRODUCTS]), alk_O24=np.array([alk[q]['O24'] for q in PRODUCTS]),
             alk_rate_common=alk[PRODUCTS[0]]['rate_c'], alk_p_sampling=alk[PRODUCTS[0]]['p_sampling'],
             alk_p_cells=alk[PRODUCTS[0]]['p_cells'], pattern_source=f'feedback_pattern.npz ({a.pattern})' if a.pattern != 'uniform' else 'uniform',
             polar_shift=shift_polar, polar_sigma=sigma_polar, polar_products_shift=wo['products_shift'],
             polar_products_shift_constrained=wo['products_shift_constrained'], polar_models_rel_own_cells=wo['models_rel_own_cells'],
             polar_jiang_covered=jr['covered'],
             polar_method='since 20 Nov 2026: products and Jiang et al. with the unconstrained 1750 state (multi-model mean), '
                          'each ESM on its own states, all three filled to the ocean of the paper (fill_tools); the mean of '
                          'the three, 1 sigma = POLAR_SIGMA_FACTOR x the spread of the ESMs',
             polar_jiang_shift=shift_jiang, polar_jiang_rel=jr['rel'], polar_jiang_points_2010=jr['own'] - jr['common'],
             polar_models_shift=shift_models, polar_models_rel_mean=wo['models_rel_mean'],
             polar_models_rel_sd=wo['models_rel_sd'], polar_models_n=wo['models_n'], polar_sigma_factor=POLAR_SIGMA_FACTOR,
             decline_outside=np.array([wo[f'decline_outside_{g}'] for g in ('Arctic', 'Southern', 'other')]),
             alk_dO50=np.array([alk[q]['dO50'] for q in PRODUCTS]), alk_dO24=np.array([alk[q]['dO24'] for q in PRODUCTS]),
             alk_chemistry='direct: Chem.deltas (TA + pattern x m at constant pCO2, Omega_arag recomputed), since 16 Nov 2026',
             abs_common=wo['abs_common'], abs_ocean=wo['abs_ocean'],
             abs_columns=np.array(['1750', '2024', 'boundary', '1750 alk', '2024 alk', 'boundary alk']),
             frac20_columns=np.array(['without', 'with alkalinity', 'area % of ocean']),
             frac20_products_common=wo['frac20_products_common'], frac20_products_union=wo['frac20_products_union'],
             alk_cell_common=wo['alk_cell_common'], alk_cell_ocean=wo['alk_cell_ocean'], polar_shift_alk=wo['polar_shift_alk'],
             polar_products_common=wo['products_common'], polar_products_ocean=wo['products_ocean'],
             polar_models=np.array(wo['models']), polar_models_rel=wo['models_rel'], polar_models_gap=wo['models_gap'],
             polar_models_continuation=wo['models_continuation'], polar_products_gap=wo['products_gap'],
             polar_level_sigma=wo['level_sigma'],
             polar_level_columns=np.array(['1750', str(YEAR), 'distance to the boundary']),
             polar_source=wo['models_source'],
             alk_pattern=pattern, lat=lat, lon=lon,
             alk_years=YEARS_ALL, alk_obs_period=np.array(OBS_PERIOD), alk_series_cutoff=SERIES_CUTOFF,
             alk_m_best=np.array([alk[q]['m_best'] for q in PRODUCTS]),
             alk_m_model=np.array([alk[q]['m_model'] for q in PRODUCTS]), alk_m_product=np.array([alk[q]['m_product'] for q in PRODUCTS]),
             alk_first_product_year=np.array([WINDOW[q][0] for q in PRODUCTS]),
             **{f'alk_R_{s_}': np.array([alk[q]['R_scen'][s_] if s_ in alk[q]['R_scen'] else np.full(YEARS_ALL.size, np.nan) for q in PRODUCTS])
                for s_ in SCENARIOS},
             **{f'alk_C_{s_}': np.array([alk[q]['C_scen'][s_] if s_ in alk[q]['C_scen'] else np.full(YEARS_ALL.size, np.nan) for q in PRODUCTS])
                for s_ in SCENARIOS},
             alk_R_best=np.array([alk[q]['R_best'] for q in PRODUCTS]), alk_C_best=np.array([alk[q]['C_best'] for q in PRODUCTS]),
             alk_R_tc=np.array([alk[q]['R_tc'] for q in PRODUCTS]), alk_rise_pre1850=np.array([alk[q]['rise_pre1850'] for q in PRODUCTS]),
             alk_dA_all=np.array([alk[q]['dA_all'] for q in PRODUCTS]),
             alk_scale=series['scale'], alk_trend_models=np.array([series['trend_models'][m] for m in series['trend_models']]),
             alk_trend_model_names=np.array(list(series['trend_models'])), alk_trend_mean=series['trend'],
             alk_G_years=series['years_g'], alk_G=series['G'], alk_series_source=f'feedback_pattern.npz ({series["created"]})',
             **{f'alk_m_{s_}': np.array([alk[q]['m_scen'][s_] if s_ in alk[q]['m_scen'] else np.full(YEARS_ALL.size, np.nan) for q in PRODUCTS])
                for s_ in SCENARIOS},
             **{f'alk_models_{s_}': np.array([','.join(alk[q]['mods_scen'].get(s_, [])) for q in PRODUCTS]) for s_ in SCENARIOS},
             **{f'alk_n_models_{s_}': np.array([alk[q]['n_scen'].get(s_, 0) for q in PRODUCTS]) for s_ in SCENARIOS},
             created=str(datetime.datetime.now()))
    print(f'written {out}')


if __name__ == '__main__':
    main()
