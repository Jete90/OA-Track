#!/usr/bin/env python
"""
make_oatrack.py  (data product OA-Track: the complete output for each fCO2-product, from which make_oatrack_product.py writes the archived record)
===============
Writes the complete output of OA-Track, the data product of the paper (Supplementary Methods, 'the data product
OA-Track'), from the files of the chain: the constrained states (calc_product_states.py:
states_<ref>_{pi,hist,<ssp>,pi_<ssp>}.nc), the
indicators of the observation-based estimates (calc_product_obs.py: obs_<ref>.nc) and the alkalinity adjustment
(structural_adjustments.npz, feedback_pattern.npz, as make_product_alkalinity.py). The internal files of the chain
(carbonate_system_*.nc of make_product.py, with the switch from the models to the products, used by the figures of
the paper) are not touched.

Per reference (OceanSODA, UExP, JMA, CMEMS, Jiang), two complete files of identical structure:
    OA-Track_<ref>.nc          with the alkalinity adjustment of the paper (the best estimate)
    OA-Track_<ref>_noalk.nc    without it
Every value of a file is complete; nothing has to be combined from another file.
    year (1750-2014)       the historical series: the constrained preindustrial ocean at the CMIP6 CO2 of the year
                           (1750-1849) and the constrained, bias-adjusted, smoothed ESMs (1850-2014, all 18; the
                           smoothing continued with SSP2-4.5, MRI-ESM2-0 with SSP5-8.5)
    year_ssp (1750-2100)   the scenario series <q>_<ssp>, each with the ESMs that provide the scenario (their own
                           regression and their own preindustrial years), n_models_<ssp> per year
    year_obs               the indicators of the observation-based estimate itself, <q>_obs, for its own years
                           (Jiang et al.: one value, their decade 2005-2014, labelled 2010)
    state (1750, 1850)     the constrained states at the CO2 of AR6 (278.3 and 285.5 ppm), with T, S, DIC, TA
    <q>_2024_trend         the 2024 value of the quadratic fit of the product years (Jiang: their decade state)
Suffixes (maps and series): '' constrained value; _unconstrained (multi-model mean state); _sigma (constraint:
fit, observed predictors, CO2); _sigma_unconstrained; _sigma_internal (series only); _sigma_mapping_c1, _c2 and
_sigma_constants (level uncertainty of the reference); _sigma_alkalinity (with the adjustment only; maps:
sqrt(adjustment^2 + s^2) per cell, s the standard deviation (ddof 1) of the adjustment of the cell over the patterns
of the four feedback models alone, each scaled to the observations with its own m(t); area means and relative changes:
|adjustment|).
The product indicators carry the mapping and constants components (and _sigma_alkalinity).
Global means on the area of the reference (_gm_) and on the common area of the paper (_gm_common_): the global
constraint of the main analysis (_global), the area mean of the maps (_map), the unconstrained mean, the direct
method for comparison, the relative changes from the 1750 of the series (CMIP6 CO2) and from the AR6 1750 state to
the 2024 trend value (_decline, _decline_from_AR6_1750) with the 1 sigma of the 1750 side (_sigma_1750: constraint
and CO2; the full uncertainty of the decline is Extended Data Table 2 of the paper) and, with the adjustment,
_sigma_alkalinity. CO2: xco2 (CMIP6, year), xco2_<ssp> (ScenarioMIP, year_ssp), xco2_obs (NOAA, year_obs),
state_xco2.

Alkalinity adjustment (as make_product_alkalinity.py): missing alkalinity pattern(x) * m(t), m from
structural_adjustments.npz: historical and scenario series the model formula (the rise scaled to the observations
minus the part the three feedback ESMs already contain), product years the product formula (the full rise relative to
the middle of the reference period), the states m at their Omega_arag (relation of the historical years 1750-1870);
Jiang et al.: the means of the four products with t_c = 2010 (their decade: 0). DIC at constant pCO2 (Newton),
every indicator recomputed with mocsy; the change is computed at the constrained state of the cell and year and
added to the constrained and the unconstrained value and to the area means; pCO2, fCO2, CO2*, T and S unchanged.
For the 1 sigma of the maps, the same change is computed in every cell and year with each single-model pattern
(make_product_alkalinity.PATTERN_ALTERNATIVES, m(t) from structural_adjustments_<key>.npz).

Combined files (the four fCO2-products), both versions: OA-Track_common[_noalk].nc on the common area and
OA-Track_all[_noalk].nc on every cell of the constraint area covered by at least one fCO2-product (product_flag):
series, scenarios and states the mean over the references, sigma = sqrt(mean sigma^2 + spread^2), internal
variability the rms, mapping and constants the mean (errors assumed correlated between the products); product
indicators the mean of the products present in the year and cell, n_products, the spread; the common file also the
mean of the references' global series and declines.

Needs: states_<ref>_*.nc and obs_<ref>.nc in data/<run>/product, constraint_results(.pkl, _jiang.pkl),
structural_adjustments.npz and structural_adjustments_<key>.npz of the four single-model patterns (run_structural.sh
structural), feedback_pattern.npz, inputs/co2 (co2_history.py).
Output: data/<run>/oatrack/
Usage (code/):  OA_RUN=u74 python -u make_oatrack.py                                   (all references, then combined)
                OA_RUN=u74 python -u make_oatrack.py --reference UExP --no-combine    (one process per reference)
                OA_RUN=u74 python -u make_oatrack.py --combine-only                   (after all four products)
                --no-alk: only the files without the adjustment (fast check of the structure)
                --out DIR: write to DIR instead of data/<run>/oatrack (for tests)
"""
from __future__ import annotations

import argparse
import datetime
import warnings

import netCDF4
import numpy as np

# Chunk caches. netCDF-C keeps up to 64 MB of uncompressed chunks per variable (the default of netCDF 4.10) for every
# variable of every open file, and a written chunk can stay there until the file is closed. With the 3574 variables of
# a file, written year by year in two versions, this can exceed the memory of a large server when several
# references run in parallel. Reading: at most 8 MB per variable (every file is read map by map); writing: two maps per
# variable (write_cache), so that a written map goes to the disk as soon as the next one comes.
netCDF4.set_chunk_cache(8 * 2 ** 20, 521, 0.75)

import oa_config
import calc_bias_adjusted as cb
import co2_history
import make_product as mp
import make_product_alkalinity as ma

OUT = oa_config.DATA / 'oatrack'
QUANTITIES, UNITS, LONG = mp.QUANTITIES, mp.UNITS, mp.LONG
MAPS = mp.MAPS                                   # (suffix, key) of the model maps
LEVEL_SER = mp.LEVEL_SER                         # key -> sigma_mapping_c1, sigma_mapping_c2, sigma_constants
OBS_MAPS = [('', 'c')] + [(f'_{v}', k) for k, v in LEVEL_SER.items()]
SERIES, SERIES_LABEL, MAP_NAMES = mp.SERIES, mp.SERIES_LABEL, mp.MAP_NAMES
DELTA_Q = ma.DELTA_Q
CHANGED = ('', '_unconstrained')                 # the suffixes the alkalinity adjustment changes


def write_cache(x, lat, lon):
    """chunk cache of a written variable: two maps (see the chunk caches above)"""
    x.set_var_chunk_cache(size=2 * len(lat) * len(lon) * 4 + 4096, nelems=7, preemption=1.0)
    return x


EVAL_YEAR = mp.EVAL_YEAR
YEARS = np.arange(1750, 2015)
YEARS_SSP = np.arange(1750, 2101)
KEEP_STATES = ('1750', '1850')
STATE_NAMES = {'c': '{q}_state', 'u': '{q}_unconstrained_state', 'sigma': '{q}_sigma_state',
               'sigma_u': '{q}_sigma_unconstrained_state', 's_iv': '{q}_sigma_internal_state',
               **{k: '{q}_' + v + '_state' for k, v in LEVEL_SER.items()}}
FILL = mp.FILL


def fname(ref, version):
    return OUT / f'OA-Track_{ref}{"" if version == "alk" else "_noalk"}.nc'


class Pair:
    """the two files (with and without the alkalinity adjustment) written in step; versions: ('noalk',) or both"""

    def __init__(self, paths, lat, lon, attrs):
        self.g = {v: cb.new_nc(p, lat, lon, dict(attrs, alkalinity_adjustment=(
            'included (the best estimate of the paper)' if v == 'alk' else 'not included'))) for v, p in paths.items()}
        self.lat, self.lon = lat, lon

    def dim(self, name, values, dtype='i4', **att):
        for g in self.g.values():
            g.createDimension(name, len(values))
            x = g.createVariable(name, dtype, (name,))
            x[:] = np.array(values, dtype=object) if dtype is str else np.asarray(values)
            for k, v in att.items():
                x.setncattr(k, v)

    def field(self, name, dims, unit, long_name, only=None):
        out = {}
        for v, g in self.g.items():
            if only and v not in only:
                continue
            chunks = tuple(1 if d not in ('lat', 'lon') else (self.lat.size if d == 'lat' else self.lon.size) for d in dims)
            x = g.createVariable(name, 'f4', dims, zlib=True, complevel=4, fill_value=FILL, chunksizes=chunks)
            write_cache(x, self.lat, self.lon)
            x.units, x.long_name = unit, long_name
            out[v] = x
        return out

    def series(self, name, dims, data, unit, long_name, only=None):
        """data: one array for all versions or {version: array}"""
        for v, g in self.g.items():
            if only and v not in only:
                continue
            x = g.createVariable(name, 'f8', dims, fill_value=np.nan)
            x.units, x.long_name = unit, long_name
            x[...] = data[v] if isinstance(data, dict) else data

    def attr(self, name, value):
        for g in self.g.values():
            g.setncattr(name, value)

    def close(self):
        for g in self.g.values():
            g.close()


# ================================================================================================ per reference
class Ref(mp.Reference):
    """the chain files of one reference (make_product.Reference) with the years of OA-Track"""

    def __init__(self, ref):
        super().__init__(ref)
        self.years_obs_out = np.array([2010]) if ref == 'Jiang' else self.years_obs.copy()
        self.istates = [i for i, s in enumerate(self.states) if s.split('_')[0] in KEEP_STATES]
        assert [self.states[i].split('_')[0] for i in self.istates] == list(KEEP_STATES), self.states

    def model_file(self, y, scen=None):
        """(file, its years) of a model year of the historical series (scen None) or of a scenario series"""
        if scen is None:
            return (self.pi, self.years_pi) if y < 1850 else (self.hist, self.years_hist)
        Fp, F = self.scen_pi[scen], self.scen[scen]
        return (Fp, np.asarray(Fp['year'][:]).astype(int)) if y < 1850 else (F, np.asarray(F['year'][:]).astype(int))

    def hist_series(self, q, mask_key, kind, what='value'):
        """a series of the states files on YEARS (1750-1849 pi, 1850-2014 hist)"""
        a = np.full(YEARS.size, np.nan)
        a[YEARS < 1850] = self.series(self.pi, q, mask_key, kind, what)
        h = self.series(self.hist, q, mask_key, kind, what)
        for i, y in enumerate(YEARS):
            if y >= 1850:
                a[i] = h[np.searchsorted(self.years_hist, y)]
        return a

    def hist_var(self, name):
        a = np.full(YEARS.size, np.nan)
        a[YEARS < 1850] = np.asarray(self.pi[name][:], 'f8')
        h = np.asarray(self.hist[name][:], 'f8')
        for i, y in enumerate(YEARS):
            if y >= 1850:
                a[i] = h[np.searchsorted(self.years_hist, y)]
        return a

    def scen_series(self, s, q, mask_key, kind, what='value'):
        return np.concatenate([self.series(self.scen_pi[s], q, mask_key, kind, what), self.series(self.scen[s], q, mask_key, kind, what)])


def alk_series(ref, R, P, scen):
    """m(t) (common-area missing alkalinity, umol/kg) of the historical series, the scenarios, the product years and
    the 2024 trend value of a reference"""
    if ref == 'Jiang':
        m_hist, m_scen, _ = P.jiang(YEARS, YEARS_SSP)
        m_obs, m_trend = np.zeros(1), 0.0
    else:
        m_hist = P.on(P.m_model[ref], YEARS)
        m_scen = {s: P.on(P.m_s[s][ref], YEARS_SSP) for s in scen if s in P.m_s}
        m_obs = P.on(P.m_product[ref], R.years_obs_out)
        m_trend = float(P.on(P.m_product[ref], np.array([EVAL_YEAR]))[0])
    assert np.all(np.isfinite(m_hist)), f'{ref}: m(t) of the model formula missing for {YEARS[~np.isfinite(m_hist)]}'
    assert np.all(np.isfinite(m_obs)), f'{ref}: m(t) of the product formula missing for {R.years_obs_out[~np.isfinite(m_obs)]}'
    for s, a in m_scen.items():
        assert np.all(np.isfinite(a)), f'{ref} {s}: m(t) missing for {YEARS_SSP[~np.isfinite(a)]}'
    return m_hist, m_scen, m_obs, m_trend


def write_block(W, R, chem, pattern, versions, tag, dim, years, getter, suffixes, m, masks, label, alts=()):
    """maps of every quantity for one block (historical series, a scenario, the product years or the states), year by
    year, in both versions; alts: (pattern, m) of the single-model patterns, for the 1 sigma of the adjustment in each
    cell. Returns the area means {(q, mask): array} of the constrained value (noalk) and of the alkalinity change"""
    V = {}
    for q in QUANTITIES:
        for suf, _ in suffixes:
            V[(q, suf)] = W.field(f'{q}{tag}{suf}', (dim, 'lat', 'lon'), UNITS[q], f'{LONG[q]}{label}: {MAP_NAMES.get(suf, suf)}')
        if 'alk' in versions and q in DELTA_Q:
            V[(q, '_sigma_alkalinity')] = W.field(f'{q}{tag}_sigma_alkalinity', (dim, 'lat', 'lon'), UNITS[q],
                                                  f'{LONG[q]}{label}: 1 sigma of the alkalinity adjustment (sqrt(adjustment^2 + spread over the single-model patterns^2))', only=('alk',))
    am = {(q, mk): np.full(len(years), np.nan) for q in QUANTITIES for mk in masks}
    dm = {(q, mk): np.zeros(len(years)) for q in DELTA_Q for mk in masks}
    for i, y in enumerate(years):
        F = {(q, k): getter(q, k, i, y) for q in QUANTITIES for _, k in suffixes}
        for key in F:
            F[key][~R.area_ok] = np.nan
        D, Dk = {}, []
        if 'alk' in versions:
            Ds = chem.deltas_multi(F[('thetao', 'c')], F[('so', 'c')], F[('talk', 'c')], F[('dic', 'c')],
                                   [pattern * m[i]] + [pk * mk_[i] for pk, mk_ in alts])
            D, Dk = Ds[0], Ds[1:]
        for q in QUANTITIES:
            for suf, k in suffixes:
                f = F[(q, k)]
                V[(q, suf)]['noalk'][i] = np.ma.masked_invalid(f.astype('f4'))
                if 'alk' in versions:
                    fa = f + D[q] if (q in D and suf in CHANGED) else f
                    V[(q, suf)]['alk'][i] = np.ma.masked_invalid(fa.astype('f4'))
            if q in D:                                              # the change and the spread of the patterns
                sp = np.std([d_[q] for d_ in Dk], axis=0, ddof=1) if len(Dk) > 1 else 0.0
                V[(q, '_sigma_alkalinity')]['alk'][i] = np.ma.masked_invalid(np.hypot(D[q], sp).astype('f4'))
            for mk, mask in masks.items():
                am[(q, mk)][i] = R.am(F[(q, 'c')], mask)
                if q in D:
                    dm[(q, mk)][i] = R.am(np.where(np.isfinite(F[(q, 'c')]), D[q], np.nan), mask)
    return am, dm


def build(ref, P, with_alk=True):
    R = Ref(ref)
    versions = ('noalk', 'alk') if with_alk else ('noalk',)
    OUT.mkdir(parents=True, exist_ok=True)
    scen = list(R.scen)
    lat, lon = R.lat, R.lon
    pattern, ALT = None, []
    if with_alk:
        pattern, pat_created = ma.load_pattern(lat, lon)
        pattern = np.where(R.area_ok, pattern, np.nan)
        m_hist, m_scen, m_obs, m_trend = alk_series(ref, R, P, scen)
        # the single-model patterns, each with its own m(t): their spread enters the 1 sigma of the maps
        for key in ma.PATTERN_ALTERNATIVES:
            Pk = ma.Params(key)
            assert Pk.series_source == P.series_source, f'{key}: {Pk.series_source}, central {P.series_source}: rerun run_structural.sh'
            pk, _ = ma.load_pattern(lat, lon, key)
            ALT.append(dict(key=key, pattern=np.where(R.area_ok, pk, np.nan), m=alk_series(ref, R, Pk, scen)))
    else:
        m_hist, m_scen, m_obs, m_trend = (np.zeros(YEARS.size), {s: np.zeros(YEARS_SSP.size) for s in scen},
                                          np.zeros(R.years_obs_out.size), 0.0)
    chem = ma.Chem(lat, lon) if with_alk else None
    print(f'\n=== {ref}: {mp.LABEL[ref]}; historical 1750-2014, scenarios {scen}, observation-based years '
          f'{R.years_obs_out[0]}-{R.years_obs_out[-1]}; versions {versions}', flush=True)
    W = Pair({v: fname(ref, v) for v in versions}, lat, lon, dict(
        title=f'OA-Track: surface ocean carbonate system 1750-2100, reference {mp.LABEL[ref]}',
        method=__doc__.split('Combined files')[0].split('\n', 3)[3].strip(),
        reference=mp.LABEL[ref], constants=oa_config.CONSTANTS, area_weights=oa_config.WEIGHTS_TEXT,
        chain_version=oa_config.VERSION, models=R.hist.models, n_models=int(R.hist.n_models),
        constraint=R.hist.constraint, observed_predictors=R.hist.observed_predictors, smoothing=R.hist.smoothing,
        eval_year=EVAL_YEAR, scenarios=', '.join(f'{s} ({int(R.scen[s].n_models)} models)' for s in scen),
        history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} make_oatrack.py, {oa_config.VERSION}'))
    W.dim('year', YEARS)
    if scen:
        W.dim('year_ssp', YEARS_SSP)
    W.dim('year_obs', R.years_obs_out, **({'long_name': 'the decade 2005-2014 of Jiang et al. (2023)'} if ref == 'Jiang' else {}))
    W.dim('state', [R.states[i].split('_')[0] for i in R.istates], dtype=str,
          long_name='constrained preindustrial states at the CO2 of IPCC AR6 (278.3 ppm in 1750, 285.5 ppm in 1850)')
    # masks, flags and CO2
    for nm, a, ln in (('mask', R.area_ok, 'the area of the reference: cells shared with all 18 ESMs and covered in every year'),
                      ('mask_common', R.mask_com & R.area_ok, 'the common area of the paper')):
        for g in W.g.values():
            x = g.createVariable(nm, 'i1', ('lat', 'lon')); x.long_name = ln; x[:] = a.astype('i1')
    for g in W.g.values():
        x = g.createVariable('constrained_flag', 'f4', ('lat', 'lon'), fill_value=FILL)
        x.flag_values, x.flag_meanings = '1 2', 'regional_regression multi_model_mean'
        x[:] = np.ma.masked_invalid(R.rd(R.hist, 'constrained_flag'))
    W.series('xco2', ('year',), R.hist_var('xco2'), 'ppm', 'atmospheric CO2 of the year, CMIP6 historical concentrations (Meinshausen et al. 2017)')
    co2 = co2_history.CO2History()
    for s in scen:
        x_ = np.r_[np.asarray(R.pi['xco2'][:], 'f8'), co2.series(np.arange(1850, 2101), s)]
        W.series(f'xco2_{s}', ('year_ssp',), x_, 'ppm', f'atmospheric CO2 of the year, {s} (CMIP6 history to 2014, ScenarioMIP afterwards)')
    noaa = co2_history.noaa()
    xo = (np.array([np.mean([noaa[y] for y in range(2005, 2015) if y in noaa])]) if ref == 'Jiang'
          else np.array([noaa.get(int(y), np.nan) for y in R.years_obs_out]))
    W.series('xco2_obs', ('year_obs',), xo, 'ppm', 'atmospheric CO2 of the year, NOAA global annual mean (Lan et al.)'
             + ('; Jiang et al.: mean of 2005-2014' if ref == 'Jiang' else ''))
    W.series('state_xco2', ('state',), np.asarray(R.pi['state_xco2'][:], 'f8')[R.istates], 'ppm', 'CO2 of the state (AR6)')
    W.series('state_xco2_sigma', ('state',), np.asarray(R.pi['state_xco2_sigma'][:], 'f8')[R.istates], 'ppm', '1 sigma of the CO2 of the state')
    masks = {'own': R.mask_own, 'common': R.mask_com}

    # ---------------------------------------------------------------- maps
    def g_hist(q, k, i, y):
        f, yrs = R.model_file(y)
        return R.model_year(f, yrs, q, y, k)
    am_h, dm_h = write_block(W, R, chem, pattern, versions, '', 'year', YEARS, g_hist, MAPS, m_hist, masks,
                             ', historical series', [(a_['pattern'], a_['m'][0]) for a_ in ALT])
    print(f'  historical series written; Omega_arag change by the alkalinity (common area) 1750 '
          f'{dm_h[("omega_arag", "common")][0]:+.4f}, 2014 {dm_h[("omega_arag", "common")][-1]:+.4f}', flush=True)
    am_s, dm_s = {}, {}
    for s in scen:
        def g_scen(q, k, i, y, s=s):
            f, yrs = R.model_file(y, s)
            return R.model_year(f, yrs, q, y, k)
        am_s[s], dm_s[s] = write_block(W, R, chem, pattern, versions, f'_{s}', 'year_ssp', YEARS_SSP, g_scen, MAPS,
                                       m_scen.get(s, np.zeros(YEARS_SSP.size)), masks, f', {s} ({int(R.scen[s].n_models)} ESMs)',
                                       [(a_['pattern'], a_['m'][1].get(s, np.zeros(YEARS_SSP.size))) for a_ in ALT])
        print(f'  {s} written', flush=True)

    def g_obs(q, k, i, y):
        return R.obs(q, y) if k == 'c' else R.obs(f'{q}_{LEVEL_SER[k]}', y)
    am_o, dm_o = write_block(W, R, chem, pattern, versions, '_obs', 'year_obs', R.years_obs_out, g_obs, OBS_MAPS, m_obs,
                             masks, f', {mp.LABEL[ref]} itself', [(a_['pattern'], a_['m'][2]) for a_ in ALT])
    # states: m at their Omega_arag (relation of the historical years 1750-1870)
    om_states = np.array([R.am(R.rd(R.pi, 'omega_arag_state', j), R.mask_com) for j in R.istates])
    m_st = ma.m_states(om_states, YEARS, am_h[('omega_arag', 'common')], m_hist) if with_alk else np.zeros(len(R.istates))

    def g_state(q, k, i, y):
        return R.rd(R.pi, STATE_NAMES[k].format(q=q), R.istates[i])
    alt_st = [(a_['pattern'], ma.m_states(om_states, YEARS, am_h[('omega_arag', 'common')], a_['m'][0])) for a_ in ALT]
    am_st, dm_st = write_block(W, R, chem, pattern, versions, '_state', 'state', list(range(len(R.istates))), g_state,
                               MAPS, m_st, masks, ', states of 1750 and 1850 (AR6 CO2)', alt_st)
    # the 2024 trend value of the reference
    trend = {}
    iy = [i for i, y in enumerate(R.years_obs_out) if y <= EVAL_YEAR]
    for q in QUANTITIES:
        if ref == 'Jiang':
            trend[q] = R.obs(q, 2010)
        else:
            stack = np.array([R.obs(q, int(R.years_obs_out[i])) for i in iy])
            trend[q] = mp.quad_fit_year(R.years_obs_out[iy], stack, EVAL_YEAR)
        trend[q] = np.where(R.area_ok, trend[q], np.nan)
    Dt = (chem.deltas(trend['thetao'], trend['so'], trend['talk'], trend['dic'], pattern * m_trend)
          if with_alk and m_trend != 0.0 else {q: np.zeros(trend[q].shape) for q in DELTA_Q})
    for q in QUANTITIES:
        x = W.field(f'{q}_{EVAL_YEAR}_trend', ('lat', 'lon'), UNITS[q],
                    f'{LONG[q]}: the decade 2005-2014 of Jiang et al.' if ref == 'Jiang' else
                    f'{LONG[q]} in {EVAL_YEAR}: quadratic fit of the years {R.years_obs_out[iy][0]}-{EVAL_YEAR} evaluated in {EVAL_YEAR}')
        x['noalk'][:] = np.ma.masked_invalid(trend[q].astype('f4'))
        if 'alk' in x:
            x['alk'][:] = np.ma.masked_invalid((trend[q] + Dt.get(q, 0.0)).astype('f4'))

    # ---------------------------------------------------------------- global series and relative changes
    summary = {}
    for q in QUANTITIES:
        name = LONG[q]
        for mk in ('own', 'common'):
            sfx = '' if mk == 'own' else '_common'
            gl = f'global mean on the {"area of the reference" if mk == "own" else "common area of the paper"}'
            dh = {'noalk': 0.0, 'alk': dm_h.get((q, mk), 0.0)}
            dst = {'noalk': 0.0, 'alk': dm_st.get((q, mk), 0.0)}
            do = {'noalk': 0.0, 'alk': dm_o.get((q, mk), 0.0)}
            S = {}
            for kind in SERIES:
                val, sig = R.hist_series(q, mk, kind), R.hist_series(q, mk, kind, 'sigma')
                st = np.asarray(R.pi[f'{q}_gm_{mk}_{kind}_state'][:], 'f8')[R.istates]
                sst = np.asarray(R.pi[f'{q}_gm_{mk}_{kind}_sigma_state'][:], 'f8')[R.istates]
                S[kind] = dict(val={v: val + dh[v] for v in versions}, sig=sig, st={v: st + dst[v] for v in versions}, sst=sst)
                W.series(f'{q}_gm{sfx}_{kind}', ('year',), S[kind]['val'], UNITS[q], f'{name}, {gl}: {SERIES_LABEL[kind]}')
                W.series(f'{q}_gm{sfx}_{kind}_sigma', ('year',), sig, UNITS[q], f'{name}, {gl}: 1 sigma of the constraint ({kind})')
                W.series(f'{q}_gm{sfx}_{kind}_state', ('state',), S[kind]['st'], UNITS[q], f'{name}, {gl}: states of 1750 and 1850 ({kind})')
                W.series(f'{q}_gm{sfx}_{kind}_state_sigma', ('state',), sst, UNITS[q], f'{name}, {gl}: states, 1 sigma ({kind})')
            if q in DELTA_Q and 'alk' in versions:
                W.series(f'{q}_gm{sfx}_sigma_alkalinity', ('year',), np.abs(dm_h[(q, mk)]), UNITS[q],
                         f'{name}, {gl}: 1 sigma of the alkalinity adjustment (= |adjustment| of the area mean)', only=('alk',))
            W.series(f'{q}_gm{sfx}_sigma_internal', ('year',), R.hist_series(q, mk, 'internal', 'sigma'), UNITS[q],
                     f'{name}, {gl}: 1 sigma of internal variability (not in the other sigmas)')
            for lv in LEVEL_SER.values():
                W.series(f'{q}_gm{sfx}_{lv}', ('year',), R.hist_series(q, mk, 'level', lv), UNITS[q], f'{name}, {gl}: {MAP_NAMES["_" + lv]}')
                W.series(f'{q}_gm{sfx}_{lv}_state', ('state',), np.asarray(R.pi[f'{q}_gm_{mk}_level_{lv}_state'][:], 'f8')[R.istates],
                         UNITS[q], f'{name}, {gl}: states, {MAP_NAMES["_" + lv]}')
            for nm, what in (('direct', 'value'), ('direct_sigma', 'sigma'), ('direct_r2', 'r2')):
                a = R.hist_series(q, mk, 'direct', what)
                W.series(f'{q}_gm{sfx}_{nm}', ('year',), {v: a + dh[v] for v in versions} if what == 'value' else a,
                         UNITS[q] if what != 'r2' else '1', f'{name}, {gl}: the direct method of the paper ({what})')
            # the observation-based estimate itself
            W.series(f'{q}_gm{sfx}_obs', ('year_obs',), {v: am_o[(q, mk)] + do[v] for v in versions}, UNITS[q],
                     f'{name}, {gl}: {mp.LABEL[ref]} itself')
            for lv in LEVEL_SER.values():
                W.series(f'{q}_gm{sfx}_obs_{lv}', ('year_obs',), np.array([R.obs_series(q, mk, lv, int(y)) for y in R.years_obs_out]),
                         UNITS[q], f'{name}, {gl}: {mp.LABEL[ref]} itself, {MAP_NAMES["_" + lv].split(";")[0]}')
            now = {v: R.am(trend[q], masks[mk]) + (R.am(np.where(np.isfinite(trend[q]), Dt.get(q, 0.0), np.nan), masks[mk])
                                                   if (v == 'alk' and q in Dt) else 0.0) for v in versions}
            W.series(f'{q}_gm{sfx}_{EVAL_YEAR}_trend', (), now, UNITS[q], f'{name}, {gl}: the {EVAL_YEAR} trend value'
                     if ref != 'Jiang' else f'{name}, {gl}: the decade 2005-2014 of Jiang et al.')
            for kind in SERIES:
                for lab, p_, sp_ in (('', {v: S[kind]['val'][v][0] for v in versions}, S[kind]['sig'][0]),
                                     ('_from_AR6_1750', {v: S[kind]['st'][v][0] for v in versions}, S[kind]['sst'][0])):
                    dec = {v: 100 * (now[v] / p_[v] - 1) for v in versions}
                    W.series(f'{q}_gm{sfx}_{kind}_decline{lab}', (), dec, '%',
                             f'{name}, {gl}: relative change from 1750{lab.replace("_", " ")} to the {EVAL_YEAR} trend value ({kind})')
                    W.series(f'{q}_gm{sfx}_{kind}_decline{lab}_sigma_1750', (), {v: 100 * now[v] / p_[v] ** 2 * sp_ for v in versions}, '%',
                             '1 sigma of the relative change from the 1750 side (constraint and CO2); the full uncertainty of the '
                             'decline of Omega_arag is given in Extended Data Table 2 of the paper')
                    if 'alk' in versions:
                        W.series(f'{q}_gm{sfx}_{kind}_decline{lab}_sigma_alkalinity', (), abs(dec['alk'] - dec['noalk']), '%',
                                 '1 sigma of the relative change from the alkalinity adjustment (= the change it makes)', only=('alk',))
                if mk == 'common' and kind == 'global':
                    summary[q] = {v: (S[kind]['st'][v][0], now[v], 100 * (now[v] / S[kind]['st'][v][0] - 1)) for v in versions}
            # scenarios
            for s in scen:
                ds = {'noalk': 0.0, 'alk': dm_s[s].get((q, mk), 0.0)}
                for kind in SERIES:
                    a = R.scen_series(s, q, mk, kind)
                    av = {v: a + ds[v] for v in versions}
                    W.series(f'{q}_gm{sfx}_{s}_{kind}', ('year_ssp',), av, UNITS[q], f'{name}, {s}, {gl}: {SERIES_LABEL[kind]}')
                    W.series(f'{q}_gm{sfx}_{s}_{kind}_sigma', ('year_ssp',), R.scen_series(s, q, mk, kind, 'sigma'), UNITS[q], f'{name}, {s}, {gl}: 1 sigma ({kind})')
                    W.series(f'{q}_gm{sfx}_{s}_{kind}_decline', ('year_ssp',), {v: 100 * (av[v] / av[v][0] - 1) for v in versions}, '%',
                             f'{name}, {s}: relative change from the 1750 of the same ESMs ({kind})')
                if q in DELTA_Q and 'alk' in versions:
                    W.series(f'{q}_gm{sfx}_{s}_sigma_alkalinity', ('year_ssp',), np.abs(dm_s[s][(q, mk)]), UNITS[q],
                             f'{name}, {s}, {gl}: 1 sigma of the alkalinity adjustment', only=('alk',))
                W.series(f'{q}_gm{sfx}_{s}_sigma_internal', ('year_ssp',), R.scen_series(s, q, mk, 'internal', 'sigma'), UNITS[q],
                         f'{name}, {s}, {gl}: 1 sigma of internal variability')
                for lv in LEVEL_SER.values():
                    W.series(f'{q}_gm{sfx}_{s}_{lv}', ('year_ssp',), R.scen_series(s, q, mk, 'level', lv), UNITS[q], f'{name}, {s}, {gl}: {MAP_NAMES["_" + lv]}')
    # model numbers, CO2 notes and diagnostics
    for s in scen:
        Fp, F = R.scen_pi[s], R.scen[s]
        ys = np.asarray(F['year'][:]).astype(int)
        nm_ = np.full(YEARS_SSP.size, int(getattr(Fp, 'n_models', F.n_models)), 'i4')
        if 'n_models' in F.variables:
            nf = np.asarray(F['n_models'][:]).astype(int)
            for i, y in enumerate(YEARS_SSP):
                if y >= 1850:
                    nm_[i] = nf[np.searchsorted(ys, y)]
        for g in W.g.values():
            x = g.createVariable(f'n_models_{s}', 'i4', ('year_ssp',)); x[:] = nm_
            x.long_name = f'number of ESMs in the constraint of {s} in each year (MPI-ESM1-2-HR and UKESM1-0-LL end in 2099)'
    for key in ('own', 'common'):
        for nm in ('r2_thetao', 'r2_dmc'):
            W.series(f'{nm}_{key}', ('year',), R.hist_var(f'{nm}_{key}'), '1',
                     f'R2 of the global state regression ({"T on dSST" if "thetao" in nm else "DIC-ALK on dSST and CUC"}), {key} area')
    for nm in ('rms', 'max'):
        W.series(f'omega_arag_regional_direct_{nm}', ('year',), R.hist_var(f'omega_arag_regional_direct_{nm}'), '1',
                 f'{nm} over the cells of the per-cell direct regression of Omega_arag minus the regional state')
    for g in W.g.values():
        mp.smooth_diag(g, R, lambda name, dims, data, unit, ln, g=g: _svar(g, name, dims, data, unit, ln))
    if with_alk:
        W.series('talk_missing', ('year',), m_hist, 'umol kg-1', 'missing alkalinity m(t) of the historical series, common-area mean (maps: m x pattern)', only=('alk',))
        W.series('talk_missing_obs', ('year_obs',), m_obs, 'umol kg-1', 'missing alkalinity of the observation-based years, common-area mean', only=('alk',))
        W.series('talk_missing_state', ('state',), m_st, 'umol kg-1', 'missing alkalinity of the states, common-area mean', only=('alk',))
        for s in m_scen:
            W.series(f'talk_missing_{s}', ('year_ssp',), m_scen[s], 'umol kg-1', f'missing alkalinity m(t), {s}, common-area mean', only=('alk',))
        x = W.field('feedback_pattern', ('lat', 'lon'), '1', 'spatial pattern of the missing alkalinity (mean 1 over the common area)', only=('alk',))
        x['alk'][:] = np.ma.masked_invalid(pattern.astype('f4'))
        W.g['alk'].setncattr('alkalinity', f'{ma.STRUCT.name} ({P.created}, {P.source}); {ma.PATTERN.name} ({pat_created}); '
                                           f'unchanged: {ma.UNCHANGED}')
        W.g['alk'].setncattr('alkalinity_sigma', 'maps: sqrt(adjustment^2 + s^2) per cell, s the standard deviation (ddof 1) over '
                                                 'the single-model patterns ' + ', '.join(a_['key'] for a_ in ALT) + ', each with '
                                                 'its own m(t) (structural_adjustments_<key>.npz); area means and relative '
                                                 'changes: |adjustment|')
        W.g['alk'].setncattr('newton', f'max |pCO2 - pCO2_0| {chem.resid:.2e} uatm')
    W.close()
    print(f'  {"":11s} {"version":>7s} {"1750 AR6":>9s} {str(EVAL_YEAR) if ref != "Jiang" else "2005-14":>9s} {"decline":>8s}   (common area, global constraint)')
    for q in ('omega_arag', 'ph', 'co3', 'talk', 'dic'):
        for v in versions:
            p_, n_, d_ = summary[q][v]
            print(f'  {q:11s} {v:>7s} {p_:9.4f} {n_:9.4f} {d_:7.2f}%')
    print(f'  check: Omega_arag {EVAL_YEAR} trend (own area, noalk) {R.am(trend["omega_arag"], R.mask_own):.6f}, paper {R.paper_now:.6f}')
    print('  written ' + ', '.join(str(fname(ref, v)) for v in versions), flush=True)
    R.close()


def _svar(g, name, dims, data, unit, long_name):
    x = g.createVariable(name, 'f8', dims, fill_value=np.nan)
    x.units, x.long_name = unit, long_name
    x[:] = data


# ================================================================================================ combined
def combine(version):
    refs = [r for r in mp.PRODUCTS if fname(r, version).exists()]
    if len(refs) < len(mp.PRODUCTS):
        raise SystemExit(f'OA-Track files ({version}) missing for {[r for r in mp.PRODUCTS if r not in refs]}')
    F = {r: netCDF4.Dataset(fname(r, version)) for r in refs}
    f0 = F[refs[0]]
    lat, lon = np.asarray(f0['lat'][:], 'f8'), np.asarray(f0['lon'][:], 'f8')
    masks = {r: np.asarray(F[r]['mask'][:]).astype(bool) for r in refs}
    common = np.all([np.asarray(F[r]['mask_common'][:]).astype(bool) for r in refs], axis=0)
    flag = np.zeros(common.shape, 'i1')
    for r in refs:
        flag += (mp.BIT[r] * masks[r]).astype('i1')
    scen = sorted(set.intersection(*[{s for s in mp.SCENARIOS if f'omega_arag_{s}' in F[r].variables} for r in refs]))
    yo = {r: np.asarray(F[r]['year_obs'][:]).astype(int) for r in refs}
    years_obs = np.arange(min(v.min() for v in yo.values()), max(v.max() for v in yo.values()) + 1)
    rd = lambda r, name: np.ma.filled(F[r][name][:].astype('f4'), np.nan)
    tag = '' if version == 'alk' else '_noalk'
    for kind, keep in (('common', common), ('all', flag > 0)):
        path = OUT / f'OA-Track_{kind}{tag}.nc'
        g = cb.new_nc(path, lat, lon, dict(
            title='OA-Track: surface ocean carbonate system 1750-2100, mean of the four fCO2-products '
                  + ('on the common area of the paper' if kind == 'common' else 'covering each cell'),
            references=', '.join(mp.LABEL[r] for r in refs), constants=oa_config.CONSTANTS, area_weights=oa_config.WEIGHTS_TEXT,
            chain_version=oa_config.VERSION, method=__doc__.split('Combined files')[1].split('Needs:')[0].strip(),
            alkalinity_adjustment='included' if version == 'alk' else 'not included', eval_year=EVAL_YEAR,
            history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} make_oatrack.py'))
        for dim, vals in (('year', YEARS), ('year_ssp', YEARS_SSP if scen else None), ('year_obs', years_obs)):
            if vals is None:
                continue
            g.createDimension(dim, vals.size)
            g.createVariable(dim, 'i4', (dim,))[:] = vals
        g.createDimension('state', 2)
        x = g.createVariable('state', str, ('state',)); x[:] = np.array(KEEP_STATES, dtype=object)
        x = g.createVariable('mask', 'i1', ('lat', 'lon')); x[:] = keep.astype('i1')
        if kind == 'all':
            x = g.createVariable('product_flag', 'i1', ('lat', 'lon'))
            x.long_name, x.flag_masks = 'fCO2-products covering the cell (sum of bits)', '1 2 4 8'
            x.flag_meanings = ' '.join(mp.LABEL[r].replace(' ', '_') for r in mp.PRODUCTS)
            x[:] = flag
        _svar(g, 'xco2', ('year',), np.asarray(f0['xco2'][:], 'f8'), 'ppm', 'atmospheric CO2 of the year, CMIP6 historical concentrations')
        for s in scen:
            _svar(g, f'xco2_{s}', ('year_ssp',), np.asarray(f0[f'xco2_{s}'][:], 'f8'), 'ppm', f'atmospheric CO2 of the year, {s}')
        noaa = co2_history.noaa()
        _svar(g, 'xco2_obs', ('year_obs',), np.array([noaa.get(int(y), np.nan) for y in years_obs]), 'ppm', 'NOAA global annual mean CO2')

        def put(name, dims, data, unit, long_name):
            x = write_cache(g.createVariable(name, 'f4', dims, zlib=True, complevel=4, fill_value=FILL), lat, lon)
            x.units, x.long_name = unit, long_name
            data = np.where(keep, data, np.nan) if data.ndim >= 2 else data
            x[:] = np.ma.masked_invalid(data.astype('f4'))

        for q in QUANTITIES:
            nm = LONG[q]
            # series, scenarios and states: mean over the references holding a value
            for tg, dim, sufs in [('', 'year', MAPS), ('_state', 'state', MAPS)] + [(f'_{s}', 'year_ssp', MAPS) for s in scen]:
                A = {suf: np.array([rd(r, f'{q}{tg}{suf}') for r in refs]) for suf, _ in sufs}
                if version == 'alk' and q in DELTA_Q:
                    A['_sigma_alkalinity'] = np.array([rd(r, f'{q}{tg}_sigma_alkalinity') for r in refs])
                with np.errstate(invalid='ignore'):
                    mean = np.nanmean(A[''], 0)
                    n = np.sum(np.isfinite(A['']), 0)
                    sp = np.where(n > 1, np.nanstd(A[''], 0, ddof=1), np.nan)
                    sig = np.sqrt(np.nanmean(A['_sigma'] ** 2, 0) + np.nan_to_num(sp) ** 2)
                    iv = np.sqrt(np.nanmean(A['_sigma_internal'] ** 2, 0))
                put(f'{q}{tg}', (dim, 'lat', 'lon'), mean, UNITS[q], f'{nm}: mean of the references')
                put(f'{q}{tg}_sigma', (dim, 'lat', 'lon'), np.where(np.isfinite(mean), sig, np.nan), UNITS[q],
                    f"{nm}: 1 sigma, the references' sigma and their spread in quadrature")
                put(f'{q}{tg}_spread', (dim, 'lat', 'lon'), sp, UNITS[q], f'{nm}: spread across the references')
                put(f'{q}{tg}_sigma_internal', (dim, 'lat', 'lon'), iv, UNITS[q], f'{nm}: internal variability, rms of the references')
                with np.errstate(invalid='ignore'):
                    for suf in ('_unconstrained', '_sigma_unconstrained', '_sigma_mapping_c1', '_sigma_mapping_c2', '_sigma_constants',
                                '_sigma_alkalinity'):
                        if suf in A:
                            put(f'{q}{tg}{suf}', (dim, 'lat', 'lon'), np.nanmean(A[suf], 0), UNITS[q],
                                f"{nm}: {MAP_NAMES.get(suf, suf).split(';')[0]}, mean of the references")
            # the fCO2-products themselves: the mean of those present in the year and cell
            O = np.full((len(refs), years_obs.size) + keep.shape, np.nan, 'f4')
            L = {lv: O.copy() for lv in LEVEL_SER.values()}
            if version == 'alk' and q in DELTA_Q:                   # 1 sigma of the adjustment, as the other maps
                L['sigma_alkalinity'] = O.copy()
            for k, r in enumerate(refs):
                ii = np.searchsorted(years_obs, yo[r])
                O[k, ii] = rd(r, f'{q}_obs')
                for lv in L:
                    L[lv][k, ii] = rd(r, f'{q}_obs_{lv}')
            with np.errstate(invalid='ignore'):
                n = np.sum(np.isfinite(O), 0)
                put(f'{q}_obs', ('year_obs', 'lat', 'lon'), np.nanmean(O, 0), UNITS[q], f'{nm}: mean of the fCO2-products present')
                put(f'{q}_obs_spread', ('year_obs', 'lat', 'lon'), np.where(n > 1, np.nanstd(O, 0, ddof=1), np.nan), UNITS[q],
                    f'{nm}: spread of the fCO2-products present')
                for lv, A_ in L.items():
                    put(f'{q}_obs_{lv}', ('year_obs', 'lat', 'lon'), np.nanmean(A_, 0), UNITS[q], f'{nm}: {lv}, mean of the fCO2-products present')
            if q == QUANTITIES[0]:
                x = write_cache(g.createVariable('n_products', 'i1', ('year_obs', 'lat', 'lon'), zlib=True, complevel=4), lat, lon)
                x.long_name = 'number of fCO2-products in the mean of the observation-based years'
                x[:] = np.where(keep[None], n, 0).astype('i1')
            put(f'{q}_{EVAL_YEAR}_trend', ('lat', 'lon'), np.nanmean(np.array([rd(r, f'{q}_{EVAL_YEAR}_trend') for r in refs]), 0),
                UNITS[q], f'{nm}: mean of the references\' {EVAL_YEAR} trend values')
            if kind == 'common':
                for ks in SERIES:
                    S = np.array([np.asarray(F[r][f'{q}_gm_common_{ks}'][:], 'f8') for r in refs])
                    SS = np.array([np.asarray(F[r][f'{q}_gm_common_{ks}_sigma'][:], 'f8') for r in refs])
                    _svar(g, f'{q}_gm_{ks}', ('year',), S.mean(0), UNITS[q], f'{nm}, common area: mean of the references ({ks})')
                    _svar(g, f'{q}_gm_{ks}_sigma', ('year',), np.sqrt(np.mean(SS ** 2, 0) + S.std(0, ddof=1) ** 2), UNITS[q],
                          f'{nm}, common area: sqrt(mean sigma^2 + spread^2) ({ks})')
                    for lab in ('', '_from_AR6_1750'):
                        d = np.array([float(F[r][f'{q}_gm_common_{ks}_decline{lab}'][...]) for r in refs])
                        _svar(g, f'{q}_gm_{ks}_decline{lab}', (), d.mean(), '%', f'{nm}: relative change from 1750{lab.replace("_", " ")} to {EVAL_YEAR}, mean of the references ({ks})')
                        _svar(g, f'{q}_gm_{ks}_decline{lab}_spread', (), d.std(ddof=1), '%', f'{nm}: spread of the references ({ks})')
                    for s in scen:
                        S = np.array([np.asarray(F[r][f'{q}_gm_common_{s}_{ks}'][:], 'f8') for r in refs])
                        SS = np.array([np.asarray(F[r][f'{q}_gm_common_{s}_{ks}_sigma'][:], 'f8') for r in refs])
                        _svar(g, f'{q}_gm_{s}_{ks}', ('year_ssp',), S.mean(0), UNITS[q], f'{nm}, {s}, common area: mean of the references ({ks})')
                        _svar(g, f'{q}_gm_{s}_{ks}_sigma', ('year_ssp',), np.sqrt(np.mean(SS ** 2, 0) + S.std(0, ddof=1) ** 2), UNITS[q],
                              f'{nm}, {s}, common area: sqrt(mean sigma^2 + spread^2) ({ks})')
                        _svar(g, f'{q}_gm_{s}_{ks}_decline', ('year_ssp',), 100 * (S.mean(0) / S.mean(0)[0] - 1), '%', f'{nm}, {s}: relative change from its 1750 ({ks})')
                # the fCO2-products themselves: the mean of those present in the year
                Go = np.full((len(refs), years_obs.size), np.nan)
                for k, r in enumerate(refs):
                    Go[k, np.searchsorted(years_obs, yo[r])] = np.asarray(F[r][f'{q}_gm_common_obs'][:], 'f8')
                with np.errstate(invalid='ignore'):
                    no = np.sum(np.isfinite(Go), 0)
                    _svar(g, f'{q}_gm_obs', ('year_obs',), np.nanmean(Go, 0), UNITS[q],
                          f'{nm}, common area: mean of the fCO2-products present in the year (n_products_gm)')
                    _svar(g, f'{q}_gm_obs_spread', ('year_obs',), np.where(no > 1, np.nanstd(Go, 0, ddof=1), np.nan), UNITS[q],
                          f'{nm}, common area: spread of the fCO2-products present in the year')
                if q == QUANTITIES[0]:
                    x = g.createVariable('n_products_gm', 'i1', ('year_obs',))
                    x.long_name = 'number of fCO2-products in the global mean of the observation-based years'
                    x[:] = no.astype('i1')
                nows = np.array([float(F[r][f'{q}_gm_common_{EVAL_YEAR}_trend'][...]) for r in refs])
                _svar(g, f'{q}_gm_{EVAL_YEAR}_trend', (), nows.mean(), UNITS[q], f'{nm}, common area: mean of the references')
        g.close()
        print(f'  written {path}', flush=True)
    for f in F.values():
        f.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', nargs='+', choices=list(cb.REFERENCES))
    ap.add_argument('--combine-only', action='store_true', help='only the combined files, from the existing OA-Track files')
    ap.add_argument('--no-combine', action='store_true')
    ap.add_argument('--no-alk', action='store_true', help='only the files without the alkalinity adjustment')
    ap.add_argument('--out', help='output folder (default data/<run>/oatrack)')
    a = ap.parse_args()
    if a.out:
        global OUT
        from pathlib import Path
        OUT = Path(a.out)
    versions = ('noalk',) if a.no_alk else ('noalk', 'alk')
    if not a.combine_only:
        P = None if a.no_alk else ma.Params()
        for r in a.reference or list(cb.REFERENCES):
            build(r, P, with_alk=not a.no_alk)
    if a.combine_only or not a.no_combine:
        for v in versions:
            combine(v)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
