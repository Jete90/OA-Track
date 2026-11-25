#!/usr/bin/env python
"""
make_product.py  (data product, final step: the product files from the states and the observation-based products)
===============
Per reference (OceanSODA-ETHZ, UExP-FNN-U, JMA-MLR, CMEMS-LSCE-FFNN; Jiang et al. 2023 to its decade 2005-2014),
carbonate_system_<reference>.nc:
    best estimate  1750-2024 (Jiang 1750-2014) per cell: the constrained preindustrial ocean at the CO2 of the
                   year 1750-1849 (calc_product_states.py, pi), the constrained models 1850 to the year before the
                   product's first year (hist), the product from its first year (the first year of the chain's product file: OceanSODA 1983, CMEMS 1985, JMA
                   1990, UExP its first year; Jiang: the decade state 2005-2014); no blend, a hard switch
                   (source flag 1, 2, 4). Constrained = regional state constraint; unconstrained = multi-model mean
                   state; both with 1 sigma (model years: fit, observed predictors, CO2; product years: the level
                   uncertainty of the product is a global attribute, see Extended Data Table 2)
    global series  two constrained versions on the product's own area and on the common area of the paper:
                   _global (the global constraint, the paper's method, the regression of the global means applied
                   in every cell) and _map (the area mean of the regional maps); _unconstrained; product years: the
                   product's area mean in all three; relative changes from 1750 (series) and from the AR6 1750 state
                   to the 2024 trend value (quadratic fit of all product years, as in the paper; Jiang: to its decade)
    named states   AR6 1750 and 1850, midpoint, Jiang's CO2, 280 ppm: maps and global means with sigma
    scenarios      1750-2100 per SSP with the models that have the scenario (their own regression and their own
                   1750): preindustrial, historical and scenario, constrained and unconstrained, with sigma; global
                   series and relative changes from the scenario's own 1750
    internal var.  _sigma_internal (maps and global series, model years): how far a single real year may deviate
                   from the forced (smoothed) value, from every model's raw-minus-smoothed anomaly of that year added
                   to the constrained state (1750-1849: the anomalies of 1850-1899 on the constrained 1750 state); not
                   in _sigma: value +- _sigma is the forced state, +- sqrt(_sigma^2 + _sigma_internal^2) a single year
    level unc.     the uncertainty of the reference itself (level_uncertainty.py), maps and global series, every year,
                   NOT in _sigma: _sigma_mapping_c1 and _c2 (the product's mapping uncertainty, months independent or
                   the same error in all months; product years: that year's errors; model years: the errors of the
                   bias window carried by the bias adjustment, at the constrained state of the year) and
                   _sigma_constants (dissociation constants and total boron, mocsy.merrors)
    diagnostics    the switch (product minus model in the switch year; the step from the last model year),
                   per year the direct method of the paper (per-model indicators regressed) against the state,
                   the R2 of the state regressions, the internal variability removed by the smoothing and the
                   distortion of the multi-model mean by the smoothing (thetao, dissic)
Combined files carbonate_system_common.nc (common area of the four products) and carbonate_system_all.nc (every
cell any product covers), each with two versions of the switch: _switch1990 (models to 1989, the mean of the four
products from 1990, the first year with all four) and _switchfirst (the products as soon as any exists, the mean of those present in the year and cell; n_products says how many).

Needs: states_<ref>_{pi,hist,<ssp>}.nc, obs_<ref>.nc (calc_product_obs.py), constraint_results.pkl, data/smoothed.
Usage:  python make_product.py                 # all references with files, then the combined files
        python make_product.py --reference UExP
        in parallel: python make_product.py --reference UExP --no-combine (one process per reference), then
                     python make_product.py --combine-only
"""
from __future__ import annotations

import argparse
import oa_config   # run: data folder, mocsy constants, GFDL density (OA_RUN)
import datetime
import pickle
import warnings

import netCDF4
import numpy as np

import calc_bias_adjusted as cb
import calc_omega_arag_obs as co
import calc_product_states as ps
import calc_smooth_states as cs
import state_constraint as sc

OUT = oa_config.DATA / 'product'
QUANTITIES = ps.QUANTITIES
UNITS = ps.UNITS
LONG = {'omega_arag': 'aragonite saturation state', 'omega_calc': 'calcite saturation state', 'ph': 'pH (total scale)',
        'hplus': 'hydrogen ion concentration (total scale)', 'hplus_free': 'hydrogen ion concentration (free scale)',
        'pco2': 'partial pressure of CO2', 'fco2': 'fugacity of CO2', 'co2': 'dissolved CO2', 'hco3': 'bicarbonate ion',
        'co3': 'carbonate ion', 'revelle': 'Revelle factor', 'dic': 'dissolved inorganic carbon', 'thetao': 'temperature',
        'so': 'salinity', 'talk': 'total alkalinity'}
PRODUCTS = ['OceanSODA', 'UExP', 'JMA', 'CMEMS']
LABEL = {r: cb.REFERENCES[r]['label'] for r in cb.REFERENCES}
BIT = {'OceanSODA': 1, 'UExP': 2, 'JMA': 4, 'CMEMS': 8}
SCENARIOS = cs.SCENARIOS
JIANG_SWITCH, JIANG_LAST = 2005, 2014
EVAL_YEAR = 2024
SWITCH_COMMON = 1990
SOURCE_FLAGS = '1 preindustrial_ocean_at_the_CO2_of_the_year 2 constrained_ESMs 4 observation_based_product'
FILL = cb.FILL
STATES = ps.PRODUCT_STATES
MAPS = [('', 'c'), ('_unconstrained', 'u'), ('_sigma', 'sigma'), ('_sigma_unconstrained', 'sigma_u'),
        ('_sigma_internal', 's_iv'), ('_sigma_mapping_c1', 's_map1'), ('_sigma_mapping_c2', 's_map2'),
        ('_sigma_constants', 's_const')]
LEVEL = [k for _, k in MAPS[5:]]
LEVEL_SER = {'s_map1': 'sigma_mapping_c1', 's_map2': 'sigma_mapping_c2', 's_const': 'sigma_constants'}
MAP_NAMES = {'': 'best estimate', '_unconstrained': 'multi-model mean (model years) or the product',
             '_sigma': '1 sigma of the constraint (model years: fit, observed predictors and CO2 in quadrature; '
                       'product years: none, see _sigma_mapping_c1, _c2 and _sigma_constants)',
             '_sigma_unconstrained': '1 sigma of the unconstrained value (model spread and CO2)',
             '_sigma_internal': '1 sigma of internal variability (model years): how far a single real year may deviate '
                                'from the forced value; not in _sigma, add in quadrature for a single year; product years: '
                                'not applicable (the products contain the real variability)',
             '_sigma_mapping_c1': '1 sigma, mapping uncertainty of the reference product, errors independent between months '
                                  '(model years: the product errors of the bias window carried by the bias adjustment); not in _sigma',
             '_sigma_mapping_c2': '1 sigma, mapping uncertainty of the reference product, the same error in all months '
                                  '(conservative); not in _sigma',
             '_sigma_constants': '1 sigma, dissociation constants and total boron (fully correlated in space and time); not in _sigma'}
SERIES = ['global', 'map', 'unconstrained']
SERIES_LABEL = {'global': 'global constraint (the regression of the global means applied in every cell; the method of the paper)',
                'map': 'area mean of the regional constrained maps', 'unconstrained': 'multi-model mean state'}
VERSIONS = ('switch1990', 'switchfirst')


def to_180(lon, a):
    l = np.where(lon > 180, lon - 360, lon)
    o = np.argsort(l)
    return l[o], a[..., o]


def quad_fit_year(years, series, year):
    """quadratic fit of (year, series) evaluated at year; series (n, ...) along axis 0 (cells with a missing year NaN)"""
    yy = np.asarray(years, 'f8') - 2000.0
    if series.ndim == 1:
        ok = np.isfinite(series)
        return float(np.polyval(np.polyfit(yy[ok], series[ok], 2), year - 2000.0))
    A = np.vander(yy, 3)
    b = series.reshape(series.shape[0], -1).astype('f8')
    bad = ~np.all(np.isfinite(b), axis=0)
    coef, *_ = np.linalg.lstsq(A, np.where(bad[None], 0.0, b), rcond=None)
    v = np.array([(year - 2000.0) ** 2, year - 2000.0, 1.0]) @ coef
    v[bad] = np.nan
    return v.reshape(series.shape[1:])


class Reference:
    """the files of one reference: states (pi, hist, scenarios), the observation-based product, masks"""

    def __init__(self, ref):
        self.ref = ref
        self.pi = netCDF4.Dataset(OUT / f'states_{ref}_pi.nc')
        self.hist = netCDF4.Dataset(OUT / f'states_{ref}_hist.nc')
        self.O = netCDF4.Dataset(OUT / f'obs_{ref}.nc')
        self.scen, self.scen_pi = {}, {}
        for s in SCENARIOS:
            p = OUT / f'states_{ref}_{s}.nc'
            if p.exists():
                self.scen[s] = netCDF4.Dataset(p)
                q = OUT / f'states_{ref}_pi_{s}.nc'
                if q.exists():
                    self.scen_pi[s] = netCDF4.Dataset(q)
                elif str(self.scen[s].models) == str(self.hist.models):
                    self.scen_pi[s] = self.pi                  # the full model set: the preindustrial file of hist
                else:
                    raise FileNotFoundError(f'{q.name} missing: {s} has another model set than hist, the preindustrial '
                                            f'years of the full set would be wrong for it')
        self.lon0 = np.asarray(self.pi['lon'][:], 'f8')
        self.lat = np.asarray(self.pi['lat'][:], 'f8')
        self.lon, _ = to_180(self.lon0, np.zeros(self.lon0.size))
        if ref == 'Jiang':
            with open(oa_config.DATA / 'constraint_results_jiang.pkl', 'rb') as f:
                J = pickle.load(f)
            masks = (J['own']['Jiang']['mask'], J['common']['Jiang']['mask'])
            self.paper_now = float(J['own']['Jiang']['obs10'])
        else:
            with open(oa_config.DATA / 'constraint_results.pkl', 'rb') as f:
                P = pickle.load(f)
            masks = (P['results'][ref]['mask'], P['results_common'][ref]['mask'])
            self.paper_now = float(P['results'][ref]['now'])
        self.mask_own0, self.mask_com0 = (np.asarray(m).astype(bool) for m in masks)      # on the original grid
        self.mask_own, self.mask_com = self.t(self.mask_own0), self.t(self.mask_com0)
        self.obs_mask = self.t(np.asarray(self.O['mask'][:]).astype(bool))
        self.area_ok = self.mask_own & self.obs_mask              # the product's own area
        self.area = oa_config.area_weights(self.lat, self.lon)
        self.years_obs = np.asarray(self.O['year'][:]).astype(int)
        self.first = JIANG_SWITCH if ref == 'Jiang' else int(self.years_obs[0])     # the product from here
        self.last = JIANG_LAST if ref == 'Jiang' else int(self.years_obs[-1])
        self.years = np.arange(1750, self.last + 1)
        self.years_pi = np.asarray(self.pi['year'][:]).astype(int)
        self.years_hist = np.asarray(self.hist['year'][:]).astype(int)
        self.states = [str(x) for x in self.pi['state'][:]]

    def t(self, a):
        return to_180(self.lon0, a)[1]

    def rd(self, f, name, i=None):
        v = f[name]
        a = np.ma.filled((v[:] if i is None else v[i]).astype('f8'), np.nan)
        return self.t(a) if v.dimensions[-1] == 'lon' else a

    def obs(self, q, y):
        """the product field of year y (Jiang: the decade state)"""
        i = 0 if self.ref == 'Jiang' else int(np.where(self.years_obs == y)[0][0])
        return self.rd(self.O, q, i)

    def obs_series(self, q, mask, lv, y):
        """the level uncertainty of the product's area mean in year y (calc_product_obs.py)"""
        name = f'{q}_gm_{mask}_{lv}'
        if name not in self.O.variables:
            return np.nan
        i = 0 if self.ref == 'Jiang' else int(np.where(self.years_obs == y)[0][0])
        return float(self.O[name][i])

    def model_year(self, f, years, q, y, key):
        """map of q (key c, u, sigma, sigma_u) of year y from a states file"""
        i = int(np.where(years == y)[0][0])
        name = {'c': q, 'u': f'{q}_unconstrained', 'sigma': f'{q}_sigma', 'sigma_u': f'{q}_sigma_unconstrained',
                's_iv': f'{q}_sigma_internal', **{k: f'{q}_{v}' for k, v in LEVEL_SER.items()}}[key]
        return self.rd(f, name, i)

    def series(self, f, q, mask, kind, what='value'):
        """global series of q from a states file: kind global/map/unconstrained/direct; what value/sigma/r2"""
        name = f'{q}_gm_{mask}_{kind}' + ('' if what == 'value' else f'_{what}')
        return np.asarray(f[name][:], 'f8')

    def am(self, field, mask):
        return float(sc.area_mean(field, self.lat, self.lon, mask & self.area_ok))

    def hist_to_years(self, h, years):
        """a historical series (years_hist) placed on `years` for the model years before the switch"""
        a = np.full(years.size, np.nan)
        for i, y in enumerate(years):
            if 1850 <= y < self.first and y <= self.years_hist[-1]:
                a[i] = h[np.searchsorted(self.years_hist, y)]
        return a

    def close(self):
        for f in [self.pi, self.hist, self.O] + list(self.scen.values()) + [g for g in self.scen_pi.values() if g is not self.pi]:
            f.close()


# ================================================================================================ per reference
def build(ref):
    R = Reference(ref)
    path = OUT / f'carbonate_system_{ref}.nc'
    n_years = R.years.size
    print(f'\n=== {ref}: {LABEL[ref]}: years 1750-{R.last}, product from {R.first}, scenarios {list(R.scen)}, '
          f'own area {100 * (R.area * R.area_ok).sum() / R.area.sum():.1f} % of the weighted sphere ({oa_config.WEIGHTS})', flush=True)
    g = cb.new_nc(path, R.lat, R.lon, dict(
        title=f'Surface ocean carbonate system 1750-{R.last} and scenarios to 2100, reference {LABEL[ref]}',
        method=__doc__.split('Combined files')[0].split('\n', 3)[3].strip(),
        reference=LABEL[ref], constants=oa_config.CONSTANTS, area_weights=oa_config.WEIGHTS_TEXT, chain_version=oa_config.VERSION,
        models=R.hist.models, n_models=int(R.hist.n_models), constraint=R.hist.constraint,
        observed_predictors=R.hist.observed_predictors, smoothing=R.hist.smoothing,
        switch_year=R.first, eval_year=EVAL_YEAR, source_flags=SOURCE_FLAGS,
        scenarios=', '.join(f'{s} ({int(R.scen[s].n_models)} models)' for s in R.scen),
        history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} make_product.py, {oa_config.VERSION}'))
    g.createDimension('year', n_years)
    g.createVariable('year', 'i4', ('year',))[:] = R.years
    g.createDimension('state', len(STATES))
    x = g.createVariable('state', str, ('state',)); x[:] = np.array(R.states, dtype=object)
    for nm in ('state_xco2', 'state_xco2_sigma'):
        x = g.createVariable(nm, 'f8', ('state',)); x.units = 'ppm'; x[:] = R.pi[nm][:]
    years_ssp = np.arange(1750, 2101)                      # the scenario series: preindustrial, historical, scenario
    if R.scen:
        g.createDimension('year_ssp', years_ssp.size)
        g.createVariable('year_ssp', 'i4', ('year_ssp',))[:] = years_ssp
    x = g.createVariable('mask', 'i1', ('lat', 'lon')); x.long_name = 'the product area: constraint area and product coverage'
    x[:] = R.area_ok.astype('i1')
    x = g.createVariable('mask_common', 'i1', ('lat', 'lon')); x.long_name = 'the common area of the paper'
    x[:] = (R.mask_com & R.area_ok).astype('i1')
    x = g.createVariable('constrained_flag', 'f4', ('lat', 'lon'), fill_value=FILL)
    x.flag_values, x.flag_meanings = '1 2', 'regional_regression multi_model_mean'
    x[:] = np.ma.masked_invalid(R.rd(R.hist, 'constrained_flag'))
    src = np.where(R.years < 1850, 1, np.where(R.years < R.first, 2, 4)).astype('i1')
    x = g.createVariable('source', 'i1', ('year',)); x.flag_values, x.flag_meanings = '1 2 4', SOURCE_FLAGS; x[:] = src
    xco2 = np.full(n_years, np.nan)
    xco2[R.years < 1850] = R.pi['xco2'][:]
    xh = np.asarray(R.hist['xco2'][:])
    for i, y in enumerate(R.years):
        if 1850 <= y <= R.years_hist[-1]:
            xco2[i] = xh[np.searchsorted(R.years_hist, y)]
    x = g.createVariable('xco2', 'f8', ('year',), fill_value=np.nan); x.units = 'ppm'
    x.long_name = 'atmospheric CO2 of the year (CMIP6 history, Meinshausen et al. 2017; the product years are not forced by it)'
    x[:] = xco2

    def var(name, dims, data, unit, long_name):
        x = g.createVariable(name, 'f4', dims, zlib=True, complevel=4, fill_value=FILL)
        x.units, x.long_name = unit, long_name
        x[:] = np.ma.masked_invalid(np.asarray(data, 'f4'))

    def svar(name, dims, data, unit, long_name):
        x = g.createVariable(name, 'f8', dims, fill_value=np.nan)
        x.units, x.long_name = unit, long_name
        x[:] = data

    def scalar(name, value, unit, long_name):
        x = g.createVariable(name, 'f8')
        x.units, x.long_name = unit, long_name
        x[...] = value

    summary, diag = {}, {}
    for q in QUANTITIES:
        name = LONG[q]
        # ---------------------------------------------------------------- maps 1750-last
        M = {k: np.full((n_years,) + R.area_ok.shape, np.nan, 'f4') for _, k in MAPS}
        for i, y in enumerate(R.years):
            if y >= R.first:
                M['c'][i] = M['u'][i] = R.obs(q, y)
                for k in LEVEL:
                    M[k][i] = R.obs(f'{q}_{LEVEL_SER[k]}', y)
                continue
            f, yrs = (R.pi, R.years_pi) if y < 1850 else (R.hist, R.years_hist)
            for _, k in MAPS:
                M[k][i] = R.model_year(f, yrs, q, y, k)
        for k in M:
            M[k][:, ~R.area_ok] = np.nan
        for suf, k in MAPS:
            var(f'{q}{suf}', ('year', 'lat', 'lon'), M[k], UNITS[q], f'{name}: {MAP_NAMES[suf]}')
        # the trend value of the product (its years up to EVAL_YEAR, the same end for every product) and the
        # switch diagnostics
        iy = np.where((R.years >= R.first) & (R.years <= EVAL_YEAR))[0]
        now = R.obs(q, 2010) if ref == 'Jiang' else quad_fit_year(R.years[iy], M['c'][iy], EVAL_YEAR)
        now = np.where(R.area_ok, now, np.nan)
        var(f'{q}_{EVAL_YEAR}_trend', ('lat', 'lon'), now, UNITS[q],
            f'{name}: the decade 2005-2014 state of Jiang et al.' if ref == 'Jiang'
            else f'{name} in {EVAL_YEAR}: quadratic fit of the product years {R.first}-{min(R.last, EVAL_YEAR)} evaluated in {EVAL_YEAR}')
        i_sw = int(np.searchsorted(R.years, R.first))
        mod_sw = (R.model_year(R.hist, R.years_hist, q, R.first, 'c') if R.first <= R.years_hist[-1]
                  else np.full(R.area_ok.shape, np.nan))
        mismatch = np.where(R.area_ok, M['c'][i_sw] - mod_sw, np.nan)
        step = M['c'][i_sw] - M['c'][i_sw - 1]
        var(f'{q}_switch_mismatch', ('lat', 'lon'), mismatch, UNITS[q], f'{name}: product minus constrained models in the switch year {R.first}')
        diag[q] = dict(mismatch=R.am(mismatch, R.mask_own), mismatch_rms=float(np.sqrt(np.nanmean(mismatch[R.area_ok] ** 2))),
                       step=R.am(step, R.mask_own))
        # ---------------------------------------------------------------- named states
        for suf, k in MAPS:
            nm = {'c': f'{q}_state', 'u': f'{q}_unconstrained_state', 'sigma': f'{q}_sigma_state',
                  'sigma_u': f'{q}_sigma_unconstrained_state', 's_iv': f'{q}_sigma_internal_state',
                  **{kk: f'{q}_{v}_state' for kk, v in LEVEL_SER.items()}}[k]
            A = np.array([R.rd(R.pi, nm, i) for i in range(len(STATES))])
            A[:, ~R.area_ok] = np.nan
            var(f'{q}_state{suf}', ('state', 'lat', 'lon'), A, UNITS[q], f'{name} of the named preindustrial states: {MAP_NAMES[suf]}')
        # ---------------------------------------------------------------- global series
        for mask_key, mask in (('own', R.mask_own), ('common', R.mask_com)):
            sfx = '' if mask_key == 'own' else '_common'
            gl = f'global mean on the {"product area" if mask_key == "own" else "common area of the paper"}'
            obs_gm = np.array([R.am(M['c'][i], mask) if y >= R.first else np.nan for i, y in enumerate(R.years)])
            S = {}
            for kind in SERIES:
                S[kind] = {}
                for w in ('value', 'sigma'):
                    a = np.full(n_years, np.nan)
                    a[R.years < 1850] = R.series(R.pi, q, mask_key, kind, w)
                    h = R.hist_to_years(R.series(R.hist, q, mask_key, kind, w), R.years)
                    a[np.isfinite(h)] = h[np.isfinite(h)]
                    if w == 'value':
                        a[R.years >= R.first] = obs_gm[R.years >= R.first]
                    S[kind][w] = a
                svar(f'{q}_gm{sfx}_{kind}', ('year',), S[kind]['value'], UNITS[q], f'{name}, {gl}: {SERIES_LABEL[kind]}; product years: the product')
                svar(f'{q}_gm{sfx}_{kind}_sigma', ('year',), S[kind]['sigma'], UNITS[q], f'{name}, {gl}: 1 sigma ({kind}); product years: see the level uncertainty attributes')
                svar(f'{q}_gm{sfx}_{kind}_state', ('state',), np.asarray(R.pi[f'{q}_gm_{mask_key}_{kind}_state'][:], 'f8'), UNITS[q], f'{name}, {gl}: named states ({kind})')
                svar(f'{q}_gm{sfx}_{kind}_state_sigma', ('state',), np.asarray(R.pi[f'{q}_gm_{mask_key}_{kind}_sigma_state'][:], 'f8'), UNITS[q], f'{name}, {gl}: named states, 1 sigma ({kind})')
            a = np.full(n_years, np.nan)
            a[R.years < 1850] = R.series(R.pi, q, mask_key, 'internal', 'sigma')
            h = R.hist_to_years(R.series(R.hist, q, mask_key, 'internal', 'sigma'), R.years)
            a[np.isfinite(h)] = h[np.isfinite(h)]
            svar(f'{q}_gm{sfx}_sigma_internal', ('year',), a, UNITS[q],
                 f'{name}, {gl}: 1 sigma of internal variability of the area mean (model years; not in the other sigmas)')
            for k, lv in LEVEL_SER.items():
                a = np.full(n_years, np.nan)
                a[R.years < 1850] = R.series(R.pi, q, mask_key, 'level', lv)
                h = R.hist_to_years(R.series(R.hist, q, mask_key, 'level', lv), R.years)
                a[np.isfinite(h)] = h[np.isfinite(h)]
                io = R.years >= R.first
                a[io] = [R.obs_series(q, mask_key, lv, y) for y in R.years[io]]
                svar(f'{q}_gm{sfx}_{lv}', ('year',), a, UNITS[q], f'{name}, {gl}: {MAP_NAMES["_" + lv]} (area mean)')
                svar(f'{q}_gm{sfx}_{lv}_state', ('state',), np.asarray(R.pi[f'{q}_gm_{mask_key}_level_{lv}_state'][:], 'f8'), UNITS[q],
                     f'{name}, {gl}: named states, {MAP_NAMES["_" + lv]} (area mean)')
            now_g = R.am(now, mask)
            scalar(f'{q}_gm{sfx}_{EVAL_YEAR}_trend', now_g, UNITS[q], f'{name}, {gl}: the {EVAL_YEAR} trend value of the product' if ref != 'Jiang' else f'{name}, {gl}: the decade 2005-2014 of Jiang et al.')
            for kind in SERIES:
                for lab, p_, sp_ in (('', S[kind]['value'][0], S[kind]['sigma'][0]),
                                     ('_from_AR6_1750', float(R.pi[f'{q}_gm_{mask_key}_{kind}_state'][0]), float(R.pi[f'{q}_gm_{mask_key}_{kind}_sigma_state'][0]))):
                    scalar(f'{q}_gm{sfx}_{kind}_decline{lab}', 100 * (now_g / p_ - 1), '%',
                           f'{name}, {gl}: relative change from 1750{lab.replace("_", " ")} to the {EVAL_YEAR} trend value ({kind})')
                    scalar(f'{q}_gm{sfx}_{kind}_decline{lab}_sigma', 100 * now_g / p_ ** 2 * sp_, '%',
                           '1 sigma of the relative change (1750 side; the product side is the level uncertainty of the product)')
            for nm, what in (('direct', 'value'), ('direct_sigma', 'sigma'), ('direct_r2', 'r2')):
                a = np.full(n_years, np.nan)
                a[R.years < 1850] = R.series(R.pi, q, mask_key, 'direct', what)
                h = R.hist_to_years(R.series(R.hist, q, mask_key, 'direct', what), R.years)
                a[np.isfinite(h)] = h[np.isfinite(h)]
                svar(f'{q}_gm{sfx}_{nm}', ('year',), a, UNITS[q] if what != 'r2' else '1',
                     f'{name}, {gl}: the direct method of the paper (per-model area means regressed on dSST and CUC){" (1 sigma)" if what == "sigma" else " (R2)" if what == "r2" else ""}')
            if mask_key == 'own':
                summary[q] = dict(pi=S['global']['value'][0], pi_map=S['map']['value'][0], u=S['unconstrained']['value'][0],
                                  ar6=float(R.pi[f'{q}_gm_own_global_state'][0]), now=now_g,
                                  dec=100 * (now_g / S['global']['value'][0] - 1),
                                  sdec=100 * now_g / S['global']['value'][0] ** 2 * S['global']['sigma'][0],
                                  jump=S['global']['value'][100] - S['global']['value'][99])
        # ---------------------------------------------------------------- scenarios 1750-2100
        for s, F in R.scen.items():
            Fp = R.scen_pi[s]
            ys_pi, ys = np.asarray(Fp['year'][:]).astype(int), np.asarray(F['year'][:]).astype(int)
            Ms = {k: np.full((years_ssp.size,) + R.area_ok.shape, np.nan, 'f4') for _, k in MAPS}
            for i, y in enumerate(years_ssp):
                f, yrs = (Fp, ys_pi) if y < 1850 else (F, ys)
                for _, k in MAPS:
                    Ms[k][i] = R.model_year(f, yrs, q, y, k)
            for k in Ms:
                Ms[k][:, ~R.area_ok] = np.nan
            ms = f'{int(F.n_models)} models that have {s}'
            for suf, k in MAPS:
                var(f'{q}_{s}{suf}', ('year_ssp', 'lat', 'lon'), Ms[k], UNITS[q], f'{name}, {s}: preindustrial, historical and scenario of the {ms}: {MAP_NAMES[suf]}')
            for mask_key in ('own', 'common'):
                sfx = '' if mask_key == 'own' else '_common'
                for kind in SERIES:
                    a = np.concatenate([R.series(Fp, q, mask_key, kind), R.series(F, q, mask_key, kind)])
                    sa = np.concatenate([R.series(Fp, q, mask_key, kind, 'sigma'), R.series(F, q, mask_key, kind, 'sigma')])
                    svar(f'{q}_gm{sfx}_{s}_{kind}', ('year_ssp',), a, UNITS[q], f'{name}, {s} ({ms}), global mean on the {mask_key} mask: {kind}')
                    svar(f'{q}_gm{sfx}_{s}_{kind}_sigma', ('year_ssp',), sa, UNITS[q], f'{name}, {s}, {mask_key} mask: 1 sigma ({kind})')
                    svar(f'{q}_gm{sfx}_{s}_{kind}_decline', ('year_ssp',), 100 * (a / a[0] - 1), '%', f'{name}, {s}: relative change from the 1750 of the same models ({kind})')
                a = np.concatenate([R.series(Fp, q, mask_key, 'internal', 'sigma'), R.series(F, q, mask_key, 'internal', 'sigma')])
                svar(f'{q}_gm{sfx}_{s}_sigma_internal', ('year_ssp',), a, UNITS[q], f'{name}, {s}, {mask_key} mask: 1 sigma of internal variability')
                for lv in LEVEL_SER.values():
                    a = np.concatenate([R.series(Fp, q, mask_key, 'level', lv), R.series(F, q, mask_key, 'level', lv)])
                    svar(f'{q}_gm{sfx}_{s}_{lv}', ('year_ssp',), a, UNITS[q], f'{name}, {s}, {mask_key} mask: {MAP_NAMES["_" + lv]}')
                for what in ('value', 'r2'):
                    a = np.concatenate([R.series(Fp, q, mask_key, 'direct', what), R.series(F, q, mask_key, 'direct', what)])
                    svar(f'{q}_gm{sfx}_{s}_direct' + ('' if what == 'value' else '_r2'), ('year_ssp',), a, UNITS[q] if what == 'value' else '1',
                         f'{name}, {s}, {mask_key} mask: the direct method of the paper' + (' (R2)' if what == 'r2' else ''))
            if q == 'omega_arag':
                a = np.concatenate([R.series(Fp, q, 'own', 'global'), R.series(F, q, 'own', 'global')])
                summary.setdefault('scen', {})[s] = (float(a[-1]), float(100 * (a[-1] / a[0] - 1)))
    # the number of models per scenario year (MPI-ESM1-2-HR and UKESM1-0-LL end in 2099) and the notes of the states files
    for s, F in R.scen.items():
        Fp = R.scen_pi[s]
        ys = np.asarray(F['year'][:]).astype(int)
        nm = np.full(years_ssp.size, int(getattr(Fp, 'n_models', F.n_models)), 'i4')
        if 'n_models' in F.variables:
            nf = np.asarray(F['n_models'][:]).astype(int)
            for i, y in enumerate(years_ssp):
                if y >= 1850:
                    nm[i] = nf[np.searchsorted(ys, y)]
        x = g.createVariable(f'n_models_{s}', 'i4', ('year_ssp',))
        x.long_name = f'number of models in the constraint of {s} in each year'
        x[:] = nm
        for att in F.ncattrs():
            if att.startswith('note_'):
                g.setncattr(f'{s}_{att}', F.getncattr(att))
    # the R2 of the state regressions and the regional direct check, per year
    for key in ('own', 'common'):
        for nm in ('r2_thetao', 'r2_dmc'):
            a = np.full(n_years, np.nan)
            a[R.years < 1850] = R.pi[f'{nm}_{key}'][:]
            h = R.hist_to_years(np.asarray(R.hist[f'{nm}_{key}'][:], 'f8'), R.years)
            a[np.isfinite(h)] = h[np.isfinite(h)]
            svar(f'{nm}_{key}', ('year',), a, '1', f'R2 of the global state regression ({"T on dSST" if "thetao" in nm else "DIC-ALK on dSST and CUC"}), {key} mask')
    for nm in ('rms', 'max'):
        a = np.full(n_years, np.nan)
        a[R.years < 1850] = R.pi[f'omega_arag_regional_direct_{nm}'][:]
        h = R.hist_to_years(np.asarray(R.hist[f'omega_arag_regional_direct_{nm}'][:], 'f8'), R.years)
        a[np.isfinite(h)] = h[np.isfinite(h)]
        svar(f'omega_arag_regional_direct_{nm}', ('year',), a, '1', f'{nm} over the cells of (per-cell direct regression of Omega_arag - regional state)')
    smooth_diag(g, R, svar)
    g.switch_diagnostics = '; '.join(f'{q}: product minus model in {R.first} {d["mismatch"]:+.4g} (rms over cells {d["mismatch_rms"]:.3g}), '
                                     f'step from {R.first - 1} {d["step"]:+.4g}' for q, d in diag.items())
    g.close()
    when = f'{EVAL_YEAR}' if ref != 'Jiang' else '2005-14'
    print(f'  {"indicator":11s} {"1750 glob":>9s} {"1750 map":>9s} {"unconstr.":>9s} {"AR6 1750":>9s} {"jump 1849/50":>12s} {when:>9s} {"decline":>8s} {"+-":>6s}  '
          f'{R.first}: product - model, step from {R.first - 1}')
    for q in QUANTITIES:
        s_, d = summary[q], diag[q]
        print(f'  {q:11s} {s_["pi"]:9.4g} {s_["pi_map"]:9.4g} {s_["u"]:9.4g} {s_["ar6"]:9.4g} {s_["jump"]:+12.4f} {s_["now"]:9.4g} {s_["dec"]:7.2f}% {s_["sdec"]:5.2f}  '
              f'{d["mismatch"]:+.3g}, {d["step"]:+.3g}')
    if 'scen' in summary:
        print('  scenarios (Omega_arag, global constraint): ' + ', '.join(f'{s} 2100 {v:.3f} ({d:+.1f} % from its 1750)' for s, (v, d) in summary['scen'].items()))
    print(f'  check: Omega_arag {when} on the own area {summary["omega_arag"]["now"]:.6f}, paper {R.paper_now:.6f}, '
          f'difference {summary["omega_arag"]["now"] - R.paper_now:+.1e}')
    if ref == 'Jiang':
        coverage_test(path, R)
    print(f'  written {path}', flush=True)
    R.close()
    return ref


def smooth_diag(g, R, svar):
    """the internal variability removed by the smoothing (model mean of the per-model area RMS) and the distortion
    of the multi-model mean by the smoothing (area mean on the own mask of raw annual minus smoothed annual, model
    mean), thetao and dissic, historical years and each scenario (from the pass-1 files on the model grid)"""
    models = [m.split()[0] for m in R.hist.models.split(', ')]
    members = {m.split()[0]: m.split()[1] for m in R.hist.models.split(', ')}
    for v in ('thetao', 'dissic'):
        rem, dis, yrs_of = {}, {}, {}
        for m in models:
            p = cs.out_path(m, members[m])
            if not p.exists():
                continue
            with netCDF4.Dataset(p) as f:
                mlat, mlon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
                jy = co.nearest_index(mlat, R.lat)
                ix = co.nearest_index(mlon, R.lon0, periodic=True)
                for key in ['hist'] + [s for s in R.scen if f'{v}_{s}' in f.variables]:
                    yrs = np.asarray(f['year_hist' if key == 'hist' else 'year_scen'][:]).astype(int)
                    rem.setdefault(key, []).append(np.asarray(f[f'{v}_{key}_resid_rms'][:], 'f8'))
                    raw = np.ma.filled(f[f'{v}_{key}_raw_annual'][:].astype('f8'), np.nan)[:, jy][:, :, ix]
                    sm = np.ma.filled(f[f'{v}_{key}'][:].astype('f8'), np.nan).mean(axis=1)[:, jy][:, :, ix]
                    dis.setdefault(key, []).append(sc.area_mean(raw - sm, R.lat, R.lon0, R.mask_own0))
                    yrs_of[key] = yrs
        for key in rem:
            yrs = yrs_of[key]
            dim = 'year_smooth_hist' if key == 'hist' else 'year_smooth_scen'
            if dim not in g.dimensions:
                g.createDimension(dim, yrs.size)
                g.createVariable(dim, 'i4', (dim,))[:] = yrs
            # mean over the models present in each year (two models end in 2099)
            svar(f'{v}_smoothing_removed_{key}', (dim,), np.nanmean(rem[key], axis=0), cb.UNITS[v],
                 f'{v}: internal variability removed by the smoothing ({key}): model mean of the area RMS of raw minus smoothed '
                 '(models present in the year)')
            svar(f'{v}_smoothing_distortion_{key}', (dim,), np.nanmean(dis[key], axis=0), cb.UNITS[v],
                 f'{v}: distortion of the multi-model mean by the smoothing ({key}): area mean of (raw annual - smoothed annual), '
                 'model mean (models present in the year)')


def coverage_test(path, R):
    """Jiang et al. (full coverage): the relative change AR6 1750 -> 2005-2014 of the constrained and the
    unconstrained ESMs on Jiang's own area and on the common area of the paper"""
    dec = (R.years_hist >= JIANG_SWITCH) & (R.years_hist <= JIANG_LAST)
    out = {}
    for kind in ('global', 'unconstrained'):
        out[kind] = {}
        for key in ('own', 'common'):
            pi_ = float(R.pi[f'omega_arag_gm_{key}_{kind}_state'][0])
            now_ = float(np.mean(np.asarray(R.hist[f'omega_arag_gm_{key}_{kind}'][:])[dec]))
            out[kind][key] = 100 * (now_ / pi_ - 1)
    print('  coverage test, Omega_arag, relative change AR6 1750 -> 2005-2014:')
    for kind, lab in (('global', 'constrained ESMs'), ('unconstrained', 'unconstrained ESMs')):
        d = out[kind]
        print(f'    {lab:20s} own area {d["own"]:7.2f} %, common area {d["common"]:7.2f} %, difference {d["common"] - d["own"]:+.2f} percentage points')
    with netCDF4.Dataset(path, 'a') as g:
        g.coverage_test = (f'Omega_arag, relative change AR6 1750 -> 2005-2014, own area (Jiang et al. and ESMs) / common area of '
                           f'the paper: constrained ESMs {out["global"]["own"]:.2f} / {out["global"]["common"]:.2f} %, unconstrained '
                           f'{out["unconstrained"]["own"]:.2f} / {out["unconstrained"]["common"]:.2f} %; the observation-based version '
                           'is in run_constraint_jiang.py')


# ================================================================================================ combined
def combine(refs):
    """carbonate_system_common.nc and _all.nc from the product files, two switch versions"""
    refs = [r for r in refs if r in PRODUCTS]
    if not refs:
        return
    files = {r: netCDF4.Dataset(OUT / f'carbonate_system_{r}.nc') for r in refs}
    f0 = files[refs[0]]
    lat, lon = np.asarray(f0['lat'][:], 'f8'), np.asarray(f0['lon'][:], 'f8')
    # the years all references have (they end with their last product year): the combined files stop at the
    # earliest last year, so that every year holds the same references
    yrs = {r: np.asarray(files[r]['year'][:]).astype(int) for r in refs}
    years = np.array(sorted(set.intersection(*[set(v.tolist()) for v in yrs.values()])), int)
    if years[0] != 1750 or np.any(np.diff(years) != 1):
        raise ValueError(f'common years not contiguous from 1750: {years[0]}-{years[-1]}')
    yidx = {r: np.searchsorted(yrs[r], years) for r in refs}
    cut = {r: f'{yrs[r][-1]}' for r in refs if yrs[r][-1] > years[-1]}
    print(f'  combined years 1750-{years[-1]}' + (f' (not used: the later years of {", ".join(f"{r} (to {y})" for r, y in cut.items())})' if cut else ''),
          flush=True)

    def ysel(r, var, a):
        """a, read from variable var of reference r, on the common years if its first dimension is year"""
        return a[yidx[r]] if files[r][var].dimensions[:1] == ('year',) else a
    masks = {r: np.asarray(files[r]['mask'][:]).astype(bool) for r in refs}
    common = np.all([np.asarray(files[r]['mask_common'][:]).astype(bool) for r in refs], axis=0)
    flag = np.zeros(lat.shape + lon.shape, 'i1')
    for r in refs:
        flag += (BIT[r] * masks[r]).astype('i1')
    first = {r: int(files[r].switch_year) for r in refs}
    scen = sorted(set.intersection(*[{s for s in SCENARIOS if f'omega_arag_{s}' in files[r].variables} for r in refs]))
    years_ssp = np.asarray(f0['year_ssp'][:]).astype(int) if scen else None
    rd = lambda r, name: ysel(r, name, np.ma.filled(files[r][name][:].astype('f8'), np.nan))
    is_product = {v: {r: (years >= (SWITCH_COMMON if v == 'switch1990' else first[r])) for r in refs} for v in VERSIONS}
    # switch1990: before SWITCH_COMMON every reference enters with its constrained model state, also in the years in
    # which its own file already holds the product (OceanSODA from 1983, CMEMS from 1985): those years are taken
    # from the states files of the reference (as in build())
    Rr = {r: Reference(r) for r in refs}
    patch_years = {r: [i for i, y in enumerate(years) if first[r] <= y < SWITCH_COMMON and y <= Rr[r].years_hist[-1]]
                   for r in refs}
    print('  switch1990, model values in the product years before ' + str(SWITCH_COMMON) + ': '
          + (', '.join(f'{r} {years[ii[0]]}-{years[ii[-1]]}' for r, ii in patch_years.items() if ii) or 'none'), flush=True)

    def to_models(A, key, q):
        """A (reference, year, lat, lon) of the product files, in place: model values in the patch years"""
        for k, r in enumerate(refs):
            R_ = Rr[r]
            for i in patch_years[r]:
                f = R_.model_year(R_.hist, R_.years_hist, q, int(years[i]), key)
                f[~R_.area_ok] = np.nan
                A[k, i] = f
        return A
    for kind, keep in (('common', common), ('all', flag > 0)):
        path = OUT / f'carbonate_system_{kind}.nc'
        g = cb.new_nc(path, lat, lon, dict(
            title=('Surface ocean carbonate system 1750-2100, mean of the four observation-based references on the common '
                   'area of the paper' if kind == 'common' else 'Surface ocean carbonate system 1750-2100, mean of the references covering each cell'),
            references=', '.join(LABEL[r] for r in refs), constants=oa_config.CONSTANTS, area_weights=oa_config.WEIGHTS_TEXT, chain_version=oa_config.VERSION,
            method=__doc__.split('Combined files')[1].split('Needs:')[0].strip(), source_flags=SOURCE_FLAGS, eval_year=EVAL_YEAR,
            switch_versions=f'switch1990: models to {SWITCH_COMMON - 1}, the mean of the four products from {SWITCH_COMMON}; '
                            'switchfirst: the mean of the products present in the year and cell, from the first year of any product, the models elsewhere',
            sigma="model years: sqrt(mean of the references' sigma^2 + spread across the references^2); product years: the spread "
                  "(none where a single product is present); every mean over the references that hold a value in the cell; "
                  "the level uncertainty of the references (_sigma_mapping_c1, _c2, _sigma_constants) separately, the mean of the "
                  "references' sigma",
            history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} make_product.py'))
        g.createDimension('year', years.size)
        g.createVariable('year', 'i4', ('year',))[:] = years
        x = g.createVariable('mask', 'i1', ('lat', 'lon')); x[:] = keep.astype('i1')
        if kind == 'all':
            x = g.createVariable('product_flag', 'i1', ('lat', 'lon'))
            x.long_name, x.flag_masks = 'references covering the cell (sum of bits)', '1 2 4 8'
            x.flag_meanings = ' '.join(LABEL[r].replace(' ', '_') for r in PRODUCTS)
            x[:] = flag
        if scen:
            g.createDimension('year_ssp', years_ssp.size)
            g.createVariable('year_ssp', 'i4', ('year_ssp',))[:] = years_ssp
        x = g.createVariable('xco2', 'f8', ('year',), fill_value=np.nan); x.units = 'ppm'; x[:] = ysel(refs[0], 'xco2', np.asarray(f0['xco2'][:], 'f8'))

        def var(name, dims, data, unit, long_name):
            x = g.createVariable(name, 'f4', dims, zlib=True, complevel=4, fill_value=FILL)
            x.units, x.long_name = unit, long_name
            x[:] = np.ma.masked_invalid(np.asarray(data, 'f4'))

        def svar(name, dims, data, unit, long_name):
            x = g.createVariable(name, 'f8', dims, fill_value=np.nan)
            x.units, x.long_name = unit, long_name
            x[:] = data

        # which references enter the mean in each year and cell, per version: the products where they exist, the
        # models where no product exists yet (switchfirst: cell by cell)
        use = {}
        for v in VERSIONS:
            u = np.zeros((len(refs), years.size) + keep.shape, bool)
            for i, y in enumerate(years):
                prod = [r for r in refs if is_product[v][r][i]]
                covered = np.any([masks[r] for r in prod], axis=0) if prod else np.zeros(keep.shape, bool)
                for k, r in enumerate(refs):
                    if r in prod:
                        u[k, i] = masks[r]
                    else:
                        u[k, i] = masks[r] & ~covered
            use[v] = u
            n = np.zeros((years.size,) + keep.shape, 'i1')
            for i, y in enumerate(years):
                for k, r in enumerate(refs):
                    if is_product[v][r][i]:
                        n[i] += (u[k, i]).astype('i1')
            x = g.createVariable(f'n_products_{v}', 'i1', ('year', 'lat', 'lon'), zlib=True, complevel=4)
            x.long_name = f'number of observation-based products in the mean ({v}); 0: models'
            x[:] = np.where(keep[None], n, 0).astype('i1')
            src = np.array([1 if y < 1850 else (4 if n[i].max() > 0 else 2) for i, y in enumerate(years)], 'i1')
            x = g.createVariable(f'source_{v}', 'i1', ('year',)); x.flag_values, x.flag_meanings = '1 2 4', SOURCE_FLAGS; x[:] = src
        for q in QUANTITIES:
            name = LONG[q]
            C = np.array([rd(r, q) for r in refs])
            U = np.array([rd(r, f'{q}_unconstrained') for r in refs])
            SG = np.array([rd(r, f'{q}_sigma') for r in refs])
            IV = np.array([rd(r, f'{q}_sigma_internal') for r in refs])
            LV = {lv: np.array([rd(r, f'{q}_{lv}') for r in refs]) for lv in LEVEL_SER.values()}
            for v in ('switchfirst', 'switch1990'):            # switch1990 last: its arrays are patched in place
                if v == 'switch1990':
                    to_models(C, 'c', q)
                    to_models(U, 'u', q)
                    to_models(SG, 'sigma', q)
                    to_models(IV, 's_iv', q)
                    for k_, lv_ in LEVEL_SER.items():
                        to_models(LV[lv_], k_, q)
                u = use[v]

                def fmean(A, power=1):
                    """mean of A^power over the references used in the year and cell that hold a finite value (a
                    reference whose field is missing in a cell does not enter as zero), and their number"""
                    ok = u & np.isfinite(A)
                    n = ok.sum(0)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        return np.where(n > 0, np.nansum(np.where(ok, A ** power, 0), 0) / np.maximum(n, 1), np.nan), n
                mean, n_c = fmean(C)
                umean, _ = fmean(U)
                msg, n_s = fmean(SG, 2)                               # product years: no sigma (NaN) by design
                iv2, n_iv = fmean(IV, 2)
                with np.errstate(invalid='ignore', divide='ignore'):
                    okc = u & np.isfinite(C)
                    sp = np.where(n_c > 1, np.sqrt(np.nansum(np.where(okc, (C - mean) ** 2, 0), 0) / np.maximum(n_c - 1, 1)), np.nan)
                    # model years: sqrt(mean sigma^2 + spread^2); product years: the spread (NaN with a single product)
                    sig = np.where((n_s > 0) | (n_c > 1), np.sqrt(np.nan_to_num(msg) + np.nan_to_num(sp) ** 2), np.nan)
                    ivm = np.where((n_iv > 0) & (n_iv == n_c), np.sqrt(iv2), np.nan)     # product years (no sigma_internal): NaN
                sig[~np.isfinite(mean)] = np.nan
                for a in (mean, umean, sp, sig, ivm):
                    a[:, ~keep] = np.nan
                var(f'{q}_{v}', ('year', 'lat', 'lon'), mean, UNITS[q], f'{name}: mean of the references ({v})')
                var(f'{q}_{v}_unconstrained', ('year', 'lat', 'lon'), umean, UNITS[q], f'{name}: multi-model mean state, mean of the references; product years: the products ({v})')
                var(f'{q}_{v}_sigma', ('year', 'lat', 'lon'), sig, UNITS[q], f"{name}: 1 sigma, the references' sigma and their spread ({v})")
                var(f'{q}_{v}_spread', ('year', 'lat', 'lon'), sp, UNITS[q], f'{name}: spread across the references ({v})')
                var(f'{q}_{v}_sigma_internal', ('year', 'lat', 'lon'), ivm, UNITS[q],
                    f'{name}: 1 sigma of internal variability, rms of the references (model years only; not in _sigma) ({v})')
                for lv, A in LV.items():
                    lm, n_l = fmean(A)
                    lm[n_l < n_c] = np.nan                            # a reference without the level sigma: NaN
                    lm[:, ~keep] = np.nan
                    var(f'{q}_{v}_{lv}', ('year', 'lat', 'lon'), lm, UNITS[q],
                        f"{name}: {MAP_NAMES['_' + lv].split(';')[0]}; mean of the references' sigma (errors assumed fully "
                        f"correlated between the references: conservative; not in _sigma) ({v})")
                    svar(f'{q}_gm_{v}_{lv}', ('year',), sc.area_mean(lm, lat, lon, keep), UNITS[q],
                         f'{name}: area mean of the cell sigma {lv} ({v}; an upper bound for the mapping uncertainty, which '
                         f'decorrelates in space; exact for the constants)')
                svar(f'{q}_gm_{v}', ('year',), sc.area_mean(mean, lat, lon, keep), UNITS[q], f'{name}: area mean of the combined map ({v})')
                svar(f'{q}_gm_{v}_spread', ('year',), sc.area_mean(sp, lat, lon, keep), UNITS[q], f'{name}: area mean of the spread across the references ({v})')
            if kind == 'common':
                for ks in SERIES:
                    S = np.array([rd(r, f'{q}_gm_common_{ks}') for r in refs])
                    SS = np.array([rd(r, f'{q}_gm_common_{ks}_sigma') for r in refs])
                    svar(f'{q}_gm_{ks}', ('year',), S.mean(0), UNITS[q], f'{name}, common area: mean over the references of their {ks} series (model years) and of their products (product years)')
                    svar(f'{q}_gm_{ks}_sigma', ('year',), np.sqrt(np.nanmean(SS ** 2, 0) + (S.std(0, ddof=1) ** 2 if len(refs) > 1 else 0)), UNITS[q], f'{name}, common area: sqrt(mean sigma^2 + spread^2) ({ks})')
                    svar(f'{q}_gm_{ks}_spread', ('year',), S.std(0, ddof=1) if len(refs) > 1 else np.full(years.size, np.nan), UNITS[q], f'{name}, common area: spread across the references ({ks})')
                    for lab in ('', '_from_AR6_1750'):
                        d = np.array([float(files[r][f'{q}_gm_common_{ks}_decline{lab}'][...]) for r in refs])
                        x = g.createVariable(f'{q}_gm_{ks}_decline{lab}', 'f8'); x.units = '%'
                        x.long_name = f'{name}: relative change from 1750{lab.replace("_", " ")} to {EVAL_YEAR}, common area, mean of the references ({ks})'
                        x[...] = d.mean()
                        x = g.createVariable(f'{q}_gm_{ks}_decline{lab}_spread', 'f8'); x.units = '%'
                        x[...] = float(d.std(ddof=1)) if len(refs) > 1 else np.nan
                for lv in LEVEL_SER.values():
                    S = np.array([rd(r, f'{q}_gm_common_{lv}') for r in refs])
                    svar(f'{q}_gm_{lv}', ('year',), S.mean(0), UNITS[q],
                         f"{name}, common area: {MAP_NAMES['_' + lv].split(';')[0]}, mean of the references' area-mean sigma (conservative)")
                nows = np.array([float(files[r][f'{q}_gm_common_{EVAL_YEAR}_trend'][...]) for r in refs])
                x = g.createVariable(f'{q}_gm_{EVAL_YEAR}_trend', 'f8'); x.units = UNITS[q]; x[...] = nows.mean()
            for s_ in scen:
                for suf in ('', '_unconstrained'):
                    A = np.array([rd(r, f'{q}_{s_}{suf}') for r in refs])
                    with np.errstate(invalid='ignore'):
                        m = np.nanmean(A, 0)
                        spd = np.where(np.sum(np.isfinite(A), 0) > 1, np.nanstd(A, 0, ddof=1), np.nan)
                    m[:, ~keep] = np.nan
                    var(f'{q}_{s_}{suf}', ('year_ssp', 'lat', 'lon'), m, UNITS[q], f'{name}, {s_}: mean of the references{suf.replace("_", " ")}')
                    if not suf:
                        var(f'{q}_{s_}_spread', ('year_ssp', 'lat', 'lon'), np.where(np.isfinite(m), spd, np.nan), UNITS[q], f'{name}, {s_}: spread across the references')
                for ks in SERIES:
                    S = np.array([np.asarray(files[r][f'{q}_gm_common_{s_}_{ks}'][:], 'f8') for r in refs])
                    svar(f'{q}_gm_{s_}_{ks}', ('year_ssp',), S.mean(0), UNITS[q], f'{name}, {s_}, common area: mean of the references ({ks})')
                    svar(f'{q}_gm_{s_}_{ks}_decline', ('year_ssp',), 100 * (S.mean(0) / S.mean(0)[0] - 1), '%', f'{name}, {s_}: relative change from the 1750 of the same models ({ks})')
        g.close()
        with netCDF4.Dataset(path) as om:
            gm = om['omega_arag_gm_switch1990'][:]
            line = f'  {kind}: Omega_arag 1750 {gm[0]:.3f} (area mean of the combined map), {years[-1]} {gm[-1]:.3f} (switch1990)'
            if kind == 'common':
                line += (f'; global constraint 1750 {om["omega_arag_gm_global"][:][0]:.3f}, decline {float(om["omega_arag_gm_global_decline"][...]):.2f} % '
                         f'(map mean {float(om["omega_arag_gm_map_decline"][...]):.2f} %, unconstrained {float(om["omega_arag_gm_unconstrained_decline"][...]):.2f} %)')
        print(line + f'\n  written {path}', flush=True)
    for f in files.values():
        f.close()
    for R_ in Rr.values():
        R_.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', nargs='+', choices=list(cb.REFERENCES))
    ap.add_argument('--no-combine', action='store_true',
                    help='only the files of the references (to run the references in parallel, one process each)')
    ap.add_argument('--combine-only', action='store_true',
                    help='only the combined files, from the existing carbonate_system_<product>.nc')
    a = ap.parse_args()
    if a.combine_only:
        done = [r for r in PRODUCTS if (OUT / f'carbonate_system_{r}.nc').exists()]
        if len(done) < len(PRODUCTS):
            raise SystemExit(f'carbonate_system files missing for {[r for r in PRODUCTS if r not in done]}')
        combine(done)
        return
    refs = a.reference or [r for r in cb.REFERENCES if all((OUT / f'{p}_{r}{s}.nc').exists()
                                                           for p, s in (('states', '_pi'), ('states', '_hist'), ('obs', '')))]
    missing = [r for r in (a.reference or []) if r not in refs]
    if missing:
        raise SystemExit(f'states or obs files missing for {missing}: run calc_product_obs.py and calc_product_states.py first')
    for r in refs:
        build(r)
    if a.no_combine:
        return
    combine([r for r in PRODUCTS if (OUT / f'carbonate_system_{r}.nc').exists()])


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
