#!/usr/bin/env python
"""
calc_product_states.py  (data product, pass 2: per run and reference)
======================
The constrained and the unconstrained carbonate system of every year, from the models' states, with the
constraint applied to the STATE (state_constraint.py): T regressed on dSST, DIC-ALK on dSST and CUC, S and
ALK the multi-model mean, globally (the global regression applied in every cell: the paper's constraint) and
regionally (the regression per cell with the local predictors); every indicator from these states with mocsy,
monthly, then annual means. Sigma from the residual states, the observed predictors and the CO2 of the year.
Every temperature that enters the chemistry is floored at the freezing point of seawater of the cell's salinity
(calc_omega_arag.floor_temperature; the regressions and their perturbations can take an ice-covered cell below it),
and a carbonate system mocsy did not solve is NaN (calc_omega_arag.valid_carbonate); a residual state with NaN drops
out of the fit sigma (state_constraint.Constraint.sigma), a model with NaN out of the multi-model mean and spread.

Parts (--part; all by default, in parallel with --jobs):
    pi        the preindustrial ocean: the constrained monthly climatology of the adjusted piControl
              (calc_pi_xco2.py <var>_clim), DIC lowered on it to the CO2 of every year 1750-1849 (CMIP6
              history) and to the named states (1750 and 1850 of AR6, midpoint, Jiang's CO2, 280 ppm); the CO2
              term from the state at xCO2 -/+ sigma (AR6 ranges for the states, co2_history.sigma for the years)
    hist      1850-2014, all models: the smoothed drift-removed states of calc_smooth_states.py plus the
              window mean of the drift and the bias of the reference (the adjusted state of the chain)
    <ssp>     1850-2100 with the models that have the scenario (their own regression): 1850-1969 from the
              smoothed historical series, 1970-2100 from the series continued with the scenario; the
              preindustrial part for the same model subset (file states_<ref>_pi_<ssp>.nc) where the subset
              differs from the full set
Every part also carries the direct method for comparison: the per-model indicators (the models' own states)
regressed on dSST and CUC, globally (value, sigma, R2) and, for Omega_arag, per cell (rms and max difference
to the regional state), and the R2 of the state regressions (T on dSST, DIC-ALK on dSST and CUC) per year.
The pi part checks the per-model Omega_arag of the climatology against the all-months means of calc_pi_xco2.py
and the direct regression against the paper's value (constraint_results.pkl).

Output: data/<run>/product/states_<ref>_<part>.nc (float32 maps, float64 series), for every indicator of
calc_pi_xco2.INDICATOR_NAMES and the state variables thetao, so, talk (dic is an indicator):
    <q>(year, lat, lon)                  regional constrained;  <q>_unconstrained  the multi-model mean state
    <q>_sigma                            1 sigma of the constrained value: fit, observed dSST and CUC, CO2 in
                                         quadrature; the components <q>_sigma_fit, _sigma_dsst, _sigma_cuc,
                                         _sigma_co2;  <q>_sigma_unconstrained = sqrt(spread^2 + co2^2),
                                         <q>_spread the model spread
    <q>_sigma_mapping_c1, _c2, _sigma_constants   level uncertainty of the reference (level_uncertainty.py): the
                                         product's mapping error of the bias window carried into the state, and
                                         the dissociation constants; NOT in _sigma (the pi part: at the 1750 state)
    <q>_gm_<own|common>_global(year)     global constraint (global regression) on the mask, with _sigma and the
                                         components;  <q>_gm_<mask>_map  area mean of the regional field, with
                                         _sigma ...;  <q>_gm_<mask>_unconstrained with _sigma, _spread;
                                         <q>_gm_<mask>_direct (value), _direct_sigma, _direct_r2
    r2_thetao_<mask>, r2_dmc_<mask>(year)   R2 of the global state regressions
    omega_arag_regional_direct_rms, _max(year)   per-cell direct regression against the regional state
    constrained_flag(lat, lon)           1 regional regression, 2 multi-model mean (no local predictors)
    pi part: dimension state for the named states (the same variables with _state), thetao_clim ... talk_clim
    (12, lat, lon) of the constrained (regional) monthly climatology at 284.317 and at 278.3 ppm, xco2(year)

Needs: constraint_results.pkl (_jiang.pkl), the xco2 files (pi part), data/smoothed (hist, ssp), the bias
files, the drift files, inputs/co2/.
Scenarios with the same model set share the years before the switch (1850-1969): the first scenario of a set
computes them, the others copy them (and the preindustrial file pi_<ssp>) from it once it is complete; a scenario
with the full model set copies them from the historical file. Every part runs in its own process, so that a
process killed by the system (out of memory) is reported instead of blocking the run, and memory is returned when
a part finishes. Threads of BLAS/OpenMP are limited to 1 per process (the parallelism is over the parts).
Usage:  python calc_product_states.py --reference OceanSODA --jobs 3
        python calc_product_states.py --reference UExP --part ssp126 ssp245 ssp370 ssp585 --jobs 3
        python calc_product_states.py --reference Jiang --part pi hist
"""
from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import oa_config   # run: data folder, mocsy constants, GFDL density (OA_RUN)
import datetime
import glob
import pickle
import shutil
import time
import traceback
import warnings
import multiprocessing as mp
from pathlib import Path

import netCDF4
import numpy as np

import calc_bias_adjusted as cb
import calc_omega_arag_obs as co
import calc_pi_xco2 as px
import calc_smooth_states as cs
import co2_history
import state_constraint as sc
import level_uncertainty as lu
from drift_tools import Drift
from pi_states import TARGETS, XCO2_SIGMA, PRODUCT_STATES

OUT = oa_config.DATA / 'product'
VARS = cb.VARS
INDICATORS = list(px.INDICATOR_NAMES)                 # omega_arag ... hplus_free, dic
STATE_OUT = ('thetao', 'so', 'talk')                   # dic is an indicator
QUANTITIES = INDICATORS + list(STATE_OUT)
UNITS = {**px.UNITS_ALL, **cb.UNITS}
YEARS_PI = np.arange(1750, 1850)
YEARS_HIST = np.arange(1850, 2015)
YEARS_SSP = np.arange(1850, 2101)
SCENARIOS = cs.SCENARIOS
FILL = cb.FILL
PI_CHUNK = 4                 # preindustrial targets per chunk (memory: about 0.5 GB per target on the 1 degree grid)
STATE_LABEL = {'1750': '1750_AR6', '1850': '1850_AR6', 'mid': 'mid_1750_1850_AR6', 'jiang': 'CO2_of_Jiang_et_al_2023',
               'x280': '280ppm'}


# ================================================================================================ setup
def model_set(ref):
    """the models of a reference (constraint results) and their members (xco2 files of the chain)"""
    cb.set_reference(ref)
    if ref == 'Jiang':
        with open(oa_config.DATA / 'constraint_results_jiang.pkl', 'rb') as f:
            R = pickle.load(f)['own']['Jiang']
    else:
        with open(oa_config.DATA / 'constraint_results.pkl', 'rb') as f:
            R = pickle.load(f)['results'][ref]
    names = list(R['names'])
    members = {}
    for m in names:
        fx = glob.glob(str(cb.OUT_DIR / 'piControl' / f'omega_arag_biasadj_xco2_Oyr_{m}_piControl_*.nc'))
        if not fx:
            raise FileNotFoundError(f'{ref}: no xco2 file for {m}')
        members[m] = fx[0].split('_piControl_')[1].split('_')[0]
    return names, members


def scenario_models(names, members, scen):
    """the models of the set that provide the scenario (their smoothed file has it)"""
    have = []
    for m in names:
        p = cs.out_path(m, members[m])
        if p.exists():
            with netCDF4.Dataset(p) as f:
                if f'thetao_{scen}' in f.variables:
                    have.append(m)
    return have


def complete(path, names, years):
    """True if the file exists, holds the given models and has a finite global value in every one of the given
    years (for a copy source: the years that are copied)"""
    if not path.exists():
        return False
    try:
        with netCDF4.Dataset(path) as f:
            models = [x.split()[0] for x in str(f.models).split(', ')]
            ys = np.asarray(f['year'][:]).astype(int)
            last = np.ma.filled(f['omega_arag_gm_own_global'][:], np.nan)
            ok = models == list(names) and set(years) <= set(ys.tolist())
            return bool(ok and np.isfinite(last[np.searchsorted(ys, years)]).all())
    except Exception:
        return False


class Reference:
    """everything of one reference that the parts share: grid, masks, predictors, models, nutrients, CO2"""

    def __init__(self, ref):
        cb.set_reference(ref)
        self.ref = ref
        self.label = cb.REF['label']
        if ref == 'Jiang':
            with open(oa_config.DATA / 'constraint_results_jiang.pkl', 'rb') as f:
                J = pickle.load(f)
            R, Rc = J['own']['Jiang'], J['common']['Jiang']
        else:
            with open(oa_config.DATA / 'constraint_results.pkl', 'rb') as f:
                P = pickle.load(f)
            R, Rc = P['results'][ref], P['results_common'][ref]
        self.R, self.Rc = R, Rc
        self.names, self.members = model_set(ref)
        self.lat, self.lon = np.asarray(R['lat'], 'f8'), np.asarray(R['lon'], 'f8')
        self.lat2 = np.broadcast_to(self.lat[:, None], (self.lat.size, self.lon.size))
        self.lon2 = np.broadcast_to(self.lon[None, :], (self.lat.size, self.lon.size))
        self.mask_own = np.asarray(R['mask']).astype(bool)
        self.mask_com = np.asarray(Rc['mask']).astype(bool)
        self.si, self.po4 = co.load_nutrients(self.lat, self.lon)
        F = R['fields']
        dobs = np.array([np.asarray(F['dsst_obs'][k], 'f8') for k in F['dsst_obs']])
        self.pred = dict(dT=np.asarray(R['dT'], 'f8'), CUC=np.asarray(R['CUC'], 'f8'),
                         dTo=float(R['dT_obs'][0]), dTs=float(R['dT_obs'][1]), CUo=float(R['CUC_obs']),
                         DTf=np.asarray(F['dT'], 'f8'), CUf=np.asarray(F['CUC'], 'f8'),
                         dTo_f=dobs.mean(0), dTs_f=dobs.std(0, ddof=1) if dobs.shape[0] > 1 else np.zeros(dobs.shape[1:]),
                         CUo_f=np.asarray(F['CUC_obs'], 'f8'))
        # mapping uncertainty of the observed CUC (cuc_mapping_uncertainty.py), own area: global value and field
        pc = oa_config.DATA / 'product_uncertainty' / f'cuc_mapping_{ref}.npz'
        if not pc.exists():
            raise FileNotFoundError(f'{pc} missing: run cuc_mapping_uncertainty.py first')
        zc = np.load(pc)
        fc = np.asarray(zc['own_field_sigma_cuc'], 'f8')
        wc = oa_config.area_weights(self.lat, self.lon) * np.isfinite(fc)
        self.pred['CUs'] = float(zc['own_sigma_cuc_C2'])
        self.pred['CUs_f'] = np.where(np.isfinite(fc), fc, float(np.nansum(np.nan_to_num(fc) * wc) / wc.sum()))
        self.co2 = co2_history.CO2History()
        self.co2.require('historical')
        self.paper = {k: v for k, v in R['states'].items()} if 'states' in R else {}
        # level uncertainty of the reference (mapping: the window-mean state errors of calc_product_obs.py; constants)
        self.level = lu.Level(ref, self.lat, self.lon, self.si, self.po4, {'own': self.mask_own, 'common': self.mask_com})
        dx = lu.read_dx(OUT / f'obs_{ref}.nc', self.lat, self.lon) if (OUT / f'obs_{ref}.nc').exists() else None
        if dx is None:
            print(f'  {ref}: no mapping errors in obs_{ref}.nc (run calc_product_obs.py first): mapping uncertainty NaN', flush=True)
            self.level.names, self.dX = [], {}
        else:
            self.level.names = [c for c in dx[0] if c in self.level.length]
            self.dX = dx[1]

    def level_sigma(self, fields12):
        """level uncertainty (mapping C1, C2, constants) of the constrained state {var: (12, ny, nx)}"""
        return self.level.model_year(fields12, self.dX)

    def constraint(self, names):
        idx = [self.names.index(m) for m in names]
        p = self.pred
        return sc.Constraint(names, p['dT'][idx], p['CUC'][idx], p['dTo'], p['dTs'], p['CUo'], p['DTf'][idx], p['CUf'][idx],
                             p['dTo_f'], p['dTs_f'], p['CUo_f'], self.mask_own, CUs=p['CUs'], CUs_f=p['CUs_f'])

    def area(self, field, key):
        return sc.area_mean(field, self.lat, self.lon, self.mask_own if key == 'own' else self.mask_com)

    # ---------------------------------------------------------------------------------------- model states
    def clim(self, model):
        """the adjusted piControl monthly climatology (12, ny, nx) per variable, and the named-state means"""
        with netCDF4.Dataset(px.xco2_path(model, self.members[model])) as f:
            X = {v: np.ma.filled(f[f'{v}_clim'][:].astype('f8'), np.nan) for v in VARS}
            named = {k: np.ma.filled(f[f'omega_arag_{k}_mean'][:].astype('f8'), np.nan) for k in PRODUCT_STATES}
        return X, named


class Smoothed:
    """the smoothed states of one model on the reference grid, adjusted: + window mean of the drift + bias"""

    def __init__(self, R: Reference, model):
        member = R.members[model]
        self.f = netCDF4.Dataset(cs.out_path(model, member))
        self.step = int(self.f.step)
        self.years_hist = np.asarray(self.f['year_hist'][:]).astype(int)
        self.years_scen = np.asarray(self.f['year_scen'][:]).astype(int)
        self.scenarios = [s for s in SCENARIOS if f'thetao_{s}' in self.f.variables]
        mlat, mlon = np.asarray(self.f['lat'][:], 'f8'), np.asarray(self.f['lon'][:], 'f8')
        self.jy = co.nearest_index(mlat, R.lat)
        self.ix = co.nearest_index(mlon, R.lon, periodic=True)
        with netCDF4.Dataset(cb.bias_path(model, member)) as b:
            self.bias = {v: np.ma.filled(b[f'bias_{v}'][:].astype('f8'), np.nan) for v in VARS}    # (step, ny, nx)
            y0, y1 = (int(x) for x in b.overlap.split('-'))
        drift = Drift(model, member)
        self.offset = {v: drift.window_mean(v, (y0, y1))[np.ix_(self.jy, self.ix)] for v in VARS}   # mean of s over W'
        if self.bias['thetao'].shape[0] != self.step:
            raise ValueError(f'{model}: bias has {self.bias["thetao"].shape[0]} slots, smoothed file {self.step}')

    def state(self, year, series, anomalies=False):
        """adjusted state of one year as {var: (12, ny, nx)} (annual models repeated); series 'hist' or <ssp>;
        with anomalies=True also the internal variability of the year (raw minus smoothed, same shape)"""
        if series == 'hist':
            i = np.searchsorted(self.years_hist, year)
            if i >= self.years_hist.size or self.years_hist[i] != year:
                raise KeyError(f'year {year} not in the historical series')
        else:
            i = np.searchsorted(self.years_scen, year)
            if i >= self.years_scen.size or self.years_scen[i] != year:
                raise KeyError(f'year {year} not in the {series} series')
        out, anom = {}, {}
        for v in VARS:
            a = np.ma.filled(self.f[f'{v}_{series}'][i].astype('f8'), np.nan)          # (step, ny_m, nx_m)
            a = a[:, self.jy][:, :, self.ix] + self.offset[v][None] + self.bias[v]
            out[v] = np.repeat(a, 12, axis=0) if self.step == 1 else a
            if anomalies:
                b = np.ma.filled(self.f[f'{v}_{series}_anom'][i].astype('f8'), np.nan)[:, self.jy][:, :, self.ix]
                anom[v] = np.repeat(b, 12, axis=0) if self.step == 1 else b
        return (out, anom) if anomalies else out

    def close(self):
        self.f.close()


# ================================================================================================ engine
class Engine:
    """chemistry on the stacked states of a model set, and the annual bookkeeping"""

    def __init__(self, R: Reference, con: sc.Constraint):
        self.R, self.con = R, con
        self.n_states = len(con.stack)
        self.small = [con.index['loc'], con.index['glob'], con.index['mean']]   # for the CO2 -/+ states

    def chemistry(self, fields, targets, iters=None):
        """fields {var: (k, ny, nx)} -> {target: {quantity: (k, ny, nx)}} for all QUANTITIES"""
        resid = [0.0]
        out = {}
        for tag, res in px.target_states(fields, self.R.si, self.R.po4, self.R.lat2, targets, resid, iters):
            res['thetao'], res['so'], res['talk'] = fields['thetao'], fields['so'], fields['talk']
            out[tag] = res
        return out

    def month(self, X, x_main, x_sig, iters=None, A=None):
        """one month: X {var: (n, ny, nx)} adjusted states of the models; x_main: {tag: xCO2} targets for the full
        stack; x_sig: sigma of the CO2 for the CO2 term of each tag (0: none). Returns {tag: {q: (n_states, ...)}}
        and {tag: {q: (3, 2, ...)}} (loc, glob, mean at -/+ sigma)"""
        S = self.con.states(X, A)
        main = self.chemistry(S, x_main, iters)
        pm = {}
        tags = [t for t in x_main if x_sig.get(t, 0.0) > 0]
        if tags:
            small = {v: S[v][self.small] for v in VARS}
            tg = {}
            for t in tags:
                tg[f'{t}-'] = x_main[t] - x_sig[t]
                tg[f'{t}+'] = x_main[t] + x_sig[t]
            res = self.chemistry(small, tg)
            for t in tags:
                pm[t] = {q: np.stack([res[f'{t}-'][q], res[f'{t}+'][q]], axis=1) for q in QUANTITIES}   # (3, 2, ny, nx)
        return main, pm


class Accumulator:
    """annual means of the stacked states (NaN if a month is missing), global means per state, then sigma"""

    def __init__(self, engine: Engine, with_pm=True):
        self.e = engine
        self.sum = {q: None for q in QUANTITIES}
        self.pm = {q: None for q in QUANTITIES} if with_pm else None
        self.count = 0

    def add(self, main, pm=None):
        for q in QUANTITIES:
            self.sum[q] = main[q].astype('f8') if self.sum[q] is None else self.sum[q] + main[q]
            if self.pm is not None and pm is not None:
                self.pm[q] = pm[q].astype('f8') if self.pm[q] is None else self.pm[q] + pm[q]
        self.count += 1

    def result(self):
        """{q: {'maps': {...}, 'own': {...}, 'common': {...}}} for the year"""
        R, con = self.e.R, self.e.con
        out = {}
        for q in QUANTITIES:
            A = self.sum[q] / self.count                                    # (n_states, ny, nx)
            PM = None if self.pm is None or self.pm[q] is None else self.pm[q] / self.count   # (3, 2, ny, nx)
            res = {}
            # maps: regional constrained, unconstrained
            c, s_fit, s_dT, s_CU, spread, u = con.sigma(A, 'loc')
            s_co2 = 0.5 * np.abs(PM[0, 1] - PM[0, 0]) if PM is not None else np.zeros_like(c)
            s_co2_u = 0.5 * np.abs(PM[2, 1] - PM[2, 0]) if PM is not None else np.zeros_like(c)
            has_iv = A.shape[0] == len(con.stack_iv)
            s_iv = con.internal(A) if has_iv else np.full(c.shape, np.nan)
            res['maps'] = dict(c=c, u=u, s_fit=s_fit, s_dT=s_dT, s_CU=s_CU, s_co2=s_co2, spread=spread, s_co2_u=s_co2_u,
                               sigma=np.sqrt(s_fit ** 2 + s_dT ** 2 + s_CU ** 2 + s_co2 ** 2),
                               sigma_u=np.hypot(spread, s_co2_u), s_iv=s_iv)
            # global series on both masks
            for key in ('own', 'common'):
                am = np.array([R.area(A[i], key) for i in range(A.shape[0])])          # (n_states,)
                g = {}
                for which in ('glob', 'loc'):
                    cg, sf, sd, scu, sp, ug = con.sigma(am, which)
                    i3 = 1 if which == 'glob' else 0
                    sco2 = 0.5 * abs(R.area(PM[i3, 1], key) - R.area(PM[i3, 0], key)) if PM is not None else 0.0
                    name = 'global' if which == 'glob' else 'map'
                    g[name] = dict(value=float(cg), s_fit=float(sf), s_dT=float(sd), s_CU=float(scu), s_co2=float(sco2),
                                   sigma=float(np.sqrt(sf ** 2 + sd ** 2 + scu ** 2 + sco2 ** 2)))
                mods = con.models(am)
                sco2u = 0.5 * abs(R.area(PM[2, 1], key) - R.area(PM[2, 0], key)) if PM is not None else 0.0
                g['unconstrained'] = dict(value=float(mods.mean()), spread=float(mods.std(ddof=1)), s_co2=float(sco2u),
                                          sigma=float(np.hypot(mods.std(ddof=1), sco2u)))
                dv, ds, dr = con.direct(mods)
                g['direct'] = dict(value=dv, sigma=ds, r2=dr)
                g['internal'] = dict(sigma=float(con.internal(am)) if has_iv else np.nan)
                res[key] = g
            if q == 'omega_arag':
                dcell = con.direct_cells(con.models(A))
                d = (dcell - c)[con.ok]
                res['regional_direct'] = (float(np.sqrt(np.nanmean(d ** 2))), float(np.nanmax(np.abs(d))))
            out[q] = res
        # R2 of the state regressions, from the area means of the model states
        for key in ('own', 'common'):
            T = np.array([R.area(self.sum['thetao'][con.index[f'mod{m}']] / self.count, key) for m in range(con.n)])
            D = np.array([R.area((self.sum['dic'][con.index[f'mod{m}']] - self.sum['talk'][con.index[f'mod{m}']]) / self.count, key)
                          for m in range(con.n)])
            out[f'r2_{key}'] = (con.gT.r2(T), con.gC.r2(D))
        return out


# ================================================================================================ output
class Writer:
    """one output file of a part: maps and series per year (and per named state for the pi part)"""

    def __init__(self, path, R: Reference, con: sc.Constraint, years, part, states=None, attrs=None, append=False):
        self.R, self.con = R, con
        if append:
            # an existing file of this layout, to recompute selected years (--years)
            self.g = netCDF4.Dataset(path, 'a')
            self.years = np.asarray(self.g['year'][:]).astype(int)
            self.states = []
            self._set_keys()
            self.vars = self.g.variables
            return
        self.g = cb.new_nc(path, R.lat, R.lon, dict(
            title=f'Constrained and unconstrained carbonate system, reference {R.label}, part {part}',
            method=__doc__.split('Output:')[0].split('\n', 3)[3].strip(),
            models=', '.join(f'{m} {R.members[m]}' for m in con.names), n_models=con.n,
            constraint='state constraint: T on dSST, DIC-ALK on dSST and CUC, S and ALK the multi-model mean; '
                       'regional (local predictors per cell) and global (global predictors)',
            observed_predictors=f'dSST {con.dTo:.4f} +- {con.dTs:.4f} K, CUC {con.CUo:.5f} +- {con.CUs:.5f} (mapping, '
                                'cuc_mapping_uncertainty.py; per cell the local field)',
            constants=oa_config.CONSTANTS, area_weights=oa_config.WEIGHTS_TEXT, chain_version=oa_config.VERSION, **(attrs or {}),
            history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} calc_product_states.py'))
        g = self.g
        self.years = np.asarray(years)
        g.createDimension('year', self.years.size)
        g.createVariable('year', 'i4', ('year',))[:] = self.years
        self.states = list(states) if states else []
        if self.states:
            g.createDimension('state', len(self.states))
            x = g.createVariable('state', str, ('state',)); x[:] = np.array([STATE_LABEL[k] for k in self.states], dtype=object)
            x = g.createVariable('state_xco2', 'f8', ('state',)); x.units = 'ppm'; x[:] = [TARGETS[k] for k in self.states]
            x = g.createVariable('state_xco2_sigma', 'f8', ('state',)); x.units = 'ppm'; x[:] = [XCO2_SIGMA[k] for k in self.states]
        x = g.createVariable('constrained_flag', 'f4', ('lat', 'lon'), fill_value=FILL)
        x.flag_values, x.flag_meanings = '1 2', 'regional_regression multi_model_mean'
        x[:] = np.ma.masked_invalid(con.flag)
        self.vars = {}
        self._set_keys()
        for dim in ['year'] + (['state'] if self.states else []):
            sfx = '' if dim == 'year' else '_state'
            for q in QUANTITIES:
                for suf, _, ln in self.map_keys:
                    x = g.createVariable(f'{q}{suf}{sfx}', 'f4', (dim, 'lat', 'lon'), zlib=True, complevel=4, fill_value=FILL)
                    x.units, x.long_name = UNITS[q], f'{q}: {ln}'
                    self.vars[f'{q}{suf}{sfx}'] = x
                for key in ('own', 'common'):
                    for suf, _, ln in self.ser_keys:
                        x = g.createVariable(f'{q}_gm_{key}_{suf}{sfx}', 'f8', (dim,), fill_value=np.nan)
                        x.units = UNITS[q] if 'r2' not in suf else '1'
                        if ln:
                            x.long_name = f'{q}, area mean on the {key} mask: {ln}'
                        self.vars[f'{q}_gm_{key}_{suf}{sfx}'] = x
            for key in ('own', 'common'):
                for nm in ('r2_thetao', 'r2_dmc'):
                    x = g.createVariable(f'{nm}_{key}{sfx}', 'f8', (dim,), fill_value=np.nan)
                    x.long_name = f'R2 of the global regression of {"T on dSST" if "thetao" in nm else "DIC-ALK on dSST and CUC"} (area means on the {key} mask)'
                    self.vars[f'{nm}_{key}{sfx}'] = x
            for nm in ('rms', 'max'):
                x = g.createVariable(f'omega_arag_regional_direct_{nm}{sfx}', 'f8', (dim,), fill_value=np.nan)
                x.long_name = f'{nm} over the cells of (per-cell direct regression of Omega_arag - regional state)'
                self.vars[f'omega_arag_regional_direct_{nm}{sfx}'] = x

    def _set_keys(self):
        self.map_keys = [('', 'c', 'regional constrained'), ('_unconstrained', 'u', 'multi-model mean state'),
                         ('_sigma', 'sigma', '1 sigma of the constrained value (fit, observed dSST and CUC, CO2)'),
                         ('_sigma_fit', 's_fit', '1 sigma: regression fit (residual states, leverage)'),
                         ('_sigma_dsst', 's_dT', '1 sigma: observed dSST'), ('_sigma_cuc', 's_CU', '1 sigma: observed CUC'),
                         ('_sigma_co2', 's_co2', '1 sigma: atmospheric CO2'),
                         ('_sigma_unconstrained', 'sigma_u', '1 sigma of the unconstrained value (model spread and CO2)'),
                         ('_spread', 'spread', 'standard deviation across the models'),
                         ('_sigma_internal', 's_iv', '1 sigma of internal variability: how far a single real year may deviate '
                          'from the forced (smoothed) state; NOT included in _sigma, add in quadrature for a single year'),
                         ('_sigma_mapping_c1', 's_map1', '1 sigma: mapping uncertainty of the reference product (its window '
                          'mean, via the bias adjustment), errors independent between months; NOT included in _sigma'),
                         ('_sigma_mapping_c2', 's_map2', '1 sigma: mapping uncertainty of the reference product, the same error '
                          'in all months (conservative); NOT included in _sigma'),
                         ('_sigma_constants', 's_const', '1 sigma: dissociation constants and total boron; NOT included in _sigma')]
        self.ser_keys = [('global', 'value', 'global constraint (global regression)'), ('global_sigma', 'sigma', '1 sigma'),
                         ('global_sigma_fit', 's_fit', ''), ('global_sigma_dsst', 's_dT', ''), ('global_sigma_cuc', 's_CU', ''),
                         ('global_sigma_co2', 's_co2', ''),
                         ('map', 'value', 'area mean of the regional constrained field'), ('map_sigma', 'sigma', '1 sigma'),
                         ('map_sigma_fit', 's_fit', ''), ('map_sigma_dsst', 's_dT', ''), ('map_sigma_cuc', 's_CU', ''),
                         ('map_sigma_co2', 's_co2', ''),
                         ('unconstrained', 'value', 'multi-model mean'), ('unconstrained_sigma', 'sigma', ''),
                         ('unconstrained_spread', 'spread', ''), ('unconstrained_sigma_co2', 's_co2', ''),
                         ('direct', 'value', 'direct regression of the per-model area means (the method of the paper)'),
                         ('direct_sigma', 'sigma', ''), ('direct_r2', 'r2', ''),
                         ('internal_sigma', 'sigma', '1 sigma of internal variability of the area mean (not in the other sigmas)'),
                         ('level_sigma_mapping_c1', 's_map1', 'mapping uncertainty of the reference, months independent (not in the other sigmas)'),
                         ('level_sigma_mapping_c2', 's_map2', 'mapping uncertainty of the reference, same error in all months (not in the other sigmas)'),
                         ('level_sigma_constants', 's_const', 'dissociation constants and total boron (not in the other sigmas)')]

    def write(self, i, res, state=False):
        sfx = '_state' if state else ''
        for q in QUANTITIES:
            r = res[q]
            for suf, k, _ in self.map_keys:
                self.vars[f'{q}{suf}{sfx}'][i] = np.ma.masked_invalid(np.asarray(r['maps'][k], 'f4'))
            for key in ('own', 'common'):
                for suf, k, _ in self.ser_keys:
                    grp = suf.split('_')[0]
                    self.vars[f'{q}_gm_{key}_{suf}{sfx}'][i] = r[key][grp][k]
        for key in ('own', 'common'):
            self.vars[f'r2_thetao_{key}{sfx}'][i], self.vars[f'r2_dmc_{key}{sfx}'][i] = res[f'r2_{key}']
        self.vars[f'omega_arag_regional_direct_rms{sfx}'][i], self.vars[f'omega_arag_regional_direct_max{sfx}'][i] = res['omega_arag']['regional_direct']

    def copy_years(self, src_path, years):
        """copy the maps and series of the given years from another file of this writer's layout (the
        historical file, for the scenario years before the switch when the model set is the full set)"""
        with netCDF4.Dataset(src_path) as f:
            ys = np.asarray(f['year'][:]).astype(int)
            js = np.searchsorted(ys, years)
            if not np.array_equal(ys[js], years):
                raise ValueError(f'{src_path}: years {years[0]}-{years[-1]} not all present')
            jd = np.searchsorted(self.years, years)
            for name, x in self.vars.items():
                if x.dimensions[0] == 'year' and name in f.variables:
                    x[jd[0]:jd[-1] + 1] = f[name][js[0]:js[-1] + 1]

    def close(self):
        self.g.close()


# ================================================================================================ parts
def part_pi(R: Reference, names, suffix=''):
    """the preindustrial part for a model set: the years 1750-1849 and the named states. The targets are computed in
    chunks and every target is written as soon as its chunk is done, so that only the accumulators of one chunk are
    in memory (one accumulator holds every stacked state of every quantity: about 0.5 GB on the 1 degree grid)"""
    con = R.constraint(names)
    eng = Engine(R, con)
    X = {v: [] for v in VARS}
    named = {k: [] for k in PRODUCT_STATES}
    for m in names:
        c, nm = R.clim(m)
        for v in VARS:
            X[v].append(c[v])
        for k in PRODUCT_STATES:
            named[k].append(nm[k])
    X = {v: np.array(a) for v, a in X.items()}                                      # (n, 12, ny, nx)
    named = {k: np.array(a) for k, a in named.items()}
    x_years = {f'y{y}': float(R.co2.value(int(y))) for y in YEARS_PI}
    s_years = {f'y{y}': float(co2_history.sigma(int(y))) for y in YEARS_PI}
    x_states = {k: TARGETS[k] for k in PRODUCT_STATES}
    s_states = {k: XCO2_SIGMA[k] for k in PRODUCT_STATES}
    targets = {**x_years, **x_states}
    sig = {**s_years, **s_states}
    print(f'  pi{suffix}: {con.n} models, {len(targets)} targets, {int(con.ok.sum())} cells with local predictors', flush=True)
    # the targets in chunks sorted by CO2 (the Newton solve of a chunk starts warm after its first target); the first
    # chunk holds pi284 (the climatology itself) and the AR6 1750 state, whose constrained climatology gives the
    # internal variability and the level uncertainty that every target needs before it is written
    order = [t for t in sorted(targets, key=lambda k: targets[k]) if t != '1750']
    chunks = [['pi284', '1750']] + [order[i:i + PI_CHUNK] for i in range(0, len(order), PI_CHUNK)]
    clim_c = {t: {v: np.full((12,) + con.shape, np.nan) for v in VARS} for t in ('pi284', '1750')}
    iters = {}
    Xm = [{v: X[v][:, mo] for v in VARS} for mo in range(12)]
    path = OUT / f'states_{R.ref}_pi{suffix}.nc'
    w = Writer(path, R, con, YEARS_PI, f'pi{suffix}', states=PRODUCT_STATES,
               attrs=dict(co2_source='CMIP6 historical concentrations (Meinshausen et al. 2017), 1750-1849; AR6 for the named states'))
    x = w.g.createVariable('xco2', 'f8', ('year',)); x.units = 'ppm'; x.long_name = 'atmospheric CO2 of the year'; x[:] = [x_years[f'y{y}'] for y in YEARS_PI]
    x = w.g.createVariable('xco2_sigma', 'f8', ('year',)); x.units = 'ppm'; x[:] = [s_years[f'y{y}'] for y in YEARS_PI]
    w.g.createDimension('month', 12)
    iv = lres = None
    lines, keep = {}, {}
    for ic, chunk in enumerate(chunks):
        tg = {t: (px.XCO2_MODEL if t == 'pi284' else targets[t]) for t in chunk}
        sg = {t: sig.get(t, 0.0) for t in chunk}
        acc = {t: Accumulator(eng) for t in chunk if t != 'pi284'}
        for mo in range(12):
            main, pm = eng.month(Xm[mo], tg, sg, iters)
            for t in chunk:
                if t != 'pi284':
                    acc[t].add(main[t], pm.get(t))
                if t in clim_c:
                    iloc = con.index['loc']
                    for v in ('thetao', 'so', 'talk'):
                        clim_c[t][v][mo] = main[t][v][iloc]
                    clim_c[t]['dissic'][mo] = main[t]['dic'][iloc]
            del main, pm
        if ic == 0:
            for t, lab in (('pi284', '284ppm'), ('1750', '1750')):
                for v in VARS:
                    x = w.g.createVariable(f'{v}_clim_{lab}', 'f4', ('month', 'lat', 'lon'), zlib=True, complevel=4, fill_value=FILL)
                    x.units = cb.UNITS[v]
                    x.long_name = (f'{v}: constrained (regional) monthly climatology at '
                                   f'{"284.317 ppm (the adjusted piControl)" if t == "pi284" else "278.3 ppm (AR6 1750)"}')
                    x[:] = np.ma.masked_invalid(clim_c[t][v])
            # internal variability: the models' raw-minus-smoothed anomalies of 1850-1899 added to the constrained 1750
            # state; the same for every preindustrial year and named state (6 ppm of CO2 hardly change it)
            iv = pi_internal(R, con, names, {v: clim_c['1750'][v] for v in VARS})
            # level uncertainty (mapping, constants): at the constrained 1750 state, the same for every preindustrial year
            lres = R.level_sigma({v: clim_c['1750'][v] for v in VARS})
            g = lres['omega_arag']['series']['own']
            print(f'  pi{suffix}: level uncertainty of Omega_arag (1750 state, own area): mapping C1 {g[0]:.4f}, '
                  f'C2 {g[1]:.4f}, constants {g[2]:.4f}', flush=True)
        for t in list(acc):
            res = acc[t].result()
            set_internal(res, iv)
            set_level(res, lres)
            if t in PRODUCT_STATES:
                w.write(PRODUCT_STATES.index(t), res, state=True)
                # check: climatology-based per-model Omega against the all-months means; state against the paper
                A = acc[t].sum['omega_arag'] / acc[t].count
                mods = con.models(A)
                d = np.array([R.area(mods[m] - named[t][m], 'own') for m in range(con.n)])
                g = res['omega_arag']['own']
                lines[t] = (f'  {t:6s}: per-model Omega from the climatology minus all-months mean: {np.mean(d):+.1e} '
                            f'(max |{np.max(np.abs(d)):.1e}|); state global {g["global"]["value"]:.4f} +- {g["global"]["sigma"]:.4f}, '
                            f'direct {g["direct"]["value"]:.4f} +- {g["direct"]["sigma"]:.4f}, map mean {g["map"]["value"]:.4f}, '
                            f'unconstrained {g["unconstrained"]["value"]:.4f}'
                            + (f'; paper {R.paper[t]["constrained"]:.4f} +- {R.paper[t]["constrained_sigma"]:.4f}'
                               if t in R.paper and not suffix else ''))
            else:
                w.write(int(t[1:]) - YEARS_PI[0], res)
                if t in ('y1750', 'y1849'):
                    keep[t] = (res['omega_arag']['own'], res['r2_own'])
            del acc[t], res
        w.g.sync()
        print(f'  pi{suffix}: targets {ic + 1}/{len(chunks)} done and written ({chunk[0]} ... {chunk[-1]})', flush=True)
    out = [lines[k] for k in PRODUCT_STATES]
    (y0, r2), (y1, _) = keep['y1750'], keep['y1849']
    out.append(f'  years: 1750 ({x_years["y1750"]:.2f} ppm) global {y0["global"]["value"]:.4f} +- {y0["global"]["sigma"]:.4f}, '
               f'1849 ({x_years["y1849"]:.2f} ppm) {y1["global"]["value"]:.4f}; direct 1750 {y0["direct"]["value"]:.4f}, '
               f'R2 T {r2[0]:.2f}, DIC-ALK {r2[1]:.2f}')
    w.g.checks = '\n'.join(out)
    w.g.newton_iterations = ', '.join(f'{n}: {c}' for n, c in sorted(iters.items()))
    w.close()
    print('\n'.join(out) + f'\n  written {path}', flush=True)


IV_YEARS = (1850, 1899)          # years of the historical anomalies that give the preindustrial internal variability


def pi_internal(R: Reference, con, names, base):
    """sigma of internal variability for the preindustrial part: every model's raw-minus-smoothed anomaly of each
    year IV_YEARS (calc_smooth_states.py) added month by month to the constrained (regional) 1750 state `base`
    {var: (12, ny, nx)}; root mean square over models and years of the change of the annual mean of every
    quantity, per cell and for the area means on both masks. Returns {q: (map, {'own': value, 'common': value})}"""
    eng = Engine(R, con)
    base_res = {}
    for mo in range(12):
        r = eng.chemistry({v: base[v][mo][None] for v in VARS}, {'state': px.XCO2_MODEL})['state']
        for q in QUANTITIES:
            base_res[q] = r[q][0] / 12.0 + base_res.get(q, 0.0)
    ss = {q: 0.0 for q in QUANTITIES}
    sg = {q: {'own': 0.0, 'common': 0.0} for q in QUANTITIES}
    count = 0
    for m in names:
        sm = Smoothed(R, m)
        try:
            yrs = [y for y in range(IV_YEARS[0], IV_YEARS[1] + 1) if y in sm.years_hist]
            ann = {q: 0.0 for q in QUANTITIES}
            for mo in range(12):
                A = {v: [] for v in VARS}
                for y in yrs:
                    _, a = sm.state(y, 'hist', anomalies=True)
                    for v in VARS:
                        A[v].append(a[v][mo])
                fields = {v: base[v][mo][None] + np.array(A[v]) for v in VARS}          # (n_years, ny, nx)
                r = eng.chemistry(fields, {'state': px.XCO2_MODEL})['state']
                for q in QUANTITIES:
                    ann[q] = ann[q] + r[q] / 12.0
        finally:
            sm.close()
        for q in QUANTITIES:
            d = ann[q] - base_res[q][None]                                            # (n_years, ny, nx)
            ss[q] = ss[q] + np.sum(d ** 2, axis=0)
            for key in ('own', 'common'):
                sg[q][key] += float(np.nansum(np.array([R.area(d[j], key) for j in range(d.shape[0])]) ** 2))
        count += len(yrs)
    out = {}
    for q in QUANTITIES:
        out[q] = (np.sqrt(ss[q] / count), {key: float(np.sqrt(sg[q][key] / count)) for key in ('own', 'common')})
    print(f'  internal variability (anomalies {IV_YEARS[0]}-{IV_YEARS[1]}, {len(names)} models, {count} model-years): '
          f'Omega_arag {out["omega_arag"][1]["own"]:.4f} (area mean), {np.nanmean(out["omega_arag"][0]):.4f} (mean over cells)', flush=True)
    return out


def set_level(res, lres):
    for q in QUANTITIES:
        r = lres[q]
        res[q]['maps'].update(s_map1=r['map1'], s_map2=r['map2'], s_const=r['const'])
        for key in ('own', 'common'):
            s1, s2, s3 = r['series'][key]
            res[q][key]['level'] = dict(s_map1=s1, s_map2=s2, s_const=s3)


def set_internal(res, iv):
    for q in QUANTITIES:
        res[q]['maps']['s_iv'] = iv[q][0]
        for key in ('own', 'common'):
            res[q][key]['internal'] = dict(sigma=iv[q][1][key])


def part_series(R: Reference, names, series, copy_from=None, only_years=None):
    """historical (series 'hist', 1850-2014) or scenario (1850-2100) part for a model set; the years before the
    switch are copied from copy_from (the historical file for the full model set, or the complete file of another
    scenario with the same model set) when given; they are identical there by construction. A year in which a
    model has no state (MPI-ESM1-2-HR and UKESM1-0-LL end in 2099) is constrained with the models that have it
    (own regression); n_models(year) records this. only_years: recompute these years in the existing file."""
    con = R.constraint(names)
    eng = Engine(R, con)
    sm = {m: Smoothed(R, m) for m in names}
    sub = {}                                   # constraints of the model subsets of single years
    years = YEARS_HIST if series == 'hist' else YEARS_SSP
    switch = cs.SCEN_FIRST
    path = OUT / f'states_{R.ref}_{series}.nc'
    if only_years:
        part_years(R, names, series, path, con, eng, sm, sub, [int(y) for y in only_years])
        return
    w = Writer(path, R, con, years, series,
               attrs=dict(smoothing=f'{sm[names[0]].f.cutoff_years:.0f}-year Enting spline per calendar month (calc_smooth_states.py)',
                          series=('historical 1850-2014' if series == 'hist' else
                                  f'historical 1850-{switch - 1} (smoothed with the default continuation) and {series} from {switch}'),
                          co2_source='CMIP6 historical and ScenarioMIP concentrations (Meinshausen et al. 2017, 2020)'))
    x = w.g.createVariable('xco2', 'f8', ('year',)); x.units = 'ppm'; x.long_name = 'atmospheric CO2 of the year'
    x_co2 = np.array([R.co2.value(int(y), None if y <= 2014 else series) for y in years])
    x[:] = x_co2
    s_co2 = np.array([co2_history.sigma(int(y)) for y in years])
    x = w.g.createVariable('xco2_sigma', 'f8', ('year',)); x.units = 'ppm'; x[:] = s_co2
    nm = w.g.createVariable('n_models', 'i4', ('year',)); nm[:] = con.n
    nm.long_name = 'number of models in the constraint of the year (fewer where a model has no state)'
    w.vars['n_models'] = nm
    copied = series != 'hist' and copy_from is not None
    if copied:
        w.copy_years(copy_from, years[years < switch])
        w.g.years_copied_from = f'{years[0]}-{switch - 1} from {Path(copy_from).name} (same models)'
    print(f'  {series}: {con.n} models, years {years[0]}-{years[-1]}'
          + (f' ({years[0]}-{switch - 1} copied from {Path(copy_from).name}: same models)' if copied else ''), flush=True)
    for i, y in enumerate(years):
        if copied and y < switch:
            continue
        res, n = year_state(R, names, series, int(y), switch, con, eng, sm, float(s_co2[i]), sub)
        if n < con.n:
            w.vars['n_models'][i] = n
            w.g.setncattr(f'note_{y}', f'{n} of {con.n} models have a state in {y}; this year is constrained with '
                                      f'these (own regression)')
            print(f'  {series} {y}: {n} of {con.n} models have a state, constrained with these', flush=True)
        w.write(i, res)
        if y % 10 == 0 or y == years[-1]:
            g = res['omega_arag']['own']
            print(f'  {series} {y}: Omega_arag global {g["global"]["value"]:.4f} +- {g["global"]["sigma"]:.4f}, map {g["map"]["value"]:.4f}, '
                  f'unconstrained {g["unconstrained"]["value"]:.4f} +- {g["unconstrained"]["sigma"]:.4f}, direct {g["direct"]["value"]:.4f} '
                  f'(R2 {g["direct"]["r2"]:.2f}); R2 T {res["r2_own"][0]:.2f}, DIC-ALK {res["r2_own"][1]:.2f}', flush=True)
    for s_ in sm.values():
        s_.close()
    w.close()
    R.level.close()
    print(f'  written {path}', flush=True)


def year_state(R, names, series, y, switch, con, eng, sm, s_co2, sub):
    """the constrained state of one year with every indicator; a model without a state in this year (all NaN) is
    left out, and the year is constrained with the others (their own regression, cached in sub)"""
    key = 'hist' if (series == 'hist' or y < switch) else series
    st = [sm[m].state(y, key, anomalies=True) for m in names]     # per model ({var: (12, ny, nx)}, anomalies)
    have = [k for k in range(len(names)) if np.isfinite(st[k][0]['thetao']).any()]
    if len(have) < len(names):
        sel = tuple(names[k] for k in have)
        if sel not in sub:
            c = R.constraint(list(sel))
            sub[sel] = (c, Engine(R, c))
        con, eng = sub[sel]
        st = [st[k] for k in have]
    acc = Accumulator(eng)
    iloc = con.index['loc']
    loc12 = {v: np.full((12,) + con.shape, np.nan) for v in VARS}      # the constrained (regional) state
    for mo in range(12):
        Xm = {v: np.array([s_[0][v][mo] for s_ in st]) for v in VARS}
        Am = {v: np.array([s_[1][v][mo] for s_ in st]) for v in VARS}
        main, pm = eng.month(Xm, {'state': px.XCO2_MODEL}, {'state': s_co2}, A=Am)
        acc.add(main['state'], pm.get('state'))
        for v in VARS:
            loc12[v][mo] = main['state']['dic' if v == 'dissic' else v][iloc]
    del st
    res = acc.result()
    set_level(res, R.level_sigma(loc12))
    return res, len(have)


def part_years(R, names, series, path, con, eng, sm, sub, years):
    """recompute the given years in an existing file of the part (for example 2100, where two models end in 2099)"""
    if not path.exists():
        raise FileNotFoundError(f'{path} missing: run the part first')
    w = Writer(path, R, con, None, series, append=True)
    if 'n_models' not in w.g.variables:
        nm = w.g.createVariable('n_models', 'i4', ('year',)); nm[:] = con.n
        nm.long_name = 'number of models in the constraint of the year (fewer where a model has no state)'
    switch = cs.SCEN_FIRST
    for y in years:
        i = int(np.searchsorted(w.years, y))
        if i >= w.years.size or w.years[i] != y:
            raise KeyError(f'{y} not in {path.name}')
        res, n = year_state(R, names, series, y, switch, con, eng, sm, float(co2_history.sigma(y)), sub)
        w.g['n_models'][i] = n
        if n < con.n:
            w.g.setncattr(f'note_{y}', f'{n} of {con.n} models have a state in {y}; this year is constrained with '
                                      f'these (own regression)')
        w.write(i, res)
        g = res['omega_arag']['own']
        print(f'  {series} {y} recomputed with {n} of {con.n} models: Omega_arag global {g["global"]["value"]:.4f} '
              f'+- {g["global"]["sigma"]:.4f}, map {g["map"]["value"]:.4f}', flush=True)
    w.g.years_recomputed = ', '.join(str(y) for y in years) + f' ({datetime.datetime.now():%Y-%m-%d})'
    for s_ in sm.values():
        s_.close()
    w.close()
    R.level.close()


def run_part(ref, part, source=None, only_years=None):
    """one part in its own process; source: for a scenario, the file to copy the pre-switch years from
    ('hist', or the scenario of the same model set computed first)"""
    try:
        R = Reference(ref)
        OUT.mkdir(parents=True, exist_ok=True)
        if part == 'pi':
            part_pi(R, R.names)
        elif part == 'hist':
            part_series(R, R.names, 'hist', only_years=only_years)
        elif only_years:
            part_series(R, scenario_models(R.names, R.members, part), part, only_years=only_years)
        else:
            have = scenario_models(R.names, R.members, part)
            copy_from = None
            if source == 'hist':
                copy_from = OUT / f'states_{ref}_hist.nc'
                if not complete(copy_from, R.names, YEARS_SSP[YEARS_SSP < cs.SCEN_FIRST]):
                    raise RuntimeError(f'{copy_from.name} is not complete: the years before {cs.SCEN_FIRST} cannot be copied')
            elif source:
                src = OUT / f'states_{ref}_{source}.nc'
                if not complete(src, have, YEARS_SSP[YEARS_SSP < cs.SCEN_FIRST]):
                    raise RuntimeError(f'{src.name} is not complete for the models of {part}')
                copy_from = src
                # the preindustrial part of the same model set is identical: copy it
                pi_src, pi_dst = OUT / f'states_{ref}_pi_{source}.nc', OUT / f'states_{ref}_pi_{part}.nc'
                shutil.copyfile(pi_src, pi_dst)
                with netCDF4.Dataset(pi_dst, 'a') as f:
                    f.title = f.title.replace(f'pi_{source}', f'pi_{part}')
                    f.copied_from = f'{pi_src.name} (same models)'
                print(f'  pi_{part}: copied from {pi_src.name} (same models)', flush=True)
            if have != R.names and not source:
                pi_path = OUT / f'states_{ref}_pi_{part}.nc'
                if complete(pi_path, have, YEARS_PI):
                    print(f'  pi_{part}: {pi_path.name} complete (same models), reused', flush=True)
                else:
                    part_pi(R, have, f'_{part}')
            part_series(R, have, part, copy_from)
        fl = px.cm.FLOORED
        print(f'  {part}: temperatures set to the freezing point of seawater before the chemistry: {fl[0]} of {fl[1]} '
              f'cell-states ({100 * fl[0] / max(fl[1], 1):.3f} %; the constrained, perturbed and single-model states of this part)', flush=True)
    except Exception:
        print(f'[FAILED] {ref} {part}\n{traceback.format_exc()}', flush=True)
        raise SystemExit(1)


def run_tasks(tasks, jobs):
    """each task (ref, part, source) in its own process, at most jobs at a time; returns {part: status}. A
    process killed by a signal (for example by the out-of-memory killer) is reported, the others go on."""
    todo, running, status = list(tasks), {}, {}
    ctx = mp.get_context('fork')
    while todo or running:
        while todo and len(running) < max(jobs, 1):
            t = todo.pop(0)
            p = ctx.Process(target=run_part, args=t, name=f'{t[0]}_{t[1]}')
            p.start()
            running[t[1]] = p
        time.sleep(5)
        for part, p in list(running.items()):
            if p.exitcode is not None:
                p.join()
                status[part] = ('done' if p.exitcode == 0 else
                                f'KILLED (signal {-p.exitcode})' if p.exitcode < 0 else 'FAILED')
                print(f'  [{status[part]}] {part}', flush=True)
                del running[part]
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', default='OceanSODA', choices=list(cb.REFERENCES))
    ap.add_argument('--part', nargs='+', default=['pi', 'hist'] + list(SCENARIOS))
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--dry', action='store_true', help='print the plan of the scenarios only')
    ap.add_argument('--years', type=int, nargs='+', help='recompute only these years in the existing files of the '
                    'given parts (hist or scenarios)')
    a = ap.parse_args()
    # the reference must have been run in the chain
    pkl = oa_config.DATA / ('constraint_results_jiang.pkl' if a.reference == 'Jiang' else 'constraint_results.pkl')
    with open(pkl, 'rb') as f:
        P = pickle.load(f)
    present = a.reference in (P['own'] if a.reference == 'Jiang' else P['results'])
    if not present:
        print(f'{a.reference}: not in {pkl} (reference not run in this chain), nothing to do')
        return
    print(f'{a.reference}: parts {a.part}', flush=True)
    status = {}
    if a.years:
        status.update(run_tasks([(a.reference, p, None, a.years) for p in a.part if p != 'pi'], a.jobs))
        finish(status)
        return
    # phase 1: pi and hist
    if not a.dry:
        status.update(run_tasks([(a.reference, p, None) for p in a.part if p in ('pi', 'hist')], a.jobs))
    # phase 2: the scenarios, grouped by model set. Full set: copy from hist. Otherwise the first scenario of a
    # set computes the shared years (2a), the others copy them from it once it is complete (2b).
    names, members = model_set(a.reference)
    first, later = [], []
    groups = {}
    for scen in [p for p in a.part if p not in ('pi', 'hist')]:
        have = scenario_models(names, members, scen)
        if len(have) < 4:
            print(f'  {scen}: {len(have)} models, fewer than 4: skipped', flush=True)
            status[scen] = 'skipped'
            continue
        if have == names:
            first.append((a.reference, scen, 'hist'))
            continue
        key = tuple(have)
        if key in groups:
            later.append((a.reference, scen, groups[key]))
        else:
            # a complete scenario of the same set from an earlier run can serve as the source
            done = [s for s in SCENARIOS if s != scen and s not in a.part
                    and complete(OUT / f'states_{a.reference}_{s}.nc', have, YEARS_SSP[YEARS_SSP < cs.SCEN_FIRST])
                    and scenario_models(names, members, s) == have]
            if done:
                groups[key] = done[0]
                later.append((a.reference, scen, done[0]))
            else:
                groups[key] = scen
                first.append((a.reference, scen, None))
    for t in first + later:
        if t[2] is None:
            pi_ok = complete(OUT / f'states_{a.reference}_pi_{t[1]}.nc', scenario_models(names, members, t[1]), YEARS_PI)
            print(f'  pi_{t[1]}: ' + ('complete, reused' if pi_ok else 'computed'), flush=True)
        print(f'  {t[1]}: ' + ('copies 1850-1969 from the historical file' if t[2] == 'hist' else
                               f'copies 1850-1969 and pi from {t[2]}' if t[2] else 'computes all years'), flush=True)
    if a.dry:
        return
    # scenarios that copy from hist only if hist is done (or was complete before this run)
    hist_bad = status.get('hist', 'done') != 'done'
    for t in [t for t in first if t[2] == 'hist' and hist_bad]:
        status[t[1]] = 'not run (hist failed)'
    first = [t for t in first if not (t[2] == 'hist' and hist_bad)]
    status.update(run_tasks(first, a.jobs))
    for t in later:
        if status.get(t[2], 'done') != 'done':
            status[t[1]] = f'not run ({t[2]} failed)'
    later = [t for t in later if status.get(t[2], 'done') == 'done']
    status.update(run_tasks(later, a.jobs))
    finish(status)


def finish(status):
    """print the summary and exit non-zero if any part did not finish"""
    print('\nsummary:')
    for part, st in status.items():
        print(f'  {st:20s} {part}')
    bad = [p for p, st in status.items() if st not in ('done', 'skipped')]
    if bad:
        print(f'not finished: {", ".join(bad)}', flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
