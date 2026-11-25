#!/usr/bin/env python
"""
make_oatrack_simple.py  (the estimate of the forced state, the temperature flag and the storage precision that make_oatrack_product.py uses; its own compact files are not part of the record)
======================
Writes, next to the full files of OA-Track (make_oatrack.py), the files most users need, each value with the few
uncertainties that a user combines with the formulas below (also in the attributes of every file):
    OA-Track_maps_<q>.nc       one file per quantity: maps of the mean of the fCO2-products that cover each cell
                               (from OA-Track_all.nc), with the alkalinity adjustment
    OA-Track_global_means.nc   all quantities: global means on the common area of the paper, mean of the four
                               fCO2-products (from OA-Track_<ref>.nc), with the alkalinity adjustment
Values (<q> = omega_arag, ph, dic, ...):
    <q>_forced            year 1750-2014       forced state from the constrained ESMs (1750-1849: the constrained
                                               preindustrial ocean at the CMIP6 CO2 of the year)
    <q>_forced_<ssp>      year_ssp 1750-2100   the same for each scenario, with the ESMs that provide it
    <q>_preindustrial     state 1750, 1850     the constrained states at the CO2 of IPCC AR6 (278.3, 285.5 ppm)
    <q>_obs               year_obs             annual values of the fCO2-products
    <q>_obs_forced        year_obs to 2024     forced state of the observations: per fCO2-product (and cell) the
                                               quadratic fit of its annual values up to 2024, evaluated in the year;
                                               2024 is the forced 2024 state of the paper (<q>_2024_trend)
Uncertainties (1 sigma, the same units as the value):
    _sigma_delta            the change relative to the reference period known only imprecisely (the constraint:
                            fit, observed predictors, CO2); root mean square over the fCO2-products (forced and
                            preindustrial only)
    _sigma_products         standard deviation between the fCO2-products (absolute values only)
    _sigma_baseline         mapping (C2) and equilibrium constants in quadrature (absolute values only); the Revelle
                            factor has no constants term (mocsy.merrors gives none), so its baseline is the mapping.
                            Left out (masked) where the mapping uncertainty of temperature is not reliable: in the
                            cells of sigma_baseline_flag (known limitation of version 1.0, see T_LIMIT below)
    _alkalinity_adjustment  the adjustment a itself, included in the value, and its own 1 sigma (|a| for a value,
                            |a2 - a1| for a change); not for pCO2, fCO2, CO2*, T and S, which it does not change
    _sigma_interannual      a single real year about the forced state (forced only; added only for a real year)
  global means only:
    <q>_obs_forced_sigma_fit                the estimate of the forced state from annual values: AR(1) error of the
                            quadratic fit, corrected with synthetic records, and half the range of the alternative
                            estimators, in quadrature, root mean square over the products (as Extended Data Table 2)
    <q>_obs_forced_change_sigma_products    standard deviation between the products of the change from 1750
The maps: the 1 sigma terms from the combined file (_sigma_delta = sqrt(_sigma^2 - _spread^2), as make_oatrack.py
combines them), the alkalinity adjustment as the difference of the files with and without it. <q>_obs_forced from the
four files of the single products: the fit of the values without the adjustment, plus the adjustment evaluated at the
forced state of the year (the fits of temperature, salinity, alkalinity and DIC, with the alkalinity increase of the
year), as make_oatrack.py computes the 2024 trend value; the global means use its area mean on the common area.

Needs: data/<run>/oatrack/OA-Track_all[_noalk].nc and OA-Track_<ref>[_noalk].nc (make_oatrack.py, unquantized), the
NOAA record of the run folder (co2_annmean_gl.txt, as the notebook), and mocsy with the nutrients that
make_product_alkalinity.Chem reads (as make_oatrack.py).
Output: --out (default data/<run>/oatrack_simple), maps quantized with DIGITS and MODE.
Usage (code/):  OA_RUN=u74 OMP_NUM_THREADS=1 python -u make_oatrack_simple.py --jobs 4
"""
import argparse
import datetime
import multiprocessing as mp_
import time
import warnings
from pathlib import Path

import netCDF4
import numpy as np

import make_product_alkalinity as ma
import oa_config
import state_constraint as sc

PRODUCTS = ('OceanSODA', 'UExP', 'JMA', 'CMEMS')
EVAL_YEAR = 2024
DIGITS = (5, 3)                        # maps: values, and every 1 sigma and adjustment
MODE = 'GranularBitRound'
BLOCK = 24                             # maps per read and write
FILL = np.float32(1.0e20)
SRC = oa_config.DATA / 'oatrack'
netCDF4.set_chunk_cache(8 * 2 ** 20, 521, 0.75)

# ------------------------------------------------------------------------------------------------ texts
SOURCE = ('OA-Track version 1.0 (Terhaar, Xue and Hollitzer: Precise tracking of ocean acidification as it nears its '
          'planetary boundary; data https://doi.org/10.17882/121840; code https://doi.org/10.5281/zenodo.24493310)')
UNC = {
    '_sigma_delta': '1 sigma of the state because the change relative to the reference period of the fCO2-products '
                    'is known only imprecisely (emergent constraint: regression across the 18 ESMs, observed warming '
                    'and carbon uptake capacity, CO2 of the year); root mean square over the fCO2-products',
    '_sigma_products': '1 sigma: standard deviation between the fCO2-products (missing where only one covers the '
                       'value); for absolute values only',
    '_sigma_baseline': '1 sigma of the observational baseline: mapping error of the fCO2-products (the same error in '
                       'all months of a year) and equilibrium constants, in quadrature; it shifts all years alike and '
                       'cancels in changes; for absolute values only',
    '_alkalinity_adjustment': 'the alkalinity adjustment a included in the value (signed; the value minus a is the '
                              'value without it); it is also its own 1 sigma uncertainty: |a| for a value, |a2 - a1| '
                              'for a change between two years',
    '_sigma_interannual': '1 sigma of interannual variability: how far a single real year may deviate from the '
                          'forced state; add it only when comparing with a single real year',
    '_sigma_fit': '1 sigma of the estimate of the forced state from annual values with natural variability: the '
                  'error of the quadratic fit (AR(1) residuals, corrected with synthetic records) and half the range '
                  'of the alternative estimators (quadratic fit against CO2, Enting splines of 15, 20 and 25 years '
                  'in time and against CO2), in quadrature; root mean square over the fCO2-products, as Extended '
                  'Data Table 2 of the paper',
    '_change_sigma_products': '1 sigma: standard deviation between the fCO2-products of the change from the '
                              'preindustrial state of 1750 to the year; for changes from 1750',
}
SHORT = {'_sigma_delta': '1 sigma of the change relative to the reference period',
         '_sigma_products': '1 sigma between the fCO2-products',
         '_sigma_baseline': '1 sigma of the observational baseline (mapping and constants)',
         '_alkalinity_adjustment': 'alkalinity adjustment (also its own 1 sigma)',
         '_sigma_interannual': '1 sigma of interannual variability',
         '_sigma_fit': '1 sigma of the estimate of the forced state',
         '_change_sigma_products': '1 sigma between the fCO2-products of the change from 1750'}
BLOCK_NAME = {'forced': 'forced state', 'forced_ssp': 'forced state, {s}', 'preindustrial': 'preindustrial state',
              'obs': 'annual values of the fCO2-products', 'obs_forced': 'forced state of the observations'}
VAL = {
    'forced': 'forced state from the 18 Earth system models, bias-adjusted to the fCO2-products and constrained by '
              'the observed warming and carbon uptake capacity (emergent constraint): 1750-1849 the constrained '
              'preindustrial ocean at the CMIP6 CO2 of the year, 1850-2014 the constrained, smoothed ESMs',
    'forced_ssp': 'forced state of scenario {s} from the constrained ESMs that provide the scenario, 1750-2100',
    'preindustrial': 'constrained preindustrial state at the atmospheric CO2 of IPCC AR6 (278.3 ppm in 1750, 285.5 ppm '
                     'in 1850), the reference of the paper for the decline',
    'obs': 'annual values of the fCO2-products (the real year-to-year variability included)',
    'obs_forced': 'forced state of the observations: the quadratic fit of the annual values of each fCO2-product up to '
                  '2024 (without the alkalinity adjustment), evaluated in the year, plus the alkalinity adjustment at '
                  'that forced state; the value of 2024 is the forced 2024 state of the paper',
}
FORMULAS = (
    'x: a value; a: its _alkalinity_adjustment; s_delta, s_products, s_baseline, s_interannual: its 1 sigma terms. '
    '(1) Absolute value: sigma = sqrt(s_delta^2 + s_products^2 + s_baseline^2 + a^2) (s_delta only for _forced and '
    '_preindustrial; add s_interannual^2 to compare a forced value with a single real year{fit}). '
    '(2) Change from a model year or preindustrial state x1 to an observed year x2 (_obs_forced or _obs, 1980-2024): '
    'difference x2 - x1 with sigma = sqrt(s_delta(x1)^2 + (a2 - a1)^2{chg}); mapping, constants and the spread of '
    'the levels of the fCO2-products cancel. Relative change 100 (x2 / x1 - 1) in percent, with '
    'sigma = sqrt((100 x2 s_delta(x1) / x1^2)^2 + (r - r0)^2{chgr}), r0 the relative change of the values without '
    'the adjustment, (x2 - a2) and (x1 - a1). (3) Change between two model years: the errors of the two states are '
    'correlated; the files give only the bounds |s_delta(2) - s_delta(1)| <= sigma <= sqrt(s_delta(1)^2 + '
    's_delta(2)^2) for that term (the exact value needs the emergent constraint applied to the difference, with the '
    'code). The full uncertainty of the global decline is Extended Data Table 2 of the paper.')
F_MAPS = FORMULAS.format(fit='', chg='', chgr='') + (
    ' Not included per cell: the uncertainty of the estimate of the forced state from the annual values (for the global '
    'mean see OA-Track_global_means.nc and Extended Data Table 2).')
F_GLOBAL = FORMULAS.format(fit='; for _obs_forced add s_fit^2',
                           chg=' + s_fit(x2)^2 + s_change_products(x2)^2 (x1 the 1750 state)',
                           chgr=' + (100 s_fit(x2) / x1)^2 + (100 s_change_products(x2) / x1)^2')


def rd(f, name, idx=Ellipsis):
    return np.ma.filled(f[name][idx].astype('f8'), np.nan)


def quantities(f):
    """the quantities of an OA-Track file, in its order: names with a state map and annual values"""
    v = f.variables
    return [n for n in v if f'{n}_state' in v and f'{n}_obs' in v and n not in ('xco2',)]


def scenarios(f, form='omega_arag_{s}'):
    """the scenarios of an OA-Track file: maps <q>_<ssp>, or global means with form 'omega_arag_gm_common_{s}_global'"""
    return [s for s in ('ssp126', 'ssp245', 'ssp370', 'ssp585') if form.format(s=s) in f.variables]


def std_products(a):
    """standard deviation (ddof 1) over the fCO2-products (axis 0); missing where fewer than two are present"""
    n = np.sum(np.isfinite(a), 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.where(n > 1, np.nanstd(a, 0, ddof=1), np.nan)


def rms_products(a):
    """root mean square over the fCO2-products (axis 0); missing where none is present"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.sqrt(np.nanmean(np.square(a), 0))


def hat_fit(years, Y, ok):
    """the quadratic fit of Y (n, ...) over `years`, evaluated in every year; NaN where not ok"""
    X = np.vander(np.asarray(years, 'f8') - 2000.0, 3)
    H = X @ np.linalg.solve(X.T @ X, X.T)                        # the hat matrix: fitted values of every year
    fit = np.tensordot(H, np.where(ok, Y, 0.0), axes=(1, 0))
    return np.where(ok, fit, np.nan)


def obs_years(f):
    """the years of the annual values of a product up to EVAL_YEAR, and their index"""
    yr = np.asarray(f['year_obs'][:]).astype(int)
    sel = yr <= EVAL_YEAR
    assert np.all(np.diff(yr[sel]) == 1), f'{f.filepath()}: years with gaps'
    return yr[sel], sel


ADJ = {}                               # per product: years, {q: adjustment maps at the forced state}, {q: common-area mean}


def adjustment_at_fit(r):
    """the alkalinity adjustment of one fCO2-product in every year up to EVAL_YEAR, evaluated at the forced state of the
    year: the quadratic fits of temperature, salinity, alkalinity and DIC without the adjustment, raised by the
    alkalinity increase of the year (the difference of the files with and without it), with the chemistry of
    make_oatrack.py (make_product_alkalinity.Chem); in EVAL_YEAR this is the adjustment of <q>_2024_trend"""
    t0 = time.time()
    with netCDF4.Dataset(SRC / f'OA-Track_{r}.nc') as f, netCDF4.Dataset(SRC / f'OA-Track_{r}_noalk.nc') as g:
        lat, lon = rd(f, 'lat'), rd(f, 'lon')
        yy, sel = obs_years(f)
        own, common = rd(f, 'mask') > 0, rd(f, 'mask_common') > 0
        state = {q: rd(g, f'{q}_obs')[sel] for q in ('thetao', 'so', 'talk', 'dic')}
        dta = (rd(f, 'talk_obs') - rd(g, 'talk_obs'))[sel]
    ok = own & np.all([np.all(np.isfinite(a), axis=0) for a in state.values()], axis=0)
    fit = {q: hat_fit(yy, a, ok) for q, a in state.items()}
    chem = ma.Chem(lat, lon)
    D = {q: np.full(dta.shape, np.nan) for q in ma.DELTA_Q}
    for j in range(yy.size):
        d = chem.deltas(fit['thetao'][j], fit['so'][j], fit['talk'][j], fit['dic'][j], dta[j])
        for q in ma.DELTA_Q:
            D[q][j] = d[q]
    gm = {q: sc.area_mean(D[q], lat, lon, common) for q in ma.DELTA_Q}
    return r, yy, {q: a.astype('f4') for q, a in D.items()}, gm, time.time() - t0


NO_CONSTANTS = ('revelle',)           # mocsy.merrors gives no uncertainty of the Revelle factor (level_uncertainty.py)
# Known limitation of version 1.0: level_uncertainty.py converts the Omega_arag uncertainty from sea surface temperature
# (UExP-FNN-U, used for all products) into a temperature error with dOmega_arag/dT at fixed fCO2 and TA. Where this
# derivative is close to zero (cold water), the temperature error, and through it the mapping uncertainty of the
# quantities that depend on it, is far too large (in the model years, where the error is propagated at another state).
# A cell is flagged where the temperature error of one of the products exceeds T_LIMIT in one of its years, more than
# any plausible uncertainty of sea surface temperature; there _sigma_baseline of the quantities in T_DEPENDENT is left
# out (for temperature in every block, for the others in the model years). Omega_arag and Omega_calc (whose error is
# the original one), alkalinity and salinity are not affected, nor are the global means.
T_LIMIT = 1.0                          # degC
T_DEPENDENT = ('thetao', 'pco2', 'fco2', 'co2', 'ph', 'hplus', 'hplus_free', 'dic', 'hco3', 'co3', 'revelle')
MODEL_BLOCKS = ('forced', 'preindustrial')     # and forced_<ssp>
TFLAG = {}                             # 'cells': the flagged cells (lat, lon), set in main


def temperature_flag():
    """the cells where the mapping uncertainty of temperature of one of the products exceeds T_LIMIT in one of its years,
    and the number per product"""
    flag, n = None, {}
    for r in PRODUCTS:
        with netCDF4.Dataset(SRC / f'OA-Track_{r}.nc') as f:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                fr = np.nanmax(rd(f, 'thetao_obs_sigma_mapping_c2'), 0) > T_LIMIT
        n[r] = int(fr.sum())
        flag = fr if flag is None else flag | fr
    return flag, n


def masked_baseline(q, blk):
    """whether _sigma_baseline of quantity q in block blk is left out in the flagged cells"""
    return q in T_DEPENDENT and (q == 'thetao' or blk.split('_ssp')[0] in MODEL_BLOCKS)


def baseline(q, c2, k):
    """_sigma_baseline: the mapping (C2) and the equilibrium constants in quadrature, the mapping alone where the
    constants give no uncertainty (k is then not used and may be None)"""
    return np.abs(c2) if q in NO_CONSTANTS else np.hypot(c2, k)


def unc_text(q, s, blk=''):
    """the description of an uncertainty term of quantity q in block blk"""
    t = UNC[s]
    if s == '_sigma_baseline' and q in NO_CONSTANTS:
        t = t.replace(' and equilibrium constants, in quadrature', ' only (mocsy gives no uncertainty of this quantity '
                                                                    'from the equilibrium constants)')
        assert t != UNC[s]
    if s == '_sigma_baseline' and blk and masked_baseline(q, blk):
        t += ('; left out where sigma_baseline_flag is 1, because the mapping uncertainty of temperature is not '
              'reliable there (known limitation of version 1.0, see the README)')
    return t


class Out:
    """an output file with its dimensions; maps quantized and compressed, series in double precision"""

    def __init__(self, path, dims, attrs, latlon=None):
        self.part = path.with_name(path.name + '.part')
        self.path = path
        self.g = netCDF4.Dataset(self.part, 'w', format='NETCDF4')
        self.g.setncatts(attrs)
        self.latlon = latlon
        if latlon is not None:
            for name, vals, u in (('lat', latlon[0], 'degrees_north'), ('lon', latlon[1], 'degrees_east')):
                self.g.createDimension(name, vals.size)
                x = self.g.createVariable(name, 'f8', (name,))
                x.units = u
                x[:] = vals
        for name, vals, att in dims:
            self.g.createDimension(name, len(vals))
            x = self.g.createVariable(name, str if isinstance(vals[0], str) else 'i4', (name,))
            x[:] = np.array(vals, dtype=object) if isinstance(vals[0], str) else np.asarray(vals)
            x.setncatts(att)

    def var(self, name, dims, units, long_name, description, kind='value', ancillary=None):
        if self.latlon is not None and dims[-2:] == ('lat', 'lon'):
            chunks = tuple(1 if d not in ('lat', 'lon') else self.g.dimensions[d].size for d in dims)
            x = self.g.createVariable(name, 'f4', dims, zlib=True, complevel=4, shuffle=True, fill_value=FILL,
                                      chunksizes=chunks, significant_digits=DIGITS[kind != 'value'], quantize_mode=MODE)
        else:
            x = self.g.createVariable(name, 'f8', dims, fill_value=np.nan)
        x.units, x.long_name, x.description = units, long_name, description
        if ancillary:
            x.ancillary_variables = ' '.join(ancillary)
        return x

    def close(self):
        self.g.close()
        self.part.rename(self.path)


# ------------------------------------------------------------------------------------------------ maps
def maps_file(q, out):
    t0 = time.time()
    A, N = netCDF4.Dataset(SRC / 'OA-Track_all.nc'), netCDF4.Dataset(SRC / 'OA-Track_all_noalk.nc')
    lat, lon = rd(A, 'lat'), rd(A, 'lon')
    scen = scenarios(A)
    alk = f'{q}_sigma_alkalinity' in A.variables
    units, lname = A[q].units, A[q].long_name.split(':')[0]
    yo = np.asarray(A['year_obs'][:]).astype(int)
    states = [str(s) for s in A['state'][:]]
    R = {r: (netCDF4.Dataset(SRC / f'OA-Track_{r}.nc'), netCDF4.Dataset(SRC / f'OA-Track_{r}_noalk.nc')) for r in PRODUCTS}
    f0 = R[PRODUCTS[0]][0]
    dims = [('year', A['year'][:], {'long_name': 'year'}), ('year_ssp', A['year_ssp'][:], {'long_name': 'year'}),
            ('year_obs', yo, {'long_name': 'year of the fCO2-products'}),
            ('state', states, {'long_name': 'preindustrial state (atmospheric CO2 of IPCC AR6: 278.3 ppm in 1750, '
                                            '285.5 ppm in 1850)'})]
    attrs = dict(
        title=f'OA-Track: {lname}, maps 1750-2100, mean of the fCO2-products covering each cell',
        source=SOURCE, references=A.references, constants=A.constants, area_weights=A.area_weights,
        alkalinity_adjustment='included (the best estimate of the paper)', license='CC BY 4.0',
        content='values: <q>_forced (year), <q>_forced_<ssp> (year_ssp), <q>_preindustrial (state), <q>_obs and '
                '<q>_obs_forced (year_obs); each with its 1 sigma terms _sigma_delta (forced and preindustrial), '
                '_sigma_products, _sigma_baseline, _alkalinity_adjustment and _sigma_interannual (forced)',
        uncertainty_formulas=F_MAPS, chain_version=A.chain_version,
        history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} make_oatrack_simple.py from OA-Track_all.nc and the files '
                'of the single fCO2-products')
    W = Out(out / f'OA-Track_maps_{q}.nc', dims, attrs, (lat, lon))
    for name, dm, src_ in (('mask', ('lat', 'lon'), 'mask'), ('product_flag', ('lat', 'lon'), 'product_flag')):
        x = W.g.createVariable(name, 'i1', dm)
        for k in A[src_].ncattrs():
            x.setncattr(k, A[src_].getncattr(k))
        x[:] = A[src_][:]
    for name, dm, src_, f_ in (('xco2', ('year',), 'xco2', A), ('xco2_obs', ('year_obs',), 'xco2_obs', A),
                               ('state_xco2', ('state',), 'state_xco2', f0)) + tuple(
                               (f'xco2_{s}', ('year_ssp',), f'xco2_{s}', A) for s in scen):
        x = W.g.createVariable(name, 'f8', dm)
        x.units, x.long_name = 'ppm', f_[src_].long_name
        x[:] = rd(f_, src_)

    def block(tag, out_name, bname, dim, n, desc, with_delta, with_iv, src_suffix=''):
        """the value and its terms, read and written BLOCK maps at a time"""
        names = [''] + (['_sigma_delta'] if with_delta else []) + ['_sigma_products', '_sigma_baseline'] + (
            ['_alkalinity_adjustment'] if alk else []) + (['_sigma_interannual'] if with_iv else [])
        anc = [f'{q}_{out_name}{s}' for s in names[1:]]
        V = {s: W.var(f'{q}_{out_name}{s}', (dim, 'lat', 'lon'), units, f'{lname}: {bname}' + (f', {SHORT[s]}' if s else ''),
                      desc if not s else unc_text(q, s, out_name), 'value' if not s else 'sigma', anc if not s else None)
             for s in names}
        drop = TFLAG['cells'][None] if masked_baseline(q, out_name) else False
        for i in range(0, n, BLOCK):
            sl = slice(i, min(i + BLOCK, n))
            v = rd(A, f'{q}{tag}', sl)
            sp = rd(A, f'{q}{tag}{src_suffix}_spread', sl)
            F = {'': v, '_sigma_products': sp,
                 '_sigma_baseline': np.where(drop, np.nan, baseline(q, rd(A, f'{q}{tag}{src_suffix}_sigma_mapping_c2', sl),
                                                                    rd(A, f'{q}{tag}{src_suffix}_sigma_constants', sl)))}
            if with_delta:
                s2 = rd(A, f'{q}{tag}_sigma', sl) ** 2 - np.nan_to_num(sp) ** 2
                F['_sigma_delta'] = np.sqrt(np.maximum(s2, 0.0))
            if alk:
                F['_alkalinity_adjustment'] = v - rd(N, f'{q}{tag}', sl)
            if with_iv:
                F['_sigma_interannual'] = rd(A, f'{q}{tag}_sigma_internal', sl)
            for s in names:
                V[s][sl] = np.ma.masked_invalid(np.where(np.isfinite(v), F[s], np.nan).astype('f4'))

    block('', 'forced', BLOCK_NAME['forced'], 'year', A.dimensions['year'].size, VAL['forced'], True, True)
    for s in scen:
        block(f'_{s}', f'forced_{s}', BLOCK_NAME['forced_ssp'].format(s=s), 'year_ssp', A.dimensions['year_ssp'].size,
              VAL['forced_ssp'].format(s=s), True, True)
    block('_state', 'preindustrial', BLOCK_NAME['preindustrial'], 'state', len(states), VAL['preindustrial'], True, False)
    block('_obs', 'obs', BLOCK_NAME['obs'], 'year_obs', yo.size, VAL['obs'], False, False)
    x = W.g.createVariable('n_products', 'i1', ('year_obs', 'lat', 'lon'), zlib=True, complevel=4)
    x.long_name = 'number of fCO2-products in the mean of _obs and _obs_forced'
    x[:] = A['n_products'][:]
    if q in T_DEPENDENT:
        x = W.g.createVariable('sigma_baseline_flag', 'i1', ('lat', 'lon'), zlib=True, complevel=4)
        x.long_name = 'cells where _sigma_baseline is left out'
        x.description = (f'1 where the mapping uncertainty of temperature of one of the fCO2-products exceeds {T_LIMIT:g} degC '
                         'in one of its years: there the conversion of the Omega_arag uncertainty from sea surface '
                         'temperature into a temperature error fails, and _sigma_baseline is left out ' +
                         ('in every block' if q == 'thetao' else 'in the model years (_forced, _forced_<ssp>, _preindustrial)') +
                         ' (known limitation of version 1.0, see the README)')
        x.flag_values, x.flag_meanings = np.array([0, 1], 'i1'), 'reliable left_out'
        x[:] = TFLAG['cells'].astype('i1')

    # the forced state of the observations: per product the fit of its values without the adjustment on its own area,
    # plus the adjustment at the forced state of the year; then the mean of the products present
    yf = yo[yo <= EVAL_YEAR]
    Fp = np.full((len(PRODUCTS), yf.size, lat.size, lon.size), np.nan)
    Ap = np.zeros_like(Fp)
    for k, r in enumerate(PRODUCTS):
        f, g = R[r]
        yy, sel = obs_years(f)
        Y = rd(g, f'{q}_obs')[sel]
        ok = (rd(f, 'mask') > 0) & np.all(np.isfinite(Y), axis=0)
        ii = np.searchsorted(yf, yy)
        Fp[k, ii] = hat_fit(yy, Y, ok)
        if alk:
            assert np.array_equal(ADJ[r][0], yy), f'{r}: years of the adjustment'
            Ap[k, ii] = ADJ[r][1][q]
    Fa = Fp + Ap
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        val = np.nanmean(Fa, 0)
        adj = np.nanmean(np.where(np.isfinite(Fp), Ap, np.nan), 0)
    spread = std_products(Fa)
    io = np.searchsorted(yo, yf)
    base = baseline(q, rd(A, f'{q}_obs_sigma_mapping_c2')[io], rd(A, f'{q}_obs_sigma_constants')[io])
    if masked_baseline(q, 'obs_forced'):
        base = np.where(TFLAG['cells'][None], np.nan, base)
    names = ['', '_sigma_products', '_sigma_baseline'] + (['_alkalinity_adjustment'] if alk else [])
    data = {'': val, '_sigma_products': spread, '_sigma_baseline': base, '_alkalinity_adjustment': adj}
    anc = [f'{q}_obs_forced{s}' for s in names[1:]]
    for s in names:
        x = W.var(f'{q}_obs_forced{s}', ('year_obs', 'lat', 'lon'), units,
                  f'{lname}: {BLOCK_NAME["obs_forced"]}' + (f', {SHORT[s]}' if s else ''),
                  VAL['obs_forced'] if not s else unc_text(q, s, 'obs_forced'), 'value' if not s else 'sigma', anc if not s else None)
        full = np.full((yo.size, lat.size, lon.size), np.nan)
        full[io] = np.where(np.isfinite(val), data[s], np.nan)
        x[:] = np.ma.masked_invalid(full.astype('f4'))
    # check: 2024 against the 2024 trend value of the combined file
    i24 = list(yf).index(EVAL_YEAR)
    ref = rd(A, f'{q}_2024_trend')
    d = np.nanmax(np.abs(val[i24] - ref) / np.maximum(np.abs(ref), 1e-30))
    W.close()
    for f, g in R.values():
        f.close(); g.close()
    A.close(); N.close()
    return f'{q:12s} maps written, {time.time() - t0:5.0f} s; obs_forced 2024 against {q}_2024_trend: max relative difference {d:.1e}'


# ------------------------------------------------------------------------------------------------ global means
# the estimators of the forced state from annual means, unchanged from the preparation cell of the notebook
ENT_CUTOFFS = (15.0, 20.0, 25.0)    # years; in CO2 converted with the mean growth rate of the fit period
YEAR = EVAL_YEAR
XCO2 = {}


def _w_poly(x, x0, deg):
    """weights of a polynomial fit of degree deg in x, evaluated at x0, and the hat matrix (all fitted values)"""
    X = np.column_stack([x ** k for k in range(deg, -1, -1)])
    H = np.linalg.solve(X.T @ X, X.T)
    v0 = np.array([x0 ** k for k in range(deg, -1, -1)])
    return v0 @ H, X @ H


def _smoother(x, cutoff):
    """Enting (1987) smoothing spline on (possibly uneven) x: penalised least squares with second divided differences,
    lambda chosen for 50 % response at the cutoff (in units of x) for the mean spacing; returns the smoother matrix"""
    n_ = x.size
    h = np.diff(x)
    D = np.zeros((n_ - 2, n_))
    for i in range(n_ - 2):
        D[i, i], D[i, i + 2] = 2 / (h[i] * (h[i] + h[i + 1])), 2 / (h[i + 1] * (h[i] + h[i + 1]))
        D[i, i + 1] = -D[i, i] - D[i, i + 2]
    hm = h.mean()
    lam = hm ** 4 / (2.0 - 2.0 * np.cos(2.0 * np.pi * hm / cutoff)) ** 2
    return np.linalg.inv(np.eye(n_) + lam * D.T @ D)


_AR1_CACHE = {}


def _ar1_calibration(n, wq, r_obs, n_sim=20000, seed=1):
    """the lag-1 autocorrelation of the residuals of a quadratic fit to n annual values is biased low, and so is the
    AR(1) standard error built from it. Monte Carlo with AR(1) noise: rho_c, the autocorrelation whose residuals give
    r_obs as median estimate, and the factor by which the standard error the chain computes (residual variance and r
    from the residuals) underestimates the true error of the fit value (weights wq) at rho_c"""
    key = (n, round(float(r_obs), 3), np.round(wq, 6).tobytes(), n_sim, seed)
    if key in _AR1_CACHE:
        return _AR1_CACHE[key]
    rng_ = np.random.default_rng(seed)
    t_ = np.arange(n, dtype='f8')
    X_ = np.vander(t_, 3)
    H_ = X_ @ np.linalg.pinv(X_)
    lag = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))

    def sim(rho):
        e_ = rng_.standard_normal((n_sim, n))
        x_ = np.empty_like(e_)
        x_[:, 0] = e_[:, 0]
        for i_ in range(1, n):
            x_[:, i_] = rho * x_[:, i_ - 1] + np.sqrt(1 - rho ** 2) * e_[:, i_]
        r_ = x_ - x_ @ H_.T
        a_, b_ = r_[:, :-1] - r_[:, :-1].mean(1, keepdims=True), r_[:, 1:] - r_[:, 1:].mean(1, keepdims=True)
        rh = np.clip((a_ * b_).sum(1) / np.sqrt((a_ * a_).sum(1) * (b_ * b_).sum(1)), 0.0, 0.99)
        return x_, r_, rh

    grid = np.linspace(0.0, 0.8, 33)
    med = [np.median(sim(rho)[2]) for rho in grid]
    rho_c = float(np.interp(r_obs, med, grid))
    x_, r_, rh = sim(rho_c)
    s2_ = (r_ ** 2).sum(1) / (n - 3)
    k_ = slice(0, 2000)                               # the chain's estimate on a subsample (memory)
    se_ = np.sqrt(s2_[k_] * np.einsum('i,kij,j->k', wq, rh[k_, None, None] ** lag[None], wq))
    _AR1_CACHE[key] = (rho_c, float((x_ @ wq).std() / np.median(se_)))
    return _AR1_CACHE[key]


def _trend(q, series=None, year=YEAR):
    """the estimators of the state of `year` from annual means, their spread and the AR(1) standard error of the
    quadratic fit"""
    yrs, y_ = _series(q) if series is None else series
    t = yrs - 2000.0
    xc = np.array([XCO2[int(v)] for v in yrs])
    gr = (xc[-1] - xc[0]) / (yrs[-1] - yrs[0])                  # ppm per year over the fit period
    i24 = list(yrs).index(year)
    e = dict(yrs=yrs, y=y_, xco2=xc)
    wq, Hq = _w_poly(t, year - 2000.0, 2)
    e['time quadratic'], e['fit time quadratic'] = float(wq @ y_), Hq @ y_
    wc, Hc = _w_poly(xc - 400.0, XCO2[year] - 400.0, 2)
    e['CO2 quadratic'], e['fit CO2 quadratic'] = float(wc @ y_), Hc @ y_
    for co in ENT_CUTOFFS:
        St, Sc = _smoother(yrs, co), _smoother(xc, co * gr)
        e[f'fit time Enting {co:.0f} yr'], e[f'fit CO2 Enting {co:.0f} yr'] = St @ y_, Sc @ y_
        e[f'time Enting {co:.0f} yr'] = float((St @ y_)[i24])
        e[f'CO2 Enting {co * gr:.0f} ppm'] = float((Sc @ y_)[i24])
    res = y_ - e['fit time quadratic']
    s2 = res @ res / (yrs.size - 3)
    r1 = float(np.clip(np.corrcoef(res[:-1], res[1:])[0, 1], 0.0, 0.99))
    C = s2 * r1 ** np.abs(np.subtract.outer(np.arange(yrs.size), np.arange(yrs.size)))
    alt = [v for k, v in e.items() if isinstance(v, float) and k[:4] in ('time', 'CO2 ')]
    rho_c, k_se = _ar1_calibration(yrs.size, wq, r1)
    e['se_uncorrected'] = float(np.sqrt(wq @ C @ wq))
    e['se'], e['r1'], e['rho_c'], e['k_se'] = e['se_uncorrected'] * k_se, r1, rho_c, k_se
    e['gr'], e['range'] = gr, max(alt) - min(alt)
    return e


def _series(q):
    raise RuntimeError('series must be given')


def noaa_co2():
    """the NOAA global annual means of the run folder, as projection.noaa_co2 reads them for the notebook"""
    path = oa_config.DATA / 'co2_annmean_gl.txt'
    co2 = np.array([[float(v) for v in l.split()[:2]] for l in open(path) if l.strip() and not l.startswith('#')])
    return dict(zip(co2[:, 0].astype(int), co2[:, 1]))


def global_quantity(q):
    """the blocks of one quantity: {name: (dims, array)}, from the four product files"""
    out, info = {}, []
    F = {r: (netCDF4.Dataset(SRC / f'OA-Track_{r}.nc'), netCDF4.Dataset(SRC / f'OA-Track_{r}_noalk.nc')) for r in PRODUCTS}
    f0 = F[PRODUCTS[0]][0]
    alk = f'{q}_gm_common_sigma_alkalinity' in f0.variables
    scen = scenarios(f0, '{q}_gm_common_{s}_global'.replace('{q}', q))
    rms, std = rms_products, std_products

    def combine(blk, dim, getter, with_delta, with_iv):
        """getter(f, g) -> dict of per-product arrays: v, delta, c2, k, iv"""
        P = [getter(*F[r]) for r in PRODUCTS]
        st = lambda key: np.array([p[key] for p in P])
        v = st('v')
        out[(blk, '')] = (dim, np.nanmean(v, 0))
        if with_delta:
            out[(blk, '_sigma_delta')] = (dim, rms(st('delta')))
        out[(blk, '_sigma_products')] = (dim, std(v))
        out[(blk, '_sigma_baseline')] = (dim, baseline(q, np.nanmean(st('c2'), 0),
                                                       None if q in NO_CONSTANTS else np.nanmean(st('k'), 0)))
        if alk:
            out[(blk, '_alkalinity_adjustment')] = (dim, np.nanmean(st('a'), 0))
        if with_iv:
            out[(blk, '_sigma_interannual')] = (dim, rms(st('iv')))
        return P

    def series_getter(s):
        b = f'{q}_gm_common' + (f'_{s}' if s else '')
        return lambda f, g: dict(v=rd(f, f'{b}_global'), delta=rd(f, f'{b}_global_sigma'), c2=rd(f, f'{b}_sigma_mapping_c2'),
                                 k=rd(f, f'{b}_sigma_constants'), iv=rd(f, f'{b}_sigma_internal'),
                                 a=rd(f, f'{b}_global') - rd(g, f'{b}_global'))
    combine('forced', 'year', series_getter(None), True, True)
    for s in scen:
        combine(f'forced_{s}', 'year_ssp', series_getter(s), True, True)
    b = f'{q}_gm_common'
    pre = combine('preindustrial', 'state',
                  lambda f, g: dict(v=rd(f, f'{b}_global_state'), delta=rd(f, f'{b}_global_state_sigma'),
                                    c2=rd(f, f'{b}_sigma_mapping_c2_state'), k=rd(f, f'{b}_sigma_constants_state'),
                                    a=rd(f, f'{b}_global_state') - rd(g, f'{b}_global_state')), True, False)
    yo = np.arange(min(int(F[r][0]['year_obs'][0]) for r in PRODUCTS), max(int(F[r][0]['year_obs'][-1]) for r in PRODUCTS) + 1)

    def on_obs(f, a):
        y = np.asarray(f['year_obs'][:]).astype(int)
        x = np.full(yo.size, np.nan)
        x[np.searchsorted(yo, y)] = a
        return x
    combine('obs', 'year_obs',
            lambda f, g: dict(v=on_obs(f, rd(f, f'{b}_obs')), c2=on_obs(f, rd(f, f'{b}_obs_sigma_mapping_c2')),
                              k=on_obs(f, rd(f, f'{b}_obs_sigma_constants')),
                              a=on_obs(f, rd(f, f'{b}_obs') - rd(g, f'{b}_obs'))), False, False)
    n_products = (np.sum([np.isfinite(on_obs(F[r][0], rd(F[r][0], f'{b}_obs'))) for r in PRODUCTS], 0))

    # the forced state of the observations, its fit uncertainty and the spread of the change from 1750
    yf = yo[yo <= EVAL_YEAR]
    V, A_, SE, RG, C2, K = (np.full((len(PRODUCTS), yf.size), np.nan) for _ in range(6))
    for k, r in enumerate(PRODUCTS):
        f, g = F[r]
        yy, sel = obs_years(f)
        y0 = rd(g, f'{b}_obs')[sel]
        ii = np.searchsorted(yf, yy)
        V[k, ii] = hat_fit(yy, y0, np.isfinite(y0).all())
        A_[k, ii] = ADJ[r][2][q] if alk else 0.0
        C2[k, ii], K[k, ii] = rd(f, f'{b}_obs_sigma_mapping_c2')[sel], rd(f, f'{b}_obs_sigma_constants')[sel]
        for j, y in zip(ii, yy):
            e = _trend(None, series=(yy.astype('f8'), y0), year=int(y))
            SE[k, j], RG[k, j] = e['se'], e['range']
            if int(y) == EVAL_YEAR:
                trend = float(rd(f, f'{b}_2024_trend'))
                info.append(f'    {r:10s} 2024: fit {V[k, j] + A_[k, j]:.6g} against {b}_2024_trend {trend:.6g}; se {e["se"]:.4g} '
                            f'(factor {e["k_se"]:.2f}), half range {e["range"] / 2:.4g}')
    Va = V + A_
    out[('obs_forced', '')] = ('year_obs', on_full(yo, yf, np.nanmean(Va, 0)))
    out[('obs_forced', '_sigma_products')] = ('year_obs', on_full(yo, yf, std(Va)))
    out[('obs_forced', '_sigma_baseline')] = ('year_obs', on_full(yo, yf, baseline(q, np.nanmean(C2, 0),
                                                                                  None if q in NO_CONSTANTS else np.nanmean(K, 0))))
    if alk:
        out[('obs_forced', '_alkalinity_adjustment')] = ('year_obs', on_full(yo, yf, np.nanmean(np.where(np.isfinite(V), A_, np.nan), 0)))
    out[('obs_forced', '_sigma_fit')] = ('year_obs', on_full(yo, yf, np.hypot(rms(SE), rms(RG / 2))))
    p1750 = np.array([p['v'][0] for p in pre])[:, None]
    out[('obs_forced', '_change_sigma_products')] = ('year_obs', on_full(yo, yf, std(Va - p1750)))
    info.append(f'    mean       2024: sigma_fit {out[("obs_forced", "_sigma_fit")][1][list(yo).index(EVAL_YEAR)]:.4g}, '
                f'{100 * out[("obs_forced", "_sigma_fit")][1][list(yo).index(EVAL_YEAR)] / out[("preindustrial", "")][1][0]:.3f} '
                'percentage points of the decline from 1750')
    for f, g in F.values():
        f.close(); g.close()
    return q, out, yo, n_products, info


def on_full(yo, yf, a):
    x = np.full(yo.size, np.nan)
    x[np.searchsorted(yo, yf)] = a
    return x


def global_file(qs, out, jobs):
    t0 = time.time()
    f0 = netCDF4.Dataset(SRC / f'OA-Track_{PRODUCTS[0]}.nc')
    XCO2.update(noaa_co2())
    with mp_.get_context('fork').Pool(jobs) as pool:
        res = pool.map(global_quantity, qs)
    yo = res[0][2]
    A = netCDF4.Dataset(SRC / 'OA-Track_all.nc')
    assert np.array_equal(np.asarray(A['year_obs'][:]).astype(int), yo), 'year_obs of OA-Track_all.nc'
    states = [str(s) for s in f0['state'][:]]
    dims = [('year', f0['year'][:], {'long_name': 'year'}), ('year_ssp', f0['year_ssp'][:], {'long_name': 'year'}),
            ('year_obs', yo, {'long_name': 'year of the fCO2-products'}),
            ('state', states, {'long_name': 'preindustrial state (atmospheric CO2 of IPCC AR6: 278.3 ppm in 1750, '
                                            '285.5 ppm in 1850)'})]
    attrs = dict(
        title='OA-Track: global means 1750-2100 on the common area of the paper, mean of the four fCO2-products',
        source=SOURCE, references=A.references, constants=f0.constants, area_weights=f0.area_weights,
        alkalinity_adjustment='included (the best estimate of the paper)',
        area='the common area of the paper: the cells that all four fCO2-products cover in every year and all 18 ESMs '
             'have; the extension to the whole ocean (Methods of the paper) is not included',
        method_forced='the global emergent constraint of the paper (the global mean of every ESM regressed on its global '
                      'warming and carbon uptake capacity)', license='CC BY 4.0',
        content='values: <q>_forced (year), <q>_forced_<ssp> (year_ssp), <q>_preindustrial (state), <q>_obs and '
                '<q>_obs_forced (year_obs); each with its 1 sigma terms _sigma_delta (forced and preindustrial), '
                '_sigma_products, _sigma_baseline, _alkalinity_adjustment and _sigma_interannual (forced); '
                '<q>_obs_forced also _sigma_fit and _change_sigma_products',
        uncertainty_formulas=F_GLOBAL, chain_version=f0.chain_version,
        history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} make_oatrack_simple.py from the files of the four fCO2-products')
    W = Out(out / 'OA-Track_global_means.nc', dims, attrs)
    for name, dm, f_ in (('xco2', ('year',), A), ('xco2_obs', ('year_obs',), A), ('state_xco2', ('state',), f0)) + tuple(
            (f'xco2_{s}', ('year_ssp',), A) for s in scenarios(A)):
        x = W.g.createVariable(name, 'f8', dm)
        x.units, x.long_name = 'ppm', f_[name].long_name
        x[:] = rd(f_, name)
    for q, o, _, n_products, info in res:
        units, lname = A[q].units, A[q].long_name.split(':')[0]
        for (blk, suf), (dim, a) in o.items():
            ssp = blk.startswith('forced_ssp')
            bname = BLOCK_NAME['forced_ssp'].format(s=blk[7:]) if ssp else BLOCK_NAME[blk]
            desc = unc_text(q, suf) if suf else (VAL['forced_ssp'].format(s=blk[7:]) if ssp else VAL[blk])
            anc = [f'{q}_{b_}{s_}' for b_, s_ in o if b_ == blk and s_] if not suf else None
            x = W.var(f'{q}_{blk}{suf}', (dim,), units, f'{lname}, global mean: {bname}' + (f', {SHORT[suf]}' if suf else ''),
                      desc, 'value' if not suf else 'sigma', anc)
            x[:] = a
        print(f'  {q}: global means', flush=True)
        for l in info:
            print(l, flush=True)
    x = W.g.createVariable('n_products', 'i1', ('year_obs',))
    x.long_name = 'number of fCO2-products in the mean of _obs and _obs_forced'
    x[:] = res[0][3]
    W.close()
    A.close(); f0.close()
    return f'global means written, {time.time() - t0:.0f} s'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(oa_config.DATA / 'oatrack_simple'))
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--only', choices=('maps', 'global'))
    ap.add_argument('--quantities', nargs='+')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(SRC / 'OA-Track_all.nc') as A:
        qs = a.quantities or quantities(A)
        adjusted = [q for q in qs if f'{q}_sigma_alkalinity' in A.variables]
    print(f'{len(qs)} quantities: {", ".join(qs)} -> {out}', flush=True)
    assert set(adjusted) <= set(ma.DELTA_Q), f'adjusted quantities without a chemistry change: {set(adjusted) - set(ma.DELTA_Q)}'
    if adjusted:
        # before the pools below, which inherit it
        with mp_.get_context('fork').Pool(min(a.jobs, len(PRODUCTS))) as pool:
            for r, yy, D, gm, dt in pool.imap_unordered(adjustment_at_fit, PRODUCTS):
                ADJ[r] = (yy, D, gm)
                print(f'{r:10s} alkalinity adjustment at the forced state of {yy[0]}-{yy[-1]}, {dt:4.0f} s; '
                      f'omega_arag common-area mean in {yy[-1]}: {gm["omega_arag"][-1]:.6f}', flush=True)
    if a.only != 'global':
        TFLAG['cells'], n = temperature_flag()                  # before the pool below, which inherits it
        print(f'sigma_baseline_flag: temperature error above {T_LIMIT:g} degC in ' +
              ', '.join(f'{r} {k}' for r, k in n.items()) + f' cells; {int(TFLAG["cells"].sum())} cells in all', flush=True)
        with mp_.get_context('fork').Pool(a.jobs) as pool:
            for line in pool.imap_unordered(maps_file_args, [(q, out) for q in qs]):
                print(line, flush=True)
    if a.only != 'maps':
        print(global_file(qs, out, a.jobs), flush=True)


def maps_file_args(args):
    return maps_file(*args)


if __name__ == '__main__':
    main()
