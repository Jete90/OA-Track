#!/usr/bin/env python
"""
make_product_alkalinity.py  (data product: the alkalinity adjustment of the best estimate)
==========================
The adjustment of the carbonate system for the surface alkalinity rise that the observation-based products and 15
of the 18 ESMs miss (structural_adjustments.py, Methods 'Alkalinity adjustment'), part of the best estimate of the
paper. Per reference, carbonate_system_<reference>_alkalinity.nc, and for the combined files,
carbonate_system_<common|all>_alkalinity.nc, beside the product files: the change of every indicator, to be added
to the corresponding variable of the product. These are internal files of the chain; the released OA-Track files,
with the adjustment applied, are written by make_oatrack.py. 1 sigma of every change = its absolute value.

Missing alkalinity, per cell and year:  dTA(t, x) = pattern(x) * m(t)
    pattern  the feedback pattern of the paper (feedback_pattern.npz, key 'pattern': four ESMs with Omega-dependent
             calcification, mean 1 over the common area), nearest neighbour on the product grid, cells without a
             value the zonal mean
    m(t)     the common-area missing alkalinity (umol/kg), relative to the time t_c the product's alkalinity
             stands for (the centre of the bias window; Jiang et al. 2023: 2010, the decade all their fields are
             adjusted to), one continuous series per product and scenario from structural_adjustments.npz
             (structural_adjustments.py): the real rise of the common-area nTA R(t), the common-area mean of the four
             ESMs with Omega-dependent calcification scaled to the observed rate of 1991-2021 (Barrett et al. 2025),
             minus the content C(t) of the multi-model mean in the model years (the feedback models' share), nothing
             subtracted in the product years (the products' TA has no trend):
               model years (before the first product year)   m = [R(t) - R(t_c)] - [C(t) - C(t_c)]
               product years                                  m = R(t) - R(t_c)      (Jiang: 0, the 2010 decade)
               scenario years                                 m = [R_s(t) - R(t_c)] - [C_s(t) - C_s(t_c)]
               named states (preindustrial CO2)               m of the model years at the state's common-area
                                                              Omega_arag (m against Omega_arag of 1750-1870)
             Jiang et al.: the means of the four products' R and C with t_c = 2010
Indicators at the adjusted state: TA + dTA, DIC solved by Newton's method to the same pCO2 (the surface ocean
takes up CO2 to stay with the atmosphere; |dpCO2| < calc_pi_xco2.PCO2_TOL), every indicator with mocsy as in
calc_pi_xco2.py (constants of the run, GLODAP Si and PO4); the change = adjusted minus unadjusted, both from the
annual-mean state of the product (T, S, TA, DIC). pCO2, fCO2, CO2* and T, S do not change.

Output (per reference; combined files: <q>_<version>_alk ... and the series over the file's mask):
    talk_missing(year), _<ssp>(year_ssp)       m(t), umol/kg (common-area mean)
    <q>_alk(year, lat, lon)                    change of <q>; <q>: talk dic omega_arag omega_calc ph hplus
                                               hplus_free co3 hco3 revelle
    <q>_state_alk(state, lat, lon), <q>_2024_trend_alk(lat, lon), <q>_<ssp>_alk(year_ssp, lat, lon)
    <q>_gm_alk, <q>_gm_common_alk (year), <q>_gm_<ssp>_alk, <q>_gm_common_<ssp>_alk (year_ssp): area means of the
    change on the product's own area and on the common area

Needs: carbonate_system_<ref>.nc (make_product.py), structural_adjustments.npz and feedback_pattern.npz.
Usage (code/):  OA_RUN=u74 python -u make_product_alkalinity.py --reference UExP         (one per process)
                OA_RUN=u74 python -u make_product_alkalinity.py --combined               (after make_product --combine-only)
"""
from __future__ import annotations

import argparse
import datetime
import warnings

import netCDF4
import numpy as np

import oa_config
import calc_bias_adjusted as cb
import calc_omega_arag_obs as co
import calc_pi_xco2 as px
import make_product as mp
import state_constraint as sc

OUT = mp.OUT
STRUCT = oa_config.DATA / 'structural' / 'structural_adjustments.npz'
PATTERN = oa_config.DATA / 'structural' / 'feedback_pattern.npz'
DELTA_Q = ['talk', 'dic', 'omega_arag', 'omega_calc', 'ph', 'hplus', 'hplus_free', 'co3', 'hco3', 'revelle']
UNCHANGED = 'pco2, fco2, co2 (constant pCO2), thetao, so'
JIANG_CENTRE = 2010.0          # Jiang et al. (2023) adjust all fields to the observational decade around 2010
EVAL_YEAR = mp.EVAL_YEAR
FILL = mp.FILL
# the patterns of the four feedback models alone (feedback_pattern.npz keys). Their spread enters the 1 sigma of the
# adjustment (make_oatrack.py, structural_adjustments.py): sqrt(adjustment^2 + s^2), s the standard deviation (ddof 1)
# of the adjustment over these four, each scaled to Barrett et al. with its own m(t) (structural_adjustments_<key>.npz)
PATTERN_ALTERNATIVES = ('pattern_GFDL-CM4', 'pattern_GFDL-ESM4', 'pattern_UKESM1-0-LL', 'pattern_esm2m')


def struct_file(key='pattern'):
    """structural_adjustments.npz of the central pattern or structural_adjustments_<key>.npz of a sensitivity"""
    return STRUCT if key == 'pattern' else STRUCT.with_name(f'structural_adjustments_{key}.npz')


# ------------------------------------------------------------------------------------------------ parameters
class Params:
    """the series m(t) of structural_adjustments.npz (central pattern; key: the structural_adjustments_<key>.npz of a
    sensitivity pattern): per product the model formula, the product formula, the best estimate and the scenarios on
    alk_years (1750-2100), and R and C for Jiang et al."""

    def __init__(self, key='pattern'):
        path = struct_file(key)
        assert path.exists(), f'{path} missing: run run_structural.sh structural'
        z = np.load(path)
        self.key = key
        self.products = [str(q) for q in z['products']]
        need = ('alk_years', 'alk_m_best', 'alk_m_model', 'alk_m_product', 'alk_R_best', 'alk_C_best', 'alk_window_centre',
                'alk_first_product_year', 'alk_rate_common', 'alk_scale')
        miss = [k for k in need if k not in z.files]
        assert not miss, f'{path}: no {miss}: rerun structural_adjustments.py (7 November 2026 or later)'
        src = str(z['pattern_source']) if 'pattern_source' in z.files else '?'
        assert src == f'feedback_pattern.npz ({key})', f'{path} is not the run of {key} ({src}): rerun structural_adjustments.py'
        self.series_source = str(z['alk_series_source']) if 'alk_series_source' in z.files else '?'
        self.years = np.asarray(z['alk_years']).astype(int)
        rows = lambda key: dict(zip(self.products, np.asarray(z[key], 'f8')))
        self.m_best, self.m_model, self.m_product = rows('alk_m_best'), rows('alk_m_model'), rows('alk_m_product')
        self.R, self.C = rows('alk_R_best'), rows('alk_C_best')
        self.R_s = {s_: rows(f'alk_R_{s_}') for s_ in mp.SCENARIOS if f'alk_R_{s_}' in z.files}
        self.C_s = {s_: rows(f'alk_C_{s_}') for s_ in mp.SCENARIOS if f'alk_C_{s_}' in z.files}
        self.m_s = {s_: rows(f'alk_m_{s_}') for s_ in mp.SCENARIOS if f'alk_m_{s_}' in z.files}
        self.tc = dict(zip(self.products, np.asarray(z['alk_window_centre'], 'f8')))
        self.first = dict(zip(self.products, np.asarray(z['alk_first_product_year']).astype(int)))
        self.M = dict(zip(self.products, np.asarray(z['alk_miss_model'], 'f8')))
        self.r = float(z['alk_rate_common'])
        self.scale = float(z['alk_scale'])
        self.created = str(z['created'])
        self.source = src

    def on(self, series, years):
        """series (on self.years) at years; NaN outside"""
        out = np.full(len(years), np.nan)
        k = np.searchsorted(self.years, years)
        ok = (k < self.years.size) & (self.years[np.minimum(k, self.years.size - 1)] == years)
        out[ok] = series[k[ok]]
        return out

    def mean4(self, d):
        return np.nanmean([d[q] for q in self.products], axis=0)

    def jiang(self, years, ys):
        """m(t) of the best estimate (model years; the decade 2005-2014 is the product: 0) and of the scenarios for
        Jiang et al.: the means of the four products' R and C, t_c = 2010"""
        C = self.mean4(self.C)
        R245 = self.mean4(self.R_s['ssp245'])                 # R_best to 2024, the continuation beyond
        R_tc = float(np.interp(JIANG_CENTRE, self.years, R245))
        C_tc = float(np.interp(JIANG_CENTRE, self.years, np.where(np.isfinite(C), C, np.nan)))
        m = self.on((R245 - R_tc) - (C - C_tc), years)
        scen = {}
        for s_ in self.m_s:
            Rs, Cs = self.mean4(self.R_s[s_]), self.mean4(self.C_s[s_])
            scen[s_] = self.on((Rs - R_tc) - (Cs - float(np.interp(JIANG_CENTRE, self.years, Cs))), ys)
        return m, scen, R_tc


def m_states(om_states, years, om, m):
    """m of the named preindustrial states: m of the model years as a function of the common-area Omega_arag of
    1750-1870 (monotonic), interpolated at the state's Omega_arag, linear extrapolation beyond the range"""
    sel = (years <= 1870) & np.isfinite(om) & np.isfinite(m)
    x, y = -om[sel], m[sel]                                  # x increasing with time (Omega falls)
    o = np.argsort(x); x, y = x[o], y[o]
    xs = -np.asarray(om_states, 'f8')
    out = np.interp(xs, x, y)
    lo, hi = xs < x[0], xs > x[-1]
    if lo.any():
        sl = (y[5] - y[0]) / (x[5] - x[0])
        out[lo] = y[0] + sl * (xs[lo] - x[0])
    if hi.any():
        sl = (y[-1] - y[-6]) / (x[-1] - x[-6])
        out[hi] = y[-1] + sl * (xs[hi] - x[-1])
    return out


def load_pattern(lat, lon, key='pattern'):
    """the feedback pattern (key of feedback_pattern.npz) on (lat, lon) by nearest neighbour; NaN cells: the zonal
    mean of the row"""
    z = np.load(PATTERN)
    jy = co.nearest_index(np.asarray(z['lat'], 'f8'), lat)
    ix = co.nearest_index(np.asarray(z['lon'], 'f8'), lon, periodic=True)
    p = np.asarray(z[key], 'f8')[np.ix_(jy, ix)]
    with np.errstate(invalid='ignore'):
        zon = np.nanmean(p, axis=1)
    zon = np.interp(lat, lat[np.isfinite(zon)], zon[np.isfinite(zon)])
    return np.where(np.isfinite(p), p, zon[:, None] * np.ones(p.shape)), str(z['created'])


# ------------------------------------------------------------------------------------------------ chemistry
class Chem:
    """indicator changes for a TA increase at constant pCO2, on one grid"""

    def __init__(self, lat, lon):
        self.si, self.po4 = co.load_nutrients(lat, lon)
        self.lat2 = np.broadcast_to(lat[:, None], (lat.size, lon.size))
        self.resid = 0.0
        self.iters = {}

    def deltas(self, t, s, ta, dic, dta):
        """t, s, ta, dic (umol/kg), dta (umol/kg): 2-D fields; returns {q: change} for DELTA_Q (NaN where invalid)"""
        return self.deltas_multi(t, s, ta, dic, [dta])[0]

    def deltas_multi(self, t, s, ta, dic, dtas):
        """as deltas for several TA changes of the same state (one list entry each, in one Newton solve); a cell is
        computed only where every change is finite, so that all results share the same cells"""
        n = len(dtas)
        out = [{q: np.full(t.shape, np.nan) for q in DELTA_Q} for _ in range(n)]
        ok = (np.isfinite(t) & np.isfinite(s) & np.isfinite(ta) & np.isfinite(dic) & (s > 0)
              & (ta > 0) & (dic > 0) & np.isfinite(self.si) & np.isfinite(self.po4))
        for dta in dtas:
            ok &= np.isfinite(dta)
        if not ok.any():
            return out
        rep = lambda a_: np.tile(a_[ok], n)
        tt, ss, la = rep(t), rep(s), rep(self.lat2)
        a0, d0 = rep(ta) * 1e-6, rep(dic) * 1e-6
        sil, phos = rep(self.si) * 1e-6, rep(self.po4) * 1e-6
        k1 = slice(0, int(ok.sum()))                                     # the base state once, then repeated
        r0 = tuple(np.tile(np.asarray(x_, 'f8'), n) for x_ in px.mvars_all(tt[k1], ss[k1], d0[k1], a0[k1], sil[k1], phos[k1], la[k1]))
        p0 = np.asarray(r0[1], 'f8')
        dt_all = np.concatenate([dta[ok] for dta in dtas])
        a1 = a0 + dt_all * 1e-6
        d = d0.copy()
        r1 = px.mvars_all(tt, ss, d, a1, sil, phos, la)
        p, rv = np.asarray(r1[1], 'f8'), np.asarray(r1[8], 'f8')
        it = 0
        for it in range(1, px.N_ITER_MAX + 1):
            with np.errstate(divide='ignore', invalid='ignore'):
                d = d + (p0 - p) * d / (rv * p)                    # Newton with the Revelle factor (as calc_pi_xco2)
            r1 = px.mvars_all(tt, ss, d, a1, sil, phos, la)
            p, rv = np.asarray(r1[1], 'f8'), np.asarray(r1[8], 'f8')
            good = px.valid_omega(np.asarray(r1[6], 'f8'), np.asarray(r1[0], 'f8')) & np.isfinite(p)
            err = np.abs(p - p0)[good]
            if err.size == 0 or float(err.max()) < px.PCO2_TOL:
                break
        self.iters[it] = self.iters.get(it, 0) + 1
        good = (px.valid_omega(np.asarray(r1[6], 'f8'), np.asarray(r1[0], 'f8')) & px.valid_omega(np.asarray(r0[6], 'f8'), np.asarray(r0[0], 'f8'))
                & np.isfinite(p) & (np.abs(p - p0) < 10 * px.PCO2_TOL))
        if good.any():
            self.resid = max(self.resid, float(np.max(np.abs(p - p0)[good])))
        free = px.total_to_free(tt, ss)
        ch = {}
        for q in ('omega_arag', 'omega_calc', 'ph', 'co3', 'hco3', 'revelle'):
            i, _, scale = px.INDICATORS[q]
            ch[q] = np.where(good, (np.asarray(r1[i], 'f8') - np.asarray(r0[i], 'f8')) * scale, np.nan)
        h0, h1 = 10.0 ** (-np.asarray(r0[0], 'f8')) * 1e9, 10.0 ** (-np.asarray(r1[0], 'f8')) * 1e9
        ch['hplus'] = np.where(good, h1 - h0, np.nan)
        ch['hplus_free'] = np.where(good, (h1 - h0) * free, np.nan)
        ch['dic'] = np.where(good, (d - d0) * 1e6, np.nan)
        ch['talk'] = np.where(good, dt_all, np.nan)
        m = int(ok.sum())
        for k in range(n):
            for q in DELTA_Q:
                out[k][q][ok] = ch[q][k * m:(k + 1) * m]
        return out


# ------------------------------------------------------------------------------------------------ output
class Writer:
    def __init__(self, path, lat, lon, attrs):
        self.g = cb.new_nc(path, lat, lon, attrs)
        self.lat, self.lon = lat, lon

    def dim(self, name, values):
        self.g.createDimension(name, len(values))
        x = self.g.createVariable(name, 'i4' if np.issubdtype(np.asarray(values).dtype, np.integer) else str, (name,))
        x[:] = np.asarray(values) if np.issubdtype(np.asarray(values).dtype, np.integer) else np.array(values, dtype=object)

    def field(self, name, dims, unit, long_name):
        chunks = tuple(1 if d not in ('lat', 'lon') else (self.lat.size if d == 'lat' else self.lon.size) for d in dims)
        x = self.g.createVariable(name, 'f4', dims, zlib=True, complevel=4, fill_value=FILL, chunksizes=chunks)
        x.units, x.long_name = unit, long_name
        return x

    def series(self, name, dims, data, unit, long_name):
        x = self.g.createVariable(name, 'f8', dims, fill_value=np.nan)
        x.units, x.long_name = unit, long_name
        x[...] = data


def unit_of(q):
    return mp.UNITS[q]


def run_maps(W, chem, label, dim, years, m, read_state, pattern, masks, tag):
    """the changes of every year of a map set: read_state(i) -> (t, s, ta, dic); m(t) per year; written as
    <q><tag>_alk(dim, lat, lon) and the series over masks {suffix: mask}"""
    fields = {q: W.field(f'{q}{tag}_alk', (dim, 'lat', 'lon'), unit_of(q),
                         f'{mp.LONG[q]}{label}: change from the alkalinity adjustment (add to the product; 1 sigma = |change|)')
              for q in DELTA_Q}
    ser = {(q, sfx): np.full(len(years), np.nan) for q in DELTA_Q for sfx in masks}
    for i in range(len(years)):
        t, s, ta, dic = read_state(i)
        dta = pattern * m[i] if np.ndim(m[i]) == 0 else m[i]
        D = chem.deltas(t, s, ta, dic, dta)
        for q in DELTA_Q:
            fields[q][i] = np.ma.masked_invalid(D[q].astype('f4'))
            for sfx, mk in masks.items():
                ser[(q, sfx)][i] = float(sc.area_mean(D[q], W.lat, W.lon, mk))
    return ser


def write_series(W, ser, dim, tag, masks_label):
    for (q, sfx), a in ser.items():
        W.series(f'{q}_gm{sfx}{tag}_alk', (dim,), a, unit_of(q),
                 f'{mp.LONG[q]}: area mean of the change from the alkalinity adjustment ({masks_label[sfx]})')


# ------------------------------------------------------------------------------------------------ per reference
def build(ref, P):
    src_path = OUT / f'carbonate_system_{ref}.nc'
    F = netCDF4.Dataset(src_path)
    rd = lambda name, i=None: np.ma.filled((F[name][:] if i is None else F[name][i]).astype('f8'), np.nan)
    lat, lon = np.asarray(F['lat'][:], 'f8'), np.asarray(F['lon'][:], 'f8')
    years = np.asarray(F['year'][:]).astype(int)
    source = np.asarray(F['source'][:]).astype(int)
    own = np.asarray(F['mask'][:]).astype(bool)
    com = np.asarray(F['mask_common'][:]).astype(bool)
    pattern, pat_created = load_pattern(lat, lon)
    pattern = np.where(own, pattern, np.nan)
    om = rd('omega_arag_gm_common_global')
    prod = source == 4
    scen = [s for s in mp.SCENARIOS if f'omega_arag_{s}' in F.variables]
    ys = np.asarray(F['year_ssp'][:]).astype(int) if scen else np.array([], int)
    if ref == 'Jiang':
        tc = JIANG_CENTRE
        m, m_scen, R_tc = P.jiang(years, ys)
        m[prod] = 0.0                                         # the decade 2005-2014 is the product itself
        M = float(-m[0])
    else:
        tc, M = P.tc[ref], P.M[ref]
        # m_best switches from the model formula to the product formula in the first year of the bias window; the file
        # switches from the constrained ESMs to the product in the product's first year: both must be the same year, so
        # that each formula goes with its own base field
        assert prod.any() and int(years[prod][0]) == P.first[ref], \
            f'{ref}: product from {years[prod][0] if prod.any() else "-"}, m(t) switches in {P.first[ref]}'
        m = P.on(P.m_best[ref], years)
        m_scen = {s: P.on(P.m_s[s][ref], ys) for s in scen if s in P.m_s}
    assert np.all(np.isfinite(m)), f'{ref}: m(t) missing for {years[~np.isfinite(m)]}'
    print(f'\n=== {ref}: t_c {tc:.1f}, missing rise 1750 -> t_c {M:.2f} umol/kg, observed common-area rate {P.r:.4f} umol/kg/yr '
          f'(scale of the feedback models {P.scale:.3f}); m(1750) {m[0]:+.2f}, m(first product year '
          f'{years[prod][0] if prod.any() else "-"}) {m[prod][0] if prod.any() else np.nan:+.2f}, m({years[-1]}) {m[-1]:+.2f} umol/kg',
          flush=True)
    chem = Chem(lat, lon)
    W = Writer(OUT / f'carbonate_system_{ref}_alkalinity.nc', lat, lon, dict(
        title=f'Alkalinity adjustment of the surface ocean carbonate system, reference {mp.LABEL[ref]} (part of the best estimate)',
        method=__doc__.split('Output (')[0].split('\n', 3)[3].strip(), reference=mp.LABEL[ref],
        constants=oa_config.CONSTANTS, area_weights=oa_config.WEIGHTS_TEXT, chain_version=oa_config.VERSION, unchanged=UNCHANGED,
        t_c=tc, missing_before_window_umol_kg=M, rate_umol_kg_yr=P.r, feedback_models_scale=P.scale,
        structural_adjustments=f'{STRUCT.name} ({P.created}, {P.source})', feedback_pattern=f'{PATTERN.name} ({pat_created})',
        history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} make_product_alkalinity.py'))
    W.dim('year', years)
    x = W.g.createVariable('source', 'i1', ('year',)); x.flag_values, x.flag_meanings = '1 2 4', mp.SOURCE_FLAGS; x[:] = source
    for nm, a in (('mask', own), ('mask_common', com)):
        W.g.createVariable(nm, 'i1', ('lat', 'lon'))[:] = a.astype('i1')
    x = W.g.createVariable('feedback_pattern', 'f4', ('lat', 'lon'), fill_value=FILL)
    x.long_name = 'spatial pattern of the missing alkalinity (mean 1 over the common area)'; x[:] = np.ma.masked_invalid(pattern)
    W.series('talk_missing', ('year',), m, 'umol kg-1', 'missing alkalinity m(t), common-area mean (the maps: m x pattern)')
    masks = {'': own, '_common': own & com}
    labels = {'': 'own area', '_common': 'common area'}
    st = lambda i: tuple(rd(v, i) for v in ('thetao', 'so', 'talk', 'dic'))
    ser = run_maps(W, chem, '', 'year', years, m, st, pattern, masks, '')
    write_series(W, ser, 'year', '', labels)
    print(f'  best estimate: {years[0]}-{years[-1]}; Omega_arag change (common area) 1750 {ser[("omega_arag", "_common")][0]:+.4f}, '
          f'{years[-1]} {ser[("omega_arag", "_common")][-1]:+.4f}', flush=True)
    # named states (preindustrial CO2)
    states = [str(s) for s in F['state'][:]]
    W.dim('state', states)
    oms = np.array([float(sc.area_mean(rd('omega_arag_state', j), lat, lon, own & com)) for j in range(len(states))])
    ms = m_states(oms, years, om, m)
    W.series('talk_missing_state', ('state',), ms, 'umol kg-1', 'missing alkalinity of the named states, common-area mean')
    sst = lambda j: tuple(rd(f'{v}_state', j) for v in ('thetao', 'so', 'talk', 'dic'))
    ser = run_maps(W, chem, ', named states', 'state', states, ms, sst, pattern, masks, '_state')
    write_series(W, ser, 'state', '_state', labels)
    # the 2024 trend map (Jiang: the decade state, no change)
    if ref != 'Jiang':
        m24 = float(m[years == EVAL_YEAR][0])
        D = chem.deltas(*(rd(f'{v}_{EVAL_YEAR}_trend') for v in ('thetao', 'so', 'talk', 'dic')), pattern * m24)
        for q in DELTA_Q:
            x = W.field(f'{q}_{EVAL_YEAR}_trend_alk', ('lat', 'lon'), unit_of(q),
                        f'{mp.LONG[q]} in {EVAL_YEAR} (quadratic fit of the product years): change from the alkalinity adjustment')
            x[:] = np.ma.masked_invalid(D[q].astype('f4'))
            W.series(f'{q}_gm_common_{EVAL_YEAR}_trend_alk', (), float(sc.area_mean(D[q], lat, lon, own & com)), unit_of(q),
                     f'{mp.LONG[q]} in {EVAL_YEAR}: common-area mean of the change')
        print(f'  {EVAL_YEAR} trend: m {m24:+.2f} umol/kg, Omega_arag change (common area) '
              f'{float(sc.area_mean(D["omega_arag"], lat, lon, own & com)):+.4f}', flush=True)
    # scenarios: the series of structural_adjustments.npz (the CMIP6 feedback models of the scenario)
    if scen:
        W.dim('year_ssp', ys)
        for s in scen:
            if s not in m_scen:
                print(f'  {s}: no m(t) in {STRUCT.name}, left out', flush=True)
                continue
            msc = m_scen[s]
            assert np.all(np.isfinite(msc)), f'{ref} {s}: m(t) missing for {ys[~np.isfinite(msc)]}'
            W.series(f'talk_missing_{s}', ('year_ssp',), msc, 'umol kg-1', f'missing alkalinity m(t), {s}, common-area mean')
            sst = lambda i, s=s: tuple(rd(f'{v}_{s}', i) for v in ('thetao', 'so', 'talk', 'dic'))
            ser = run_maps(W, chem, f', {s}', 'year_ssp', ys, msc, sst, pattern, masks, f'_{s}')
            write_series(W, ser, 'year_ssp', f'_{s}', labels)
            j24 = int(np.searchsorted(ys, EVAL_YEAR))
            print(f'  {s}: m 2024 {msc[j24]:+.2f}, 2100 {msc[-1]:+.2f} umol/kg; Omega_arag change (common area) 2024 '
                  f'{ser[("omega_arag", "_common")][j24]:+.4f}, 2100 {ser[("omega_arag", "_common")][-1]:+.4f}', flush=True)
    W.g.newton = (f'max |pCO2 - pCO2_0| {chem.resid:.2e} uatm; Newton iterations used: '
                  + ', '.join(f'{k_} ({v_} fields)' for k_, v_ in sorted(chem.iters.items())))
    print(f'  Newton: {W.g.newton}', flush=True)
    W.g.close()
    F.close()
    print(f'  written {OUT / f"carbonate_system_{ref}_alkalinity.nc"}', flush=True)


# ------------------------------------------------------------------------------------------------ combined files
class CellPath:
    """m x pattern of year i for a combined version: the product formula where products enter the mean, the model
    formula elsewhere"""

    def __init__(self, nprod, pattern, m_mod, m_prod):
        self.nprod, self.pattern, self.m_mod, self.m_prod = nprod, pattern, m_mod, m_prod

    def __getitem__(self, i):
        n = np.asarray(self.nprod[i]).astype(int)
        return self.pattern * np.where(n > 0, self.m_prod[i], self.m_mod[i])


def build_combined(P):
    """carbonate_system_<common|all>_alkalinity.nc: the same for the combined maps; m(t) the means of the four
    products' series; per cell and year the product formula where n_products > 0, the model formula elsewhere"""
    tc = float(np.mean([P.tc[q] for q in mp.PRODUCTS]))
    M = float(np.mean([P.M[q] for q in mp.PRODUCTS]))
    for kind in ('common', 'all'):
        F = netCDF4.Dataset(OUT / f'carbonate_system_{kind}.nc')
        rd = lambda name, i=None: np.ma.filled((F[name][:] if i is None else F[name][i]).astype('f8'), np.nan)
        lat, lon = np.asarray(F['lat'][:], 'f8'), np.asarray(F['lon'][:], 'f8')
        years = np.asarray(F['year'][:]).astype(int)
        keep = np.asarray(F['mask'][:]).astype(bool)
        pattern, pat_created = load_pattern(lat, lon)
        pattern = np.where(keep, pattern, np.nan)
        chem = Chem(lat, lon)
        W = Writer(OUT / f'carbonate_system_{kind}_alkalinity.nc', lat, lon, dict(
            title=f'Alkalinity adjustment of the combined carbonate system ({kind}; part of the best estimate)',
            method=__doc__.split('Output (')[0].split('\n', 3)[3].strip(), constants=oa_config.CONSTANTS,
            area_weights=oa_config.WEIGHTS_TEXT, chain_version=oa_config.VERSION, unchanged=UNCHANGED, t_c=tc, missing_before_window_umol_kg=M,
            rate_umol_kg_yr=P.r, feedback_models_scale=P.scale, parameters='means of the four products',
            structural_adjustments=f'{STRUCT.name} ({P.created}, {P.source})', feedback_pattern=f'{PATTERN.name} ({pat_created})',
            history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} make_product_alkalinity.py'))
        W.dim('year', years)
        W.g.createVariable('mask', 'i1', ('lat', 'lon'))[:] = keep.astype('i1')
        masks, labels = {'': keep}, {'': f'{kind} mask'}
        print(f'\n=== combined {kind}: t_c {tc:.1f}, M {M:.2f}', flush=True)
        m_mod_all = P.on(P.mean4(P.m_model), years)
        m_prod_all = P.on(P.mean4(P.m_product), years)
        assert np.all(np.isfinite(m_prod_all)), f'combined: product formula missing for {years[~np.isfinite(m_prod_all)]}'
        for v in mp.VERSIONS:
            m_mod = np.where(np.isfinite(m_mod_all), m_mod_all, m_prod_all)      # no model years after 2014
            m_prod = m_prod_all
            W.series(f'talk_missing_{v}_models', ('year',), m_mod, 'umol kg-1', f'missing alkalinity, model formula ({v})')
            W.series(f'talk_missing_{v}_products', ('year',), m_prod, 'umol kg-1', f'missing alkalinity, product formula ({v})')
            m_cell = CellPath(F[f'n_products_{v}'], pattern, m_mod, m_prod)
            st = lambda i, v=v: tuple(rd(f'{q}_{v}', i) for q in ('thetao', 'so', 'talk', 'dic'))
            ser = run_maps(W, chem, f' ({v})', 'year', years, m_cell, st, pattern, masks, f'_{v}')
            write_series(W, ser, 'year', f'_{v}', labels)
            print(f'  {v}: Omega_arag change 1750 {ser[("omega_arag", "")][0]:+.4f}, {years[-1]} {ser[("omega_arag", "")][-1]:+.4f}', flush=True)
        scen = [s for s in mp.SCENARIOS if f'omega_arag_{s}' in F.variables]
        if scen:
            ys = np.asarray(F['year_ssp'][:]).astype(int)
            W.dim('year_ssp', ys)
            for s in scen:
                if s not in P.m_s:
                    print(f'  {s}: no m(t) in {STRUCT.name}, left out', flush=True)
                    continue
                msc = P.on(P.mean4(P.m_s[s]), ys)
                W.series(f'talk_missing_{s}', ('year_ssp',), msc, 'umol kg-1', f'missing alkalinity m(t), {s}')
                st = lambda i, s=s: tuple(rd(f'{q}_{s}', i) for q in ('thetao', 'so', 'talk', 'dic'))
                ser = run_maps(W, chem, f', {s}', 'year_ssp', ys, msc, st, pattern, masks, f'_{s}')
                write_series(W, ser, 'year_ssp', f'_{s}', labels)
                print(f'  {s}: m 2100 {msc[-1]:+.2f} umol/kg, Omega_arag change 2100 {ser[("omega_arag", "")][-1]:+.4f}', flush=True)
        W.g.newton = f'max |pCO2 - pCO2_0| {chem.resid:.2e} uatm'
        W.g.close()
        F.close()
        print(f'  written {OUT / f"carbonate_system_{kind}_alkalinity.nc"}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', nargs='+', choices=mp.PRODUCTS + ['Jiang'])
    ap.add_argument('--combined', action='store_true', help='the combined files (after make_product.py --combine-only)')
    a = ap.parse_args()
    P = Params()
    for ref in a.reference or []:
        build(ref, P)
    if a.combined:
        build_combined(P)
    if not a.reference and not a.combined:
        ap.error('give --reference and/or --combined')


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
