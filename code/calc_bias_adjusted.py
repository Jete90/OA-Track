#!/usr/bin/env python
"""
calc_bias_adjusted.py
=====================
Bias-adjusted, drift-corrected surface Omega_arag for one CMIP6 model, using an
observation-based product as reference (OceanSODA-ETHZ v2025 by default).

1. The model's top-level T, S, DIC and ALK are sampled onto the product grid
   (nearest neighbour). DIC and ALK are converted from mol m-3 to umol kg-1
   with the gsw surface density, so they are in the same units as the product
   (GFDL-CM4 and GFDL-ESM4: with 1035 kg m-3, the density with which these
   models write their per-kg tracers to mol m-3; cm.REF_DENSITY).
2. For every time step where both overlap (the reference window W, oa_config.WINDOWS:
   1983-2014 for OceanSODA, 1980-2014 for UExP), the difference product - model is taken for each variable.
3. The climatology of that difference is the bias: one field per calendar
   month for monthly model output, one field for annual-mean output. A cell
   needs at least MIN_FRAC of the overlap years, otherwise its bias is NaN.
   The bias is computed from the raw historical run; because the drift
   correction below averages to zero over W, it is the same for the corrected run.
4. Every time step t of historical and piControl is corrected for the piControl
   drift and the bias,

       X~(t) = X(t) - [s(k(t)) - mean of s over W'] + bias(month of t)

   with s the Enting spline of the piControl annual means (calc_drift.py),
   k(t) the piControl year of a piControl month and the parallel piControl year
   (via the branch point) of a historical month, and W' the piControl years
   parallel to W. Omega_arag is recomputed with mocsy (optcon='mol/kg', otherwise
   as in calc_omega_arag.py) for the raw, the bias-only and the corrected
   state (like-for-like, same chemistry): omega_arag_raw, omega_arag_biasonly,
   omega_arag_adj. omega_arag_adj is the state of the chain; its piControl is
   the state from which calc_pi_xco2.py computes the preindustrial CO2 states.

Global means are taken over one joint mask: the product mask, restricted to
cells where all three Omega are valid in every year of both experiments.

Reference: --reference OceanSODA (default), UExP, JMA, CMEMS or Jiang; each has its own output folder.
Jiang (Jiang et al. 2023, decade 2010 = January 2005 to December 2014, make_jiang_ref.py) is a single
annual state: its bias is one field per cell, the reference minus the model mean over 2005-2014, added
to every month (the seasonal cycle comes from the model).

Output:
    data/bias/bias_clim_<model>_<member>_<reference>.nc
    data/omega_arag_biasadj[_<reference>]/<exp>/omega_arag_biasadj_Oyr_<model>_<exp>_<member>_<reference>grid.nc
    (OceanSODA keeps the folder data/omega_arag_biasadj and the file names used so far)

Needs data/<run>/drift/drift_<model>_<member>.nc (calc_drift.py) for every model.

Usage:
    python calc_bias_adjusted.py                      # all runs with historical and piControl
    python calc_bias_adjusted.py --jobs 4             # same, 4 runs in parallel
    python calc_bias_adjusted.py --models GFDL-ESM4 CESM2
    python calc_bias_adjusted.py --force              # recompute existing output
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

import calc_omega_arag as cm        # model reading, density, units (imports mocsy)
import calc_omega_arag_obs as co    # OceanSODA path, nutrients, mocsy call in mol/kg
import calc_omega_arag_products as cp   # generic product reader and product definitions
from drift_tools import Drift       # piControl drift correction (calc_drift.py output)
from make_jma_ts import year_month

ROOT = cm.ROOT
BIAS_DIR = oa_config.DATA / 'bias'
OUT_DIR = oa_config.DATA / 'omega_arag_biasadj'
OVERLAP = oa_config.WINDOWS['OceanSODA']     # default; every reference has its own window below
MIN_FRAC = 0.8
CHUNK = 60                          # multiple of 12
VARS = ('thetao', 'so', 'dissic', 'talk')
OBS_NAME = {'thetao': 'temperature', 'so': 'salinity', 'dissic': 'dic', 'talk': 'talk'}
UNITS = {'thetao': 'degC', 'so': 'psu', 'dissic': 'umol kg-1', 'talk': 'umol kg-1'}
FILL = np.float32(1.0e20)
OMEGA_VARS = ('omega_arag_raw', 'omega_arag_biasonly', 'omega_arag_adj')

# observational references; output of each goes to its own folder, OceanSODA keeps the old names
REFERENCES = {
    'OceanSODA': dict(file=co.SRC, order='time,lat,lon', t_kelvin=False,
                      vars=dict(t='temperature', s='salinity', dic='dic', ta='talk'),
                      omega=co.OUT, label='OceanSODA-ETHZ v2025',
                      window=oa_config.WINDOWS['OceanSODA'],      # first complete year 1983
                      outdir=oa_config.DATA / 'omega_arag_biasadj'),
    'UExP': dict(**{k: cp.PRODUCTS['UExP-FNN-U-v2026-1'][k] for k in ('file', 'order', 't_kelvin', 'vars')},
                 omega=co.OBS_DIR / 'omega_arag_Oyr_UExP-FNN-U-v2026-1.nc', label='UExP-FNN-U v2026-1',
                 window=oa_config.WINDOWS['UExP'],           # product starts in 1980
                 outdir=oa_config.DATA / 'omega_arag_biasadj_UExP'),
    'JMA': dict(**{k: cp.PRODUCTS['JMA-MLR-v2025'][k] for k in ('file', 'ts_file', 'order', 't_kelvin', 'vars')},
                omega=co.OBS_DIR / 'omega_arag_Oyr_JMA-MLR-v2025.nc', label='JMA-MLR v2025',
                window=oa_config.WINDOWS['JMA'],           # product starts in 1990
                outdir=oa_config.DATA / 'omega_arag_biasadj_JMA'),
    'CMEMS': dict(**{k: cp.PRODUCTS['CMEMS-LSCE-FFNN'][k] for k in ('file', 'order', 't_kelvin', 'vars')},
                  omega=co.OBS_DIR / 'omega_arag_Oyr_CMEMS-LSCE-FFNN.nc', label='CMEMS-LSCE-FFNN',
                  window=oa_config.WINDOWS['CMEMS'],       # product starts in 1985
                  outdir=oa_config.DATA / 'omega_arag_biasadj_CMEMS'),
    # Jiang et al. (2023): observation-based state of the decade 2010 (Jan 2005 to Dec 2014), one annual field
    'Jiang': dict(file=co.OBS_DIR / 'omega_arag_Oyr_Jiang2023-2010.nc', order='time,lat,lon', t_kelvin=False,
                  vars=dict(t='temperature', s='salinity', dic='dic', ta='talk'), annual=True,
                  omega=co.OBS_DIR / 'omega_arag_Oyr_Jiang2023-2010.nc', label='Jiang et al. (2023), decade 2010',
                  window=oa_config.WINDOWS['Jiang'],
                  outdir=oa_config.DATA / 'omega_arag_biasadj_Jiang'),
}
REF_NAME = 'OceanSODA'
REF = REFERENCES[REF_NAME]


def set_reference(name):
    """Select the observational reference for all functions of this module."""
    global REF_NAME, REF, OUT_DIR
    REF_NAME, REF = name, REFERENCES[name]
    OUT_DIR = REF['outdir']
    return REF


def bias_path(model, member):
    return BIAS_DIR / f'bias_clim_{model}_{member}_{REF_NAME}.nc'


# ===========================================================================
# MODEL
# ===========================================================================

class ModelRun:
    """Top-level T, S, DIC, ALK of one run, sampled onto the OceanSODA grid."""

    def __init__(self, model, exp, member):
        self.tag = f'{model} {exp} {member}'
        self.model = model
        self.exp = exp
        self.member = member
        paths = {v: cm.input_path(v, model, exp, member) for v in VARS}
        missing = [v for v, p in paths.items() if p is None]
        if missing:
            raise FileNotFoundError(f'{self.tag}: no {missing} in {cm.RAW_DIR / exp}')
        self.ds = {v: netCDF4.Dataset(p) for v, p in paths.items()}
        self.var = {v: self.ds[v].variables[v] for v in VARS}
        self.lat = np.asarray(self.ds['dissic']['lat'][:], 'f8')
        self.lon = np.asarray(self.ds['dissic']['lon'][:], 'f8')

        nts = {v: self.var[v].shape[0] for v in VARS}
        nt = min(nts.values())
        t = self.ds['dissic']['time']
        self.units, self.cal = t.units, getattr(t, 'calendar', 'standard')
        years, months = cm.year_month(t[:nt], self.units, self.cal)
        self.step = 12 if nt > 1 and years[1] == years[0] else 1
        if self.step == 12 and months[0] != 1:
            raise ValueError(f'{self.tag}: monthly file does not start in January')
        self.nt = nt - nt % self.step
        self.years = years[:self.nt]

        t0 = cm.read(self.var['thetao'], 0, 1)
        self.toff = -273.15 if np.nanmedian(t0) > 100 else 0.0
        self.sscale = 1000.0 if np.nanmedian(cm.read(self.var['so'], 0, 1)) < 1 else 1.0
        self.cunits = {v: getattr(self.var[v], 'units', 'mol m-3') for v in ('dissic', 'talk')}

    def read(self, i0, i1, jy, ix, lon2, lat2):
        """Steps i0:i1 as {var: (n, ny, nx)} in degC, psu, umol kg-1."""
        def sample(v):
            return cm.read(self.var[v], i0, i1)[:, jy][:, :, ix]
        t = sample('thetao') + self.toff
        s = sample('so') * self.sscale
        rho = cm.surface_density(t, s, lon2, lat2)
        rho_m = cm.model_density(self.model, t, s, lon2, lat2)   # REF_DENSITY for GFDL, else in-situ
        out = {'thetao': t, 'so': s}
        for v in ('dissic', 'talk'):
            out[v] = cm.to_mol_m3(sample(v), self.cunits[v], rho) / rho_m * 1e6
        return out

    def close(self):
        for f in self.ds.values():
            f.close()


# ===========================================================================
# OBSERVATIONS AND BIAS
# ===========================================================================

def obs_fields(y0, y1, step):
    """Reference T (degC), S, DIC, ALK (umol/kg) for y0..y1, monthly or as annual means.
    Annual references (Jiang): the single state of the file, shape (1, ny, nx), whatever y0, y1 and step."""
    if REF.get('annual'):
        with netCDF4.Dataset(REF['file']) as f:
            v = REF['vars']
            return {'thetao': np.ma.filled(f[v['t']][:1].astype('f8'), np.nan),
                    'so': np.ma.filled(f[v['s']][:1].astype('f8'), np.nan),
                    'dissic': np.ma.filled(f[v['dic']][:1].astype('f8'), np.nan),
                    'talk': np.ma.filled(f[v['ta']][:1].astype('f8'), np.nan)}
    main = cp.Reader(REF['file'], REF['order'])
    ts = cp.Reader(REF['ts_file'], 'time,lat,lon') if REF.get('ts_file') else main
    try:
        ym = year_month(main.f['time'])
        idx = np.array([i for i, (y, m) in enumerate(ym) if y0 <= y <= y1])
        if idx.size != 12 * (y1 - y0 + 1) or np.any(np.diff(idx) != 1) or ym[idx[0]][1] != 1:
            raise ValueError(f'{REF_NAME}: {idx.size} months for {y0}-{y1}')
        i0, i1 = int(idx[0]), int(idx[-1]) + 1
        v = REF['vars']
        out = {'thetao': ts(v['t'], i0, i1) - (273.15 if REF['t_kelvin'] else 0.0),
               'so': ts(v['s'], i0, i1),
               'dissic': main(v['dic'], i0, i1),
               'talk': main(v['ta'], i0, i1)}
    finally:
        main.close()
        if ts is not main:
            ts.close()
    if step == 1:
        out = {k: a.reshape(-1, 12, *a.shape[1:]).mean(axis=1) for k, a in out.items()}
    return out


def window(run):
    """reference window restricted to the years of the historical run"""
    w0, w1 = REF.get('window', OVERLAP)
    return max(w0, int(run.years[0])), min(w1, int(run.years[-1]))


def bias_climatology(run, jy, ix, lon2, lat2):
    y0, y1 = window(run)
    obs = obs_fields(y0, y1, run.step)
    i0 = int(np.where(run.years == y0)[0][0])
    i1 = int(np.where(run.years == y1)[0][-1]) + 1
    mod = run.read(i0, i1, jy, ix, lon2, lat2)
    bias = {}
    if REF.get('annual'):
        # one field: reference minus the model mean over all steps of the window, used for every slot
        for v in VARS:
            n = np.isfinite(mod[v]).sum(axis=0)
            b = obs[v][0] - np.nanmean(mod[v], axis=0)
            b[n < MIN_FRAC * mod[v].shape[0]] = np.nan
            bias[v] = np.repeat(b[None], run.step, axis=0)        # (step, ny, nx), identical slots
        return bias, (y0, y1)
    for v in VARS:
        d = (obs[v] - mod[v]).reshape(-1, run.step, *obs[v].shape[1:])
        n = np.isfinite(d).sum(axis=0)
        b = np.nanmean(d, axis=0)
        b[n < MIN_FRAC * d.shape[0]] = np.nan
        bias[v] = b                                   # (step, ny, nx)
    return bias, (y0, y1)


def adjusted_fields(run, i0, i1, bias, drift, win, jy, ix, lon2, lat2):
    """Steps i0:i1 of the run as raw, bias-only and corrected (drift + bias) fields, {var: (n, ny, nx)}.
    corrected = raw - [s(k(t)) - mean of s over W'] + bias(month)   (module doc, point 4)."""
    f = run.read(i0, i1, jy, ix, lon2, lat2)
    slot = np.arange(i0, i1) % run.step
    years = run.years[i0:i1]
    fb = {v: f[v] + bias[v][slot] for v in VARS}
    fa = {v: fb[v] - drift.term(v, run.exp, years, win, jy, ix) for v in VARS}
    return f, fb, fa


# ===========================================================================
# CHEMISTRY
# ===========================================================================

def omega(fields, si, po4, lat2):
    t, s, dic, alk = (fields[v] for v in VARS)
    ok = (np.isfinite(t) & np.isfinite(s) & np.isfinite(dic) & np.isfinite(alk)
          & (s > 0) & (dic > 0) & (alk > 0))
    ok &= np.isfinite(si) & np.isfinite(po4)
    om = np.full(t.shape, np.nan)
    if ok.any():
        om[ok] = co.omega_cells(t[ok], s[ok], dic[ok] * 1e-6, alk[ok] * 1e-6,
                                np.broadcast_to(si * 1e-6, t.shape)[ok],
                                np.broadcast_to(po4 * 1e-6, t.shape)[ok],
                                np.broadcast_to(lat2, t.shape)[ok])
    return om


def omega_series(run, bias, drift, win, si, po4, jy, ix, lon2, lat2):
    """Annual raw, bias-only and corrected Omega on the product grid."""
    ny, nx = lat2.shape
    nyr = run.nt // run.step
    out = {k: np.full((nyr, ny, nx), np.nan, 'f4') for k in OMEGA_VARS}
    for i0 in range(0, run.nt, CHUNK):
        i1 = min(i0 + CHUNK, run.nt)
        states = dict(zip(OMEGA_VARS, adjusted_fields(run, i0, i1, bias, drift, win, jy, ix, lon2, lat2)))
        for k, fields in states.items():
            om = omega(fields, si, po4, lat2)
            out[k][i0 // run.step:i1 // run.step] = om.reshape(-1, run.step, ny, nx).mean(axis=1)
        print(f'  {run.tag}: {run.years[i0]}-{run.years[i1 - 1]} done', flush=True)
    return run.years[::run.step][:nyr], out


def masked_mean(field, lat, lon, mask):
    """area-weighted mean of field (..., ny, nx) over mask (oa_config.area_weights: cell area or cell area x ocean fraction)"""
    w = oa_config.area_weights(lat, lon) * mask
    return np.nansum(np.where(mask, field, 0) * w, axis=(-2, -1)) / w.sum()


# ===========================================================================
# OUTPUT
# ===========================================================================

def new_nc(path, lat, lon, attrs):
    f = netCDF4.Dataset(path, 'w', format='NETCDF4')
    f.createDimension('lat', lat.size)
    f.createDimension('lon', lon.size)
    for name, vals, u in (('lat', lat, 'degrees_north'), ('lon', lon, 'degrees_east')):
        v = f.createVariable(name, 'f8', (name,))
        v.units = u
        v[:] = vals
    f.setncatts(attrs)
    return f


def out_path(exp, model, member):
    return OUT_DIR / exp / f'omega_arag_biasadj_Oyr_{model}_{exp}_{member}_{REF_NAME}grid.nc'


def find_runs(models=None):
    """(model, member) with all inputs for historical and piControl."""
    runs = []
    for model, exp, member in cm.find_runs(['historical'], models):
        if all(cm.input_path(v, model, 'piControl', member) for v in VARS):
            runs.append((model, member))
        else:
            print(f'[skip] {model} {member}: no piControl')
    return runs


def run_one(model, member, force=False):
    tag = f'{model} {member}'
    if not force and all(out_path(e, model, member).exists() for e in ('historical', 'piControl')):
        print(f'[skip] {tag}: output exists', flush=True)
        return tag, 'exists'
    try:
        process(model, member)
        return tag, 'done'
    except Exception:
        print(f'[FAILED] {tag}\n{traceback.format_exc()}', flush=True)
        return tag, 'FAILED'


def process(model, member):
    args = argparse.Namespace(model=model, member=member)
    BIAS_DIR.mkdir(parents=True, exist_ok=True)

    with netCDF4.Dataset(REF['omega']) as f:
        olat = np.asarray(f['lat'][:], 'f8')
        olon = np.asarray(f['lon'][:], 'f8')
        omask = np.asarray(f['mask'][:]).astype(bool)
        oyears = np.asarray(f['year'][:])
        ofield = np.ma.filled(f['omega_arag'][:].astype('f8'), np.nan)
    lon2, lat2 = np.meshgrid(olon, olat)
    si, po4 = co.load_nutrients(olat, olon)

    drift = Drift(args.model, args.member)             # calc_drift.py output, required (fails first if missing)
    hist = ModelRun(args.model, 'historical', args.member)
    jy = co.nearest_index(hist.lat, olat)
    ix = co.nearest_index(hist.lon, olon, periodic=True)
    print(f'{hist.tag}: {"monthly" if hist.step == 12 else "annual"} input', flush=True)
    bias, (y0, y1) = bias_climatology(hist, jy, ix, lon2, lat2)

    stamp = f'{datetime.datetime.now():%Y-%m-%d %H:%M} calc_bias_adjusted.py, {oa_config.VERSION}'
    fb = new_nc(bias_path(args.model, args.member), olat, olon,
                dict(title=f'{REF["label"]} minus {args.model} historical, climatology',
                     overlap=f'{y0}-{y1}',
                     slots='calendar months' if hist.step == 12 else 'annual mean',
                     history=stamp))
    fb.createDimension('slot', hist.step)
    for v in VARS:
        x = fb.createVariable(f'bias_{v}', 'f4', ('slot', 'lat', 'lon'), zlib=True, fill_value=FILL)
        x.units = UNITS[v]
        x[:] = np.ma.masked_invalid(bias[v])
    fb.close()

    area = oa_config.area_weights(olat, olon)
    w = area * omask
    print(f'\nbias {y0}-{y1}, area mean over {REF_NAME} mask:')
    for v in VARS:
        b = bias[v].mean(axis=0)
        fin = np.isfinite(b) & omask
        print(f'  {v:7s} {np.nansum(np.where(fin, b, 0) * w) / (w * fin).sum():+8.3f} {UNITS[v]}')

    res = {}
    for exp in ('historical', 'piControl'):
        run = hist if exp == 'historical' else ModelRun(args.model, exp, args.member)
        res[exp] = omega_series(run, bias, drift, (y0, y1), si, po4, jy, ix, lon2, lat2)
        run.close()

    mask = omask.copy()
    for _, out in res.values():
        for a in out.values():
            mask &= np.all(np.isfinite(a), axis=0)
    print(f'\njoint mask: {100 * (area * mask).sum() / (area * omask).sum():.1f} % of the {REF_NAME} mask')

    gms = {}
    for exp, (years, out) in res.items():
        (OUT_DIR / exp).mkdir(parents=True, exist_ok=True)
        path = out_path(exp, args.model, args.member)
        g = new_nc(path, olat, olon, dict(
            title=f'Bias-adjusted, drift-corrected surface aragonite saturation state, {args.model} {exp}',
            method=(f'T, S, DIC, ALK corrected for the piControl drift (Enting spline of the piControl annual means, '
                    f'50-year cutoff, relative to the piControl years parallel to {y0}-{y1}; '
                    f'historical years aligned at the branch point, drift file {drift.tag}) and by the '
                    f'{REF["label"]} minus model climatology over {y0}-{y1}; mocsy 2.4.0, optcon=mol/kg, '
                    'optt=Tpot, optp=m, default ' + oa_config.CONSTANTS + '; GLODAPv2.2016b surface Si and PO4'),
            variables=('omega_arag_raw: model as is; omega_arag_biasonly: bias only (no drift correction); '
                       'omega_arag_adj: drift correction and bias (the state of the chain)'),
            bias_file=bias_path(args.model, args.member).name, reference=REF['label'],
            branch_year=drift.by, hist_first_year=drift.hist0, history=stamp))
        g.createDimension('year', years.size)
        x = g.createVariable('year', 'i4', ('year',))
        x[:] = years
        x = g.createVariable('mask', 'i1', ('lat', 'lon'))
        x.long_name = 'joint mask used for the global means'
        x[:] = mask.astype('i1')
        for name, data in out.items():
            x = g.createVariable(name, 'f4', ('year', 'lat', 'lon'), zlib=True, fill_value=FILL)
            x.units = '1'
            x[:] = np.ma.masked_invalid(data)
            gm = masked_mean(data, olat, olon, mask)
            x = g.createVariable(f'{name}_gm', 'f8', ('year',))
            x.long_name = 'area-weighted mean over the joint mask'
            x[:] = gm
            gms[(exp, name)] = (years, gm)
        g.close()

    obs_gm = masked_mean(ofield, olat, olon, mask)
    yh = gms[('historical', 'omega_arag_raw')][0]
    sel_m = (yh >= y0) & (yh <= y1)
    sel_o = (oyears >= y0) & (oyears <= y1)
    print(f'\n{args.model} {args.member}: mean model minus {REF_NAME} {y0}-{y1}, joint mask:')
    for name in OMEGA_VARS:
        d = gms[('historical', name)][1][sel_m].mean() - obs_gm[sel_o].mean()
        pi = gms[('piControl', name)][1].mean()
        first = gms[('historical', name)][1][0]
        last = gms[('historical', name)][1][-1]
        print(f'  {name:20s}: offset {d:+.3f}, piControl {pi:.3f}, boundary {0.8 * pi:.3f}, '
              f'{yh[0]} {first:.3f}, {yh[-1]} {last:.3f} ({100 * (last / pi - 1):+.1f} %)')
    d_pi = gms[('piControl', 'omega_arag_adj')][1].mean() - gms[('piControl', 'omega_arag_biasonly')][1].mean()
    d_h0 = gms[('historical', 'omega_arag_adj')][1][0] - gms[('historical', 'omega_arag_biasonly')][1][0]
    print(f'  drift correction: piControl mean {d_pi:+.4f}, historical {yh[0]} {d_h0:+.4f} '
          f'(corrected minus bias-only)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', default='OceanSODA', choices=list(REFERENCES))
    ap.add_argument('--models', nargs='+', help='default: all models')
    ap.add_argument('--model', help='single model (with --member), as in run_bias_all.sh')
    ap.add_argument('--member', default='r1i1p1f1')
    ap.add_argument('--jobs', type=int, default=1, help='runs in parallel')
    ap.add_argument('--force', action='store_true', help='recompute existing output')
    a = ap.parse_args()

    set_reference(a.reference)
    print(f'reference: {REF["label"]} -> {OUT_DIR}', flush=True)
    runs = [(a.model, a.member)] if a.model else find_runs(a.models)
    print(f'{len(runs)} runs', flush=True)
    tasks = [(m, r, a.force) for m, r in runs]
    if a.jobs > 1:
        with Pool(a.jobs) as p:
            results = p.starmap(run_one, tasks)
    else:
        results = [run_one(*t) for t in tasks]
    print('\nsummary:')
    for tag, status in results:
        print(f'  {status:7s} {tag}')


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
