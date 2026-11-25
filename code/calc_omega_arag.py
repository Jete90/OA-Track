#!/usr/bin/env python
"""
calc_omega_arag.py
==================
Surface aragonite saturation state (Omega_arag) for the CMIP6 historical and
piControl runs linked by prepare_inputs.sh, computed with mocsy 2.4
(Orr and Epitalon, 2015):

  - top level (5 m) of the files in CMIP6_DATA, which are already on r360x180
    (remapdis) and standard depth levels
  - DIC and alkalinity in mol m-3, optcon='mol/m3', optt='Tpot', optp='m',
    depth 0 m, patm 1 atm, mocsy default constants. Exception: GFDL-CM4 and
    GFDL-ESM4 carry per-kg tracers and write them as tracer * 1035 kg m-3
    (REF_DENSITY); for them DIC and ALK are divided by 1035 and passed with
    optcon='mol/kg' (Si and PO4 then directly in mol/kg)
  - Si and PO4 from the GLODAPv2.2016b surface climatology for every model,
    umol kg-1 converted to mol m-3 with the gsw surface density of the same
    month and cell
  - values outside (0, OMEGA_MAX) are mocsy failures and set to NaN

The time step is detected from the file. For annual input (the CMIP6_DATA
files are annual means despite the Omon label), Omega is computed from the
annual-mean fields. For monthly input, Omega is computed per month and the
annual field is the mean of the twelve monthly values (NaN if any month is
missing). The annual file also holds the area-weighted global mean over each
model's own ocean cells.

Output (data/omega_arag/<exp>/):
    omega_arag_Oyr_<model>_<exp>_<member>_r360x180.nc    annual field + global mean
    omega_arag_Omon_<model>_<exp>_<member>_r360x180.nc   monthly field (monthly input only)

Usage
-----
    python calc_omega_arag.py
    python calc_omega_arag.py --models CESM2 GFDL-ESM4 --experiments historical
    python calc_omega_arag.py --jobs 6 --no-monthly
"""
from __future__ import annotations

import argparse
import oa_config   # run: data folder, mocsy constants, GFDL density (OA_RUN)
import datetime
import re
import warnings
from multiprocessing import Pool
from pathlib import Path

import sys

import gsw
import netCDF4
import numpy as np

MOCSY_PATH = oa_config.MOCSY_PATH        # the compiled mocsy (oa_config.py)
if MOCSY_PATH not in sys.path:
    sys.path.append(MOCSY_PATH)
import mocsy  # noqa: E402
MOCSY_OPTB, MOCSY_OPTK1K2 = oa_config.OPTB, oa_config.OPTK1K2   # total boron, K1/K2 of the run (oa_config.py)

ROOT = oa_config.BASE
RAW_DIR = oa_config.DATA / 'raw'
OBS_DIR = oa_config.DATA / 'obs'
OUT_DIR = oa_config.DATA / 'omega_arag'
GRID = 'r360x180'
EXPERIMENTS = ('historical', 'piControl')
VARS = ('thetao', 'so', 'dissic', 'talk')
CHUNK = 60            # time steps per mocsy call, multiple of 12
OMEGA_MAX = 20.0      # larger values are not a valid carbonate system and are set to missing
PH_RANGE = (6.0, 10.0)   # a carbonate system mocsy solved (below about -5 degC and for DIC >= TA it returns pH -19.6,
                         # pCO2 24,500 uatm, Omega 0.002, Revelle 0 without a fill value)
FLOORED = [0, 0]         # cells set to the freezing point, cells passed (floor_temperature; per process)


def freezing_point(s):
    """freezing temperature of seawater at the surface (UNESCO 1983), degC; s clipped to 0-50"""
    s = np.clip(np.asarray(s, 'f8'), 0.0, 50.0)
    return -0.0575 * s + 1.710523e-3 * s ** 1.5 - 2.154996e-4 * s ** 2


def floor_temperature(t, s, count=True):
    """t not below the freezing point of its salinity: surface water cannot be colder than its freezing point, but the
    bias adjustment, the drift correction, the regressions and their perturbations of the chain can take an ice-covered
    cell below it (to -5 degC), where mocsy fails (PH_RANGE). The count goes to FLOORED (count=True)."""
    t = np.asarray(t, 'f8')
    tf = freezing_point(s)
    low = np.isfinite(t) & (t < tf)
    if count:
        FLOORED[0] += int(low.sum()); FLOORED[1] += int(np.isfinite(t).sum())
    return np.where(low, tf, t)


def valid_carbonate(ph, om):
    """a solved carbonate system: Omega_arag in (0, OMEGA_MAX) and pH in PH_RANGE"""
    ph, om = np.asarray(ph, 'f8'), np.asarray(om, 'f8')
    return np.isfinite(om) & (om > 0) & (om < OMEGA_MAX) & np.isfinite(ph) & (ph > PH_RANGE[0]) & (ph < PH_RANGE[1])
FILL = np.float32(1.0e20)

# plausible global medians of surface values after unit conversion
EXPECTED = {'dissic': (1.5, 2.8), 'talk': (1.8, 3.0), 'so': (25.0, 40.0)}

RUN_RE = re.compile(
    r'^dissic_Omon_(?P<model>[^_]+)_(?P<exp>[^_]+)_(?P<member>[^_]+)_[^_]+_[^_]+\.nc$')


# ===========================================================================
# FILES
# ===========================================================================

def input_path(var, model, exp, member):
    hits = sorted((RAW_DIR / exp).glob(f'{var}_Omon_{model}_{exp}_{member}_*.nc'))
    return hits[0] if hits else None


def find_runs(experiments, models=None):
    runs = []
    for exp in experiments:
        for p in sorted((RAW_DIR / exp).glob('dissic_Omon_*.nc')):
            m = RUN_RE.match(p.name)
            if m is None or (models and m['model'] not in models):
                continue
            missing = [v for v in VARS
                       if input_path(v, m['model'], exp, m['member']) is None]
            if missing:
                print(f"[skip] {m['model']} {exp} {m['member']}: no {missing}")
                continue
            runs.append((m['model'], exp, m['member']))
    return runs


def read(v, i0, i1):
    """Time steps i0:i1 of the top level as float64 with NaN for missing."""
    raw = v[i0:i1, 0] if v.ndim == 4 else v[i0:i1]
    a = np.ma.filled(np.ma.asarray(raw).astype('f8'), np.nan)
    a[np.abs(a) > 1e10] = np.nan
    return a


def load_nutrients(lat, lon):
    """GLODAPv2.2016b surface Si and PO4 (umol kg-1) on the model grid; the few small negative values of the
    mapped climatology are set to 0, as in calc_omega_arag_obs.load_nutrients (same nutrients everywhere)."""
    out = []
    for var in ('silicate', 'PO4'):
        path = OBS_DIR / f'{var}_GLODAPv2.2016b_surface_{GRID}.nc'
        with netCDF4.Dataset(path) as f:
            a = np.ma.filled(f.variables[var][:].astype('f8'), np.nan).squeeze()
            if not (np.allclose(f['lat'][:], lat) and np.allclose(f['lon'][:], lon)):
                raise ValueError(f'{path.name}: grid differs from model grid')
        out.append(np.where(a < 0, 0.0, a))
    return out


def create_nc(path, lat, lon, time, tunits, calendar, attrs):
    f = netCDF4.Dataset(path, 'w', format='NETCDF4')
    f.createDimension('time', None)
    f.createDimension('lat', lat.size)
    f.createDimension('lon', lon.size)
    for name, dim, vals, a in (
            ('time', 'time', time, dict(units=tunits, calendar=calendar,
                                        standard_name='time', axis='T')),
            ('lat', 'lat', lat, dict(units='degrees_north',
                                     standard_name='latitude', axis='Y')),
            ('lon', 'lon', lon, dict(units='degrees_east',
                                     standard_name='longitude', axis='X'))):
        v = f.createVariable(name, 'f8', (dim,))
        v.setncatts(a)
        v[:] = vals
    v = f.createVariable('omega_arag', 'f4', ('time', 'lat', 'lon'),
                         zlib=True, complevel=1, fill_value=FILL)
    v.long_name = 'surface aragonite saturation state'
    v.units = '1'
    f.setncatts(attrs)
    return f


# ===========================================================================
# UNITS AND CHEMISTRY
# ===========================================================================

def to_mol_m3(a, units, rho):
    """CMIP6 DIC or alkalinity to mol m-3 (native CMIP6 unit is mol m-3)."""
    u = units.replace(' ', '').replace('^', '').lower()
    scale = 1e-6 if u.startswith(('umol', 'µmol', 'micromol')) else \
        1e-3 if u.startswith('mmol') else 1.0
    return a * scale * rho if 'kg' in u else a * scale


def check_range(a, name, tag):
    med = np.nanmedian(a)
    lo, hi = EXPECTED[name]
    if not lo < med < hi:
        raise ValueError(f'{tag}: median {name} = {med:.4g} outside ({lo}, {hi}), check units')


# Models that carry DIC and alkalinity per kg and write the CMIP6 mol m-3 output as tracer * constant density
# (kg m-3). All other models: per-volume tracers (or per kg written with in-situ density), converted with the
# in-situ density.
REF_DENSITY = oa_config.REF_DENSITY   # GFDL-CM4/-ESM4: 1035 (RHO_0 in generic_BLING/COBALT.F90); oa_config.py


def model_density(model, t, s, lon2, lat2):
    """density (kg m-3) that converts this model's CMIP6 dissic/talk (mol m-3) back to per kg"""
    if model in REF_DENSITY:
        return np.full(np.shape(t), REF_DENSITY[model])
    return surface_density(t, s, lon2, lat2)


def surface_density(t, s, lon2, lat2):
    """Surface density from gsw, for the umol/kg to mol m-3 conversions."""
    z = np.zeros_like(t)
    sa = gsw.SA_from_SP(s, z, lon2, lat2)
    ct = gsw.CT_from_pt(sa, gsw.pt0_from_t(sa, t, z))
    return gsw.sigma0(sa, ct) + 1000.0


def omega_cells(t, s, dic, alk, sil, phos, lat, optcon='mol/m3'):
    """Omega_arag for 1-D arrays of valid cells, all mol m-3 (or mol kg-1 with optcon='mol/kg') except t and s."""
    n = t.size
    res = mocsy.mvars(temp=floor_temperature(t, s), sal=s, alk=alk, dic=dic, sil=sil, phos=phos,
                      patm=np.ones(n, dtype=float),
                      depth=np.zeros(n, dtype=float),
                      lat=lat, optcon=optcon, optt='Tpot', optp='m', optb=MOCSY_OPTB, optk1k2=MOCSY_OPTK1K2)
    om = np.asarray(res[6], dtype=float)   # ph, pco2, fco2, co2, hco3, co3, OmegaA, ...
    return np.where(valid_carbonate(res[0], om), om, np.nan)


def year_month(vals, units, calendar):
    """Year and month of each time value. Also handles 'years since' and 'months since',
    which cftime rejects for most calendars (IPSL piControl uses 'years since')."""
    vals = np.asarray(vals, dtype='f8')
    u = units.strip().lower()
    if u.startswith(('years since', 'year since', 'months since', 'month since')):
        ref = units.split('since')[1].strip()
        y0 = int(ref[0:4])
        m0 = int(ref[5:7]) if len(ref) >= 7 and ref[4] == '-' else 1
        k = np.floor(vals + 1e-6).astype(int)
        if u.startswith('year'):
            return y0 + k, np.full(k.shape, m0)
        k = k + (m0 - 1)
        return y0 + k // 12, k % 12 + 1
    dates = netCDF4.num2date(vals, units, calendar)
    return np.array([d.year for d in dates]), np.array([d.month for d in dates])


def global_mean(field, lat):
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], field.shape[-2:])    # native model grid, diagnostic: cell area (oa_config.WEIGHTS applies to the product grid)
    fin = np.isfinite(field)
    return (np.where(fin, field * w, 0).sum((-2, -1))
            / np.where(fin, w, 0).sum((-2, -1)))


# ===========================================================================
# ONE RUN
# ===========================================================================

def process(model, exp, member, force=False, monthly=True):
    tag = f'{model} {exp} {member}'
    out_dir = OUT_DIR / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    f_mon = out_dir / f'omega_arag_Omon_{model}_{exp}_{member}_{GRID}.nc'
    f_yr = out_dir / f'omega_arag_Oyr_{model}_{exp}_{member}_{GRID}.nc'
    if f_yr.exists() and not force:
        print(f'[skip] {tag}: {f_yr.name} exists', flush=True)
        return tag, None

    paths = {v: input_path(v, model, exp, member) for v in VARS}
    src = {v: netCDF4.Dataset(paths[v]) for v in VARS}
    try:
        var = {v: src[v].variables[v] for v in VARS}
        lat = np.asarray(src['dissic']['lat'][:], dtype='f8')
        lon = np.asarray(src['dissic']['lon'][:], dtype='f8')
        for v in VARS:
            if not (np.allclose(src[v]['lat'][:], lat) and np.allclose(src[v]['lon'][:], lon)):
                raise ValueError(f'{tag}: grid of {v} differs from dissic')
        ny, nx = lat.size, lon.size
        lon2, lat2 = np.meshgrid(lon, lat)
        si, po4 = load_nutrients(lat, lon)

        nts = {v: var[v].shape[0] for v in VARS}
        nt = min(nts.values())
        if len(set(nts.values())) > 1:
            print(f'[warn] {tag}: lengths {nts}, using first {nt} steps', flush=True)

        tvar = src['dissic']['time']
        tunits, cal = tvar.units, getattr(tvar, 'calendar', 'standard')
        yy, mm = year_month(tvar[:2], tunits, cal)
        step = 12 if (yy[1], mm[1]) != (yy[0] + 1, mm[0]) else 1
        nt -= nt % step
        monthly = monthly and step == 12
        tvals = np.asarray(tvar[:nt], dtype='f8')
        if var['dissic'].ndim == 4:
            zname = var['dissic'].dimensions[1]
            print(f"[info] {tag}: {'monthly' if step == 12 else 'annual'}, "
                  f"top level {zname} = {float(src['dissic'][zname][0]):g}", flush=True)

        # temperature and salinity units from the first year
        t0, s0 = read(var['thetao'], 0, step), read(var['so'], 0, step)
        kelvin = getattr(var['thetao'], 'units', '').strip() in ('K', 'degK', 'kelvin') \
            or np.nanmedian(t0) > 100
        toff = -273.15 if kelvin else 0.0
        sscale = 1000.0 if np.nanmedian(s0) < 1 else 1.0
        cunits = {v: getattr(var[v], 'units', 'mol m-3') for v in ('dissic', 'talk')}

        attrs = dict(
            title='Surface aragonite saturation state',
            model=model, experiment=exp, member=member, grid=GRID,
            method=('mocsy 2.4.0 mvars, optcon=mol/m3, optt=Tpot, optp=m, depth 0 m, '
                    'patm 1 atm, default ' + oa_config.CONSTANTS + '; Si and PO4 from GLODAPv2.2016b '
                    'surface climatology converted with gsw surface density; '
                    f'values outside (0, {OMEGA_MAX:g}) set to missing'),
            source=' '.join(str(paths[v]) for v in VARS),
            units_in=f"thetao {getattr(var['thetao'], 'units', '?')}, "
                     f"so {getattr(var['so'], 'units', '?')}, "
                     f"dissic {cunits['dissic']}, talk {cunits['talk']}",
            input_step='monthly' if step == 12 else 'annual means (Omega from annual-mean fields)',
            dic_alk_density=(f'per kg = CMIP6 mol m-3 / {REF_DENSITY[model]:g} kg m-3 (model reference density, '
                             'mocsy optcon=mol/kg)' if model in REF_DENSITY else 'mocsy in-situ density (optcon=mol/m3)'),
            history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} calc_omega_arag.py')

        fm = create_nc(f_mon, lat, lon, tvals, tunits, cal, attrs) if monthly else None
        oyr = np.full((nt // step, ny, nx), np.nan, dtype='f4')
        n_ok = n_cull = 0
        for i0 in range(0, nt, CHUNK):
            i1 = min(i0 + CHUNK, nt)
            t = read(var['thetao'], i0, i1) + toff
            s = read(var['so'], i0, i1) * sscale
            rho = surface_density(t, s, lon2, lat2)
            dic = to_mol_m3(read(var['dissic'], i0, i1), cunits['dissic'], rho)
            alk = to_mol_m3(read(var['talk'], i0, i1), cunits['talk'], rho)
            if i0 == 0:
                check_range(s, 'so', tag)
                check_range(dic, 'dissic', tag)
                check_range(alk, 'talk', tag)

            ok = (np.isfinite(t) & np.isfinite(s) & np.isfinite(rho)
                  & np.isfinite(dic) & np.isfinite(alk)
                  & (s > 0) & (dic > 0) & (alk > 0))
            ok &= np.isfinite(si) & np.isfinite(po4)
            om = np.full(t.shape, np.nan)
            if ok.any():
                if model in REF_DENSITY:      # per-kg tracers written as tracer * REF_DENSITY
                    rref = REF_DENSITY[model]
                    om[ok] = omega_cells(t[ok], s[ok], dic[ok] / rref, alk[ok] / rref,
                                         np.broadcast_to(si * 1e-6, t.shape)[ok],
                                         np.broadcast_to(po4 * 1e-6, t.shape)[ok],
                                         np.broadcast_to(lat2, t.shape)[ok], optcon='mol/kg')
                else:
                    om[ok] = omega_cells(t[ok], s[ok], dic[ok], alk[ok],
                                         (si * rho * 1e-6)[ok], (po4 * rho * 1e-6)[ok],
                                         np.broadcast_to(lat2, t.shape)[ok])
            n_ok += int(ok.sum())
            n_cull += int((ok & ~np.isfinite(om)).sum())

            if fm is not None:
                fm['omega_arag'][i0:i1] = np.ma.masked_invalid(om.astype('f4'))
            oyr[i0 // step:i1 // step] = om.reshape(-1, step, ny, nx).mean(axis=1)
        if fm is not None:
            fm.close()

        gm = global_mean(oyr, lat)
        fy = create_nc(f_yr, lat, lon, tvals.reshape(-1, step).mean(axis=1), tunits, cal, attrs)
        fy['omega_arag'][:] = np.ma.masked_invalid(oyr)
        v = fy.createVariable('omega_arag_gm', 'f8', ('time',))
        v.long_name = 'area-weighted global mean surface aragonite saturation state'
        v.units = '1'
        v[:] = gm
        fy.close()
    finally:
        for f in src.values():
            f.close()

    cull = n_cull / max(n_ok, 1)
    print(f'[done] {tag}: {nt // step} yr, global mean {gm[0]:.3f} -> {gm[-1]:.3f}, '
          f'culled {100 * cull:.3f} %', flush=True)
    return tag, (nt // step, gm[0], gm[-1], cull)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+')
    ap.add_argument('--experiments', nargs='+', default=list(EXPERIMENTS))
    ap.add_argument('--force', action='store_true', help='recompute existing output')
    ap.add_argument('--no-monthly', action='store_true',
                    help='for monthly input, write annual files only')
    ap.add_argument('--jobs', type=int, default=1, help='runs in parallel')
    args = ap.parse_args()

    runs = find_runs(args.experiments, args.models)
    print(f'{len(runs)} runs', flush=True)
    tasks = [(m, e, r, args.force, not args.no_monthly) for m, e, r in runs]
    if args.jobs > 1:
        with Pool(args.jobs) as p:
            results = p.starmap(process, tasks)
    else:
        results = [process(*t) for t in tasks]

    print(f"\n{'run':50s} {'years':>5s} {'first':>7s} {'last':>7s} {'cull %':>7s}")
    for tag, r in results:
        if r is not None:
            print(f'{tag:50s} {r[0]:5d} {r[1]:7.3f} {r[2]:7.3f} {100 * r[3]:7.3f}')


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
