#!/usr/bin/env python
"""
calc_omega_arag_obs.py
======================
Surface aragonite saturation state from OceanSODA-ETHZ v2025 (1982-2024),
recomputed with mocsy 2.4.0 from the product's talk, dic, temperature and
salinity, in the same set-up as calc_omega_arag.py for the CMIP6 models:

  - optt='Tpot', optp='m', depth 0 m, patm 1 atm, mocsy default constants
  - optcon='mol/kg' here, because OceanSODA gives talk and dic in umol/kg
  - Si and PO4 from the same GLODAPv2.2016b surface fields as for the models
  - values outside (0, OMEGA_MAX) set to NaN

Annual value: Omega_arag of the annual-mean state, i.e. mocsy applied to the
annual means of T, S, DIC and TA, as for the CMIP6 models, whose inputs are
annual means. A cell enters a year only if all twelve months are valid (T, S,
DIC, TA finite and positive where required, and the monthly Omega_arag within
(0, OMEGA_MAX)).

For comparison two further annual fields are stored:
  omega_arag_monthly_mean   mean of the twelve monthly Omega_arag (minus omega_arag
                            it is the effect of averaging after instead of before
                            the chemistry)
  omega_arag_monthly_mean_dw  the same, months weighted by their number of days
                            (as xarray users often average monthly output)
  omega_arag_product        mean of the product's own monthly omega_ar
                            (PyCO2SYS, Mehrbach refit constants); compare it
                            with omega_arag_monthly_mean for the constants alone

Years with less than half the usual coverage are dropped (1982, because
January 1982 has no dic). Global means are area-weighted over a FIXED mask of
cells valid in every remaining year, so changing coverage cannot create a trend.

Output: data/obs/omega_arag_Oyr_OceanSODA-ETHZ-v2025.nc
"""
from __future__ import annotations

import calendar
import datetime
import oa_config   # run: data folder, mocsy constants, GFDL density (OA_RUN)
import sys
import warnings
from pathlib import Path

import netCDF4
import numpy as np

MOCSY_PATH = oa_config.MOCSY_PATH        # the compiled mocsy (oa_config.py)
if MOCSY_PATH not in sys.path:
    sys.path.append(MOCSY_PATH)
import mocsy  # noqa: E402
import calc_omega_arag as cm   # the freezing-point floor and the validity check of the chemistry
MOCSY_OPTB, MOCSY_OPTK1K2 = oa_config.OPTB, oa_config.OPTK1K2   # total boron, K1/K2 of the run (oa_config.py)

ROOT = oa_config.BASE
SRC = (oa_config.OBS_DIR / 'OceanSODA_ETHZ-v2025/0220059/6.6/data/0-data/'
       'OceanSODA_ETHZ-v2025.OCADS.01-1982-2024.nc')
OBS_DIR = oa_config.DATA / 'obs'
OUT = OBS_DIR / 'omega_arag_Oyr_OceanSODA-ETHZ-v2025.nc'
GRID = 'r360x180'
CHUNK = 60
OMEGA_MAX = 20.0
FILL = np.float32(1.0e20)


def read(v, i0, i1):
    a = np.ma.filled(np.ma.asarray(v[i0:i1]).astype('f8'), np.nan)
    a[np.abs(a) > 1e10] = np.nan
    return a


def nearest_index(src, dst, periodic=False):
    """Index into src of the nearest value for every dst value."""
    d = dst[:, None] - src[None, :]
    if periodic:
        d = (d + 180.0) % 360.0 - 180.0
    return np.abs(d).argmin(axis=1)


_NUTRIENT_REPORTED = set()


def load_nutrients(lat, lon):
    """GLODAP surface Si and PO4 (umol kg-1) mapped onto the given grid. The mapped climatology has a few small
    negative values (unphysical, from the objective mapping); they are set to 0, as in the uncertainty scripts
    (product_uncertainty_mc_C.py, product_constants_uncertainty.py), so that every mocsy call of the study uses
    the same nutrients. The number of clipped cells is printed once per grid."""
    out = []
    for var in ('silicate', 'PO4'):
        with netCDF4.Dataset(OBS_DIR / f'{var}_GLODAPv2.2016b_surface_{GRID}.nc') as f:
            a = np.ma.filled(f.variables[var][:].astype('f8'), np.nan).squeeze()
            jy = nearest_index(np.asarray(f['lat'][:], 'f8'), lat)
            ix = nearest_index(np.asarray(f['lon'][:], 'f8'), lon, periodic=True)
        b = a[np.ix_(jy, ix)]
        neg = b < 0
        key = (var, lat.size, lon.size)
        if neg.any() and key not in _NUTRIENT_REPORTED:
            print(f'  GLODAP {var}: {int(neg.sum())} cells < 0 (min {np.nanmin(b):.3f} umol/kg) set to 0', flush=True)
            _NUTRIENT_REPORTED.add(key)
        out.append(np.where(neg, 0.0, b))
    return out


def omega_cells(t, s, dic, alk, sil, phos, lat):
    """Omega_arag for 1-D arrays of valid cells, all mol kg-1 except t and s."""
    n = t.size
    res = mocsy.mvars(temp=cm.floor_temperature(t, s), sal=s, alk=alk, dic=dic, sil=sil, phos=phos,
                      patm=np.ones(n, dtype=float),
                      depth=np.zeros(n, dtype=float),
                      lat=lat, optcon='mol/kg', optt='Tpot', optp='m', optb=MOCSY_OPTB, optk1k2=MOCSY_OPTK1K2)
    om = np.asarray(res[6], dtype=float)
    return np.where(cm.valid_carbonate(res[0], om), om, np.nan)


def annual_state_omega(t, s, dic, alk, om_month, si, po4, lat2):
    """Omega_arag of the annual-mean state for whole years of monthly fields (n_months = 12 k, ny, nx); dic and alk
    in mol kg-1, si and po4 in umol kg-1. Valid only where all twelve monthly Omega_arag om_month are valid
    (the rule of the monthly mean). Returns (annual-state Omega, mean of the monthly Omega), both (k, ny, nx)."""
    ny, nx = t.shape[1:]
    yr = lambda a: a.reshape(-1, 12, ny, nx)
    om_mm = yr(om_month).mean(axis=1)                    # NaN unless all twelve months are valid
    ok = np.isfinite(om_mm)
    ta, sa, da, aa = (yr(a).mean(axis=1) for a in (t, s, dic, alk))
    om_st = np.full(om_mm.shape, np.nan)
    if ok.any():
        om_st[ok] = omega_cells(ta[ok], sa[ok], da[ok], aa[ok],
                                np.broadcast_to(si * 1e-6, ok.shape)[ok],
                                np.broadcast_to(po4 * 1e-6, ok.shape)[ok],
                                np.broadcast_to(lat2, ok.shape)[ok])
    return om_st, om_mm


def global_mean(field, lat, lon, mask):
    """area-weighted mean over mask (oa_config.area_weights)"""
    w = oa_config.area_weights(lat, lon) * mask
    return np.nansum(field * w, axis=(-2, -1)) / w.sum()


def main():
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    link = OBS_DIR / 'raw' / SRC.name
    link.parent.mkdir(exist_ok=True)
    if not link.exists():
        link.symlink_to(SRC)

    with netCDF4.Dataset(SRC) as f:
        lat = np.asarray(f['lat'][:], 'f8')
        lon = np.asarray(f['lon'][:], 'f8')
        tv = f['time']
        dates = netCDF4.num2date(tv[:], tv.units, getattr(tv, 'calendar', 'standard'))
        nt = len(dates) - len(dates) % 12
        years = np.array([d.year for d in dates[:nt:12]])
        ny, nx = lat.size, lon.size
        lat2 = np.broadcast_to(lat[:, None], (ny, nx))
        si, po4 = load_nutrients(lat, lon)

        om_yr = np.full((nt // 12, ny, nx), np.nan, 'f4')
        mm_yr = np.full((nt // 12, ny, nx), np.nan, 'f4')
        mw_yr = np.full((nt // 12, ny, nx), np.nan, 'f4')
        prod_yr = np.full((nt // 12, ny, nx), np.nan, 'f4')
        n_ok = n_cull = 0
        for i0 in range(0, nt, CHUNK):
            i1 = min(i0 + CHUNK, nt)
            t = read(f['temperature'], i0, i1)
            s = read(f['salinity'], i0, i1)
            dic = read(f['dic'], i0, i1) * 1e-6
            alk = read(f['talk'], i0, i1) * 1e-6
            ok = (np.isfinite(t) & np.isfinite(s) & np.isfinite(dic) & np.isfinite(alk)
                  & (s > 0) & (dic > 0) & (alk > 0))
            ok &= np.isfinite(si) & np.isfinite(po4)
            om = np.full(t.shape, np.nan)
            om[ok] = omega_cells(t[ok], s[ok], dic[ok], alk[ok],
                                 np.broadcast_to(si * 1e-6, t.shape)[ok],
                                 np.broadcast_to(po4 * 1e-6, t.shape)[ok],
                                 np.broadcast_to(lat2, t.shape)[ok])
            n_ok += int(ok.sum())
            n_cull += int((ok & ~np.isfinite(om)).sum())
            om_yr[i0 // 12:i1 // 12], mm_yr[i0 // 12:i1 // 12] = annual_state_omega(t, s, dic, alk, om, si, po4, lat2)
            days = np.array([calendar.monthrange(d.year, d.month)[1] for d in dates[i0:i1]], 'f8').reshape(-1, 12)
            wd = (days / days.sum(axis=1, keepdims=True))[:, :, None, None]
            mw_yr[i0 // 12:i1 // 12] = (om.reshape(-1, 12, ny, nx) * wd).sum(axis=1)      # NaN if any month is NaN
            prod_yr[i0 // 12:i1 // 12] = read(f['omega_ar'], i0, i1).reshape(-1, 12, ny, nx).mean(axis=1)
            print(f'{dates[i0].year}-{dates[i1 - 1].year} done', flush=True)

    # drop years with almost no data (January 1982 has no dic, so 1982 is empty)
    n_valid = np.isfinite(om_yr).sum((1, 2))
    keep = n_valid > 0.5 * n_valid.max()
    if not keep.all():
        print(f'dropping years with <50 % coverage: {years[~keep].tolist()}')
    years, om_yr, mm_yr, mw_yr, prod_yr = years[keep], om_yr[keep], mm_yr[keep], mw_yr[keep], prod_yr[keep]

    mask = (np.all(np.isfinite(om_yr), axis=0) & np.all(np.isfinite(mm_yr), axis=0)
            & np.all(np.isfinite(prod_yr), axis=0))
    gm = global_mean(om_yr, lat, lon, mask)
    gm_mm = global_mean(mm_yr, lat, lon, mask)
    gm_mw = global_mean(mw_yr, lat, lon, mask)
    gm_prod = global_mean(prod_yr, lat, lon, mask)

    with netCDF4.Dataset(OUT, 'w', format='NETCDF4') as g:
        g.createDimension('year', years.size)
        g.createDimension('lat', ny)
        g.createDimension('lon', nx)
        for name, dim, vals, units in (('year', 'year', years, '1'),
                                       ('lat', 'lat', lat, 'degrees_north'),
                                       ('lon', 'lon', lon, 'degrees_east')):
            v = g.createVariable(name, 'f8' if name != 'year' else 'i4', (dim,))
            v.units = units
            v[:] = vals
        for name, data, ln in (
                ('omega_arag', om_yr, 'surface aragonite saturation state of the annual-mean state, mocsy'),
                ('omega_arag_monthly_mean', mm_yr, 'mean of the twelve monthly aragonite saturation states, mocsy'),
                ('omega_arag_monthly_mean_dw', mw_yr, 'mean of the twelve monthly aragonite saturation states weighted '
                 'with the number of days of each month, mocsy'),
                ('omega_arag_product', prod_yr, 'mean of the twelve monthly OceanSODA omega_ar')):
            v = g.createVariable(name, 'f4', ('year', 'lat', 'lon'), zlib=True,
                                 complevel=1, fill_value=FILL)
            v.long_name, v.units = ln, '1'
            v[:] = np.ma.masked_invalid(data)
        v = g.createVariable('mask', 'i1', ('lat', 'lon'))
        v.long_name = 'cells valid in every month, used for the global means'
        v[:] = mask.astype('i1')
        for name, data, ln in (('omega_arag_gm', gm, 'area-weighted global mean of omega_arag over mask'),
                               ('omega_arag_monthly_mean_gm', gm_mm, 'same for omega_arag_monthly_mean'),
                               ('omega_arag_monthly_mean_dw_gm', gm_mw, 'same for omega_arag_monthly_mean_dw'),
                               ('omega_arag_product_gm', gm_prod, 'same for OceanSODA omega_ar')):
            v = g.createVariable(name, 'f8', ('year',))
            v.long_name, v.units = ln, '1'
            v[:] = data
        g.setncatts(dict(
            title='Surface aragonite saturation state from OceanSODA-ETHZ v2025',
            method=('mocsy 2.4.0 mvars, optcon=mol/kg, optt=Tpot, optp=m, depth 0 m, patm 1 atm, '
                    'default ' + oa_config.CONSTANTS + '; Si and PO4 from GLODAPv2.2016b surface climatology; '
                    f'values outside (0, {OMEGA_MAX:g}) set to missing; annual omega_arag = Omega_arag of the '
                    'annual means of T, S, DIC and TA, where all 12 months are valid'),
            source=str(SRC),
            history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} calc_omega_arag_obs.py'))

    area = oa_config.area_weights(lat, lon)
    print(f'\nmask covers {100 * (area * mask).sum() / area.sum():.1f} % of the weighted sphere area, '
          f'culled {100 * n_cull / max(n_ok, 1):.3f} % of cell-months')
    print(f'mocsy   {years[0]} {gm[0]:.3f}  {years[-1]} {gm[-1]:.3f}  '
          f'trend {10 * np.polyfit(years, gm, 1)[0]:+.4f} per decade')
    print(f'product {years[0]} {gm_prod[0]:.3f}  {years[-1]} {gm_prod[-1]:.3f}  '
          f'trend {10 * np.polyfit(years, gm_prod, 1)[0]:+.4f} per decade')
    d = gm_mm - gm
    print(f'day-weighted minus equal-weighted mean of monthly: {(gm_mw - gm_mm).min():+.5f} to {(gm_mw - gm_mm).max():+.5f}')
    print(f'mean of monthly minus annual state: {d.min():+.4f} to {d.max():+.4f} '
          f'({years[0]} {d[0]:+.4f}, {years[-1]} {d[-1]:+.4f})')


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
