#!/usr/bin/env python
"""
calc_omega_arag_products.py
===========================
Surface Omega_arag from further observation-based products, computed exactly as for
OceanSODA in calc_omega_arag_obs.py (mocsy 2.4.0, optcon='mol/kg', optt='Tpot', optp='m',
GLODAPv2.2016b surface Si and PO4, values outside (0, 20) culled), so that all products and
the models share one chemistry:

  UExP-FNN-U v2026-1   DIC, TA, subskin T (K) and S from the product file (1980-2025)
  JMA-MLR v2025        DIC, TA from the product; T = MGDSST, S = 35*DIC/nDIC
                       (from make_jma_ts.py)                                    (1990-2024)
  CMEMS-LSCE-FFNN      DIC, TA from the product; T = OSTIA, S = CNR SSS (climatology before 1993)
                       (1 deg file from make_cmems_ts.py)                       (1985-2025)

Per product: annual Omega_arag of the annual-mean state (mocsy on the annual means of T, S, DIC,
TA, as for the CMIP6 models, whose inputs are annual means), only where all 12 months are valid
(monthly Omega_arag within (0, 20)), years with less than half the usual coverage dropped, and a
fixed mask of cells valid in every remaining year. Kept for comparison: the mean of the twelve
monthly Omega_arag (omega_arag_monthly_mean) and the mean of the product's own monthly Omega
(omega_arag_product, where provided; compare it with omega_arag_monthly_mean for the constants).

Output: data/obs/omega_arag_Oyr_<product>.nc  (same layout as the OceanSODA file)

Usage:
    python calc_omega_arag_products.py                 # all products
    python calc_omega_arag_products.py --products UExP-FNN-U-v2026-1
"""
from __future__ import annotations

import argparse
import oa_config   # run: data folder, mocsy constants, GFDL density (OA_RUN)
import datetime
import warnings
from pathlib import Path

import netCDF4
import numpy as np

import calc_omega_arag_obs as co          # nutrients, mocsy call, reader (imports mocsy)
from make_jma_ts import year_month

OBS = oa_config.OBS_DIR
PRODUCTS = {
    'UExP-FNN-U-v2026-1': dict(
        file=OBS / 'UExP-FNN-U' / 'Fordetal_UExP-FNN-U_surface-carbonate-system_v2026-1.nc',
        order='lon,lat,time',
        vars=dict(t='subskin_temp', s='subskin_salinity', dic='dic', ta='ta'),
        t_kelvin=True, omega_product='saturation_aragonite'),
    'JMA-MLR-v2025': dict(
        file=OBS / 'JMA-MLR_v2025' / 'JMA_co2map_1990-2024_v2025.nc',
        ts_file=OBS / 'JMA-MLR_v2025' / 'JMA_co2map_TS_1990-2024_v2025.nc',
        order='time,lat,lon',
        vars=dict(t='sst', s='sss', dic='dic', ta='ta'),
        t_kelvin=False, omega_product=None),
    'CMEMS-LSCE-FFNN': dict(
        file=OBS / 'CMEMS-LSCE-FFNN' / 'CMEMS-LSCE-FFNN_1deg_1985-2025.nc',   # from make_cmems_ts.py
        order='time,lat,lon',
        vars=dict(t='sst', s='sss', dic='dic', ta='ta'),
        t_kelvin=False, omega_product='omega_ar'),
}
CHUNK = 60


class Reader:
    """Reads months i0:i1 of a variable as (time, lat, lon) float64 with NaN for missing."""

    def __init__(self, path, order):
        self.f = netCDF4.Dataset(path)
        self.order = order.split(',')

    def __call__(self, name, i0, i1):
        v = self.f[name]
        dims = [d for d in v.dimensions]
        sl = tuple(slice(i0, i1) if d == 'time' else slice(None) for d in dims)
        a = np.ma.filled(np.ma.asarray(v[sl]).astype('f8'), np.nan)
        a = np.transpose(a, [dims.index('time'), dims.index('lat'), dims.index('lon')])
        a[np.abs(a) > 1e10] = np.nan
        return a

    def close(self):
        self.f.close()


def process(name, cfg):
    print(f'\n=== {name}', flush=True)
    main = Reader(cfg['file'], cfg['order'])
    ts = Reader(cfg['ts_file'], 'time,lat,lon') if cfg.get('ts_file') else main
    lat = np.asarray(main.f['lat'][:], 'f8')
    lon = np.asarray(main.f['lon'][:], 'f8')
    ym = year_month(main.f['time'])
    if ts is not main:
        ym_ts = year_month(ts.f['time'])
        if ym_ts != ym:
            raise ValueError(f'{name}: time axes of product and T/S file differ')
        if not (np.allclose(ts.f['lat'][:], lat) and np.allclose(ts.f['lon'][:], lon)):
            raise ValueError(f'{name}: grids of product and T/S file differ')

    # whole years only, starting in January
    first = next(i for i, (y, m) in enumerate(ym) if m == 1)
    nt = (len(ym) - first) // 12 * 12
    years = np.array([ym[first + 12 * k][0] for k in range(nt // 12)])
    ny, nx = lat.size, lon.size
    lat2 = np.broadcast_to(lat[:, None], (ny, nx))
    si, po4 = co.load_nutrients(lat, lon)
    v = cfg['vars']

    om_yr = np.full((nt // 12, ny, nx), np.nan, 'f4')
    mm_yr = np.full((nt // 12, ny, nx), np.nan, 'f4')
    prod_yr = np.full((nt // 12, ny, nx), np.nan, 'f4') if cfg['omega_product'] else None
    n_ok = n_cull = 0
    for c0 in range(0, nt, CHUNK):
        c1 = min(c0 + CHUNK, nt)
        i0, i1 = first + c0, first + c1
        t = ts(v['t'], i0, i1) - (273.15 if cfg['t_kelvin'] else 0.0)
        s = ts(v['s'], i0, i1)
        dic = main(v['dic'], i0, i1) * 1e-6
        alk = main(v['ta'], i0, i1) * 1e-6
        ok = (np.isfinite(t) & np.isfinite(s) & np.isfinite(dic) & np.isfinite(alk)
              & (s > 0) & (dic > 0) & (alk > 0))
        ok &= np.isfinite(si) & np.isfinite(po4)
        om = np.full(t.shape, np.nan)
        if ok.any():
            om[ok] = co.omega_cells(t[ok], s[ok], dic[ok], alk[ok],
                                    np.broadcast_to(si * 1e-6, t.shape)[ok],
                                    np.broadcast_to(po4 * 1e-6, t.shape)[ok],
                                    np.broadcast_to(lat2, t.shape)[ok])
        n_ok += int(ok.sum())
        n_cull += int((ok & ~np.isfinite(om)).sum())
        om_yr[c0 // 12:c1 // 12], mm_yr[c0 // 12:c1 // 12] = co.annual_state_omega(t, s, dic, alk, om, si, po4, lat2)
        if prod_yr is not None:
            prod_yr[c0 // 12:c1 // 12] = main(cfg['omega_product'], i0, i1).reshape(-1, 12, ny, nx).mean(axis=1)
        print(f'  {ym[i0][0]}-{ym[i1 - 1][0]} done', flush=True)
    main.close()
    if ts is not main:
        ts.close()

    n_valid = np.isfinite(om_yr).sum((1, 2))
    keep = n_valid > 0.5 * n_valid.max()
    if not keep.all():
        print(f'  dropping years with <50 % coverage: {years[~keep].tolist()}')
    years, om_yr, mm_yr = years[keep], om_yr[keep], mm_yr[keep]
    mask = np.all(np.isfinite(om_yr), axis=0) & np.all(np.isfinite(mm_yr), axis=0)
    if prod_yr is not None:
        prod_yr = prod_yr[keep]
        mask &= np.all(np.isfinite(prod_yr), axis=0)

    gm = co.global_mean(om_yr, lat, lon, mask)
    gm_mm = co.global_mean(mm_yr, lat, lon, mask)
    out =co.OBS_DIR / f'omega_arag_Oyr_{name}.nc'
    with netCDF4.Dataset(out, 'w', format='NETCDF4') as g:
        g.createDimension('year', years.size)
        g.createDimension('lat', ny)
        g.createDimension('lon', nx)
        for n_, d_, a_, u_, t_ in (('year', 'year', years, '1', 'i4'), ('lat', 'lat', lat, 'degrees_north', 'f8'),
                                   ('lon', 'lon', lon, 'degrees_east', 'f8')):
            x = g.createVariable(n_, t_, (d_,))
            x.units = u_
            x[:] = a_
        fields = [('omega_arag', om_yr, 'surface aragonite saturation state of the annual-mean state, mocsy'),
                  ('omega_arag_monthly_mean', mm_yr, 'mean of the twelve monthly aragonite saturation states, mocsy')]
        if prod_yr is not None:
            fields.append(('omega_arag_product', prod_yr, f'mean of the twelve monthly product {cfg["omega_product"]}'))
        for n_, a_, ln in fields:
            x = g.createVariable(n_, 'f4', ('year', 'lat', 'lon'), zlib=True, complevel=1, fill_value=co.FILL)
            x.long_name, x.units = ln, '1'
            x[:] = np.ma.masked_invalid(a_)
        x = g.createVariable('mask', 'i1', ('lat', 'lon'))
        x.long_name = 'cells valid in every year, used for the global means'
        x[:] = mask.astype('i1')
        x = g.createVariable('omega_arag_gm', 'f8', ('year',))
        x.long_name = 'area-weighted global mean of omega_arag over mask'
        x[:] = gm
        x = g.createVariable('omega_arag_monthly_mean_gm', 'f8', ('year',))
        x.long_name = 'same for omega_arag_monthly_mean'
        x[:] = gm_mm
        if prod_yr is not None:
            x = g.createVariable('omega_arag_product_gm', 'f8', ('year',))
            x[:] = co.global_mean(prod_yr, lat, lon, mask)
        g.setncatts(dict(
            title=f'Surface aragonite saturation state from {name}',
            method=('mocsy 2.4.0 mvars, optcon=mol/kg, optt=Tpot, optp=m, depth 0 m, patm 1 atm, default '
                    '' + oa_config.CONSTANTS + '; Si and PO4 from GLODAPv2.2016b surface climatology; annual omega_arag = '
                    'Omega_arag of the annual means of T, S, DIC and TA, where all 12 months are valid'),
            source=str(cfg['file']) + (f'; T/S {cfg["ts_file"]}' if cfg.get('ts_file') else ''),
            history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} calc_omega_arag_products.py'))

    area = oa_config.area_weights(lat, lon)
    print(f'  {years[0]}-{years[-1]}, mask covers {100 * (area * mask).sum() / area.sum():.1f} % of the weighted sphere, '
          f'culled {100 * n_cull / max(n_ok, 1):.3f} % of cell-months')
    print(f'  mocsy {years[0]} {gm[0]:.3f}  {years[-1]} {gm[-1]:.3f}  '
          f'trend {10 * np.polyfit(years, gm, 1)[0]:+.4f} per decade')
    d = gm_mm - gm
    print(f'  mean of monthly minus annual state: {d.min():+.4f} to {d.max():+.4f} '
          f'({years[0]} {d[0]:+.4f}, {years[-1]} {d[-1]:+.4f})')
    print(f'  written {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--products', nargs='+', default=list(PRODUCTS))
    a = ap.parse_args()
    for p in a.products:
        cfg = PRODUCTS[p]
        missing = [str(f) for f in (cfg['file'], cfg.get('ts_file')) if f is not None and not Path(f).exists()]
        if missing:
            print(f'\n=== {p}: skipped, missing {missing}')
            continue
        process(p, cfg)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
