#!/usr/bin/env python
"""
calc_omega_arag_anymonth.py
===========================
Annual surface Omega_arag of the four observation-based products from ALL available months:
a cell enters a year as soon as it has one valid month (valid: T, S, DIC, TA present and the
monthly Omega_arag within (0, 20)); the annual value is Omega_arag of the mean state over the
valid months (means of T, S, DIC and TA over those months), the annual-state definition of
calc_omega_arag_obs.py / calc_omega_arag_products.py. Same chemistry as those scripts.
Only used for the "all cells, no consistent mask" time series in Supplementary Fig. 1e. Note that cells
with data only in some months are seasonally biased (e.g. summer-only polar cells).

Years with fewer than half of the maximum number of valid cells are dropped, as in the main files.

Output: data/obs/omega_arag_anymonth_Oyr_<product>.nc  (year, lat, lon, omega_arag, n_months)

Usage:
    python calc_omega_arag_anymonth.py
    python calc_omega_arag_anymonth.py --products UExP-FNN-U-v2026-1
"""
from __future__ import annotations

import argparse

import netCDF4
import numpy as np

import calc_omega_arag_obs as co
import calc_omega_arag_products as cp
import oa_config

PRODUCTS = {'OceanSODA-ETHZ-v2025': dict(file=co.SRC, vars=dict(t='temperature', s='salinity', dic='dic', ta='talk'),
                                         t_kelvin=False)}
PRODUCTS.update({k: v for k, v in cp.PRODUCTS.items()})
CHUNK = 60


def rd(reader, var, i0, i1):
    """(time, lat, lon) slice; falls back to the plain time-first reader if dims are not named time/lat/lon."""
    try:
        return reader(var, i0, i1)
    except ValueError:
        return co.read(reader.f[var], i0, i1)


def process(name, cfg):
    print(f'\n=== {name}', flush=True)
    main = cp.Reader(cfg['file'], 'time,lat,lon')
    ts = cp.Reader(cfg['ts_file'], 'time,lat,lon') if cfg.get('ts_file') else main
    lat = np.asarray(main.f['lat'][:], 'f8')
    lon = np.asarray(main.f['lon'][:], 'f8')
    ym = [tuple(x) for x in cp.year_month(main.f['time'])]     # handles 'months since' (JMA)
    first = next(i for i, (y, m) in enumerate(ym) if m == 1)
    nt = (len(ym) - first) // 12 * 12
    years = np.array([ym[first + 12 * k][0] for k in range(nt // 12)])
    ny, nx = lat.size, lon.size
    lat2 = np.broadcast_to(lat[:, None], (ny, nx))
    si, po4 = co.load_nutrients(lat, lon)
    v = cfg['vars']

    om_yr = np.full((nt // 12, ny, nx), np.nan, 'f4')
    n_mon = np.zeros((nt // 12, ny, nx), 'i1')
    for c0 in range(0, nt, CHUNK):
        c1 = min(c0 + CHUNK, nt)
        i0, i1 = first + c0, first + c1
        t = rd(ts, v['t'], i0, i1) - (273.15 if cfg['t_kelvin'] else 0.0)
        s = rd(ts, v['s'], i0, i1)
        dic = rd(main, v['dic'], i0, i1) * 1e-6
        alk = rd(main, v['ta'], i0, i1) * 1e-6
        ok = (np.isfinite(t) & np.isfinite(s) & np.isfinite(dic) & np.isfinite(alk)
              & (s > 0) & (dic > 0) & (alk > 0))
        ok &= np.isfinite(si) & np.isfinite(po4)
        om = np.full(t.shape, np.nan)
        if ok.any():
            om[ok] = co.omega_cells(t[ok], s[ok], dic[ok], alk[ok],
                                    np.broadcast_to(si * 1e-6, t.shape)[ok],
                                    np.broadcast_to(po4 * 1e-6, t.shape)[ok],
                                    np.broadcast_to(lat2, t.shape)[ok])
        valid = np.isfinite(om).reshape(-1, 12, ny, nx)       # months with a valid Omega_arag
        cnt = valid.sum(axis=1)
        has = cnt > 0
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = {k: np.where(valid, a.reshape(-1, 12, ny, nx), 0.0).sum(axis=1) / cnt
                    for k, a in (('t', t), ('s', s), ('dic', dic), ('alk', alk))}
        oy = np.full(cnt.shape, np.nan)
        if has.any():
            oy[has] = co.omega_cells(mean['t'][has], mean['s'][has], mean['dic'][has], mean['alk'][has],
                                     np.broadcast_to(si * 1e-6, has.shape)[has],
                                     np.broadcast_to(po4 * 1e-6, has.shape)[has],
                                     np.broadcast_to(lat2, has.shape)[has])
        om_yr[c0 // 12:c1 // 12] = oy
        n_mon[c0 // 12:c1 // 12] = cnt
        print(f'  {ym[i0][0]}-{ym[i1 - 1][0]} done', flush=True)
    main.close()
    if ts is not main:
        ts.close()

    n_valid = np.isfinite(om_yr).sum((1, 2))
    keep = n_valid > 0.5 * n_valid.max()
    if not keep.all():
        print(f'  dropping years with <50 % coverage: {years[~keep].tolist()}')
    years, om_yr, n_mon = years[keep], om_yr[keep], n_mon[keep]

    out = co.OBS_DIR / f'omega_arag_anymonth_Oyr_{name}.nc'
    with netCDF4.Dataset(out, 'w', format='NETCDF4') as g:
        g.description = ('Annual surface Omega_arag of the mean state (T, S, DIC, TA) over all available months '
                         '(>= 1 valid month per year); '
                         'mocsy 2.4.0, GLODAPv2.2016b Si/PO4; see calc_omega_arag_anymonth.py')
        for n_, vals in (('year', years), ('lat', lat), ('lon', lon)):
            g.createDimension(n_, vals.size)
            g.createVariable(n_, 'f8' if n_ != 'year' else 'i4', (n_,))[:] = vals
        x = g.createVariable('omega_arag', 'f4', ('year', 'lat', 'lon'), zlib=True, complevel=1, fill_value=co.FILL)
        x[:] = np.ma.masked_invalid(om_yr)
        x = g.createVariable('n_months', 'i1', ('year', 'lat', 'lon'), zlib=True, complevel=1)
        x.long_name = 'number of valid months in the year'
        x[:] = n_mon
    area = oa_config.area_weights(lat, lon)
    part = (n_mon > 0) & (n_mon < 12)
    print(f'  {years[0]}-{years[-1]} written to {out}; cells with 1-11 months: '
          f'{100 * (area * part).sum((1, 2)).mean() / (area * (n_mon > 0)).sum((1, 2)).mean():.1f} % of the area with data')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--products', nargs='*', default=list(PRODUCTS))
    for p in ap.parse_args().products:
        process(p, PRODUCTS[p])
