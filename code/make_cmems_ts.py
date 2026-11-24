#!/usr/bin/env python
"""
make_cmems_ts.py
================
CMEMS-LSCE-FFNN on a 1x1 degree monthly grid, with the temperature and salinity the product
itself uses (QUID 015_008, Table 2):

  DIC, TA, Omega_ar   CMEMS-LSCE-FFNN multi-year (1985-2024) + near-real-time (2025),
                      0.25 deg, averaged over 4x4 blocks
  T                   OSTIA reanalysis 010_011 throughout (now reaching 2026-03), monthly 1 deg files
                      from download_ostia_monthly.sh
  S                   CNR multi-observation SSS (015_013), multi-year 1993-2024 + NRT 2025,
                      1/8 deg averaged over 8x8 blocks; 1985-1992 monthly climatology of 1993-2024
                      (as in the product)

A 1 deg cell is kept where at least half of the fine cells are valid.
Output: <OBS_DIR>/CMEMS-LSCE-FFNN/CMEMS-LSCE-FFNN_1deg_1985-2025.nc (oa_config.py)
        dims (time, lat, lon), lat -89.5..89.5, lon -179.5..179.5, variables dic, ta (umol/kg),
        sst (degC), sss (psu), omega_ar (product), sss_is_clim (1 for climatological months)

Usage (env_jens):
    python make_cmems_ts.py
"""
from __future__ import annotations

import datetime
from pathlib import Path

import netCDF4
import numpy as np

import oa_config
OBS = oa_config.OBS_DIR
CAR = {'my': OBS / 'CMEMS-LSCE-FFNN' / 'cmems_obs-mob_glo_bgc-car_my_irr-i.nc',
       'nrt': OBS / 'CMEMS-LSCE-FFNN' / 'cmems_obs-mob_glo_bgc-car_nrt_irr-i.nc'}
SSS = {'my': OBS / 'CMEMS-SSS-CNR' / 'cmems_obs-mob_glo_phy-sss_my_multi_P1M.nc',
       'nrt': OBS / 'CMEMS-SSS-CNR' / 'cmems_obs-mob_glo_phy-sss_nrt_multi_P1M.nc'}
OSTIA = OBS / 'OSTIA' / 'monthly_1deg'
OUT = OBS / 'CMEMS-LSCE-FFNN' / 'CMEMS-LSCE-FFNN_1deg_1985-2025.nc'
Y0, Y1 = 1985, 2025
SSS_CLIM = (1993, 2024)
LAT = np.arange(-89.5, 90, 1.0)
LON = np.arange(-179.5, 180, 1.0)
FILL = np.float32(1e20)


def coord(f, *names):
    for n in names:
        if n in f.variables:
            return np.asarray(f[n][:], 'f8')
    raise KeyError(names)


def year_month(tvar):
    d = netCDF4.num2date(tvar[:], tvar.units, getattr(tvar, 'calendar', 'standard'))
    return [(x.year, x.month) for x in d]


class Fine:
    """Monthly fields of a regular fine lat/lon grid, block-averaged to 1 deg."""

    def __init__(self, path):
        self.f = netCDF4.Dataset(path)
        self.lat = coord(self.f, 'latitude', 'lat')
        lon = coord(self.f, 'longitude', 'lon')
        self.k = int(round(1.0 / abs(self.lat[1] - self.lat[0])))
        self.flip = self.lat[0] > self.lat[-1]
        lon = np.where(lon >= 180, lon - 360, lon)
        self.order = np.argsort(lon)
        lon = lon[self.order]
        lat = self.lat[::-1] if self.flip else self.lat
        for name, a, n in (('lat', lat, 180), ('lon', lon, 360)):
            if a.size != n * self.k or abs(a[0] - (a[1] - a[0]) / 2 - (-90 if name == 'lat' else -180)) > 1e-3:
                raise ValueError(f'{path.name}: {name} is not a regular grid starting at the edge')
        self.ym = {ym: i for i, ym in enumerate(year_month(self.f['time']))}

    def __call__(self, name, year, month):
        v = self.f[name]
        i = self.ym[(year, month)]
        sl = tuple(i if d == 'time' else (0 if d == 'depth' else slice(None)) for d in v.dimensions)
        a = np.ma.filled(np.ma.asarray(v[sl]).astype('f8'), np.nan).squeeze()
        a[np.abs(a) > 1e10] = np.nan
        if self.flip:
            a = a[::-1]
        a = a[:, self.order]
        k = self.k
        b = a.reshape(180, k, 360, k)
        n = np.isfinite(b).sum(axis=(1, 3))
        m = np.nansum(b, axis=(1, 3)) / np.maximum(n, 1)
        return np.where(n >= k * k / 2, m, np.nan)

    def close(self):
        self.f.close()


def ostia(year, month):
    p = OSTIA / f'ostia_sst_{year}{month:02d}_1deg.nc'
    with netCDF4.Dataset(p) as f:
        lat, lon = coord(f, 'lat', 'latitude'), coord(f, 'lon', 'longitude')
        if not (np.allclose(lat, LAT) and np.allclose(lon, LON)):
            raise ValueError(f'{p.name}: unexpected grid')
        a = np.ma.filled(np.ma.asarray(f['analysed_sst'][:]).astype('f8'), np.nan).squeeze()
    a[np.abs(a) > 1e10] = np.nan
    return a - 273.15


def main():
    months = [(y, m) for y in range(Y0, Y1 + 1) for m in range(1, 13)]
    missing = [f'{y}{m:02d}' for y, m in months if not (OSTIA / f'ostia_sst_{y}{m:02d}_1deg.nc').exists()]
    if missing:
        raise SystemExit(f'OSTIA missing for {len(missing)} months, e.g. {missing[:5]}')

    car = {k: Fine(p) for k, p in CAR.items()}
    sss = {k: Fine(p) for k, p in SSS.items()}
    print('grids: carbon 1/%d deg, SSS 1/%d deg' % (car['my'].k, sss['my'].k), flush=True)

    # SSS: multi-year where available, NRT after; climatology of 1993-2024 for 1985-1992
    def sss_month(y, m):
        for k in ('my', 'nrt'):
            if (y, m) in sss[k].ym:
                return sss[k]('sos', y, m)
        return None

    clim = np.zeros((12, 180, 360))
    ncl = np.zeros((12, 180, 360))
    for y in range(SSS_CLIM[0], SSS_CLIM[1] + 1):
        for m in range(1, 13):
            a = sss_month(y, m)
            ok = np.isfinite(a)
            clim[m - 1][ok] += a[ok]
            ncl[m - 1] += ok
        print(f'  SSS climatology {y}', flush=True)
    clim = np.where(ncl >= 0.9 * (SSS_CLIM[1] - SSS_CLIM[0] + 1), clim / np.maximum(ncl, 1), np.nan)

    g = netCDF4.Dataset(OUT, 'w', format='NETCDF4')
    g.createDimension('time', len(months))
    g.createDimension('lat', 180)
    g.createDimension('lon', 360)
    t = g.createVariable('time', 'f8', ('time',))
    t.units, t.calendar = 'days since 1985-01-01', 'standard'
    t[:] = netCDF4.date2num([datetime.datetime(y, m, 15) for y, m in months], t.units, t.calendar)
    for n, a, u in (('lat', LAT, 'degrees_north'), ('lon', LON, 'degrees_east')):
        x = g.createVariable(n, 'f8', (n,))
        x.units = u
        x[:] = a
    out = {}
    for n, u, ln in (('dic', 'umol kg-1', 'DIC, CMEMS-LSCE-FFNN tco2'),
                     ('ta', 'umol kg-1', 'total alkalinity, CMEMS-LSCE-FFNN talk'),
                     ('omega_ar', '1', 'aragonite saturation state of the product'),
                     ('sst', 'degC', 'OSTIA SST reanalysis (010_011)'),
                     ('sss', 'psu', 'CNR SSS 015_013; 1985-1992 monthly climatology 1993-2024')):
        x = g.createVariable(n, 'f4', ('time', 'lat', 'lon'), zlib=True, fill_value=FILL)
        x.units, x.long_name = u, ln
        out[n] = x
    flag = g.createVariable('sss_is_clim', 'i1', ('time',))
    g.setncatts(dict(
        title='CMEMS-LSCE-FFNN carbonate system with its own T and S inputs, 1 deg monthly',
        source=f'{CAR["my"].name}, {CAR["nrt"].name}; {SSS["my"].name}, {SSS["nrt"].name}; OSTIA monthly 1 deg',
        method='block means of the fine grids, cell kept where at least half of the fine cells are valid',
        history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} make_cmems_ts.py'))

    for i, (y, m) in enumerate(months):
        src = 'my' if (y, m) in car['my'].ym else 'nrt'
        out['dic'][i] = np.ma.masked_invalid(car[src]('tco2', y, m))
        out['ta'][i] = np.ma.masked_invalid(car[src]('talk', y, m))
        out['omega_ar'][i] = np.ma.masked_invalid(car[src]('omega_ar', y, m))
        out['sst'][i] = np.ma.masked_invalid(ostia(y, m))
        s = sss_month(y, m)
        flag[i] = s is None
        out['sss'][i] = np.ma.masked_invalid(clim[m - 1] if s is None else s)
        if m == 12:
            print(f'  {y} done ({src}, SSS {"climatology" if s is None else "observed"})', flush=True)
    g.close()
    for d in (car, sss):
        for r in d.values():
            r.close()
    print(f'written {OUT}')


if __name__ == '__main__':
    main()
