#!/usr/bin/env python
"""
make_jma_ts.py
==============
Temperature and salinity for the JMA Ocean CO2 Map (JMA-MLR, v2025), as used by JMA
(Iida et al. 2021, Table 1):

  SST  MGDSST daily 0.25 deg (download_mgdsst.sh) -> monthly means -> 1x1 deg block means
  SSS  recovered from the product itself: S = 35 * DIC / nDIC, which is exactly the
       MOVE/MRI.COM salinity JMA used to normalise DIC; checked against 35 * TA / nTA

Output: <JMA_DIR>/JMA_co2map_TS_1990-2024_v2025.nc with sst, sss, and the check sss_ta,
on the JMA 1x1 grid and monthly time axis.

Usage:  python make_jma_ts.py
"""
from __future__ import annotations

import datetime
import gzip
from pathlib import Path

import netCDF4
import numpy as np

import oa_config
MGD_DIR = oa_config.OBS_DIR / 'MGDSST/daily'
JMA_DIR = oa_config.OBS_DIR / 'JMA-MLR_v2025'
JMA_FILE = JMA_DIR / 'JMA_co2map_1990-2024_v2025.nc'
OUT = JMA_DIR / 'JMA_co2map_TS_1990-2024_v2025.nc'
MIN_VALID = 8          # of 16 quarter-degree cells needed for a 1-degree value


def year_month(t):
    """(year, month) for every time step; handles 'months since' units (not supported by num2date)."""
    units = t.units.strip()
    if units.lower().startswith('months since'):
        ref = units.split('since')[1].strip()
        y0, m0 = int(ref[0:4]), int(ref[5:7])
        k = np.round(np.asarray(t[:], 'f8')).astype(int) + (m0 - 1)
        return [(y0 + int(i) // 12, int(i) % 12 + 1) for i in k]
    dates = netCDF4.num2date(t[:], units, getattr(t, 'calendar', 'standard'))
    return [(d.year, d.month) for d in dates]


def read_mgdsst(path):
    """(year, month, day), SST (720, 1440) in degC, north to south, NaN for land/ice flags."""
    raw = gzip.open(path, 'rb').read() if path.suffix == '.gz' else path.read_bytes()
    lines = raw.replace(b'\r', b'').split(b'\n')
    head = lines[0]
    ymd = int(head[0:4]), int(head[4:8]), int(head[8:12])
    rows = [ln[:4320].ljust(4320) for ln in lines[1:721]]
    if len(rows) != 720:
        raise ValueError(f'{path.name}: {len(rows)} data records')
    vals = np.frombuffer(b''.join(rows), dtype='S3').astype(float).reshape(720, 1440)
    vals[(vals == 888) | (vals == 999)] = np.nan
    return ymd, vals / 10.0


def to_one_degree(sst):
    """0.25 deg north-to-south -> 1 deg south-to-north block means (180, 360)."""
    b = sst[::-1].reshape(180, 4, 360, 4)
    n = np.isfinite(b).sum(axis=(1, 3))
    m = np.nanmean(b, axis=(1, 3))
    m[n < MIN_VALID] = np.nan
    return m


def monthly_mgdsst(years):
    files = sorted(p for y in years for p in (MGD_DIR / str(y)).glob('*') if p.is_file())
    acc, cnt = {}, {}
    for i, p in enumerate(files):
        try:
            (y, m, _), sst = read_mgdsst(p)
        except Exception as e:                       # noqa: BLE001
            print(f'skip {p.name}: {e}')
            continue
        key = (y, m)
        if key not in acc:
            acc[key] = np.zeros_like(sst)
            cnt[key] = np.zeros_like(sst)
        ok = np.isfinite(sst)
        acc[key][ok] += sst[ok]
        cnt[key][ok] += 1
        if i % 500 == 0:
            print(f'  {i}/{len(files)} daily files', flush=True)
    out = {}
    for k in acc:
        mean = np.where(cnt[k] > 0, acc[k] / np.maximum(cnt[k], 1), np.nan)
        out[k] = to_one_degree(mean)
    return out


def main():
    with netCDF4.Dataset(JMA_FILE) as f:
        lat = np.asarray(f['lat'][:], 'f8')
        lon = np.asarray(f['lon'][:], 'f8')
        ym = year_month(f['time'])
        get = lambda v: np.ma.filled(f[v][:].astype('f8'), np.nan)
        dic, ndic, ta, nta = get('dic'), get('ndic'), get('ta'), get('nta')

    sss = 35.0 * dic / ndic
    sss_ta = 35.0 * ta / nta
    d = np.abs(sss - sss_ta)
    print(f'SSS from DIC vs TA: median |diff| {np.nanmedian(d):.4f}, 99th pct {np.nanpercentile(d, 99):.4f}')

    years = sorted({y for y, _ in ym})
    mgd = monthly_mgdsst(years)
    # MGDSST 1 deg grid: lat -89.5..89.5, lon 0.5..359.5; map onto the JMA grid by nearest index
    glat = np.arange(-89.5, 90, 1.0)
    glon = np.arange(0.5, 360, 1.0)
    jy = np.abs(lat[:, None] - glat[None, :]).argmin(axis=1)
    ix = np.abs(((lon[:, None] - glon[None, :]) + 180) % 360 - 180).argmin(axis=1)
    sst = np.full(dic.shape, np.nan)
    missing = []
    for i, k in enumerate(ym):
        if k in mgd:
            sst[i] = mgd[k][np.ix_(jy, ix)]
        else:
            missing.append(k)
    if missing:
        print(f'no MGDSST for {len(missing)} months, e.g. {missing[:5]}')

    with netCDF4.Dataset(OUT, 'w', format='NETCDF4') as g:
        g.createDimension('time', len(ym))
        g.createDimension('lat', lat.size)
        g.createDimension('lon', lon.size)
        v = g.createVariable('time', 'f8', ('time',))
        v.units, v.calendar = 'days since 1990-01-01', 'standard'
        v[:] = netCDF4.date2num([datetime.datetime(y, m, 15) for y, m in ym], v.units, v.calendar)
        for n, a, u in (('lat', lat, 'degrees_north'), ('lon', lon, 'degrees_east')):
            x = g.createVariable(n, 'f8', (n,))
            x.units = u
            x[:] = a
        for n, a, u, ln in (('sst', sst, 'degC', 'MGDSST monthly mean, 1x1 block mean of 0.25 deg'),
                            ('sss', sss, 'psu', 'salinity recovered as 35*DIC/nDIC (JMA MOVE/MRI.COM SSS)'),
                            ('sss_ta', sss_ta, 'psu', 'check: 35*TA/nTA')):
            x = g.createVariable(n, 'f4', ('time', 'lat', 'lon'), zlib=True, fill_value=np.float32(1e20))
            x.units, x.long_name = u, ln
            x[:] = np.ma.masked_invalid(a)
        g.setncatts(dict(source=f'{JMA_FILE.name}; MGDSST from {MGD_DIR}',
                         history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} make_jma_ts.py'))
    print(f'written {OUT}')


if __name__ == '__main__':
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
