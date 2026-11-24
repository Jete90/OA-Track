#!/usr/bin/env python
"""
make_jiang_ref.py
=================
Observation-based surface state of Jiang et al. (2023, NCEI 0259391, median of 14 ESMs bias-adjusted to SOCAT,
GLODAPv2 and CODAP-NA) for their decade 2010 (mean of January 2005 to December 2014), as a reference for the chain
of this study (calc_bias_adjusted.py --reference Jiang), in the same format as the product files of
calc_omega_arag_obs.py: T, S, DIC, ALK as given by Jiang et al. (their DIC and alkalinity, not recomputed) and
Omega_arag recomputed with mocsy and the constants of the run (oa_config.py; run l10 = the constants of Jiang et al.:
Lueker et al. 2000, Lee et al. 2010), GLODAPv2 Si and PO4 as for every product.

Input:  data/<run>/jiang2023/{Temperature,Salinity,DIC,TA}_median_historical.nc  (linked from inputs/jiang2023)
Output: data/<run>/obs/omega_arag_Oyr_Jiang2023-2010.nc  (year = 2010, one field)

Usage:  python make_jiang_ref.py
"""
import datetime

import netCDF4
import numpy as np

import oa_config
import calc_omega_arag_obs as co

DIR = oa_config.DATA / 'jiang2023'
OUT = co.OBS_DIR / 'omega_arag_Oyr_Jiang2023-2010.nc'
FILES = dict(temperature='Temperature', salinity='Salinity', dic='DIC', talk='TA')
DECADE = 2010


def read(name):
    with netCDF4.Dataset(DIR / f'{name}_median_historical.nc') as f:
        v = [k for k in f.variables if f[k].ndim == 3][0]
        t = np.asarray(f['time'][:], 'f8')
        i = int(np.where(t == DECADE)[0][0]) if np.any(t == DECADE) else t.size - 1     # 2010 is the last decade
        a = np.ma.filled(f[v][i].astype('f8'), np.nan)
        lon = np.asarray(f['longitude'][:], 'f8')
        lat = np.asarray(f['latitude'][:], 'f8')
    lon1 = (lon[0] if lon.ndim == 2 else lon) % 360.
    lat1 = lat[:, 0] if lat.ndim == 2 else lat
    a[np.abs(a) > 1e10] = np.nan
    return a, lat1, lon1, v, i


fields = {}
for k, name in FILES.items():
    a, lat, lon, v, i = read(name)
    fields[k] = a
    print(f'{name}: variable {v}, time index {i}')
o = np.argsort(lon)
lon = lon[o]
fields = {k: a[:, o] for k, a in fields.items()}
if np.nanmean(fields['temperature']) > 100:
    fields['temperature'] -= 273.15

si, po4 = co.load_nutrients(lat, lon)
lat2 = np.broadcast_to(lat[:, None], fields['dic'].shape)
ok = np.ones(lat2.shape, bool)
for a in list(fields.values()) + [si, po4]:
    ok &= np.isfinite(a)
om = np.full(lat2.shape, np.nan)
om[ok] = co.omega_cells(fields['temperature'][ok], fields['salinity'][ok], fields['dic'][ok] * 1e-6,
                        fields['talk'][ok] * 1e-6, si[ok] * 1e-6, po4[ok] * 1e-6, lat2[ok])
mask = np.isfinite(om)
w0 = oa_config.area_weights(lat, lon)
w = w0 * mask
print(f'Omega_arag {DECADE} (2005-2014), global mean over {100 * w.sum() / w0.sum():.0f} % '
      f'of the weighted grid ({oa_config.WEIGHTS}): {np.nansum(om * w) / w.sum():.3f}  ({oa_config.CONSTANTS})')

co.OBS_DIR.mkdir(parents=True, exist_ok=True)
with netCDF4.Dataset(OUT, 'w') as g:
    g.title = 'Jiang et al. (2023), surface state of the decade 2010 (January 2005 to December 2014)'
    g.source = 'NCEI accession 0259391, median of 14 CMIP6 ESMs bias-adjusted to SOCAT, GLODAPv2 and CODAP-NA'
    g.method = ('T, S, DIC, ALK of Jiang et al.; Omega_arag recomputed with mocsy, optt=Tpot, optp=m, '
                + oa_config.CONSTANTS + '; GLODAPv2.2016b surface Si and PO4')
    g.history = f'{datetime.datetime.now():%Y-%m-%d %H:%M} make_jiang_ref.py'
    g.createDimension('time', 1)
    g.createDimension('year', 1)
    g.createDimension('lat', lat.size)
    g.createDimension('lon', lon.size)
    for n_, vals, u in (('lat', lat, 'degrees_north'), ('lon', lon, 'degrees_east')):
        x = g.createVariable(n_, 'f8', (n_,))
        x.units = u
        x[:] = vals
    x = g.createVariable('time', 'f8', ('time',))
    x.units = 'days since 2010-01-01'
    x[:] = [0.0]
    g.createVariable('year', 'i4', ('year',))[:] = [DECADE]
    for k, a in fields.items():
        x = g.createVariable(k, 'f4', ('time', 'lat', 'lon'), zlib=True, fill_value=np.float32(1e20))
        x.units = 'degC' if k == 'temperature' else ('psu' if k == 'salinity' else 'umol kg-1')
        x[:] = np.ma.masked_invalid(a[None])
    x = g.createVariable('omega_arag', 'f4', ('year', 'lat', 'lon'), zlib=True, fill_value=np.float32(1e20))
    x[:] = np.ma.masked_invalid(om[None])
    x = g.createVariable('mask', 'i1', ('lat', 'lon'))
    x[:] = mask.astype('i1')
print(f'saved {OUT}')
