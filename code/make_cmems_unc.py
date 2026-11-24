#!/usr/bin/env python
"""
make_cmems_unc.py
=================
The uncertainty fields of CMEMS-LSCE-FFNN (1 sigma) on the 1 x 1 degree monthly grid of CMEMS-LSCE-FFNN_1deg_1985-2025.nc
(make_cmems_ts.py), with the same block means of the 0.25 degree fields (a 1 degree cell is kept where at least half
of the fine cells are valid), for the mapping uncertainty of the product: product_uncertainty_mc_C.py and
level_uncertainty.py treat CMEMS-LSCE-FFNN like the other products (its own pCO2 and TA errors through mocsy).
    spco2_uncertainty -> fco2_unc (uatm; Pa converted; fCO2 = 0.9965 pCO2, the surface ratio at 1 atm, which is
                         0.996 to 0.997 over the ocean's temperature range)
    talk_uncertainty  -> ta_unc (umol kg-1), tco2_uncertainty -> dic_unc (umol kg-1, for reference)
The units are taken from the files; anything other than Pa, uatm, umol kg-1 or mol kg-1 stops the script.
Needs: cmems_obs-mob_glo_bgc-car_{my,nrt}_irr-i_uncertainty.nc (download_obs_products.sh cmems_unc).
Output: <OBS_DIR>/CMEMS-LSCE-FFNN/CMEMS-LSCE-FFNN_1deg_uncertainty_1985-2025.nc (oa_config.py)
Usage:  python make_cmems_unc.py
"""
from __future__ import annotations

import datetime

import netCDF4
import numpy as np

from make_cmems_ts import Fine, OBS, Y0, Y1, LAT, LON, FILL

UNC = {'my': OBS / 'CMEMS-LSCE-FFNN' / 'cmems_obs-mob_glo_bgc-car_my_irr-i_uncertainty.nc',
       'nrt': OBS / 'CMEMS-LSCE-FFNN' / 'cmems_obs-mob_glo_bgc-car_nrt_irr-i_uncertainty.nc'}
OUT = OBS / 'CMEMS-LSCE-FFNN' / 'CMEMS-LSCE-FFNN_1deg_uncertainty_1985-2025.nc'
FCO2_PER_PCO2 = 0.9965
VARS = {'fco2_unc': ('spco2_uncertainty', 'uatm', 'fCO2, 1 sigma (CMEMS-LSCE-FFNN spco2_uncertainty x 0.9965)'),
        'ta_unc': ('talk_uncertainty', 'umol kg-1', 'total alkalinity, 1 sigma (CMEMS-LSCE-FFNN talk_uncertainty)'),
        'dic_unc': ('tco2_uncertainty', 'umol kg-1', 'DIC, 1 sigma (CMEMS-LSCE-FFNN tco2_uncertainty)')}


def factor(units, kind):
    u = str(units).strip().lower().replace('µ', 'u')
    if kind == 'uatm':
        if u in ('pa',):
            return 1e6 / 101325.0 * FCO2_PER_PCO2
        if u in ('uatm', 'microatm', 'micro atm'):
            return FCO2_PER_PCO2
    else:
        if u in ('umol kg-1', 'umol/kg', 'micromol kg-1', 'micro mol kg-1', 'umol.kg-1'):
            return 1.0
        if u in ('mol kg-1', 'mol/kg', 'mol.kg-1'):
            return 1e6
    raise SystemExit(f'unexpected units {units!r} for {kind}: add the conversion')


def main():
    fine = {k: Fine(p) for k, p in UNC.items()}
    fac = {}
    for n, (src, kind, _) in VARS.items():
        if src not in fine['my'].f.variables:
            raise SystemExit(f'{src} not in {UNC["my"]}: variables {list(fine["my"].f.variables)}')
        fac[n] = factor(fine['my'].f[src].units, kind)
        print(f'{src}: units {fine["my"].f[src].units!r} -> x {fac[n]:.6g} ({kind})', flush=True)
    months = [(y, m) for y in range(Y0, Y1 + 1) for m in range(1, 13)]
    g = netCDF4.Dataset(OUT, 'w', format='NETCDF4')
    g.createDimension('time', len(months))
    g.createDimension('lat', LAT.size)
    g.createDimension('lon', LON.size)
    t = g.createVariable('time', 'f8', ('time',))
    t.units, t.calendar = 'days since 1985-01-01', 'standard'
    t[:] = netCDF4.date2num([datetime.datetime(y, m, 15) for y, m in months], t.units, t.calendar)
    for n, a, u in (('lat', LAT, 'degrees_north'), ('lon', LON, 'degrees_east')):
        x = g.createVariable(n, 'f8', (n,))
        x.units = u
        x[:] = a
    out = {}
    for n, (src, kind, ln) in VARS.items():
        x = g.createVariable(n, 'f4', ('time', 'lat', 'lon'), zlib=True, fill_value=FILL)
        x.units, x.long_name = ('uatm' if kind == 'uatm' else 'umol kg-1'), ln
        out[n] = x
    g.setncatts(dict(title='CMEMS-LSCE-FFNN uncertainties (1 sigma), 1 deg monthly, grid of CMEMS-LSCE-FFNN_1deg_1985-2025.nc',
                     source=f'{UNC["my"].name}, {UNC["nrt"].name}',
                     method='block means of the 0.25 deg fields, cell kept where at least half of the fine cells are valid',
                     history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} make_cmems_unc.py'))
    for i, (y, m) in enumerate(months):
        src = 'my' if (y, m) in fine['my'].ym else ('nrt' if (y, m) in fine['nrt'].ym else None)
        for n, (v, _, _) in VARS.items():
            out[n][i] = np.ma.masked_invalid(fine[src](v, y, m) * fac[n]) if src else np.ma.masked_all((LAT.size, LON.size))
        if m == 12:
            print(f'  {y} done ({src})', flush=True)
    g.close()
    for r in fine.values():
        r.close()
    print(f'written {OUT}')


if __name__ == '__main__':
    main()
