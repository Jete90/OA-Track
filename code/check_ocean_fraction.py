#!/usr/bin/env python
"""
check_ocean_fraction.py  (the area weights of version 4: what the ocean-fraction map does to the grid)
=======================
Reads inputs/ESACCI_water_bodies/ocean_fraction_1deg.nc (oa_config.OCEAN_FRACTION_FILE) and prints: the ocean share of the sphere it
implies, the number of partly covered cells (0 < fraction < 1) and of cells without ocean, and the share of the
sphere area the fraction removes from whole-cell weighting. With a product or mask file as argument (obs_<ref>.nc of
the product folder, or constraint_results.pkl), also the cells of that coverage whose ocean fraction is below 0.05
(cells the map does not call ocean: inland seas, lakes, or a land-sea mismatch; they get almost no weight), with
their positions, and the coverage in percent of the ocean with both weightings.
Usage (code/):  python check_ocean_fraction.py [<obs_<ref>.nc | constraint_results.pkl>]
"""
import pickle
import sys

import netCDF4
import numpy as np

import oa_config

f = oa_config.OCEAN_FRACTION_FILE
assert f.exists(), f'{f} missing'
with netCDF4.Dataset(f) as g:
    lat = np.asarray(g['lat'][:], 'f8')
    lon = np.asarray(g['lon'][:], 'f8')
    frac = np.clip(np.ma.filled(g['ocean_fraction'][:].astype('f8'), 0.0), 0.0, 1.0)
cw = np.cos(np.deg2rad(lat))[:, None] * np.ones(lon.size)
print(f'{f.name}: {lat.size} x {lon.size}, lat {lat[0]} to {lat[-1]}, lon {lon[0]} to {lon[-1]}')
print(f'  ocean share of the sphere: {100 * (frac * cw).sum() / cw.sum():.2f} % (361 of 510 million km2 = 70.8 %)')
part = (frac > 0) & (frac < 1)
print(f'  cells partly ocean: {int(part.sum())} ({100 * (cw * part).sum() / (cw * (frac > 0)).sum():.1f} % of the ocean cells by area), '
      f'their mean fraction {frac[part].mean():.2f}; cells without ocean: {int((frac == 0).sum())}')
print(f'  area the ocean weights remove from whole-cell weighting of the ocean cells: '
      f'{100 * (cw * (frac > 0) * (1 - frac)).sum() / (cw * (frac > 0)).sum():.1f} %')

if len(sys.argv) > 1:
    p = sys.argv[1]
    if p.endswith('.pkl'):
        with open(p, 'rb') as h:
            P = pickle.load(h)
        # every reference carries its own grid (JMA-MLR runs on 0.5..359.5 E, the others on -179.5..179.5 E)
        masks = {'common area': (np.asarray(P['lat_c'], 'f8'), np.asarray(P['lon_c'], 'f8'), np.asarray(P['cm']).astype(bool))}
        masks.update({f'own area {r}': (np.asarray(P['results'][r]['lat'], 'f8'), np.asarray(P['results'][r]['lon'], 'f8'),
                                        np.asarray(P['results'][r]['mask']).astype(bool)) for r in P['results']})
    else:
        with netCDF4.Dataset(p) as g:
            masks = {'product coverage': (np.asarray(g['lat'][:], 'f8'), np.asarray(g['lon'][:], 'f8'), np.asarray(g['mask'][:]).astype(bool))}
    for name, (mlat, mlon, m) in masks.items():
        fr = oa_config._ocean_fraction(mlat, mlon)          # the lookup of the chain itself, on the grid of the mask
        cwm = np.cos(np.deg2rad(mlat))[:, None] * np.ones(mlon.size)
        l180 = np.where(mlon > 180, mlon - 360, mlon)
        low = m & (fr < 0.05)
        print(f'\n{name} ({p}): {int(m.sum())} cells; ocean fraction < 0.05 in {int(low.sum())} of them '
              f'({100 * (cwm * low).sum() / (cwm * m).sum():.2f} % of its whole-cell area); fraction < 0.5 in {int((m & (fr < 0.5)).sum())}')
        print(f'  whole-cell area of the coverage: {100 * (cwm * m).sum() / cwm.sum():.2f} % of the sphere; ocean-weighted: '
              f'{100 * (cwm * fr * m).sum() / (frac * cw).sum():.2f} % of the ocean of the map')
        if low.any():
            jj, ii = np.where(low)
            print('  cells with fraction < 0.05 (lat, lon, fraction), first 30: '
                  + ', '.join(f'({mlat[j]:.1f}, {l180[i]:.1f}, {fr[j, i]:.2f})' for j, i in list(zip(jj, ii))[:30]))
