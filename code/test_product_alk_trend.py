#!/usr/bin/env python
"""
test_product_alk_trend.py  (do the products already contain an alkalinity trend beyond salinity?)
=========================
The alkalinity adjustment assumes that the products miss the rise of surface alkalinity from weaker calcification
(Barrett et al.: 0.063 umol/kg per year in the observations, 0.072 on the common area). Three products estimate
alkalinity mainly from salinity; JMA-MLR maps it by multiple linear regression on temperature, salinity, sea surface
height and chlorophyll, and could therefore carry a trend. Here, for each product (obs_<ref>.nc of calc_product_obs.py,
the annual-mean states), on the common area of the paper (constraint_results.pkl, results_common), with the weights
of the run:
  the area-mean ALK, S and T of every year, and over 1990-2024 (the years all four share) and over each product's
  own record: the trend of ALK, of the salinity-normalised ALK (ALK x 35 / S), and of the residual of ALK after a
  regression on S and T over that period (the part neither salinity nor temperature explain), in umol/kg per year,
  and the share of the expected missing rise (0.072 umol/kg per year) that this residual trend would cover.
No figures.
Usage (code/):  OA_RUN=u74 python -u test_product_alk_trend.py
"""
import pickle

import netCDF4
import numpy as np

import oa_config
import calc_omega_arag_obs as co

REFS = ('OceanSODA', 'UExP', 'JMA', 'CMEMS')
RATE = 0.0718          # umol/kg per year, the common-area rate 0.063 / p of structural_adjustments
COMMON = (1990, 2024)

pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
r = pk['results_common']['UExP']
lat, lon, mask = np.asarray(r['lat'], 'f8'), np.asarray(r['lon'], 'f8'), np.asarray(r['mask'], bool)
w = oa_config.area_weights(lat, lon)


def series(ref):
    with netCDF4.Dataset(oa_config.DATA / 'product' / f'obs_{ref}.nc') as f:
        yy = np.asarray(f['year'][:]).astype(int)
        jy = co.nearest_index(np.asarray(f['lat'][:], 'f8'), lat)
        ix = co.nearest_index(np.asarray(f['lon'][:], 'f8'), lon, periodic=True)
        out = {}
        for v in ('talk', 'so', 'thetao'):
            a = np.ma.filled(f[v][:].astype('f8'), np.nan)[:, jy][:, :, ix]
            ok = mask & np.all(np.isfinite(a), axis=0)
            out[v] = np.array([np.sum(x[ok] * w[ok]) / np.sum(w[ok]) for x in a])
    return yy, out


def slope(t, y):
    return float(np.polyfit(t, y, 1)[0])


print(f'common area: {100 * w[mask].sum() / w.sum():.1f} % of the weights; expected missing rise {RATE} umol/kg per year')
print(f'{"product":10s} {"period":11s} {"ALK":>8s} {"nALK":>8s} {"resid":>8s} {"share":>6s} {"S":>9s} {"T":>8s}   '
      f'(trends per year: ALK, ALK x 35 / S, ALK residual after regression on S and T, umol/kg; S in psu, T in degC)')
for ref in REFS:
    try:
        yy, s = series(ref)
    except Exception as e:
        print(f'{ref:10s} failed: {type(e).__name__}: {e}')
        continue
    for lab, (y0, y1) in (('1990-2024', COMMON), ('own', (int(yy.min()), min(int(yy.max()), 2024)))):
        k = (yy >= y0) & (yy <= y1)
        t = yy[k].astype('f8')
        A, S, T = s['talk'][k], s['so'][k], s['thetao'][k]
        X = np.column_stack([np.ones(k.sum()), S, T])
        b, *_ = np.linalg.lstsq(X, A, rcond=None)
        res = A - X @ b
        sr = slope(t, res)
        print(f'{ref:10s} {y0}-{y1} {slope(t, A):+8.4f} {slope(t, A * 35 / S):+8.4f} {sr:+8.4f} {sr / RATE:+6.2f} '
              f'{slope(t, S):+9.5f} {slope(t, T):+8.4f}')
