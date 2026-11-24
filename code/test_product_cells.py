#!/usr/bin/env python
"""
test_product_cells.py  (two questions behind the irregular cells of diagnose_product.py, 7 November 2026)
=====================
1  mocsy at cold temperatures: the state of a Jiang cell with an extreme dSST sigma (81.5 N, 7.5 E, 1862: S 35.03,
   TA 2310.4, DIC 2108.2 umol/kg) with the temperature lowered from -1.0 to -6.0 degC: pH, pCO2, Omega_arag, Revelle.
   Shows where mocsy stops giving a carbonate system and what it returns instead.
2  the model without salinity in the historical years (section A): the smoothed, adjusted state of every model of
   OceanSODA-ETHZ at the cells -49.5 N 70.5 E, 2.5 N -50.5 E and -3.5 N 116.5 E in 1850 and 1900 (historical) and in
   1800 (control): T, S, DIC, alkalinity.
Usage (code/):  OA_RUN=u74 python -u test_product_cells.py
"""
import numpy as np

import calc_pi_xco2 as px
import calc_product_states as cps

print('===== 1  mocsy with the temperature lowered')
S, TA, DIC = 35.03, 2310.4e-6, 2108.2e-6
T = np.arange(-1.0, -6.01, -0.25)
n = T.size
out = px.mvars_all(T, np.full(n, S), np.full(n, DIC), np.full(n, TA), np.zeros(n), np.zeros(n), np.full(n, 81.5))
ph, pco2, om, rev = (np.asarray(out[k], float) for k in (0, 1, 6, 8))
for i in range(n):
    print(f'  T {T[i]:6.2f} degC: pH {ph[i]:.4g}, pCO2 {pco2[i]:.6g}, Omega_arag {om[i]:.6g}, Revelle {rev[i]:.6g}')
print('  (mocsy fill values, if any, appear as 1e+20 or similar)')

print('\n===== 2  the smoothed adjusted model states at the cells of section A (OceanSODA)')
R = cps.Reference('OceanSODA')
lat, lon = np.asarray(R.lat, 'f8'), np.asarray(R.lon, 'f8')
l180 = np.where(lon > 180, lon - 360, lon)
cells = [(-49.5, 70.5), (2.5, -50.5), (-3.5, 116.5)]
idx = [(int(np.argmin(np.abs(lat - a))), int(np.argmin(np.abs(l180 - b)))) for a, b in cells]
print('  cells (lat, lon) -> grid indices:', list(zip(cells, idx)))
for m in R.names:
    try:
        sm = cps.Smoothed(R, m)
        rows = []
        for year, series in ((1800, 'hist'), (1850, 'hist'), (1900, 'hist')):
            try:
                X = sm.state(year, series)
            except Exception as e:
                rows.append(f'{year}: not read ({e})')
                continue
            parts = []
            for (a, b), (j, i) in zip(cells, idx):
                v = {k: float(np.nanmean(X[k][:, j, i])) if k in X else np.nan for k in ('thetao', 'so', 'dissic', 'talk')}
                nm = {k: int(np.isnan(X[k][:, j, i]).sum()) if k in X else 12 for k in ('so', 'dissic')}
                parts.append(f'({a:.1f},{b:.1f}) T {v["thetao"]:6.2f} S {v["so"]:6.2f} DIC {v["dissic"]:7.1f} TA {v["talk"]:7.1f}'
                             f' [NaN months S {nm["so"]}, DIC {nm["dissic"]}]')
            rows.append(f'{year}: ' + ' | '.join(parts))
        sm.close()
        print(f'  {m}')
        for r in rows:
            print(f'    {r}')
    except Exception as e:
        print(f'  {m}: not read ({e})')
