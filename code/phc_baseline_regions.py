#!/usr/bin/env python
"""
phc_baseline_regions.py
=======================
Do the observation-based estimates of the Planetary Health Check (2026) differ more outside our common area? In
the step from its areas to the common area (Fig. 3c), only a part comes from the cells that only UExP-FNN-U has
(phc_area.py). Test whether the rest comes from a larger difference between the two observational
baselines outside the common area. All on the UExP-FNN-U grid with the weights of its file (area x mask_sfc), on the
cells both have, split into the common area and the rest:
  UExP-FNN-U, mean of the 12 months of 2024 and of 2005-2014 (cells with all months);
  Jiang et al. (2023), 1750 and decade 2010 (2005-2014) of Aragonite_median_historical.nc;
  per region: area share, the decline of the Planetary Health Check (2024 of UExP-FNN-U / 1750 of Jiang et al.),
  the decline within the data of Jiang et al. to their decade 2010 (consistent), the same with UExP-FNN-U 2005-2014
  instead (mixed baselines), and the relative difference of the two baselines in 2005-2014.
If the two baselines differ much more outside the common area, the area step of Fig. 3c partly contains baseline
differences. Reads only, writes nothing.
Usage (code/):  OA_RUN=u74 python -u phc_baseline_regions.py
"""
import pickle

import netCDF4
import numpy as np

import oa_config
import regional_common as rc

UEXP = str(oa_config.OBS_DIR / 'UExP-FNN-U/Fordetal_UExP-FNN-U_surface-carbonate-system_v2026-1.nc')
JFILE = f'{rc.ROOT}/jiang2023/Aragonite_median_historical.nc'

with netCDF4.Dataset(UEXP) as f:
    v = f['saturation_aragonite']
    assert v.dimensions == ('lon', 'lat', 'time')
    d = np.datetime64('1970-01-15') + np.asarray(f['time'][:], 'i8').astype('timedelta64[D]')
    yr = d.astype('datetime64[Y]').astype(int) + 1970

    def tmean(y0, y1):
        it = np.where((yr >= y0) & (yr <= y1))[0]
        assert it.size == 12 * (y1 - y0 + 1), (y0, y1, it.size)
        return np.ma.filled(v[:, :, it[0]:it[-1] + 1].astype('f8'), np.nan).mean(axis=2).T        # NaN unless all months
    U24, U10 = tmean(2024, 2024), tmean(2005, 2014)
    W = (np.ma.filled(f['area'][:].astype('f8'), np.nan) * np.ma.filled(f['mask_sfc'][:].astype('f8'), np.nan)).T
    ulat = np.asarray(f['lat'][:], 'f8')
    ulon = (np.asarray(f['lon'][:], 'f8') + 180.0) % 360.0 - 180.0

with netCDF4.Dataset(JFILE) as f:
    jy = list(np.asarray(f['time'][:]).astype(int))
    jlat = np.asarray(f['latitude'][:], 'f8')[:, 0]
    jlon = (np.asarray(f['longitude'][:], 'f8')[0, :] + 180.0) % 360.0 - 180.0
    J = np.ma.filled(f['aragonite'][:].astype('f8'), np.nan)
J[~np.isfinite(J) | (J <= 0) | (J > 10)] = np.nan
iy = np.abs(ulat[:, None] - jlat[None, :]).argmin(1)
ix = np.abs(((ulon[:, None] - jlon[None, :]) + 180) % 360 - 180).argmin(1)
J50, J10 = J[jy.index(1750)][np.ix_(iy, ix)], J[jy.index(2010)][np.ix_(iy, ix)]

# common area of this study (constraint_results.pkl, results_common) onto the UExP-FNN-U grid
with open(oa_config.DATA / 'constraint_results.pkl', 'rb') as f:
    r = pickle.load(f)['results_common']['UExP']
clat, clon = np.asarray(r['lat'], 'f8'), (np.asarray(r['lon'], 'f8') + 180.0) % 360.0 - 180.0
cm = np.asarray(r['mask']).astype(bool)
C = cm[np.ix_(np.abs(ulat[:, None] - clat[None, :]).argmin(1),
              np.abs(((ulon[:, None] - clon[None, :]) + 180) % 360 - 180).argmin(1))]

ok = np.isfinite(W) & (W > 0)
for a in (U24, U10, J50, J10):
    ok &= np.isfinite(a)
mean = lambda a, m: float(np.sum(a[m] * W[m]) / W[m].sum())
dec = lambda now, pi: 100 * (1 - now / pi)
print(f'cells with all four fields: {100 * W[ok].sum() / W[np.isfinite(W) & (W > 0)].sum():.1f} % of the weighted '
      'UExP-FNN-U area')
for lab, m in (('shared cells, all', ok), ('common area', ok & C), ('outside the common area', ok & ~C)):
    u24, u10, j50, j10 = (mean(a, m) for a in (U24, U10, J50, J10))
    print(f'\n{lab}: {100 * W[m].sum() / W[ok].sum():.1f} % of the shared cells')
    print(f'  UExP-FNN-U 2024 {u24:.4f}, 2005-2014 {u10:.4f}; Jiang et al. 1750 {j50:.4f}, 2005-2014 {j10:.4f}')
    print(f'  baselines in 2005-2014: UExP-FNN-U minus Jiang et al. {u10 - j10:+.4f} ({100 * (u10 / j10 - 1):+.2f} %)')
    print(f'  decline of the Planetary Health Check (UExP-FNN-U 2024 / Jiang et al. 1750): {dec(u24, j50):.2f} %')
    print(f'  decline to 2005-2014: within Jiang et al. {dec(j10, j50):.2f} %, with UExP-FNN-U as present state '
          f'{dec(u10, j50):.2f} % (baseline effect {dec(u10, j50) - dec(j10, j50):+.2f} points)')
