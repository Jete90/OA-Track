#!/usr/bin/env python
"""
phc_area.py  (18 November 2026)
===========
Why does averaging 1750 and 2024 over different areas make the decline of the Planetary Health Check (2026) larger?
In its framework (UExP-FNN-U v2026-1, 2024 = mean of the 12 months, cells with all 12; 1750 = the field of Jiang et
al. 2023, NCEI 0259391), both on the UExP-FNN-U grid and with the weights of the UExP-FNN-U file (area x mask_sfc), so
that only the cells differ:
  UExP-FNN-U 2024 on its full area, on the cells it shares with Jiang et al., and on the cells only it has;
  Jiang et al. 1750 on its full area, on the shared cells, and on the cells only it has;
  the decline for the four combinations (each state on its full area or on the shared cells);
  the cells only one of them has by latitude band (share of the weighted area, mean Omega_arag).
Reads only, writes nothing.
Usage (code/):  OA_RUN=u74 python -u phc_area.py
"""
import netCDF4
import numpy as np

import oa_config
import regional_common as rc

UEXP = str(oa_config.OBS_DIR / 'UExP-FNN-U/Fordetal_UExP-FNN-U_surface-carbonate-system_v2026-1.nc')
JFILE = f'{rc.ROOT}/jiang2023/Aragonite_median_historical.nc'
BANDS = [(-90, -60), (-60, -40), (-40, -20), (-20, 20), (20, 40), (40, 60), (60, 66), (66, 90)]

with netCDF4.Dataset(UEXP) as f:
    v = f['saturation_aragonite']
    assert v.dimensions == ('lon', 'lat', 'time')
    d = np.datetime64('1970-01-15') + np.asarray(f['time'][:], 'i8').astype('timedelta64[D]')
    it = np.where(d.astype('datetime64[Y]').astype(int) + 1970 == 2024)[0]
    assert it.size == 12
    U = np.ma.filled(v[:, :, it[0]:it[-1] + 1].astype('f8'), np.nan).mean(axis=2).T            # (lat, lon)
    W = (np.ma.filled(f['area'][:].astype('f8'), np.nan) * np.ma.filled(f['mask_sfc'][:].astype('f8'), np.nan)).T
    ulat = np.asarray(f['lat'][:], 'f8')
    ulon = (np.asarray(f['lon'][:], 'f8') + 180.0) % 360.0 - 180.0

with netCDF4.Dataset(JFILE) as f:
    jy = np.asarray(f['time'][:]).astype(int)
    jlat = np.asarray(f['latitude'][:], 'f8')[:, 0]
    jlon = (np.asarray(f['longitude'][:], 'f8')[0, :] + 180.0) % 360.0 - 180.0
    J = np.ma.filled(f['aragonite'][list(jy).index(1750)].astype('f8'), np.nan)
J[~np.isfinite(J) | (J <= 0) | (J > 10)] = np.nan
# Jiang et al. onto the UExP-FNN-U grid (both 1 degree: nearest cell)
iy = np.abs(ulat[:, None] - jlat[None, :]).argmin(1)
ix = np.abs(((ulon[:, None] - jlon[None, :]) + 180) % 360 - 180).argmin(1)
J = J[np.ix_(iy, ix)]

w_ok = np.isfinite(W) & (W > 0)
mU, mJ = w_ok & np.isfinite(U), w_ok & np.isfinite(J)
both, onlyU, onlyJ = mU & mJ, mU & ~mJ, mJ & ~mU
tot = W[mU | mJ].sum()
mean = lambda a, m: float(np.sum(a[m] * W[m]) / W[m].sum()) if m.any() else np.nan
share = lambda m: 100 * W[m].sum() / tot
dec = lambda now, pi: 100 * (1 - now / pi)

print(f'weighted area of the union = 100 %: UExP-FNN-U {share(mU):.1f} %, Jiang et al. {share(mJ):.1f} %, shared '
      f'{share(both):.1f} %, only UExP-FNN-U {share(onlyU):.1f} %, only Jiang et al. {share(onlyJ):.1f} %')
print(f'UExP-FNN-U 2024: full area {mean(U, mU):.4f}, shared {mean(U, both):.4f}, only UExP-FNN-U {mean(U, onlyU):.4f}')
print(f'Jiang et al. 1750: full area {mean(J, mJ):.4f}, shared {mean(J, both):.4f}, only Jiang et al. {mean(J, onlyJ):.4f}')
for lu, mu in (('full', mU), ('shared', both)):
    for lj, mj in (('full', mJ), ('shared', both)):
        print(f'  decline, 2024 on the {lu:6s} area of UExP-FNN-U, 1750 on the {lj:6s} area of Jiang et al.: '
              f'{dec(mean(U, mu), mean(J, mj)):.2f} %')
la = np.broadcast_to(ulat[:, None], U.shape)
print('cells only one of them has, by latitude band (share of the union; mean Omega_arag of that state there):')
for b0, b1 in BANDS:
    bm = (la >= b0) & (la < b1)
    print(f'  {b0:+4d} to {b1:+4d}: only UExP-FNN-U {share(onlyU & bm):5.2f} % (2024 {mean(U, onlyU & bm):.2f}), '
          f'only Jiang et al. {share(onlyJ & bm):5.2f} % (1750 {mean(J, onlyJ & bm):.2f}); shared {share(both & bm):5.1f} %')
