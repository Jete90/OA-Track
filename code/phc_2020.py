#!/usr/bin/env python
"""
phc_2020.py
===========
The decline of the Planetary Health Check (2026) for other years than 2024, in its own framework: global mean of
saturation_aragonite of UExP-FNN-U (mean of the 12 months, cells with all 12), weighted with the weights of the file
(area x mask_sfc), relative to its preindustrial 3.57 (1750, Jiang et al. 2023). Exactly as the reproduction of the
published 2.85 (20.2 %) in the notebook (phc_mean), for every year 2015-2024 and for the mean of 2015-2024, the decade
of Jiang et al. that Findlay et al. (2025) call 2020. For both the pre-release used by the Planetary Health Check
(v2026-pre1) and the released v2026-1. Reads only, writes nothing.
Usage (code/):  OA_RUN=u74 python -u phc_2020.py
"""
import netCDF4
import numpy as np

import oa_config
import regional_common as rc

PHC_PI = 3.57
FILES = {'v2026-pre1': f'{rc.BASE}/inputs/UExP-FNN-U_v2026-pre1/Fordetal_UExP-FNN-U_surface-carbonate-system_v2026-pre1.nc',
         'v2026-1': str(oa_config.OBS_DIR / 'UExP-FNN-U/Fordetal_UExP-FNN-U_surface-carbonate-system_v2026-1.nc')}
YEARS = range(2015, 2025)


def dec(x):
    return 100 * (1 - x / PHC_PI)


for ver, path in FILES.items():
    with netCDF4.Dataset(path) as f:
        v = f['saturation_aragonite']
        assert v.dimensions == ('lon', 'lat', 'time') and f['area'].dimensions == f['mask_sfc'].dimensions == ('lon', 'lat')
        d = np.datetime64('1970-01-15') + np.asarray(f['time'][:], 'i8').astype('timedelta64[D]')
        yr = d.astype('datetime64[Y]').astype(int) + 1970
        w = np.ma.filled(f['area'][:].astype('f8'), np.nan) * np.ma.filled(f['mask_sfc'][:].astype('f8'), np.nan)
        out = {}
        for y in YEARS:
            it = np.where(yr == y)[0]
            assert it.size == 12, (ver, y, it.size)
            a = np.ma.filled(v[:, :, it[0]:it[-1] + 1].astype('f8'), np.nan).mean(axis=2)    # NaN unless 12 months
            m = np.isfinite(a) & np.isfinite(w)
            out[y] = float(np.sum(a[m] * w[m]) / w[m].sum())
    print(f'{ver} (weights of the file, decline from {PHC_PI}):')
    print('  ' + ', '.join(f'{y} {x:.4f} ({dec(x):.2f} %)' for y, x in out.items()))
    mdec = np.mean([out[y] for y in YEARS])
    t = np.array(list(YEARS), 'f8')
    b = np.polyfit(t, [dec(out[y]) for y in YEARS], 1)
    print(f'  mean 2015-2024 {mdec:.4f} ({dec(mdec):.2f} %); linear trend of the decline 2015-2024 {b[0]:.3f} points per year, '
          f'fitted 2020 {np.polyval(b, 2020):.2f} %, fitted 2024 {np.polyval(b, 2024):.2f} %')
