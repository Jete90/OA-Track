#!/usr/bin/env python
"""
test_spna_crossing.py  (how much earlier does the regional constraint put the local crossing of the boundary in the
=====================   subpolar North Atlantic?)
For each observation-based product: the area mean over a box of the subpolar North Atlantic (default 45-65 N,
60-10 W) of the whole series of the product file, 1750 to 2024 (preindustrial years, constrained model years, the
product years), once constrained (omega_arag) and once unconstrained (omega_arag_unconstrained: multi-model mean in the
model years, the same product in the product years), and of the 1750 state (named state 1750_AR6) constrained and
unconstrained. The crossing year is the first year in which the 10-year running mean of the series falls below 80% of
the 1750 value of the same kind. Also the decline to 2024 (forced state, omega_arag_2024_trend) and the share of the
box beyond a 20% decline in 2024, with both 1750 states. Area weights of the run. Only reads the product files.
Usage (code/):  OA_RUN=u74 python -u test_spna_crossing.py [--data <data folder>] [--box S N W E]
"""
import argparse
from pathlib import Path

import netCDF4
import numpy as np

import oa_config

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=str(oa_config.DATA))
ap.add_argument('--box', nargs=4, type=float, default=(45.0, 65.0, -60.0, -10.0), metavar=('S', 'N', 'W', 'E'))
a = ap.parse_args()
S, N, W, E = a.box


def crossing(years, y, level, n=10):
    """first year (centre of the n-year running mean) in which the running mean of y is below level"""
    k = np.ones(n) / n
    rm = np.convolve(y, k, mode='valid')
    yc = years[n // 2: n // 2 + rm.size]
    i = np.where(rm < level)[0]
    return float(yc[i[0]]) if i.size else np.nan


print(f'box {S}..{N} N, {W}..{E} E; data {a.data}; area weights: {oa_config.WEIGHTS_TEXT}')
for ref in ('OceanSODA', 'UExP', 'JMA', 'CMEMS'):
    p = Path(a.data) / 'product' / f'carbonate_system_{ref}.nc'
    if not p.exists():
        print(f'{ref}: {p} missing')
        continue
    with netCDF4.Dataset(p) as f:
        lat, lon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        l180 = np.where(lon > 180, lon - 360, lon)
        jj = np.where((lat >= S) & (lat <= N))[0]
        ii = np.where((l180 >= W) & (l180 <= E))[0]
        names = [str(s) for s in np.asarray(f['state'][:]).ravel()]          # vlen str variable
        k = names.index('1750_AR6')
        years = np.asarray(f['year'][:], int)
        src = np.asarray(f['source'][:], int)
        get = lambda v, idx: np.ma.filled(f[v][idx].astype('f8'), np.nan)
        J, I = slice(jj[0], jj[-1] + 1), slice(ii[0], ii[-1] + 1)
        keep = years <= 2024
        c1750 = get('omega_arag_state', (k, J, I))
        u1750 = get('omega_arag_state_unconstrained', (k, J, I))
        ser_c = get('omega_arag', (np.where(keep)[0], J, I))
        ser_u = get('omega_arag_unconstrained', (np.where(keep)[0], J, I))
        t24 = get('omega_arag_2024_trend', (J, I))
    w = oa_config.area_weights(lat[jj[0]:jj[-1] + 1], lon[ii[0]:ii[-1] + 1])
    ok = np.isfinite(c1750) & np.isfinite(u1750) & np.all(np.isfinite(ser_c), axis=0) & np.all(np.isfinite(ser_u), axis=0) & np.isfinite(t24)
    am = lambda x: float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))
    yy = years[keep]
    sc = np.array([am(ser_c[i]) for i in range(ser_c.shape[0])])
    su = np.array([am(ser_u[i]) for i in range(ser_u.shape[0])])
    c, u, f24 = am(c1750), am(u1750), am(t24)
    yc, yu = crossing(yy, sc, 0.8 * c), crossing(yy, su, 0.8 * u)
    # within the product years: the forced series (quadratic fit), first year below 80 % of each 1750 value
    pr = src[keep] == 4
    cf = np.polyfit(yy[pr] - 2000.0, sc[pr], 2)
    ty = np.arange(int(yy[pr][0]), 2025)
    fv = np.polyval(cf, ty - 2000.0)
    fx = lambda lev: float(ty[np.where(fv < lev)[0][0]]) if np.any(fv < lev) else np.nan
    yfc, yfu = fx(0.8 * c), fx(0.8 * u)
    beyond = lambda base: 100 * np.sum(w[ok] * (t24[ok] < 0.8 * base[ok])) / np.sum(w[ok])
    first_prod = int(yy[src[keep] == 4][0]) if np.any(src[keep] == 4) else -1
    print(f'{ref:10s} {int(ok.sum())} cells, product from {first_prod}: 1750 constrained {c:.3f}, unconstrained {u:.3f} '
          f'(difference {c - u:+.3f}); 2024 forced {f24:.3f}; decline to 2024 {100 * (1 - f24 / c):.1f}% / {100 * (1 - f24 / u):.1f}%; '
          f'crossing (10-yr mean) {yc:.0f} / {yu:.0f}; forced crossing {yfc:.0f} / {yfu:.0f} ({yfc - yfu:+.0f} yr); share beyond 20% in 2024 {beyond(c1750):.0f}% / {beyond(u1750):.0f}%  '
          f'(constrained / unconstrained)')
