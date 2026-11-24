#!/usr/bin/env python
"""
test_cell_crossing.py  (by how much does the cell-by-cell constraint move the local crossing of the boundary?)
=====================
For each observation-based product, cell by cell: the forced series of the product years (quadratic fit per cell, as
the forced 2024 state) and the 1750 state (named state 1750_AR6) constrained and unconstrained. The crossing year is
the first year of the product record in which the forced series is below 80% of the 1750 value; cells beyond the
boundary already in the first product year are counted as "before", cells not yet beyond it in 2024 as "not yet".
Prints, area-weighted: the change of the 1750 state and of the decline to 2024 (constrained minus unconstrained,
percentiles), the shift of the crossing year where both cross within the record (percentiles, largest shifts with
their positions), and the cells whose status in 2024 changes with the constraint. Only reads the product files.
Usage (code/):  OA_RUN=u74 python -u test_cell_crossing.py [--data <data folder>]
"""
import argparse
from pathlib import Path

import netCDF4
import numpy as np

import oa_config

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=str(oa_config.DATA))
a = ap.parse_args()


def wpct(x, w, qs):
    """weighted percentiles"""
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return [float(np.interp(q / 100.0, c, x)) for q in qs]


print(f'data {a.data}; area weights: {oa_config.WEIGHTS_TEXT}')
for ref in ('OceanSODA', 'UExP', 'JMA', 'CMEMS'):
    p = Path(a.data) / 'product' / f'carbonate_system_{ref}.nc'
    if not p.exists():
        print(f'{ref}: {p} missing')
        continue
    with netCDF4.Dataset(p) as f:
        lat, lon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        names = [str(s) for s in np.asarray(f['state'][:]).ravel()]
        k = names.index('1750_AR6')
        years = np.asarray(f['year'][:], int)
        src = np.asarray(f['source'][:], int)
        pr = np.where((src == 4) & (years <= 2024))[0]
        get = lambda v, idx: np.ma.filled(f[v][idx].astype('f8'), np.nan)
        c = get('omega_arag_state', (k, slice(None), slice(None)))
        u = get('omega_arag_state_unconstrained', (k, slice(None), slice(None)))
        obs = get('omega_arag', (pr, slice(None), slice(None)))
    yp = years[pr]
    ny, nx = c.shape
    ok = np.isfinite(c) & np.isfinite(u) & np.all(np.isfinite(obs), axis=0)
    Y = obs[:, ok]                                            # (years, cells)
    co = np.polyfit(yp - 2000.0, Y, 2)                        # (3, cells)
    ty = np.arange(yp[0], 2025)
    F = (co[0][None, :] * (ty - 2000.0)[:, None] ** 2 + co[1][None, :] * (ty - 2000.0)[:, None] + co[2][None, :])
    f24 = F[-1]

    def cross(base):
        below = F < 0.8 * base[ok][None, :]
        first = np.where(below.any(axis=0), ty[np.argmax(below, axis=0)], np.nan)
        before = below[0]
        notyet = ~below[-1]
        return first, before, notyet

    yc, bc, nc = cross(c)
    yu, bu, nu = cross(u)
    w = oa_config.area_weights(lat, lon)[ok]
    dc = (c - u)[ok]
    ddec = 100 * (f24 / u[ok] - f24 / c[ok])                  # decline constrained minus unconstrained (points)
    both = np.isfinite(yc) & np.isfinite(yu) & ~bc & ~bu
    dy = (yc - yu)[both]
    W = w.sum()
    print(f'\n=== {ref}: {int(ok.sum())} cells, product years {yp[0]}-{yp[-1]}')
    p5, p50, p95 = wpct(dc, w, (5, 50, 95))
    print(f'  1750 constrained - unconstrained: 5/50/95 % {p5:+.3f} / {p50:+.3f} / {p95:+.3f}; min {dc.min():+.3f}, max {dc.max():+.3f}')
    p5, p50, p95 = wpct(ddec, w, (5, 50, 95))
    print(f'  decline to 2024, constrained - unconstrained (points): 5/50/95 % {p5:+.2f} / {p50:+.2f} / {p95:+.2f}; '
          f'min {ddec.min():+.2f}, max {ddec.max():+.2f}')
    if dy.size:
        q = wpct(dy, w[both], (5, 25, 50, 75, 95))
        print(f'  crossing year shift where both cross within the record ({100 * w[both].sum() / W:.0f}% of the area): '
              f'5/25/50/75/95 % {q[0]:+.0f} / {q[1]:+.0f} / {q[2]:+.0f} / {q[3]:+.0f} / {q[4]:+.0f} yr; '
              f'|shift| >= 5 yr on {100 * w[both][np.abs(dy) >= 5].sum() / W:.1f}%, >= 10 yr on {100 * w[both][np.abs(dy) >= 10].sum() / W:.1f}% of the area')
        jj, ii = np.where(ok)
        jb, ib = jj[both], ii[both]
        l180 = np.where(lon > 180, lon - 360, lon)
        o = np.argsort(dy)
        fmt = lambda idx: ', '.join(f'({lat[jb[i]]:.1f}, {l180[ib[i]]:.1f}): {dy[i]:+.0f}' for i in idx)
        print(f'  earliest shifts (lat, lon: yr): {fmt(o[:8])}')
        print(f'  latest shifts  (lat, lon: yr): {fmt(o[::-1][:8])}')
    st = lambda m: 100 * w[m].sum() / W
    print(f'  beyond the boundary in 2024: constrained {st(~nc):.1f}%, unconstrained {st(~nu):.1f}% of the area; '
          f'crossed only with the constraint {st(~nc & nu):.1f}%, only without {st(nc & ~nu):.1f}%; '
          f'beyond already in {yp[0]}: {st(bc):.1f}% / {st(bu):.1f}%')
