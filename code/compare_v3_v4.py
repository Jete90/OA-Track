#!/usr/bin/env python
"""
compare_v3_v4.py  (the effect of the area weights on the main analysis: whole cells (v3) against cell area x ocean fraction (v4))
================
Reads constraint_results.pkl of two production folders and prints, per product and mask (own area, common area): the
constrained 1750 value, the present-day value, the decline, the boundary distance and the mask's share of the ocean,
each for both runs and as difference. The masks are the same cells in both runs; only the weights differ.
Usage (code/):  python compare_v3_v4.py [<pkl of v3> <pkl of v4>]
"""
import pickle
import sys

import numpy as np

import oa_config
import regional_common as rc

A = sys.argv[1] if len(sys.argv) > 2 else str(oa_config.BASE.parent / 'OA_boundary_production_v3' / 'data' / 'u74' / 'constraint_results.pkl')
B = sys.argv[2] if len(sys.argv) > 2 else str(oa_config.DATA / 'constraint_results.pkl')
P = {}
for k, p in (('v3 cell', A), ('v4 ocean', B)):
    with open(p, 'rb') as f:
        P[k] = pickle.load(f)
    print(f'{k}: {p}')


def share(mask, lat, lon, ocean_weights):
    """share of the ocean (union of the products' coverage is not at hand here: the share of the weighted sphere)"""
    w = oa_config.area_weights(lat, lon) if ocean_weights else np.cos(np.deg2rad(lat))[:, None] * np.ones(lon.size)
    return 100.0 * (w * mask).sum() / w.sum()


for which, label in (('results_common', 'common area'), ('results', 'own area')):
    print(f'\n=== {label} ===')
    print(f'{"product":10s} {"run":9s} {"1750":>8s} {"sigma":>7s} {"now":>8s} {"decline %":>9s} {"now-bound":>9s} {"sphere %":>8s}')
    for q in P['v3 cell'][which]:
        rows = {}
        for k in P:
            r = P[k][which].get(q)
            if r is None:
                continue
            c, s = (float(v) for v in r['constrained'][rc.LAB][:2])
            now = float(r['now'])
            lat, lon = np.asarray(r['lat'], 'f8'), np.asarray(r['lon'], 'f8')
            rows[k] = (c, s, now, 100 * (1 - now / c), now - 0.8 * c, share(np.asarray(r['mask']).astype(bool), lat, lon, k.startswith('v4')))
            print(f'{q:10s} {k:9s} {c:8.4f} {s:7.4f} {now:8.4f} {rows[k][3]:9.2f} {rows[k][4]:+9.4f} {rows[k][5]:8.2f}')
        if len(rows) == 2:
            a, b = rows['v3 cell'], rows['v4 ocean']
            print(f'{"":10s} {"v4 - v3":9s} {b[0] - a[0]:+8.4f} {b[1] - a[1]:+7.4f} {b[2] - a[2]:+8.4f} {b[3] - a[3]:+9.3f} {b[4] - a[4]:+9.4f} {"":8s}')
print('\n(sphere %: the mask weighted as in its run, in percent of the whole weighted sphere; the shares of the ocean of the '
      'paper use the union of the products as denominator and come from the notebook)')
