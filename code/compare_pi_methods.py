#!/usr/bin/env python
"""
compare_pi_methods.py
=====================
Old against new preindustrial states, per reference and model, from the constraint results of two production
folders: the earlier method (Omega_arag of the annual-mean drift-corrected state at 284.3 ppm times a global
factor per model for the CO2 step) against version 2 (time mean of the monthly Omega_arag of the drift-
corrected, bias-adjusted piControl with DIC lowered cell by cell). Prints per reference the global 1750 value
per model (own mask), the constrained values and their sigma, and for the common-mask results the RMS and
maximum of the per-cell difference of the 1750 fields.

Usage:  python compare_pi_methods.py <old production folder> [<new production folder>] [--run u74]
        (default new folder: the one this script is in)
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('old')
ap.add_argument('new', nargs='?', default=str(Path(__file__).resolve().parents[1]))
ap.add_argument('--run', default='u74')
a = ap.parse_args()

R = {}
for tag, base in (('old', a.old), ('new', a.new)):
    # data/<run>/ (v2) or, in older folders, data/ for u74
    cands = [Path(base) / 'data' / a.run / 'constraint_results.pkl'] + \
            ([Path(base) / 'data' / 'constraint_results.pkl'] if a.run == 'u74' else [])
    path = next((p_ for p_ in cands if p_.exists()), None)
    if path is None:
        raise SystemExit(f'{tag}: no constraint_results.pkl in ' + ' or '.join(str(p_.parent) for p_ in cands))
    with open(path, 'rb') as f:
        R[tag] = pickle.load(f)
    print(f'{tag}: {path}')
print(f'run {a.run}: old {a.old}\n{"":10s} new {a.new}\n')

for which, label in (('results', 'own mask'), ('results_common', 'common mask')):
    print(f'===== {label}')
    for ref in R['old'][which]:
        if ref not in R['new'][which]:
            print(f'{ref}: not in the new results')
            continue
        o, n = R['old'][which][ref], R['new'][which][ref]
        names = [m for m in o['names'] if m in n['names']]
        yo = np.array([o['y1750'][o['names'].index(m)] for m in names])
        yn = np.array([n['y1750'][n['names'].index(m)] for m in names])
        print(f'\n{ref}: {len(names)} models; unconstrained 1750 old {yo.mean():.4f} new {yn.mean():.4f} '
              f'(spread old {yo.std(ddof=1):.4f} new {yn.std(ddof=1):.4f}); per model new - old: '
              f'mean {np.mean(yn - yo):+.4f}, range {np.min(yn - yo):+.4f} to {np.max(yn - yo):+.4f}')
        for lab in o['constrained']:
            co, cn = o['constrained'][lab], n['constrained'].get(lab)
            if cn is None:
                continue
            print(f'  {lab:16s} constrained 1750 old {co[0]:.4f} +- {co[1]:.4f}, new {cn[0]:.4f} +- {cn[1]:.4f} '
                  f'({cn[0] - co[0]:+.4f}); boundary old {0.8 * co[0]:.4f} new {0.8 * cn[0]:.4f}')
        if np.isfinite(o.get('now', np.nan)) and np.isfinite(n.get('now', np.nan)):
            lab = 'dT + CUC'
            print(f'  decline to 2024 ({lab}): old {100 * (1 - o["now"] / o["constrained"][lab][0]):.2f} %, '
                  f'new {100 * (1 - n["now"] / n["constrained"][lab][0]):.2f} %')
        if 'states' in o and 'states' in n:
            ks = [k for k in o['states'] if k in n['states']]
            print('  states (unconstrained, old -> new): ' + ', '.join(
                f'{k} {o["states"][k]["y_raw"]:.3f}->{n["states"][k]["y_raw"]:.3f}' for k in ks))
        if 'fields' in o and 'fields' in n and o['mask'].shape == n['mask'].shape:
            fo, fn = np.asarray(o['fields']['y1750'], 'f8'), np.asarray(n['fields']['y1750'], 'f8')
            m = o['mask'] & n['mask']
            d = np.array([(fn[n['names'].index(mm)] - fo[o['names'].index(mm)])[m] for mm in names])
            print(f'  per-cell 1750 fields, new - old over {m.sum()} cells: RMS {np.sqrt(np.nanmean(d ** 2)):.4f}, '
                  f'mean {np.nanmean(d):+.4f}, max |d| {np.nanmax(np.abs(d)):.4f}')
