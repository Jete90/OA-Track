# Sensitivity of the results to the carbonate constants: the runs of oa_config.py (u74 main run; l10 total boron
# Lee et al. 2010; m10 K1, K2 Millero 2010; m10l10 both). Per product and mask: constrained 1750 value, model mean
# and spread, observed CUC and dSST, present-day value, decline since 1750, boundary (80 % of 1750) and the distance
# of the present-day value to it, each with the difference to u74.   Usage:  python compare_runs.py
import os
import pickle
import numpy as np
import oa_config
import regional_common as rc

PK = {}
for k, v in oa_config.RUNS.items():
    p = oa_config.BASE / v['data'] / 'constraint_results.pkl'
    if p.exists():
        with open(p, 'rb') as f:
            PK[k] = pickle.load(f)
    else:
        print(f'{k}: {p} missing')
REF = 'u74'

for which in ('results_common', 'results'):
    print(f'\n=== {which} ===')
    print(f'{"product":10s} {"run":7s} {"1750 constrained":>17s} {"model mean ± sd":>16s} {"CUC_obs":>8s} '
          f'{"dSST":>6s} {"now":>7s} {"decline %":>9s} {"boundary":>8s} {"now-bound":>9s} '
          f'{"d1750":>8s} {"d decline":>9s} {"d now-bound":>11s}')
    for q in PK[REF][which]:
        r0 = PK[REF][which][q]
        c0, n0 = r0['constrained'][rc.LAB][0], float(r0['now'])
        for k, P in PK.items():
            r = P[which].get(q)
            if r is None:
                continue
            c, s = r['constrained'][rc.LAB][:2]
            y = np.asarray(r['y1750'], 'f8')
            now = float(r['now'])
            dec, dec0 = 100 * (1 - now / c), 100 * (1 - n0 / c0)
            print(f'{q:10s} {k:7s} {c:8.4f} ± {s:.4f} {y.mean():8.4f} ± {y.std(ddof=1):.3f} '
                  f'{float(r["CUC_obs"]):8.4f} {float(r["dT_obs"][0]):6.3f} {now:7.4f} {dec:9.2f} {0.8 * c:8.4f} '
                  f'{now - 0.8 * c:+9.4f} {c - c0:+8.4f} {dec - dec0:+9.3f} {(now - 0.8 * c) - (n0 - 0.8 * c0):+11.4f}')
        print()
