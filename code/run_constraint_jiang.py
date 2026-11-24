#!/usr/bin/env python
"""
run_constraint_jiang.py
=======================
The full chain of this study with the observation-based state of Jiang et al. (2023, decade 2010 = January 2005
to December 2014) as a fifth reference: bias adjustment to that state (one annual offset per cell), piControl drift,
1750 CO2, emergent constraint (dSST and CUC over 2005-2014). It shows whether their observations give the same
relative decline as the four products.

The relative decline from 1750 to the decade 2010 (2005-2014 mean of the products; the single state of Jiang):
    the four products on the common area of the paper, unchanged (from constraint_results.pkl; Jiang never
    reduces it), and Jiang on its own area (its coverage and the ESMs) and on the common area of the paper (cells
    outside Jiang's coverage dropped, printed if any). Since Jiang et al. cover the whole ocean, the difference of
    the two Jiang declines shows what leaving out the area outside the common area does to the decline
    (coverage test; the ESM version of the test is printed by make_product.py).

Needs: data/<run>/constraint_results.pkl (run_constraint.py) for the common mask, and the Jiang reference of
the chain (make_jiang_ref.py, calc_bias_adjusted.py, calc_pi_xco2.py and calc_mechanism.py with --reference
Jiang; rerun.sh does all of it, this is its step 8). Output: data/<run>/constraint_results_jiang.pkl.
Jiang et al. use the constants of run l10 (Lueker et al. 2000, Lee et al. 2010).

Usage:  python run_constraint_jiang.py       (OA_RUN as for the chain)
"""
import os
import pickle
from pathlib import Path

import numpy as np
import oa_config

os.environ['OA_REFS'] = 'Jiang'
_HERE = Path(__file__).resolve().parent
with open(oa_config.DATA / 'constraint_results.pkl', 'rb') as f:
    MAIN = pickle.load(f)

print('##### constraint_refs.py: Jiang et al. (2023) on its own mask', flush=True)
exec(compile((_HERE / 'constraint_refs.py').read_text(), 'constraint_refs.py', 'exec'))
own = results

# common area of the paper, restricted to the coverage of Jiang et al. (mapped onto the OceanSODA grid)
lat_c, lon_c, cm0 = MAIN['lat_c'], MAIN['lon_c'], MAIN['cm']
rj = own['Jiang']
cmJ = cm0 & rj['mask'][np.ix_(_nn(rj['lat'], lat_c), _nn(rj['lon'], lon_c, True))]
area = oa_config.area_weights(lat_c, lon_c)
print(f'\ncommon area with Jiang et al.: {100 * (area * cmJ).sum() / (area * cm0).sum():.1f} % of the common area '
      f'of the paper', flush=True)

LAB = 'dT + CUC'
# Jiang et al. on the common area of the paper (restricted to its coverage only where it has no data), for the
# coverage test; the four products keep the full common area of the paper (constraint_results.pkl)
print('\n##### Jiang et al. on the common area of the paper', flush=True)
REFS = {'Jiang': REFS_ALL['Jiang']}
com = run_constraint(common=(lat_c, lon_c, cmJ))


def decline(x):
    yc, s, _ = x['constrained'][LAB]
    return yc, s, x['obs10'], 100 * (1 - x['obs10'] / yc)


print(f'\nrelative decline from 1750 to the decade 2010 (2005-2014), constrained ({LAB})')
print(f'  {"reference":10s} {"area":34s} {"1750":>7s} {"2005-2014":>10s} {"decline":>8s}')
dec = {}
for r in ('OceanSODA', 'UExP', 'JMA', 'CMEMS'):
    if r in MAIN['results_common']:
        yc, s, o, d = decline(MAIN['results_common'][r])
        dec[r] = d
        print(f'  {r:10s} {"common area of the paper":34s} {yc:7.3f} {o:10.3f} {d:7.1f}%  (1750 +- {s:.3f})')
p4 = [dec[r] for r in ('OceanSODA', 'UExP', 'JMA', 'CMEMS') if r in dec]
yc, s, o, d_own = decline(own['Jiang'])
dec['Jiang'] = d_own
print(f'  {"Jiang":10s} {"own area (Jiang and ESMs)":34s} {yc:7.3f} {o:10.3f} {d_own:7.1f}%  (1750 +- {s:.3f})')
yc, s, o, d_com = decline(com['Jiang'])
dec['Jiang_common'] = d_com
print(f'  {"Jiang":10s} {"common area of the paper":34s} {yc:7.3f} {o:10.3f} {d_com:7.1f}%  (1750 +- {s:.3f})')
print(f'  four products: {np.mean(p4):.1f}% (range {min(p4):.1f} to {max(p4):.1f}); Jiang et al.: {d_own:.1f}% on its own '
      f'area, {d_com:.1f}% on the common area')
fo = 100 * (area * cmJ).sum() / (area * own['Jiang']['mask'][np.ix_(_nn(rj['lat'], lat_c), _nn(rj['lon'], lon_c, True))]).sum()
print(f'\ncoverage test (Jiang et al. has full coverage): the common area is {fo:.0f} % of its own area; restricting it to '
      f'the common area changes the decline by {d_com - d_own:+.2f} percentage points')

with open(oa_config.DATA / 'constraint_results_jiang.pkl', 'wb') as f:
    pickle.dump(dict(own=own, common=com, cmJ=cmJ, lat_c=lat_c, lon_c=lon_c, decline_2010=dec,
                     coverage_test=dict(own=d_own, common=d_com, common_fraction_of_own=fo)), f)
print(f'\nsaved to {oa_config.DATA / "constraint_results_jiang.pkl"}')
