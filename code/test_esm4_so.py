#!/usr/bin/env python
"""
test_esm4_so.py  (why GFDL-ESM4 has no salinity in a few coastal cells of the historical series)
===============
Reads the raw GFDL-ESM4 files (historical, piControl and the four scenarios, the members of the chain) at the source cells of the product
cells -49.5 N 70.5 E, 2.5 N -50.5 E and -3.5 N 116.5 E (nearest neighbour, as the chain samples them) and counts the
missing months of thetao, so, dissic and talk over the whole run; prints the years with missing salinity.
Usage (code/):  OA_RUN=u74 python -u test_esm4_so.py
"""
import numpy as np

import calc_bias_adjusted as cb
import calc_omega_arag_obs as co
import calc_product_states as cps
import calc_product_members as pm

R = cps.Reference('OceanSODA')
member = R.members['GFDL-ESM4'] if isinstance(R.members, dict) else R.members[list(R.names).index('GFDL-ESM4')]
olat, olon = np.asarray(R.lat, 'f8'), np.asarray(R.lon, 'f8')
l180 = np.where(olon > 180, olon - 360, olon)
cells = [(-49.5, 70.5), (2.5, -50.5), (-3.5, 116.5)]
runs = [('historical', member), ('piControl', member)]
for scen in ('ssp126', 'ssp245', 'ssp370', 'ssp585'):
    mem, note = pm.scenario_member('GFDL-ESM4', scen, member)
    print(f'{scen}: member {mem}' + (f' ({note})' if note else ''))
    if mem is not None:
        runs.append((scen, mem))
for exp, mem_ in runs:
    run = cb.ModelRun('GFDL-ESM4', exp, mem_)
    jy = co.nearest_index(run.lat, olat)
    ix = co.nearest_index(run.lon, olon, periodic=True)
    print(f'\n{run.tag}: grid {run.lat.size} x {run.lon.size}, {run.nt} steps, step {run.step}, years {run.years[0]}-{run.years[-1]}')
    for a, b in cells:
        j = int(np.argmin(np.abs(olat - a)))
        i = int(np.argmin(np.abs(l180 - b)))
        js, is_ = int(jy[j]), int(ix[i])
        print(f'  product cell ({a}, {b}) -> source cell lat {run.lat[js]:.3f}, lon {run.lon[is_]:.3f}')
        for v in ('thetao', 'so', 'dissic', 'talk'):
            var = run.var[v]
            x = var[:run.nt, 0, js, is_] if var.ndim == 4 else var[:run.nt, js, is_]     # 4-D: top level
            x = np.ma.filled(np.ma.asarray(x).astype('f8'), np.nan).reshape(-1)
            bad = ~np.isfinite(x)
            yrs = sorted(set(int(y) for y in run.years[bad]))
            print(f'    {v:7s} missing {int(bad.sum())} of {x.size} steps' + (f'; years {yrs[:20]}{" ..." if len(yrs) > 20 else ""}' if yrs else '')
                  + f'; range {np.nanmin(x):.4g} to {np.nanmax(x):.4g}')
        # the neighbours of the salinity source cell
        var = run.var['so']
        s = var[:run.nt, 0, js - 1:js + 2, is_ - 1:is_ + 2] if var.ndim == 4 else var[:run.nt, js - 1:js + 2, is_ - 1:is_ + 2]
        s = np.ma.filled(np.ma.asarray(s).astype('f8'), np.nan)
        print(f'    so, missing steps in the 3 x 3 source cells around it:\n' + '\n'.join('      ' + ' '.join(f'{int(v):5d}' for v in row) for row in (~np.isfinite(s)).sum(0)))
    run.close()
