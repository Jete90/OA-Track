#!/usr/bin/env python
"""
test_model_masks.py  (do the raw files of every model cover the same cells in every experiment and variable?)
===================
For every model and member of the chain, and every experiment (historical, piControl, the four scenarios) and
variable (thetao, so, dissic, talk): the surface mask of the first time step on the 1-degree grid (nearest neighbour,
as the chain samples the fields) compared with the historical thetao mask of the same model. Prints the cells wet in
the historical thetao but missing in the file (the ones calc_smooth_states turns into NaN for the whole series) and
the cells missing in the historical thetao but wet in the file. A missing file prints "no file".
Usage (code/):  OA_RUN=u74 python -u test_model_masks.py
"""
import numpy as np

import calc_bias_adjusted as cb
import calc_omega_arag as cm
import calc_omega_arag_obs as co
import calc_product_members as pm
import calc_product_states as cps

VARS = ('thetao', 'so', 'dissic', 'talk')
SCEN = ('ssp126', 'ssp245', 'ssp370', 'ssp585')
R = cps.Reference('OceanSODA')
olat, olon = np.asarray(R.lat, 'f8'), np.asarray(R.lon, 'f8')
area = np.cos(np.deg2rad(olat))[:, None] * np.ones(olon.size)


def surface_mask(path, var, lat_o, lon_o):
    import netCDF4
    with netCDF4.Dataset(path) as f:
        v = f.variables[var]
        lat = np.asarray(f['lat'][:], 'f8')
        lon = np.asarray(f['lon'][:], 'f8')
        a = v[0, 0] if v.ndim == 4 else v[0]
        a = np.ma.filled(np.ma.asarray(a).astype('f8'), np.nan)
    jy = co.nearest_index(lat, lat_o)
    ix = co.nearest_index(lon, lon_o, periodic=True)
    return np.isfinite(a[jy][:, ix])


members = R.members if isinstance(R.members, dict) else dict(zip(R.names, R.members))
for model in R.names:
    member = members[model]
    ref_path = cm.input_path('thetao', model, 'historical', member)
    if ref_path is None:
        print(f'{model}: no historical thetao')
        continue
    ref = surface_mask(ref_path, 'thetao', olat, olon)
    print(f'\n{model} {member}: historical thetao {int(ref.sum())} wet cells ({100 * (area * ref).sum() / area.sum():.2f} % of the sphere)')
    exps = [('historical', member), ('piControl', member)]
    for s in SCEN:
        mem, note = pm.scenario_member(model, s, member)
        if mem is not None:
            exps.append((s, mem))
    for exp, mem in exps:
        parts = []
        for v in VARS:
            p = cm.input_path(v, model, exp, mem)
            if p is None:
                parts.append(f'{v}: no file')
                continue
            m = surface_mask(p, v, olat, olon)
            miss, extra = int((ref & ~m).sum()), int((~ref & m).sum())
            grid = p.name.split('_')[5] if len(p.name.split('_')) > 5 else '?'
            parts.append(f'{v} ({grid}): -{miss} +{extra}' if (miss or extra) else f'{v} ({grid}): same')
        print(f'  {exp:10s} {mem:10s} ' + '; '.join(parts))
print('\n(-n: cells wet in the historical thetao but missing in the file; +n: wet in the file only; gn/gr: the grid label of the file name)')
