#!/usr/bin/env python
"""
test_mri_area.py  (how much of the common area is lost to the incomplete MRI-ESM2-0 salinity?)
================
The constraint area is the set of cells every one of the 18 ESMs covers (all four variables, historical and
piControl); the own area of a product is its coverage within it, the common area the coverage of all four. MRI-ESM2-0
lacks 2,619 coastal cells in the salinity of its piControl and SSP5-8.5 files (not in the other variables), so these
cells are missing from the constraint area. This script rebuilds the model masks from the raw files (first time step,
nearest neighbour onto the 1-degree grid, as the chain samples them), once as they are and once with the MRI salinity
mask replaced by its temperature mask, intersects them with the products' coverage (obs_<ref>.nc) and compares both
with the areas of the main analysis (constraint_results.pkl). Prints the cells and the area fractions of the ocean
(the whole-ocean mask of TableS1_summary.csv is approximated by the union of the four products' coverage).
Usage (code/):  OA_RUN=u74 python -u test_mri_area.py
"""
import pickle

import netCDF4
import numpy as np

import calc_omega_arag as cm
import calc_omega_arag_obs as co
import calc_product_states as cps
import oa_config

VARS = ('thetao', 'so', 'dissic', 'talk')
REFS = ('OceanSODA', 'UExP', 'JMA', 'CMEMS')
R = cps.Reference('OceanSODA')
olat, olon = np.asarray(R.lat, 'f8'), np.asarray(R.lon, 'f8')
area = np.cos(np.deg2rad(olat))[:, None] * np.ones(olon.size)
members = R.members if isinstance(R.members, dict) else dict(zip(R.names, R.members))


def surface_mask(path, var):
    with netCDF4.Dataset(path) as f:
        v = f.variables[var]
        lat = np.asarray(f['lat'][:], 'f8')
        lon = np.asarray(f['lon'][:], 'f8')
        a = v[0, 0] if v.ndim == 4 else v[0]
        a = np.ma.filled(np.ma.asarray(a).astype('f8'), np.nan)
    jy = co.nearest_index(lat, olat)
    ix = co.nearest_index(lon, olon, periodic=True)
    return np.isfinite(a[jy][:, ix])


def fr(m, ref_mask):
    """percent of the area of ref_mask"""
    return 100.0 * (area * m).sum() / (area * ref_mask).sum()


# model masks as they are, and with the MRI salinity replaced by its temperature mask
M_now = np.ones(area.shape, bool)
M_fix = np.ones(area.shape, bool)
for model in R.names:
    member = members[model]
    for exp in ('historical', 'piControl'):
        for v in VARS:
            p = cm.input_path(v, model, exp, member)
            if p is None:
                print(f'{model} {exp} {v}: no file')
                continue
            m = surface_mask(p, v)
            M_now &= m
            if model == 'MRI-ESM2-0' and v == 'so':
                m = surface_mask(cm.input_path('thetao', model, exp, member), 'thetao')
            M_fix &= m
print(f'constraint area from the raw masks: {int(M_now.sum())} cells as they are, {int(M_fix.sum())} with the MRI salinity complete '
      f'({int((M_fix & ~M_now).sum())} cells more)')

# the products' coverage
obs = {}
for ref in REFS:
    with netCDF4.Dataset(oa_config.DATA / 'product' / f'obs_{ref}.nc') as f:
        obs[ref] = np.asarray(f['mask'][:]).astype(bool)
ocean = np.zeros(area.shape, bool)
for m in obs.values():
    ocean |= m
print(f'ocean (union of the four products): {int(ocean.sum())} cells')

with open(oa_config.DATA / 'constraint_results.pkl', 'rb') as f:
    P = pickle.load(f)
cm_paper = np.asarray(P['cm']).astype(bool)
print(f'\ncommon area of the main analysis: {int(cm_paper.sum())} cells, {fr(cm_paper, ocean):.2f} % of the ocean')
common_now = M_now.copy()
common_fix = M_fix.copy()
for m in obs.values():
    common_now &= m
    common_fix &= m
print(f'common area rebuilt from the raw masks: {int(common_now.sum())} cells ({fr(common_now, ocean):.2f} %), '
      f'{int((common_now ^ cm_paper).sum())} cells differ from the main analysis (the chain also drops cells with a missing month)')
print(f'common area with the MRI salinity complete: {int(common_fix.sum())} cells ({fr(common_fix, ocean):.2f} %): '
      f'{int((common_fix & ~common_now).sum())} cells more, {fr(common_fix & ~common_now, ocean):.3f} % of the ocean')
for ref in REFS:
    own_paper = np.asarray(P['results'][ref]['mask']).astype(bool)
    own_now, own_fix = M_now & obs[ref], M_fix & obs[ref]
    print(f'  {ref:10s} own area: main analysis {int(own_paper.sum())} cells ({fr(own_paper, ocean):.2f} %); rebuilt {int(own_now.sum())}; '
          f'with MRI complete {int(own_fix.sum())} (+{int((own_fix & ~own_now).sum())} cells, {fr(own_fix & ~own_now, ocean):.3f} % of the ocean)')
add = common_fix & ~common_now
if add.any():
    jj, ii = np.where(add)
    l180 = np.where(olon > 180, olon - 360, olon)
    print('\n  cells the common area would gain (lat, lon), first 40: ' + ', '.join(f'({olat[j]:.1f}, {l180[i]:.1f})' for j, i in list(zip(jj, ii))[:40]))
    print(f'  latitude bands of the gained cells: {np.histogram(olat[jj], bins=[-90, -60, -30, 0, 30, 60, 90])[0].tolist()} '
          f'(90S-60S, 60S-30S, 30S-0, 0-30N, 30N-60N, 60N-90N)')
