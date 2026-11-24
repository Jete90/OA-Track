#!/usr/bin/env python
"""
nutrient_sensitivity.py
=======================
How much the choice of the nutrients (silicate and phosphate) changes the observed state, the constrained 1750
state and the decline, per product on its own and on the common area.

Every mocsy call of the study uses the GLODAPv2.2016b surface climatology of Si and PO4 (calc_omega_arag_obs.
load_nutrients), for the observed and for the 1750 states, models and products alike. At given DIC and TA the
nutrients enter only through their (small) contribution to the alkalinity, so their effect on Omega_arag is small,
and because the same nutrients are used in both states most of it cancels in the decline. This script quantifies
both, with the states of the constraint (as test_state_constraint.py: the products' window state, and every model's
1750 state = window state minus its change, DIC lowered to 278.3 ppm):

    base     GLODAP Si and PO4
    zero     no nutrients (an upper bound: as if a product had derived its DIC without them)
    x0.8     GLODAP x 0.8 and x 1.2: a 20 % error of the climatology in every cell, of either sign
    x1.2
    --alt    a second climatology (e.g. World Ocean Atlas): netCDF files with silicate and phosphate (umol/kg) on
             a regular lat/lon grid, given as --alt-si FILE VAR and --alt-po4 FILE VAR

For each variant: the area mean Omega_arag of the window state, the constrained 1750 value (direct regression of
the 18 models' 1750 values on dSST and CUC, as in the paper) and the decline 1 - window / 1750 (percentage points),
and their differences to the base. The window state stands for the present day here; the nutrient effect on the
2024 state is the same to the precision that matters.

Usage (code folder):  OA_RUN=u74 python -u nutrient_sensitivity.py [--reference OceanSODA ...]
                      [--alt-si FILE VAR --alt-po4 FILE VAR]
Output: data/<run>/product_uncertainty/nutrient_sensitivity.npz and the printout
"""
import argparse
import pickle
from pathlib import Path

import netCDF4
import numpy as np

import oa_config
import calc_omega_arag_obs as co
import test_state_constraint as ts


def read_alt(spec, lat, lon):
    path, var = spec
    with netCDF4.Dataset(path) as f:
        a = np.ma.filled(f[var][:].astype('f8'), np.nan).squeeze()
        names = list(f.variables)
        la = next(k for k in ('lat', 'latitude') if k in names)
        lo = next(k for k in ('lon', 'longitude') if k in names)
        jy = co.nearest_index(np.asarray(f[la][:], 'f8'), lat)
        ix = co.nearest_index(np.asarray(f[lo][:], 'f8') % 360.0, np.asarray(lon, 'f8') % 360.0, periodic=True)
    if a.ndim == 3:                                   # (depth, lat, lon): the surface
        a = a[0]
    return np.clip(a[np.ix_(jy, ix)], 0.0, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', nargs='+', default=['OceanSODA', 'UExP', 'JMA', 'CMEMS'])
    ap.add_argument('--alt-si', nargs=2, metavar=('FILE', 'VAR'))
    ap.add_argument('--alt-po4', nargs=2, metavar=('FILE', 'VAR'))
    a = ap.parse_args()
    data = Path(oa_config.DATA)
    with open(data / 'constraint_results.pkl', 'rb') as f:
        PK = pickle.load(f)
    out = {}
    for ref in a.reference:
        R = PK['results'][ref]
        obs, chg, lat, lon = ts.read_changes(R, data, ref)
        names = list(R['names'])
        n = len(names)
        si0, po0 = (np.clip(x, 0.0, None) for x in co.load_nutrients(lat, lon))
        variants = {'base': (si0, po0), 'zero': (0 * si0, 0 * po0), 'x0.8': (0.8 * si0, 0.8 * po0),
                    'x1.2': (1.2 * si0, 1.2 * po0)}
        if a.alt_si and a.alt_po4:
            variants['alt'] = (read_alt(a.alt_si, lat, lon), read_alt(a.alt_po4, lat, lon))
        dT, CUC = np.asarray(R['dT'], 'f8'), np.asarray(R['CUC'], 'f8')
        wC, *_ = ts.weights([dT, CUC], [float(R['dT_obs'][0]), float(R['CUC_obs'])])
        # the same cells in every variant: those of the base chemistry where all variants have nutrients
        ok0 = np.all([np.isfinite(obs[v]) for v in ts.VARS], 0) & np.all([np.all(np.isfinite(chg[v]), 0) for v in ts.VARS], 0)
        for s_, p_ in variants.values():
            ok0 &= np.isfinite(s_) & np.isfinite(p_)
        res = {}
        for area, key in (('own', 'results'), ('common', 'results_common')):
            mask = np.asarray(PK[key][ref]['mask']).astype(bool) & ok0
            res[area] = {}
            for vname, (s_, p_) in variants.items():
                chem = ts.Chem(lat, lon, mask)               # 1750 states: DIC lowered to 278.3 ppm
                ok = chem.ok
                chem.kw['sil'], chem.kw['phos'] = s_[ok] * 1e-6, p_[ok] * 1e-6
                w = oa_config.area_weights(lat, lon) * ok
                gm = lambda fld: float(np.nansum(np.where(ok, fld, 0) * w) / w[np.isfinite(fld) & ok].sum())
                # window state at its own CO2 (no shift)
                r = chem.mocsy.mvars(temp=obs['thetao'][ok], sal=obs['so'][ok], alk=obs['talk'][ok] * 1e-6,
                                     dic=obs['dissic'][ok] * 1e-6, **chem.kw)
                om_obs = np.full(ok.shape, np.nan)
                om_obs[ok] = np.asarray(r[6], float)
                y = np.array([gm(chem.omega({v: obs[v] - chg[v][m] for v in ts.VARS})) for m in range(n)])
                now, pi = gm(om_obs), float(wC @ y)
                res[area][vname] = dict(now=now, pi=pi, dec=100 * (1 - now / pi), unconstrained=float(y.mean()))
            b = res[area]['base']
            print(f'===== {ref}, {area} area ({int(ok.sum())} cells): base window {b["now"]:.4f}, 1750 {b["pi"]:.4f}, '
                  f'decline {b["dec"]:.2f} %', flush=True)
            for vname, x in res[area].items():
                if vname == 'base':
                    continue
                print(f'  {vname:5s}: window {x["now"] - b["now"]:+.4f}, 1750 {x["pi"] - b["pi"]:+.4f}, '
                      f'decline {x["dec"] - b["dec"]:+.3f} percentage points', flush=True)
        out[ref] = res
    path = data / 'product_uncertainty' / 'nutrient_sensitivity.npz'
    path.parent.mkdir(exist_ok=True)
    np.savez(path, **{f'{q}_{ar}_{v}_{k}': x for q, r in out.items() for ar, d in r.items() for v, e in d.items()
                      for k, x in e.items()})
    print(f'saved {path}')


if __name__ == '__main__':
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
