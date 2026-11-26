#!/usr/bin/env python
"""
check_map_means.py
==================
Compares the means of the maps of OA-Track over the common area (weights cell_weight) with its global means, for
every quantity and the README of OA-Track:
    obs           the annual values and forced states of the maps of the obs-based products (each fCO2-product and
                  the mean) against the global means: largest difference relative to the rms and as a fraction of the
                  1 sigma of the global mean
    change        the area mean of the relative change maps (obs_annual_forced_change, state 1750, 2024, mean) against
                  the change of the global means, and the change of the area means of the level maps
    coverage      the cells with a value: the mean of the ESM maps (ssp245 2020, historical 2014, preindustrial 1750)
                  against the mean of the obs-based maps (forced state 2020), and the cells with a weight but without
                  a value
Usage (code/):  OA_RUN=u74 python -u check_map_means.py [--src <dir>]
"""
import argparse
from pathlib import Path

import netCDF4
import numpy as np

import oatrack_names as N


def arr(v, idx=Ellipsis):
    return np.ma.filled(np.ma.asarray(v[idx], dtype='f8'), np.nan)


def strs(v):
    return [str(x) for x in np.asarray(v[:]).ravel()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=None, help='default: data/<run>/oatrack_product')
    a = ap.parse_args()
    if a.src:
        src = Path(a.src)
    else:
        import oa_config
        src = Path(oa_config.DATA) / 'oatrack_product'
    for q in N.QUANTITIES:
        with netCDF4.Dataset(src / f'OA-Track_{q}_maps_obs_based_products.nc') as o, \
                netCDF4.Dataset(src / f'OA-Track_{q}_global_mean.nc') as g, \
                netCDF4.Dataset(src / f'OA-Track_{q}_maps_esms.nc') as e:
            w = arr(o['cell_weight'])
            common = arr(o['mask_common_area']) == 1
            wc = np.where(common, w, 0.0)
            mean = lambda x: np.einsum('...ij,ij->...', np.nan_to_num(x), wc) / wc.sum()
            ia = strs(g['area']).index('common_area')
            yo = np.asarray(o['year'][:]).astype(int)
            yg = np.asarray(g['year'][:]).astype(int)
            iy = np.searchsorted(yg, yo)
            out = []
            for blk in ('obs_annual', 'obs_annual_forced'):
                m = mean(arr(o[f'{q}_{blk}']))                                    # product, year
                ref = arr(g[f'{q}_{blk}_gm'], (ia,))[:, iy]
                sig = arr(g[f'{q}_{blk}_gm_sigma'], (ia,))[:, iy]
                ok = np.isfinite(ref)
                d = np.abs(m - ref)[ok]
                out.append(f'{blk} {d.max() / np.sqrt(np.mean(ref[ok] ** 2)):.1e} rel., '
                           f'{np.nanmax((np.abs(m - ref) / sig)[ok]):.3f} sigma')
            # the change 1750-2024 of the mean: mean of the cell changes against the change of the means
            j = int(np.where(yo == 2024)[0][0])
            ch = mean(arr(o[f'{q}_obs_annual_forced_change'], (0, 0, j)))              # product mean, state 1750
            gch = float(arr(g[f'{q}_obs_annual_forced_gm_change'], (ia, 0, 0, iy[j])))
            lv1 = mean(arr(o[f'{q}_preindustrial'], (0, 0)))
            lv2 = mean(arr(o[f'{q}_obs_annual_forced'], (0, j)))
            of_levels = (lv2 - lv1) if q in N.ABSOLUTE_CHANGE else 100 * (lv2 / lv1 - 1)
            out.append(f'change 1750-2024 (mean): mean of the cell changes {ch:+.4g}, change of the global means '
                       f'{gch:+.4g}, change of the area means of the levels {of_levels:+.4g}')
            # coverage of the mean
            yE = np.asarray(e['year'][:]).astype(int)
            esm = np.isfinite(arr(e[f'{q}_esms_annual_forced'], (N.SCENARIOS.index('ssp245'), int(np.where(yE == 2020)[0][0]))))
            hist = np.isfinite(arr(e[f'{q}_esms_annual_forced'], (N.SCENARIOS.index('historical'), int(np.where(yE == 2014)[0][0]))))
            pre = np.isfinite(arr(e[f'{q}_preindustrial'], (0, 0)))
            obs = np.isfinite(arr(o[f'{q}_obs_annual_forced'], (0, int(np.where(yo == 2020)[0][0]))))
            out.append(f'cells: weight {(w > 0).sum()}, obs mean {obs.sum()}, ESM mean ssp245 {esm.sum()} '
                       f'(hist {hist.sum()}, pre {pre.sum()}), ESM without obs {(esm & ~obs).sum()}, obs without ESM '
                       f'{(obs & ~esm).sum()}, weight without value {((w > 0) & ~esm).sum()}')
        print(f'{q}: ' + '; '.join(out), flush=True)


if __name__ == '__main__':
    main()
