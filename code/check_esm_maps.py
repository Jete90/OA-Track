#!/usr/bin/env python
"""
check_esm_maps.py
=================
Checks the maps of the ESMs (OA-Track_<q>_maps_esms.nc) of make_oatrack_product.py --part esms before they are packed,
independently of that script, against the definition in oatrack_names.py, the global means and the maps of the
observation-based products:
    variables     every variable of oatrack_names.py with its dimensions, and nothing else; the coordinates
    historical    empty after 2014 in every variable with the dimension scenario
    coverage      every value filled in every cell of the common area (and the number of other cells with a weight
                  but without a value, for information)
    1 sigma       _sigma and _sigma_upper_limit finite where the value is (the 1 sigma of the levels may be missing
                  where sigma_flag = 1) and not negative; _sigma_alkalinity finite where the value is, not negative,
                  not below |_alkalinity_adjustment| and not above the full 1 sigma, to the stored precision: three
                  significant digits, and for the adjustment, the difference of two single-precision values of the
                  chain (with minus without), 2 float32 eps times the value
    flags         constrained_flag 0, 1 or 2, and 0 in the filled cells
    area means    the common-area means of the maps (weights cell_weight) against the area means of the maps that the
                  chain computed (make_oatrack.py, data/<run>/oatrack/OA-Track_<r>.nc: <q>_gm_common_map,
                  _<ssp>_map, _map_state, the mean of the four fCO2-products for the mean): the ESM years of the mean
                  in every scenario and year, the preindustrial states of every fCO2-product and the mean (largest
                  difference relative to the rms of the reference)
    information   the maps use the cell-by-cell constraint, the global means the global constraint of the paper:
                  their largest difference in the units of the quantity, where, and as a fraction of the 1 sigma of
                  the global mean (not a failure)
    obs maps      the preindustrial states (with their 1 sigma and adjustment) identical to those of
                  OA-Track_<q>_maps_obs_based_products.nc
Usage (code/):  OA_RUN=u74 python -u check_esm_maps.py [--src <dir>] [--chain <dir>] [--jobs 4] [--quantities ...]
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import netCDF4
import numpy as np

import oatrack_names as N

LAST_HISTORICAL = 2014
TOL_MEAN = 1e-4          # common-area means against those of the chain (maps with five significant digits)
EPS32 = float(np.finfo(np.float32).eps)
REFS = ['OceanSODA', 'UExP', 'JMA', 'CMEMS']
TOL_SIGMA = 1e-2         # comparisons between 1 sigma terms (stored with three significant digits)
SUFFIXES = ('_sigma_upper_limit', '_sigma_alkalinity', '_alkalinity_adjustment', '_sigma')


def arr(v, idx=Ellipsis):
    return np.ma.filled(np.ma.asarray(v[idx], dtype='f8'), np.nan)


def strs(v):
    return [str(x) for x in np.asarray(v[:]).ravel()]


def split(name):
    """value name and suffix of a variable"""
    for s in SUFFIXES:
        if name.endswith(s):
            return name[:-len(s)], s
    return name, ''


def check(args):
    q, src, chain = args
    t0 = time.time()
    F = next(F for F in N.files() if F.name == f'OA-Track_{q}_maps_esms.nc')
    want = {r['name']: tuple(r['dimlist']) for r in F.rows}
    bad, info = [], []
    with netCDF4.Dataset(src / F.name) as d, netCDF4.Dataset(src / f'OA-Track_{q}_global_mean.nc') as g, \
            netCDF4.Dataset(src / f'OA-Track_{q}_maps_obs_based_products.nc') as o:
        have = {n: tuple(v.dimensions) for n, v in d.variables.items()}
        if have != want:
            bad.append(f'variables: missing {sorted(set(want) - set(have))}, extra {sorted(set(have) - set(want))}, '
                       f'other dimensions {sorted(n for n in set(want) & set(have) if want[n] != have[n])}')
            return q, bad, info, time.time() - t0
        if strs(d['scenario']) != N.SCENARIOS or strs(d['product']) != N.PRODUCTS:
            bad.append(f'coordinates: scenario {strs(d["scenario"])}, product {strs(d["product"])}')
        years = np.asarray(d['year'][:]).astype(int)
        if years[0] != N.YEARS_GLOBAL[0] or years[-1] != N.YEARS_GLOBAL[1] or len(years) != 351:
            bad.append(f'years {years[0]}-{years[-1]} ({len(years)})')
        states = strs(d['state'])
        ih = N.SCENARIOS.index('historical')
        late = years > LAST_HISTORICAL

        w = arr(d['cell_weight'])
        common = arr(d['mask_common_area']) == 1
        wc = np.where(common, w, 0.0)
        weighted = w > 0
        filled = arr(d['filled_flag']) == 1
        cf = arr(d['constrained_flag'])
        if not set(np.unique(cf[np.isfinite(cf)]).tolist()) <= {0.0, 1.0, 2.0}:
            bad.append(f'constrained_flag values {np.unique(cf[np.isfinite(cf)])}')
        if (cf[filled] != 0).any():
            bad.append(f'constrained_flag not 0 in {(cf[filled] != 0).sum()} filled cells')
        sflag = (arr(d['sigma_flag']) == 1) if 'sigma_flag' in d.variables else np.zeros(w.shape, bool)
        info.append(f'common area {common.sum()} cells, filled {filled.sum()}, constrained_flag 1/2: '
                    f'{(cf == 1).sum()}/{(cf == 2).sum()}, sigma_flag {sflag.sum()}')

        # every map variable, grouped by its value, one scenario at a time
        groups = {}
        for n, dims in want.items():
            if 'lat' in dims and n.startswith(f'{q}_'):
                base, suf = split(n)
                groups.setdefault(base, {})[suf] = n
        level = lambda base: not base.endswith('_change')
        no_value = {}
        below = [0.0]
        for base, members in groups.items():
            dims = want[base]
            slices = [None] if 'scenario' not in dims else range(len(N.SCENARIOS))
            for s in slices:
                idx = tuple(s if dd == 'scenario' else slice(None) for dd in dims)
                a = {suf: arr(d[n], idx) for suf, n in members.items()}
                v = a['']
                tag = base + ('' if s is None else f' {N.SCENARIOS[s]}')
                ydim = (dims.index('year') - (1 if s is not None and dims.index('scenario') < dims.index('year') else 0)
                        if 'year' in dims else None)
                if s == ih:                                            # historical: nothing after 2014
                    sel = [slice(None)] * v.ndim
                    sel[ydim] = late
                    for suf, x in a.items():
                        if np.isfinite(x[tuple(sel)]).any():
                            bad.append(f'{tag}{suf}: values after {LAST_HISTORICAL}')
                    sel[ydim] = ~late
                    a = {suf: x[tuple(sel)] for suf, x in a.items()}
                    v = a['']
                miss = ~np.isfinite(v) & common
                if miss.any():
                    bad.append(f'{tag}: {miss.sum()} missing values in the common area')
                no_value[tag] = int((~np.isfinite(v) & weighted & ~common).sum() / max(1, v.size // w.size))
                ok = np.isfinite(v)
                for suf in ('_sigma', '_sigma_upper_limit'):
                    if suf in a:
                        x = a[suf]
                        allowed = sflag & level(base) & (q in N.T_DEPENDENT) & (suf == '_sigma')
                        if (ok & ~np.isfinite(x) & ~allowed).any():
                            bad.append(f'{tag}{suf}: {(ok & ~np.isfinite(x) & ~allowed).sum()} missing where the value is')
                        if (x < 0).any():
                            bad.append(f'{tag}{suf}: negative')
                        if (np.isfinite(x) & ~ok).any():
                            bad.append(f'{tag}{suf}: values where the value is missing')
                full = a.get('_sigma', a.get('_sigma_upper_limit'))
                if '_alkalinity_adjustment' in a:
                    ad, sa = a['_alkalinity_adjustment'], a['_sigma_alkalinity']
                    for suf, x in (('_alkalinity_adjustment', ad), ('_sigma_alkalinity', sa)):
                        if (ok & ~np.isfinite(x)).any():
                            bad.append(f'{tag}{suf}: {(ok & ~np.isfinite(x)).sum()} missing where the value is')
                    both = np.isfinite(ad) & np.isfinite(sa)
                    excess = np.abs(ad[both]) - sa[both] * (1 + TOL_SIGMA)
                    allowed = 2 * EPS32 * np.abs(v[both])
                    if (sa[both] < 0).any() or (excess > allowed).any():
                        bad.append(f'{tag}: _sigma_alkalinity below |_alkalinity_adjustment| by up to '
                                   f'{np.max(excess - allowed):.1e}')
                    if excess.size and excess.max() > 0:
                        below[0] = max(below[0], float(excess.max()))
                    both &= np.isfinite(full)
                    if (sa[both] > full[both] * (1 + TOL_SIGMA) + 1e-12).any():
                        bad.append(f'{tag}: _sigma_alkalinity above the full 1 sigma')
        worst = max(no_value.values())
        if below[0] > 0:
            info.append(f'_sigma_alkalinity below |_alkalinity_adjustment| by up to {below[0]:.1e} (within the precision of the adjustment)')
        info.append(f'cells with a weight but no value outside the common area: up to {worst}')

        # common-area means against those of the chain, and the difference to the global constraint
        areas = strs(g['area'])
        ia, ip = areas.index('common_area'), N.PRODUCTS.index('mean')
        W = wc.sum()
        mean = lambda x: np.einsum('...ij,ij->...', np.nan_to_num(x), wc) / W
        C = [netCDF4.Dataset(chain / f'OA-Track_{r}.nc') for r in REFS]
        yh = np.asarray(C[0]['year'][:]).astype(int)
        ys = np.asarray(C[0]['year_ssp'][:]).astype(int)
        rel, glob = [], []
        for s, scen in enumerate(N.SCENARIOS):
            name, yc = (f'{q}_gm_common_map', yh) if scen == 'historical' else (f'{q}_gm_common_{scen}_map', ys)
            chain_mean = np.mean([arr(c[name]) for c in C], 0)
            iy = np.searchsorted(years, yc)
            m = mean(arr(d[f'{q}_esms_annual_forced'], (s,))[iy])
            rel.append(np.max(np.abs(m - chain_mean)) / np.sqrt(np.mean(chain_mean ** 2)))
            gv = arr(g[f'{q}_esms_annual_forced_gm'], (ia, ip, s))[iy]
            gs = arr(g[f'{q}_esms_annual_forced_gm_sigma'], (ia, ip, s))[iy]
            k = int(np.nanargmax(np.abs(m - gv)))
            glob.append((abs(m[k] - gv[k]), m[k] - gv[k], scen, int(yc[k]), abs(m[k] - gv[k]) / gs[k]))
        r_esm = max(rel)
        pv = mean(arr(d[f'{q}_preindustrial']))                          # product, state
        pc = np.array([arr(c[f'{q}_gm_common_map_state']) for c in C])   # reference, state
        pc = np.concatenate([pc.mean(0)[None], pc])                     # mean, then the four, as PRODUCTS
        assert strs(C[0]['state']) == states
        r_pre = np.max(np.abs(pv - pc)) / np.sqrt(np.mean(pc ** 2))
        for c in C:
            c.close()
        for r, what in ((r_esm, 'ESM years of the mean'), (r_pre, 'preindustrial states')):
            if not r < TOL_MEAN:
                bad.append(f'{what}: common-area mean differs from that of the chain by {r:.1e}')
        info.append(f'common-area means against those of the chain: ESM years {r_esm:.1e}, preindustrial {r_pre:.1e}')
        _, dv, scen, yk, fr = max(glob)
        pg = arr(g[f'{q}_preindustrial_gm'], (ia,))
        ps = arr(g[f'{q}_preindustrial_gm_sigma'], (ia,))
        j = np.unravel_index(np.nanargmax(np.abs(pv - pg)), pv.shape)
        info.append(f'cell-by-cell minus global constraint (common area, {N.UNITS[q]}): ESM years of the mean up to '
                    f'{dv:+.3g} ({scen} {yk}, {fr:.2f} of the 1 sigma of the global mean), preindustrial up to '
                    f'{(pv - pg)[j]:+.3g} ({N.PRODUCTS[j[0]]} {states[j[1]]}, {abs(pv - pg)[j] / ps[j]:.2f} of the 1 sigma)')

        # the preindustrial states as in the maps of the obs-based products
        same = []
        for n in groups[f'{q}_preindustrial'].values():
            x, y = arr(d[n]), arr(o[n])
            same.append(x.shape == y.shape and np.array_equal(x, y, equal_nan=True))
        if not all(same):
            bad.append('preindustrial states differ from OA-Track_<q>_maps_obs_based_products.nc')
        info.append(f'preindustrial states identical to the obs-based maps: {sum(same)} of {len(same)} variables')
    return q, bad, info, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=None, help='default: data/<run>/oatrack_product')
    ap.add_argument('--chain', default=None, help='default: data/<run>/oatrack')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--quantities', nargs='+', default=N.QUANTITIES)
    a = ap.parse_args()
    if a.src is None or a.chain is None:
        import oa_config
    src = Path(a.src) if a.src else Path(oa_config.DATA) / 'oatrack_product'
    chain = Path(a.chain) if a.chain else Path(oa_config.DATA) / 'oatrack'
    failed = []
    with mp.get_context('fork').Pool(a.jobs) as pool:
        for q, bad, info, dt in pool.imap_unordered(check, [(q, src, chain) for q in a.quantities]):
            print(f'{q}: {"ok" if not bad else "FAILED"} ({dt:.0f} s)', flush=True)
            for line in info:
                print(f'    {line}', flush=True)
            for line in bad:
                print(f'    !! {line}', flush=True)
            if bad:
                failed.append(q)
    print(f'maps of the ESMs: {len(a.quantities) - len(failed)} of {len(a.quantities)} pass'
          + (f', failed: {" ".join(failed)}' if failed else ''), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
