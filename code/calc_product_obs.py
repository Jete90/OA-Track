#!/usr/bin/env python
"""
calc_product_obs.py
===================
The observation-based part of the data product: all carbonate system indicators of every reference product
(OceanSODA-ETHZ, UExP-FNN-U, JMA-MLR, CMEMS-LSCE-FFNN; monthly T, S, DIC, TA of the product, mocsy with the
constants of the run, GLODAPv2 Si and PO4, as for the models) for every year, computed from the annual-mean state
(annual means of T, S, DIC and TA, all 12 months valid), as for the CMIP6 models, whose inputs are annual means;
and the same for the decade 2010 state of Jiang et al. (2023). The level uncertainty (mapping errors) is
propagated month by month. The Omega_arag computed here is checked against the product file of the chain
(obs/omega_arag_Oyr_<product>.nc, identical chemistry) and, where the product provides it, against the
product's own Omega_arag.

Output: data/<run>/product/obs_<reference>.nc
    <indicator>(year, lat, lon)   indicators of the annual-mean state: omega_arag, omega_calc, ph, hplus, hplus_free, pco2, fco2, co2, hco3, co3,
                                  revelle, dic and the state thetao, so, talk (Jiang: one field, year 2010)
    mask(lat, lon)                cells valid in every year
    level uncertainty of the product (level_uncertainty.py), not included in anything else:
    <indicator>_sigma_mapping_c1, _c2(year, lat, lon)   mapping uncertainty, months independent (C1) or the same
                                  error in all months (C2); <indicator>_sigma_constants  dissociation constants
    <indicator>_gm_<own|common>_sigma_mapping_c1, _c2, _sigma_constants(year)   the same for the area means on the
                                  constraint area (own) and the common area of the paper, each with the product mask
    level_dx_<var>(component, month, lat, lon)   the product's mapping errors as state errors, window mean (the
                                  bias-adjustment window), for the model years (calc_product_states.py)

Usage:  python calc_product_obs.py                      # all references
        python calc_product_obs.py --reference UExP
"""
from __future__ import annotations

import argparse
import oa_config   # run: data folder, mocsy constants, GFDL density (OA_RUN)
import datetime
import warnings
from pathlib import Path

import netCDF4
import numpy as np

import calc_bias_adjusted as cb
import calc_omega_arag_obs as co
import calc_omega_arag_products as cp
import calc_pi_xco2 as px
import level_uncertainty as lu
import pickle
from make_jma_ts import year_month

OUT = oa_config.DATA / 'product'
STATE_VARS = ('thetao', 'so', 'talk')
CHUNK = 60


def process(ref):
    cfg = cb.REFERENCES[ref]
    print(f'\n=== {ref}: {cfg["label"]}', flush=True)
    if cfg.get('annual'):                               # Jiang: one state
        with netCDF4.Dataset(cfg['file']) as f:
            lat, lon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
            v = cfg['vars']
            st = {'thetao': np.ma.filled(f[v['t']][:1].astype('f8'), np.nan),
                  'so': np.ma.filled(f[v['s']][:1].astype('f8'), np.nan),
                  'dissic': np.ma.filled(f[v['dic']][:1].astype('f8'), np.nan),
                  'talk': np.ma.filled(f[v['ta']][:1].astype('f8'), np.nan)}
            years = np.asarray(f['year'][:]).astype(int)
        blocks = [(years, st)]
    else:
        main = cp.Reader(cfg['file'], cfg['order'])
        ts = cp.Reader(cfg['ts_file'], 'time,lat,lon') if cfg.get('ts_file') else main
        lat = np.asarray(main.f['lat'][:], 'f8')
        lon = np.asarray(main.f['lon'][:], 'f8')
        ym = year_month(main.f['time'])
        first = next(i for i, (y, m) in enumerate(ym) if m == 1)
        nt = (len(ym) - first) // 12 * 12
        years = np.array([ym[first + 12 * k][0] for k in range(nt // 12)])
        v = cfg['vars']

        def blocks_gen():
            for c0 in range(0, nt, CHUNK):
                c1 = min(c0 + CHUNK, nt)
                i0, i1 = first + c0, first + c1
                st = {'thetao': ts(v['t'], i0, i1) - (273.15 if cfg['t_kelvin'] else 0.0),
                      'so': ts(v['s'], i0, i1), 'dissic': main(v['dic'], i0, i1), 'talk': main(v['ta'], i0, i1)}
                yield years[c0 // 12:c1 // 12], st
        blocks = blocks_gen()
    ny, nx = lat.size, lon.size
    lat2 = np.broadcast_to(lat[:, None], (ny, nx))
    si, po4 = co.load_nutrients(lat, lon)
    names = px.INDICATOR_NAMES + list(STATE_VARS)
    out = {k: np.full((years.size,) + (ny, nx), np.nan, 'f4') for k in names}
    # the level uncertainty: masks of the area means (constraint areas with the product mask of the chain file)
    lev = lu.Level(ref, lat, lon, si, po4, level_masks(ref, cfg, lat, lon))
    for k_, note in lev.available().items():
        print(f'  level uncertainty, {k_}: {note}', flush=True)
    # the years of the chain's Omega_arag file of this product (complete years with enough coverage, the
    # selection of calc_omega_arag_obs.py / calc_omega_arag_products.py): the product uses exactly these years
    with netCDF4.Dataset(cfg['omega']) as f:
        chain_years = np.asarray(f['year'][:]).astype(int)
    keep = np.isin(years, chain_years)
    if not keep.all():
        print(f'  years not in the chain file (incomplete or too little coverage), left out: '
              f'{", ".join(str(y) for y in years[~keep])}', flush=True)
    window = lu.window_of(ref, years[keep])
    LK = ('map1', 'map2', 'const')
    lmap = {q: {k: np.full((years.size, ny, nx), np.nan, 'f4') for k in LK} for q in names}
    lser = {q: {m: np.full((years.size, 3), np.nan) for m in lev.masks} for q in names}
    dxs = {c: {x: np.zeros((12, ny, nx)) for x in lu.DX} for c in lev.names}
    nwin = 0
    lres = None
    for yrs, st in blocks:
        step = st['thetao'].shape[0] // yrs.size              # 12, or 1 for the annual state
        # the annual-mean state (NaN unless every month is valid), then the indicators of that state, as for the
        # CMIP6 models, whose inputs are annual means
        sa = {k: a.reshape(-1, step, ny, nx).mean(axis=1) for k, a in st.items()}
        _, res = next(px.target_states(sa, si, po4, lat2, {'state': px.XCO2_MODEL}, [0.0]))
        sl = slice(int(np.where(years == yrs[0])[0][0]), int(np.where(years == yrs[-1])[0][0]) + 1)
        for k in px.INDICATOR_NAMES:
            out[k][sl] = res[k]
        for k in STATE_VARS:
            out[k][sl] = sa[k]
        for j, y in enumerate(yrs):
            i = int(np.where(years == y)[0][0])
            if not keep[i]:
                continue
            f12 = {v: st[v][j * step:(j + 1) * step] for v in lu.VARS}
            if step == 1:
                f12 = {v: np.repeat(a, 12, axis=0) for v, a in f12.items()}
            lres, dX = lev.product_year(f12, int(y))
            for q in names:
                for k in LK:
                    lmap[q][k][i] = lres[q][k]
                for m in lev.masks:
                    lser[q][m][i] = lres[q]['series'][m]
            if window[0] <= y <= window[1]:
                for c in lev.names:
                    for x in lu.DX:
                        dxs[c][x] += np.where(np.isfinite(dX[c][x]), dX[c][x], np.nan)
                nwin += 1
        g0 = lres['omega_arag']['series'].get('own', (np.nan,) * 3) if lres is not None else (np.nan,) * 3
        print(f'  {yrs[0]}-{yrs[-1]} done (level uncertainty of Omega_arag {yrs[-1]}, own area: mapping C1 {g0[0]:.4f}, '
              f'C2 {g0[1]:.4f}, constants {g0[2]:.4f})', flush=True)
    dxs = {c: {x: a / max(nwin, 1) for x, a in d.items()} for c, d in dxs.items()}
    lev.close()
    if not cfg.get('annual'):
        main.close()
        if ts is not main:
            ts.close()

    # only the chain's years from here on
    years = years[keep]
    out = {k: a[keep] for k, a in out.items()}
    lmap = {q: {k: a[keep] for k, a in d.items()} for q, d in lmap.items()}
    lser = {q: {m: a[keep] for m, a in d.items()} for q, d in lser.items()}
    # checks against the Omega_arag file of the chain (same chemistry) and the product's own Omega_arag
    om = out['omega_arag']
    with netCDF4.Dataset(cfg['omega']) as f:
        yo = np.asarray(f['year'][:]).astype(int)
        oc = np.ma.filled(f['omega_arag'][:].astype('f8'), np.nan)
        op = np.ma.filled(f['omega_arag_product'][:].astype('f8'), np.nan) if 'omega_arag_product' in f.variables else None
    sel = np.isin(years, yo)
    d = om[sel] - oc[np.isin(yo, years)]
    print(f'  Omega_arag against the chain file: max |difference| {np.nanmax(np.abs(d)):.2e}, years {years[sel][0]}-{years[sel][-1]}')
    if op is not None:
        d = om[sel] - op[np.isin(yo, years)]
        print(f'  Omega_arag against the product\'s own: mean {np.nanmean(d):+.4f}, RMS {np.sqrt(np.nanmean(d ** 2)):.4f}')
    mask = np.all(np.isfinite(om), axis=0)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'obs_{ref}.nc'
    units = {**px.UNITS_ALL, **cb.UNITS}
    g = cb.new_nc(path, lat, lon, dict(
        title=f'Annual surface carbonate system of {cfg["label"]}',
        method=('indicators with mocsy from the annual means of the product T, S, DIC and TA (all 12 months valid); '
                + oa_config.CONSTANTS + '; GLODAPv2.2016b surface Si and PO4'),
        source=str(cfg['file']) + (f'; T/S {cfg["ts_file"]}' if cfg.get('ts_file') else ''),
        history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} calc_product_obs.py, {oa_config.VERSION}'))
    g.createDimension('year', years.size)
    g.createVariable('year', 'i4', ('year',))[:] = years
    for k, a in out.items():
        x = g.createVariable(k, 'f4', ('year', 'lat', 'lon'), zlib=True, complevel=4, fill_value=cb.FILL)
        x.units = units[k]
        x[:] = np.ma.masked_invalid(a)
    x = g.createVariable('mask', 'i1', ('lat', 'lon'))
    x.long_name = 'cells valid in every year'
    x[:] = mask.astype('i1')
    LN = {'map1': ('_sigma_mapping_c1', 'mapping uncertainty of the product, 1 sigma, errors independent between months (C1)'),
          'map2': ('_sigma_mapping_c2', 'mapping uncertainty of the product, 1 sigma, the same error in all months (C2, conservative)'),
          'const': ('_sigma_constants', 'uncertainty from the dissociation constants and total boron, 1 sigma (fully correlated)')}
    for q in names:
        for k, (suf, ln) in LN.items():
            x = g.createVariable(f'{q}{suf}', 'f4', ('year', 'lat', 'lon'), zlib=True, complevel=4, fill_value=cb.FILL)
            x.units, x.long_name = units[q], f'{q}: {ln}'
            x[:] = np.ma.masked_invalid(lmap[q][k])
            for m in lev.masks:
                x = g.createVariable(f'{q}_gm_{m}{suf}', 'f8', ('year',), fill_value=np.nan)
                x.units = units[q]
                x.long_name = f'{q}, area mean on the {m} area: {ln}; spatial correlation exp(-d/L) of every component'
                x[:] = lser[q][m][:, list(LN).index(k)]
    lu.write_dx(g, lev, dxs, window)
    g.level_uncertainty = lu.__doc__.split('mapping ', 1)[1].strip()
    g.level_components = '; '.join(lev.report())
    g.close()
    print('  level uncertainty components: ' + '\n    '.join([''] + lev.report()), flush=True)
    w = oa_config.area_weights(lat, lon) * mask
    gm = lambda a: np.nansum(np.where(mask, a, 0) * w) / w.sum()
    print(f'  {years[0]}: Omega_arag {gm(om[0]):.3f}, pH {gm(out["ph"][0]):.3f}; {years[-1]}: Omega_arag {gm(om[-1]):.3f}, '
          f'pH {gm(out["ph"][-1]):.3f}; written {path}', flush=True)


def level_masks(ref, cfg, lat, lon):
    """the area masks of the level uncertainty: the constraint areas of the chain (own, common) with the cells the
    chain's Omega_arag file has in every year; without a chain result the product mask alone (both)"""
    with netCDF4.Dataset(cfg['omega']) as f:
        om = np.ma.filled(f['omega_arag'][:].astype('f8'), np.nan)
        olat, olon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
    pm = np.all(np.isfinite(om), axis=0)[co.nearest_index(olat, lat)][:, co.nearest_index(olon, lon, periodic=True)]
    try:
        if ref == 'Jiang':
            with open(oa_config.DATA / 'constraint_results_jiang.pkl', 'rb') as f:
                J = pickle.load(f)
            ms = (J['own']['Jiang']['mask'], J['common']['Jiang']['mask'])
        else:
            with open(oa_config.DATA / 'constraint_results.pkl', 'rb') as f:
                P = pickle.load(f)
            ms = (P['results'][ref]['mask'], P['results_common'][ref]['mask'])
        own, com = (np.asarray(m).astype(bool) for m in ms)
        if own.shape != pm.shape:
            raise ValueError(f'mask shape {own.shape} differs from the product grid {pm.shape}')
    except (OSError, KeyError, ValueError) as e:
        print(f'  level uncertainty: no constraint masks ({type(e).__name__}: {e}); the product mask for both areas')
        own = com = np.ones(pm.shape, bool)
    return {'own': own & pm, 'common': com & pm}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', nargs='+', default=list(cb.REFERENCES), choices=list(cb.REFERENCES))
    a = ap.parse_args()
    for ref in a.reference:
        cfg = cb.REFERENCES[ref]
        if not Path(cfg['file']).exists():
            print(f'\n=== {ref}: skipped, missing {cfg["file"]}')
            continue
        cb.set_reference(ref)
        process(ref)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
