#!/usr/bin/env python
"""
summarize_product.py
====================
A compact text summary of every file of the data product, for checks without the files themselves (they are too
large to pass on). Per file: the global attributes, the dimensions, and per variable its dimensions, units and
description, the share of missing values, minimum and maximum; for fields on (lat, lon) the area mean (the weights of
the run, oa_config.area_weights, over the cells with a value) and the share of cells with a value, per index of the leading dimension; for series
the values. Long dimensions (year, year_ssp, ...) are printed at the years in YEARS only, short ones (state) in full.

Output: one text file per netCDF file, logs/<run>/product_summary/<file>.txt
Usage (code/):  OA_RUN=u74 nohup python -u summarize_product.py > ../logs/u74/summarize_product.log 2>&1 &
                (all carbonate_system_*.nc of data/<run>/product, four at a time (--jobs); or give files as arguments)
"""
import argparse
import datetime
import multiprocessing
import sys
from pathlib import Path

import netCDF4
import numpy as np

import oa_config

YEARS = (1750, 1800, 1849, 1850, 1900, 1950, 1980, 1983, 1985, 1989, 1990, 2000, 2004, 2005, 2010, 2014, 2015,
         2020, 2024, 2025, 2030, 2050, 2075, 2099, 2100)
SHORT = 12            # leading dimensions up to this length are printed in full


def fmt(x):
    return 'nan' if not np.isfinite(x) else f'{x:.6g}'


def coord(f, dim):
    """the values of a coordinate variable, or the index"""
    if dim in f.variables and f[dim].ndim == 1:
        v = f[dim][:]
        return [str(a) for a in (v.tolist() if hasattr(v, 'tolist') else v)]
    return [str(i) for i in range(len(f.dimensions[dim]))]


def picks(labels):
    """indices to print along a leading dimension"""
    if len(labels) <= SHORT:
        return list(range(len(labels)))
    years = {str(y) for y in YEARS}
    sel = [i for i, s in enumerate(labels) if s in years]
    return sel or [0, len(labels) // 2, len(labels) - 1]


def summarize(path, out):
    t0 = datetime.datetime.now()
    with netCDF4.Dataset(path) as f, open(out, 'w') as o:
        w = lambda s='': o.write(s + '\n')
        w(f'# {path}  ({path.stat().st_size / 1e9:.2f} GB), summary written {t0:%Y-%m-%d %H:%M}')
        for a in f.ncattrs():
            w(f'# attr {a}: {str(getattr(f, a))[:400]}')
        w('# dims ' + ', '.join(f'{d}={len(v)}' for d, v in f.dimensions.items()))
        latn = next((d for d in f.dimensions if d.lower().startswith('lat')), None)
        lonn = next((d for d in f.dimensions if d.lower().startswith('lon')), None)
        wlat = None
        if latn and latn in f.variables and lonn and lonn in f.variables:
            wlat = oa_config.area_weights(f[latn][:], f[lonn][:])      # (ny, nx), the weights of the run
        for name, v in f.variables.items():
            desc = getattr(v, 'long_name', getattr(v, 'description', ''))
            w(f'\n{name} {v.dimensions} {v.dtype} [{getattr(v, "units", "")}] {str(desc)[:300]}')
            if v.dtype == str or v.dtype.kind in 'SUO':
                w('  ' + ', '.join(str(a) for a in np.ravel(v[:])[:SHORT]))
                continue
            try:
                a = np.ma.filled(v[:].astype('f8'), np.nan)
            except Exception as e:                                   # noqa: BLE001 (any read problem is reported)
                w(f'  read failed: {type(e).__name__}: {e}')
                continue
            fin = np.isfinite(a)
            if a.size == 0 or not fin.any():
                w(f'  all missing ({a.size} values)')
                continue
            w(f'  missing {100 * (1 - fin.mean()):.2f} %, min {fmt(np.nanmin(a))}, max {fmt(np.nanmax(a))}'
              + (f', value {fmt(float(a))}' if a.ndim == 0 else ''))
            dims = v.dimensions
            spatial = latn in dims and lonn in dims and dims[-2:] == (latn, lonn) and wlat is not None
            if spatial and a.ndim == 2:
                ok = np.isfinite(a)
                wm = np.where(ok, wlat, 0.0)
                w(f'  area mean {fmt(np.sum(np.where(ok, a, 0) * wm) / np.sum(wm))}, cells with a value {100 * ok.mean():.1f} %')
            elif spatial and a.ndim == 3:
                lab = coord(f, dims[0])
                for i in picks(lab):
                    ok = np.isfinite(a[i])
                    if not ok.any():
                        w(f'  {dims[0]} {lab[i]}: all missing')
                        continue
                    wm = np.where(ok, wlat, 0.0)
                    w(f'  {dims[0]} {lab[i]}: area mean {fmt(np.sum(np.where(ok, a[i], 0) * wm) / np.sum(wm))}, '
                      f'cells {100 * ok.mean():.1f} %, min {fmt(np.nanmin(a[i]))}, max {fmt(np.nanmax(a[i]))}')
            elif a.ndim == 1 and len(a) > 1 and name != dims[0]:            # series, not the coordinate itself
                lab = coord(f, dims[0])
                w('  ' + ', '.join(f'{lab[i]}: {fmt(a[i])}' for i in picks(lab)))
            elif a.ndim >= 2 and not spatial:
                lab = coord(f, dims[0])
                for i in picks(lab):
                    w(f'  {dims[0]} {lab[i]}: mean {fmt(np.nanmean(a[i]))}, min {fmt(np.nanmin(a[i]))}, max {fmt(np.nanmax(a[i]))}')
        w(f'\n# done in {(datetime.datetime.now() - t0).total_seconds():.0f} s')


DEST = oa_config.BASE / 'logs' / oa_config.RUN / 'product_summary'


def one(p):
    out = DEST / (p.stem + '.txt')
    t0 = datetime.datetime.now()
    try:
        summarize(p, out)
        return f'  {p.name}: {(datetime.datetime.now() - t0).total_seconds():.0f} s -> {out.name}'
    except Exception as e:                                           # noqa: BLE001 (one bad file must not stop the rest)
        return f'  {p.name}: failed: {type(e).__name__}: {e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*')
    ap.add_argument('--jobs', type=int, default=4, help='files read in parallel (memory: about 0.5 GB each)')
    a = ap.parse_args()
    files = [Path(p) for p in a.files] or sorted((oa_config.DATA / 'product').glob('carbonate_system_*.nc'))
    DEST.mkdir(parents=True, exist_ok=True)
    print(f'{len(files)} files -> {DEST}, {a.jobs} in parallel', flush=True)
    with multiprocessing.Pool(max(1, a.jobs)) as pool:
        for line in pool.imap_unordered(one, files):
            print(line, flush=True)
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
