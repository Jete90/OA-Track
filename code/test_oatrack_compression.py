#!/usr/bin/env python
"""
test_oatrack_compression.py  (20 November 2026: how small can OA-Track get for SEANOE, 150 GB per record?)
===========================
OA-Track of v4 is 279 GB (14 files). This test copies a sample of map variables of one OA-Track file in several ways
and reports the size of each against the file as it is, and the largest error each introduces:
    as_is      zlib level 4 with shuffle, one map per chunk (as make_oatrack.py writes it; the reference)
    zlib9      the same, zlib level 9 (lossless)
    q5_3       quantized to 5 significant digits (values) and 3 (every _sigma*, _spread), zlib 4, shuffle
    q4_2       4 and 2 significant digits
    gbr5_3     as q5_3 with GranularBitRound instead of BitGroom (netCDF-C 4.9 and later)
Quantization (netCDF-C 4.9, CF-compliant, documented in the variable attributes) sets the bits below the kept
digits to values that compress well; the stored value differs from the original by less than half a unit of the
last kept digit. Only reads the input; writes into --out (a new folder).
Usage:  python -u test_oatrack_compression.py --file <v4>/data/u74/oatrack/OA-Track_JMA.nc --out <new folder>
"""
import argparse
import os
import time

import netCDF4
import numpy as np

QUANT = ['omega_arag', 'ph', 'hplus', 'dic', 'talk', 'thetao', 'pco2', 'revelle']
SUFFIXES = {'omega_arag': ['', '_unconstrained', '_sigma', '_sigma_mapping_c2', '_sigma_internal', '_sigma_constants']}
VARIANTS = {'as_is': dict(complevel=4), 'zlib9': dict(complevel=9),
            'q5_3': dict(complevel=4, digits=(5, 3), mode='BitGroom'), 'q4_2': dict(complevel=4, digits=(4, 2), mode='BitGroom'),
            'gbr5_3': dict(complevel=4, digits=(5, 3), mode='GranularBitRound')}


def is_sigma(name):
    return '_sigma' in name or name.endswith('_spread')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--file', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--block', default='_ssp245', help="the block to sample: '' (historical), _ssp245, _obs, _state")
    ap.add_argument('--years', type=int, default=120, help='number of time steps of the sample')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    total = os.path.getsize(a.file)
    with netCDF4.Dataset(a.file) as f:
        names = [n for n in (f'{q}{a.block}{x}' for q in QUANT for x in SUFFIXES.get(q, [''])) if n in f.variables]
        print(f'{a.file}: {total / 2**30:.1f} GiB, netCDF4 {netCDF4.__version__}, netCDF-C {netCDF4.__netcdf4libversion__}')
        print(f'sample: {len(names)} variables of the block "{a.block}", first {a.years} time steps: {", ".join(names)}', flush=True)
        data = {}
        for n in names:
            v = f[n]
            data[n] = (v.dimensions, v[:a.years] if v.ndim == 3 else v[:], v.units, getattr(v, '_FillValue', None))
        lat, lon = np.asarray(f['lat'][:]), np.asarray(f['lon'][:])
    res = {}
    for key, opt in VARIANTS.items():
        path = os.path.join(a.out, f'sample_{key}.nc')
        t0 = time.time()
        try:
            with netCDF4.Dataset(path, 'w') as g:
                g.createDimension('lat', lat.size)
                g.createDimension('lon', lon.size)
                for n, (dims, arr, unit, fill) in data.items():
                    for d, s in zip(dims, arr.shape):
                        if d not in g.dimensions:
                            g.createDimension(d, s)
                    kw = dict(zlib=True, complevel=opt['complevel'], shuffle=True, fill_value=fill,
                              chunksizes=tuple(1 if d not in ('lat', 'lon') else s for d, s in zip(dims, arr.shape)))
                    if 'digits' in opt:
                        kw.update(significant_digits=opt['digits'][1] if is_sigma(n) else opt['digits'][0],
                                  quantize_mode=opt['mode'])
                    x = g.createVariable(n, 'f4', dims, **kw)
                    x.units = unit
                    x[:] = arr
        except Exception as e:                    # e.g. GranularBitRound not in this netCDF-C
            print(f'  {key:7s} not possible: {e}')
            continue
        size = os.path.getsize(path)
        err = {}
        with netCDF4.Dataset(path) as g:
            for n, (dims, arr, unit, fill) in data.items():
                o = np.ma.filled(arr.astype('f8'), np.nan)
                q = np.ma.filled(g[n][:].astype('f8'), np.nan)
                ok = np.isfinite(o)
                d = np.abs(q[ok] - o[ok])
                rel = d / np.maximum(np.abs(o[ok]), 1e-30)
                err[n] = (float(d.max()) if d.size else np.nan, float(rel.max()) if d.size else np.nan)
        res[key] = size
        worst = max(err, key=lambda k: err[k][1])
        print(f'  {key:7s} {size / 2**20:8.1f} MiB  ({size / res["as_is"]:.2f} of as_is, {time.time() - t0:5.1f} s); largest '
              f'relative error {err[worst][1]:.1e} ({worst}); absolute: Omega_arag '
              f'{err.get(f"omega_arag{a.block}", (np.nan,))[0]:.1e}, its 1 sigma {err.get(f"omega_arag{a.block}_sigma", (np.nan,))[0]:.1e}', flush=True)
    print('\nextrapolated to the 279 GB of OA-Track v4 (same ratio for every variable): '
          + ', '.join(f'{k} {279 * s / res["as_is"]:.0f} GB' for k, s in res.items()))


if __name__ == '__main__':
    main()
