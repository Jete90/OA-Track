#!/usr/bin/env python
"""
quantize_oatrack.py  (24 November 2026)
===================
The release copy of OA-Track for SEANOE (one record of about 175 GB): every file of data/<run>/oatrack/ copied into a
new folder with the same dimensions, variables, attributes, chunking and compression, and every map (a float32
variable on lat, lon) quantized (netCDF-C 4.9 and later, CF-compliant; GranularBitRound) to 5 significant digits, every
1 sigma and spread (_sigma*, _spread) to 3. The stored value differs from the computed one by less than half a unit of
the last kept digit (Omega_arag about 0.00003, its 1 sigma about 0.0005), far below every uncertainty of the data;
each variable records its digits in the attribute _QuantizeGranularBitRoundNumberOfSignificantDigits and each file
the rule in the global attribute storage_precision. Series, coordinates, masks and texts are copied unchanged.
The same rule as make_oatrack.py of version 5 (test_oatrack_compression.py: 279 GiB of version 4 become about 164 GiB).

Only reads the source folder. Writes <file>.part and renames it when the file is complete and checked, so an
interrupted run can simply be started again (complete files are skipped).
Check per file: every unquantized variable equal to the source (sampled), every quantized one within its rounding.

Usage (code/ of v4, env_jens):
  OMP_NUM_THREADS=1 python -u quantize_oatrack.py --src ../data/u74/oatrack --dst ../data/u74/oatrack_seanoe --jobs 3
"""
import argparse
import multiprocessing as mp
import time
from pathlib import Path

import netCDF4
import numpy as np

DIGITS = (5, 3)                       # values, every 1 sigma and spread
MODE = 'GranularBitRound'
BLOCK = 48                            # maps per read and write
TEXT = (f'maps quantized ({MODE}, netCDF-C) to {DIGITS[0]} significant digits, every 1 sigma and spread to {DIGITS[1]}; '
        'the stored value differs from the computed one by less than half a unit of the last kept digit, far below '
        'every uncertainty; zlib with shuffle on top (quantize_oatrack.py, 24 November 2026)')


def is_map(v):
    return v.dtype == np.float32 and len(v.dimensions) >= 2 and v.dimensions[-2:] == ('lat', 'lon')


def digits(name):
    return DIGITS[1] if ('_sigma' in name or name.endswith('_spread')) else DIGITS[0]


def copy_file(src, dst):
    t0 = time.time()
    part = dst.with_name(dst.name + '.part')
    if part.exists():
        part.unlink()
    with netCDF4.Dataset(src) as f, netCDF4.Dataset(part, 'w', format=f.data_model) as g:
        g.setncatts({k: f.getncattr(k) for k in f.ncattrs()})
        g.storage_precision = TEXT
        g.history = (f'{time.strftime("%Y-%m-%d %H:%M")} quantize_oatrack.py (copy of {src.name}); '
                     + str(getattr(f, 'history', '')))
        for d in f.dimensions.values():
            g.createDimension(d.name, None if d.isunlimited() else len(d))
        nq = 0
        for name, v in f.variables.items():
            filt = v.filters() or {}
            chunk = v.chunking()
            fill = v.getncattr('_FillValue') if '_FillValue' in v.ncattrs() else None
            kw = dict(zlib=bool(filt.get('zlib')), complevel=filt.get('complevel', 4) or 4, shuffle=bool(filt.get('shuffle')),
                      chunksizes=None if chunk in (None, 'contiguous') else chunk, fill_value=fill)
            dtype = v.datatype if v.dtype != str else str
            if dtype is str:
                kw = {}
            q = is_map(v)
            if q:
                kw.update(zlib=True, shuffle=True, significant_digits=digits(name), quantize_mode=MODE)
                nq += 1
            w = g.createVariable(name, dtype, v.dimensions, **kw)
            w.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != '_FillValue'})
            v.set_auto_maskandscale(False)              # raw values; maps: the fill cells masked again below
            if not q:
                w.set_auto_maskandscale(False)
            if v.ndim >= 3 and q:
                n = v.shape[0]
                for i in range(0, n, BLOCK):
                    j = min(i + BLOCK, n)                  # explicit end: an unlimited dimension would grow otherwise
                    a = v[i:j]
                    if fill is not None:
                        a = np.ma.masked_equal(a, fill)
                    w[i:j] = a
            else:
                a = v[...]
                if q and fill is not None:
                    a = np.ma.masked_equal(a, fill)
                w[...] = a
    # check: sampled values against the source
    worst = {}
    with netCDF4.Dataset(src) as f, netCDF4.Dataset(part) as g:
        for name, v in f.variables.items():
            w = g[name]
            if v.dtype == str:
                assert list(np.asarray(v[:]).ravel()) == list(np.asarray(w[:]).ravel()), f'{name}: text differs'
                continue
            idx = (slice(None),) if v.ndim < 3 else (slice(None, None, max(1, v.shape[0] // 4)),)
            a = np.ma.filled(v[idx].astype('f8'), np.nan)
            b = np.ma.filled(w[idx].astype('f8'), np.nan)
            assert np.array_equal(np.isnan(a), np.isnan(b)), f'{name}: different missing cells'
            ok = np.isfinite(a)
            if is_map(v):
                rel = np.abs(b[ok] - a[ok]) / np.maximum(np.abs(a[ok]), 1e-30)
                lim = 10.0 ** (1 - digits(name))     # one unit of the last kept digit, relative (rounding: at most half)
                r = float(rel.max()) if rel.size else 0.0
                assert r <= lim, f'{name}: relative error {r:.2e} above {lim:.2e}'
                worst[name] = r
            else:
                assert np.array_equal(a[ok], b[ok]), f'{name}: changed although not a map'
            assert w.quantization() is None or w.quantization()[0] == digits(name), name
    part.rename(dst)
    big = max(worst, key=worst.get) if worst else '-'
    return (f'{src.name}: {src.stat().st_size / 2**30:.1f} -> {dst.stat().st_size / 2**30:.1f} GiB, {nq} maps quantized, '
            f'largest relative change {worst.get(big, 0):.1e} ({big}), {(time.time() - t0) / 60:.0f} min')


def work(args):
    src, dst = args
    try:
        return copy_file(Path(src), Path(dst))
    except Exception as e:                        # report and go on with the other files
        return f'{Path(src).name}: FAILED {type(e).__name__}: {e}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='the OA-Track folder (only read)')
    ap.add_argument('--dst', required=True, help='a new folder for the release copy')
    ap.add_argument('--jobs', type=int, default=3, help='files in parallel (at most 3 on the shared server)')
    ap.add_argument('--files', nargs='*', help='only these file names')
    a = ap.parse_args()
    src, dst = Path(a.src).resolve(), Path(a.dst).resolve()
    assert src != dst, 'source and destination are the same folder'
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.glob('OA-Track_*.nc') if not a.files or p.name in a.files)
    todo = [(str(p), str(dst / p.name)) for p in files if not (dst / p.name).exists()]
    print(f'{len(files)} files in {src}, {len(files) - len(todo)} already in {dst}, {len(todo)} to copy, {a.jobs} at a time',
          flush=True)
    with mp.get_context('spawn').Pool(min(a.jobs, 3, max(1, len(todo)))) as pool:
        for line in pool.imap_unordered(work, todo):
            print(time.strftime('%Y-%m-%d %H:%M'), line, flush=True)
    done = sorted(dst.glob('OA-Track_*.nc'))
    tot = sum(p.stat().st_size for p in done)
    print(f'{len(done)} of {len(files)} files in {dst}: {tot / 2**30:.0f} GiB ({tot / 1e9:.0f} GB)', flush=True)


if __name__ == '__main__':
    main()
