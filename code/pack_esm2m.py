#!/usr/bin/env python
"""
pack_esm2m.py  (the GFDL-ESM2M-COBALTv2 output of the paper for its own Zenodo record)
=============
Copies the part of the GFDL-ESM2M-COBALTv2 simulations that the analysis reads (feedback_pattern.py:
monthly surface alkalinity 'talk' and salinity 'salt' on the 1 degree grid) into a new
folder with the same folder structure and file names, so that the code runs unchanged with OA_ESM2M set to it:
    pi_control/OA_<decade>.01.01_to_<decade+10>.01.01_r360_180.nc                    1700-2099 (40 files)
    coupled/climate_change_RCP26/<member>/OA_variables/OA_<decade>...nc          1860-2099 (24 files per member)
        member: ens2 to ens6
Each file keeps talk and salt at the first level only (dimensions as in the source, the level dimension of length 1,
so that f[var][:, 0] reads the same field), the coordinates, the time axis with its bounds and every attribute;
lossless (zlib 4 with shuffle, one map per chunk). Every other field of the files is left out.
Check per file: talk and salt equal to the source at the first level, bit for bit, and the same coordinates.
At the end: MANIFEST.md5 and one tar each for pi_control and coupled in <dst>_zenodo/ (Zenodo takes at most 100 files
per record; the netCDF files are compressed already, so the tars are not).

Only reads the source. Writes <file>.part and renames it when the file is complete and checked, so an interrupted run
can simply be started again (complete files are skipped).

Usage (code/):
  OMP_NUM_THREADS=1 python -u pack_esm2m.py --dst ../data/esm2m_zenodo/GFDL-ESM2M-COBALTv2 --jobs 3
"""
import argparse
import hashlib
import multiprocessing as mp
import tarfile
import time
from pathlib import Path

import netCDF4
import numpy as np

import feedback_pattern as fp

VARS = ('talk', 'salt')                  # what feedback_pattern.esm2m_ta_s reads
EXPS = fp.ESM2M_EXPS                     # coupled
SCEN = fp.ESM2M_SCEN                     # climate_change_RCP26
MEMBERS = fp.ESM2M_MEMBERS               # ens2 to ens6
PERIOD = fp.PERIOD_ESM2M                 # 1860-2099
CONTROL = fp.CONTROL_ESM2M               # 1700-2099


def decade_file(folder, dec):
    """the file name of feedback_pattern.esm2m_annual"""
    return folder / f'OA_{dec}.01.01_to_{dec + 10}.01.01_r360_180.nc'


def file_list():
    rel = [decade_file(Path('pi_control'), d) for d in range(CONTROL[0] // 10 * 10, CONTROL[1] + 1, 10)]
    for exp in EXPS:
        for mem in MEMBERS:
            rel += [decade_file(Path(exp) / SCEN / mem / 'OA_variables', d) for d in range(PERIOD[0] // 10 * 10, PERIOD[1] + 1, 10)]
    return rel


def copy_file(src, dst):
    t0 = time.time()
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + '.part')
    if part.exists():
        part.unlink()
    with netCDF4.Dataset(src) as f, netCDF4.Dataset(part, 'w', format='NETCDF4') as g:
        dims4 = f[VARS[0]].dimensions
        assert len(dims4) == 4 and all(f[v].dimensions == dims4 for v in VARS), f'{src}: {[f[v].dimensions for v in VARS]}'
        tdim, zdim, ydim, xdim = dims4
        g.setncatts({k: f.getncattr(k) for k in f.ncattrs()})
        g.history = (f'{time.strftime("%Y-%m-%d %H:%M")} pack_esm2m.py: talk and salt at the first level of {src.name}, '
                     'all other fields left out; ' + str(getattr(f, 'history', '')))
        keep = []
        for name, v in f.variables.items():
            d = set(v.dimensions)
            if name in VARS:
                keep.append((name, 'field'))
            elif zdim in d and len(v.dimensions) <= 2 and not ({tdim, ydim, xdim} & d):
                keep.append((name, 'level'))                 # the level coordinate and its bounds: the first level
            elif not ({ydim, xdim, zdim} & d):
                keep.append((name, 'all'))                   # time, its bounds and other axes without space
            elif not ({tdim, zdim} & d) and len(v.dimensions) <= 2:
                keep.append((name, 'all'))                   # lat, lon and their bounds
        used = {dn for name, _ in keep for dn in f[name].dimensions}
        for dn in f.dimensions:
            if dn in used:
                dd = f.dimensions[dn]
                g.createDimension(dn, None if dd.isunlimited() else (1 if dn == zdim else len(dd)))
        for name, how in keep:
            v = f[name]
            v.set_auto_maskandscale(False)
            fill = v.getncattr('_FillValue') if '_FillValue' in v.ncattrs() else None
            if how == 'field':
                chunks = (1, 1, len(f.dimensions[ydim]), len(f.dimensions[xdim]))
                w = g.createVariable(name, v.datatype, v.dimensions, zlib=True, complevel=4, shuffle=True,
                                     chunksizes=chunks, fill_value=fill)
            else:
                w = g.createVariable(name, v.datatype, v.dimensions, fill_value=fill)
            w.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != '_FillValue'})
            w.set_auto_maskandscale(False)
            if how == 'field':
                w[:] = v[:, 0:1]
            elif how == 'level':
                w[:] = v[0:1] if v.dimensions[0] == zdim else v[:, 0:1]
            elif v.ndim == 0:
                w.assignValue(v.getValue())
            else:
                w[:] = v[:]
    # check: the fields at the first level and the coordinates, bit for bit
    with netCDF4.Dataset(src) as f, netCDF4.Dataset(part) as g:
        for name in list(VARS) + [n for n in ('lat', 'lon', 'time') if n in f.variables]:
            a, b = f[name], g[name]
            a.set_auto_maskandscale(False)
            b.set_auto_maskandscale(False)
            x = np.asarray(a[:, 0] if name in VARS else a[:])
            y = np.asarray(b[:, 0] if name in VARS else b[:])
            assert x.shape == y.shape and x.tobytes() == y.tobytes(), f'{dst.name}: {name} differs from the source'
    part.rename(dst)
    return f'{dst}: {src.stat().st_size / 2**20:.0f} -> {dst.stat().st_size / 2**20:.0f} MiB, {(time.time() - t0):.0f} s'


def work(args):
    src, dst = args
    try:
        return copy_file(Path(src), Path(dst))
    except Exception as e:                            # report and go on with the other files
        return f'{Path(src)}: FAILED {type(e).__name__}: {e}'


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=str(fp.ESM2M_DIR), help='the ESM2M folder of the study (only read)')
    ap.add_argument('--dst', required=True, help='a new folder for the copy')
    ap.add_argument('--jobs', type=int, default=3, help='files in parallel (at most 3 on the shared server)')
    ap.add_argument('--no-tar', action='store_true', help='only the copy and MANIFEST.md5, no tars')
    a = ap.parse_args()
    src, dst = Path(a.src).resolve(), Path(a.dst).resolve()
    assert src != dst and src not in dst.parents, 'the copy must not go into the source folder'
    rel = file_list()
    missing = [str(r) for r in rel if not (src / r).is_file()]
    if missing:
        raise SystemExit(f'{len(missing)} of {len(rel)} source files missing, e.g. {missing[:3]}')
    todo = [(str(src / r), str(dst / r)) for r in rel if not (dst / r).exists()]
    print(f'{len(rel)} files ({len(rel) - len(todo)} done already, {len(todo)} to copy, {a.jobs} at a time): '
          f'{", ".join(VARS)} at the first level, from {src} to {dst}', flush=True)
    failed = 0
    if todo:
        with mp.get_context('spawn').Pool(min(a.jobs, 3, len(todo))) as pool:
            for line in pool.imap_unordered(work, todo):
                failed += 'FAILED' in line
                print(time.strftime('%Y-%m-%d %H:%M'), line, flush=True)
    done = [dst / r for r in rel if (dst / r).exists()]
    if failed or len(done) != len(rel):
        raise SystemExit(f'{failed} failed, {len(done)} of {len(rel)} files complete: run again')
    tot = sum(p.stat().st_size for p in done)
    print(f'all {len(rel)} files complete: {tot / 2**30:.1f} GiB ({tot / 1e9:.1f} GB)', flush=True)
    man = dst / 'MANIFEST.md5'
    man.write_text(''.join(f'{md5(dst / r)}  {r}\n' for r in rel))
    print(f'written {man}', flush=True)
    if a.no_tar:
        return
    out = dst.parent / (dst.name + '_zenodo')
    out.mkdir(exist_ok=True)
    for top in ['pi_control'] + list(EXPS):
        t = out / f'{dst.name}_{top}.tar'
        with tarfile.open(t.with_name(t.name + '.part'), 'w') as tf:
            for r in rel:
                if r.parts[0] == top:
                    tf.add(dst / r, arcname=str(Path(dst.name) / r))
        t.with_name(t.name + '.part').rename(t)
        print(f'{t}: {t.stat().st_size / 1e9:.1f} GB, md5 {md5(t)}', flush=True)
    (out / 'MANIFEST.md5').write_bytes(man.read_bytes())
    print(f'for Zenodo: the {len(EXPS) + 1} tars and MANIFEST.md5 in {out}', flush=True)


if __name__ == '__main__':
    main()
