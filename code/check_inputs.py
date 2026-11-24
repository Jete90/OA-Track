#!/usr/bin/env python
"""
check_inputs.py
===============
The input checks of the code review (code_review_2026-11-05.md, section C), in one run. Nothing is changed; every
item ends with OK, NOTE (to read) or PROBLEM.

 C1  which files the chain reads: data/<run>/raw/<exp> and where its links point
 C2  multiprocessing start method (several scripts rely on 'fork')
 C3  COBE-SST3 grid as assumed in constraint_refs.py (0.25 deg, south to north, from 0.125 E)
 C4  drift: every model's 251 parallel control years inside its piControl drift file
 C5  units of T, S, DIC, TA in the observation-based products (DIC, TA in umol/kg; T in degC unless t_kelvin)
 C6  JMA-MLR time axis: every year 1990-2024 has the months 1-12 once, in the product file and in its T/S file
 C7  model DIC and TA units and the density used per model (GFDL: 1035 kg m-3; others in-situ), with the
     variables' comments, to compare with the model documentation
 C8  which mocsy is imported
 C9  grid coordinates of models (r360x180) and products (x.5 degrees)

Usage (code/):  OA_RUN=u74 python -u check_inputs.py
"""
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

import netCDF4
import numpy as np

import oa_config
import calc_omega_arag as cm
import calc_bias_adjusted as cb
import calc_pi_xco2 as cpx
import drift_tools as dt
import make_jma_ts as mj

EXPS = ('historical', 'piControl', 'ssp126', 'ssp245', 'ssp370', 'ssp585')


def head(s):
    print(f'\n===== {s}', flush=True)


def c1():
    head('C1 inputs the chain reads')
    print(f'  RAW_DIR = {cm.RAW_DIR}  (real path {Path(cm.RAW_DIR).resolve()})')
    for exp in EXPS:
        d = Path(cm.RAW_DIR) / exp
        fs = sorted(d.glob('*.nc')) if d.exists() else []
        if not fs:
            print(f'  {exp:10s} no files')
            continue
        targets = {str(Path(os.path.realpath(f)).parent) for f in fs}
        links = sum(f.is_symlink() for f in fs)
        print(f'  {exp:10s} {len(fs)} files ({links} links) -> {", ".join(sorted(targets)[:3])}'
              + (' ...' if len(targets) > 3 else ''))
    other = Path(oa_config.BASE) / 'data' / 'raw'
    print(f'  data/raw (written by prepare_inputs.sh): {"exists" if other.exists() else "does not exist"}; the chain '
          f'reads {cm.RAW_DIR}: NOTE, prepare_inputs.sh does not change the inputs of a run')


def c2():
    head('C2 multiprocessing')
    print(f'  Python {sys.version.split()[0]}, default start method {mp.get_start_method(allow_none=False)}')
    code = Path(__file__).parent
    explicit = [p.name for p in code.glob('*.py') if "get_context('fork')" in p.read_text(errors='ignore')]
    pools = [p.name for p in code.glob('*.py') if 'mp.Pool(' in p.read_text(errors='ignore')
             or 'multiprocessing.Pool(' in p.read_text(errors='ignore')]
    print(f'  scripts that ask for fork explicitly: {", ".join(sorted(explicit)) or "none"}')
    print(f'  scripts with Pool on the default method: {", ".join(sorted(pools)) or "none"}')
    ok = mp.get_start_method() == 'fork'
    print('  OK (fork is the default here)' if ok else '  PROBLEM: default is not fork; the Pool scripts must ask for it')


def c3():
    head('C3 COBE-SST3 grid')
    txt = (Path(__file__).parent / 'constraint_refs.py').read_text()
    path = re.search(r"^COBE3 = '(.+)'", txt, re.M).group(1).format(1990)
    with netCDF4.Dataset(path) as f:
        v = f['sst']
        dims = v.dimensions
        coord = lambda d: np.asarray(f[d][:], 'f8') if d in f.variables else None
        lat_d = next(d for d in dims if d.lower().startswith(('lat', 'y')))
        lon_d = next(d for d in dims if d.lower().startswith(('lon', 'x')))
        lat, lon = coord(lat_d), coord(lon_d)
        shp = v.shape
        print(f'  dimensions of sst: {dims}; coordinate variables in the file: {list(f.variables)}')
    if lat is None or lon is None:
        print('  PROBLEM: no coordinate variables for the dimensions of sst; check the orientation by hand')
        return
    if dims[-2:] != (lat_d, lon_d):
        print(f'  PROBLEM: sst is not (..., lat, lon) but {dims}; the reshape in constraint_refs.py assumes it')
    good = (lat.size == 720 and lon.size == 1440 and lat[0] < lat[-1] and abs(lat[0] + 89.875) < 1e-3
            and abs(lon[0] - 0.125) < 1e-3)
    print(f'  {path}: sst {shp}, lat {lat[0]}..{lat[-1]}, lon {lon[0]}..{lon[-1]}')
    print('  OK (as assumed: reshaped to 1 deg, -89.5..89.5, 0.5..359.5)' if good else
          '  PROBLEM: the grid differs from the assumption in constraint_refs.py (COBE-SST3 block)')


def c4():
    head('C4 drift: parallel control years inside the drift files')
    for model, member in cpx.find_runs(None):
        try:
            d = dt.Drift(model, member)
        except FileNotFoundError as e:
            print(f'  {model:16s} PROBLEM: {e}')
            continue
        need = np.arange(d.by, d.by + 251)
        miss = need[~np.isin(need, d.years)]
        state = 'OK' if miss.size == 0 else ('NOTE' if miss.size <= 2 else 'PROBLEM')
        print(f'  {model:16s} branch {d.by}, control years {d.years[0]}-{d.years[-1]}, needed {need[0]}-{need[-1]}: '
              f'{state}' + (f' (missing {list(miss)}: nearest year used)' if miss.size else ''))


def c5():
    head('C5 units of the observation-based products')
    for ref, cfg in cb.REFERENCES.items():
        if ref == 'Jiang':
            continue
        files = {'main': cfg['file']}
        if cfg.get('ts_file'):
            files['T/S'] = cfg['ts_file']
        for key, var in cfg['vars'].items():
            src = files['T/S'] if (key in ('t', 's') and 'T/S' in files) else files['main']
            with netCDF4.Dataset(src) as f:
                if var not in f.variables:
                    print(f'  {ref:9s} {key:3s} {var}: not in {Path(src).name}')
                    continue
                v = f[var]
                u = getattr(v, 'units', '?')
                k = 0                                      # first time step with data (a product may start empty)
                a = np.ma.filled(v[0].astype('f8'), np.nan)
                while not np.isfinite(a).any() and k < min(v.shape[0] - 1, 60):
                    k += 1
                    a = np.ma.filled(v[k].astype('f8'), np.nan)
            med = float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan
            if key in ('dic', 'ta'):
                ok = 1500 < med < 2600
            elif key == 't':
                ok = (270 < med < 310) if cfg.get('t_kelvin') else (-3 < med < 35)
            else:
                ok = 25 < med < 40
            print(f'  {ref:9s} {key:3s} {var:14s} units "{u}", median of time step {k} (first with data) {med:.4g}: '
                  f'{"OK" if ok else "PROBLEM"}')


def c6():
    head('C6 JMA-MLR time axis')
    for path in (mj.JMA_FILE, mj.OUT):
        with netCDF4.Dataset(path) as f:
            ym = mj.year_month(f['time'])
        bad = []
        for y in range(1990, 2025):
            ms = sorted(m for yy, m in ym if yy == y)
            if ms != list(range(1, 13)):
                bad.append((y, ms))
        print(f'  {Path(path).name}: {len(ym)} steps, {ym[0]} to {ym[-1]}: '
              + ('OK' if not bad else f'PROBLEM in {[b[0] for b in bad][:5]}'))


def c7():
    head('C7 model DIC and TA units and density')
    for model, member in cpx.find_runs(None):
        out = []
        for var in ('dissic', 'talk'):
            p = cm.input_path(var, model, 'historical', member)
            if p is None:
                out.append(f'{var}: no file')
                continue
            with netCDF4.Dataset(p) as f:
                v = f[var]
                u = getattr(v, 'units', '?')
                com = getattr(v, 'comment', '')[:60]
            out.append(f'{var} "{u}"' + (f' ({com})' if com else ''))
        dens = (f'{cm.REF_DENSITY[model]:g} kg m-3 (reference density)' if model in cm.REF_DENSITY
                else 'in-situ density')
        print(f'  {model:16s} {dens}; ' + '; '.join(out))
    print('  "?" = no units attribute in the input file: calc_omega_arag.py then takes mol m-3 (the CMIP6 unit) and '
          'stops if the median after conversion is outside 1.5-2.8 (DIC) or 1.8-3.0 (TA) mol m-3, so a missing '
          'attribute cannot hide a factor 1000; a per-kg field read as per-m3 (2.5 %) would pass the range check')
    print('  NOTE: compare with the model documentation for HAMOCC (MPI-ESM1-2, NorESM2): tracers per volume or per kg')


def c8():
    head('C8 mocsy')
    import mocsy
    print(f'  {mocsy.__file__}')
    for k in ('__version__', 'version'):
        if hasattr(mocsy, k):
            print(f'  {k}: {getattr(mocsy, k)}')
    print(f'  mvars: {mocsy.mvars.__doc__.splitlines()[0] if mocsy.mvars.__doc__ else "?"}')
    print('  NOTE: must be the mocsy 2.4.0 build of the chain (oa_config.CONSTANTS: '
          f'{oa_config.CONSTANTS[:80]}...)')


def c9():
    head('C9 grid coordinates')
    model, member = cpx.find_runs(None)[0]
    p = cm.input_path('dissic', model, 'historical', member)
    with netCDF4.Dataset(p) as f:
        print(f'  model {model}: lat {f["lat"][0]}..{f["lat"][-1]}, lon {f["lon"][0]}..{f["lon"][-1]} ({f["lon"].size})')
    for ref, cfg in cb.REFERENCES.items():
        if ref == 'Jiang':
            continue
        with netCDF4.Dataset(cfg['file']) as f:
            la = next(f[k] for k in ('lat', 'latitude') if k in f.variables)
            lo = next(f[k] for k in ('lon', 'longitude') if k in f.variables)
            print(f'  {ref:9s}: lat {la[0]}..{la[-1]}, lon {lo[0]}..{lo[-1]} ({lo.size})')
    print('  NOTE: models on whole-degree longitudes, products on x.5: nearest neighbour picks the western cell, a '
          'half-cell shift removed by the cell-by-cell bias adjustment (README_v2, Step 2)')


if __name__ == '__main__':
    for fn in (c1, c2, c3, c4, c5, c6, c7, c8, c9):
        try:
            fn()
        except Exception as e:                          # one failing check does not stop the others
            print(f'  PROBLEM: the check itself failed: {type(e).__name__}: {e}', flush=True)
