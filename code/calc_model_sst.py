#!/usr/bin/env python
"""
calc_model_sst.py
=================
Top-level ocean temperature (thetao) of every model's historical run, averaged over the periods the
constraint needs, so that the model warming is defined exactly as the observed one:

    dT = mean(window) - mean(1850-1900)          (models and observations alike)

Periods: 1850-1900 and the window of every reference (oa_config.PERIODS: 1980-2014, 1983-2014, 1985-2014, 1990-2014,
2005-2014).
Fields stay on the model input grid (r360x180, the same grid as the other model input); a cell is NaN
unless it is valid in every time step of the period.

Output: data/model_sst/sst_<model>_<member>.nc  with variables sst_<y0>_<y1> (degC)

Usage:
    python calc_model_sst.py              # all models with bias-adjusted piControl output (OceanSODA)
    python calc_model_sst.py --models GFDL-ESM4
"""
from __future__ import annotations

import argparse
import oa_config   # run: data folder, mocsy constants, GFDL density (OA_RUN)
import traceback
import warnings
from pathlib import Path

import netCDF4
import numpy as np

import calc_omega_arag as cm

ROOT = cm.ROOT
OUT = oa_config.DATA / 'model_sst'
PERIODS = oa_config.PERIODS        # 1850-1900 and the window of every reference (oa_config.WINDOWS)
CHUNK = 120


def process(model, member):
    path = cm.input_path('thetao', model, 'historical', member)
    if path is None:
        raise FileNotFoundError(f'no thetao for {model} historical {member}')
    with netCDF4.Dataset(path) as f:
        v = f['thetao']
        t = f['time']
        years, _ = cm.year_month(t[:], t.units, getattr(t, 'calendar', 'standard'))
        lat = np.asarray(f['lat'][:], 'f8')
        lon = np.asarray(f['lon'][:], 'f8')
        first = cm.read(v, 0, 1)
        off = -273.15 if np.nanmedian(first) > 100 else 0.0
        out = {}
        for p0, p1 in PERIODS:
            idx = np.where((years >= p0) & (years <= p1))[0]
            if idx.size == 0:
                raise ValueError(f'{model}: no data for {p0}-{p1}')
            acc, cnt, bad = None, 0, None
            for a in range(idx[0], idx[-1] + 1, CHUNK):
                b = min(a + CHUNK, idx[-1] + 1)
                x = cm.read(v, a, b)
                x = x[:, 0] if x.ndim == 4 else x          # (time, lev, lat, lon) -> top level
                x = x + off
                nan = ~np.isfinite(x)
                bad = nan.any(axis=0) if bad is None else bad | nan.any(axis=0)
                s = np.where(nan, 0.0, x).sum(axis=0)
                acc = s if acc is None else acc + s
                cnt += b - a
            m = acc / cnt
            m[bad] = np.nan
            out[f'sst_{p0}_{p1}'] = m
    OUT.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(OUT / f'sst_{model}_{member}.nc', 'w') as g:
        g.description = 'historical top-level thetao, period means (degC); calc_model_sst.py'
        for n_, vals in (('lat', lat), ('lon', lon)):
            g.createDimension(n_, vals.size)
            g.createVariable(n_, 'f8', (n_,))[:] = vals
        for k, a in out.items():
            x = g.createVariable(k, 'f4', ('lat', 'lon'), zlib=True, fill_value=np.float32(1e20))
            x.units = 'degC'
            x[:] = np.ma.masked_invalid(a)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((lat.size, lon.size))    # native model grid, diagnostic: cell area (oa_config.WEIGHTS applies to the product grid)
    d = out['sst_1983_2014'] - out['sst_1850_1900']
    ok = np.isfinite(d)
    print(f'[done] {model} {member}: global dSST 1983-2014 minus 1850-1900 {np.sum(d[ok] * w[ok]) / w[ok].sum():.3f} K',
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+')
    a = ap.parse_args()
    runs = [(model, member) for model, exp, member in cm.find_runs(['historical'], a.models)]   # raw inputs
    print(f'{len(runs)} runs -> {OUT}', flush=True)
    for r in runs:
        try:
            process(*r)
        except Exception:
            print(f'[FAILED] {r}\n{traceback.format_exc()}', flush=True)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
