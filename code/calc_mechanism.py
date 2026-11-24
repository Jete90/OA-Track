#!/usr/bin/env python
"""
calc_mechanism.py
=================
Mean surface states for testing why the bias-adjusted preindustrial Omega_arag
depends on the raw model bias.

With the bias correction, the adjusted piControl state is
    OceanSODA(1983-2014) - [model(1983-2014) - model(piControl)]
so its spread across models comes only from each model's historical change.
This script saves, on the OceanSODA grid (umol/kg, degC, psu):

    per model:  T, S, DIC, ALK time means for historical 1983-2014 and piControl,
                raw Revelle factor and raw Omega_arag of the 1983-2014 mean state
    once:       OceanSODA T, S, DIC, ALK annual means averaged over 1983-2014

Output: data/mechanism/state_<model>_<member>.nc, data/mechanism/state_OceanSODA.nc

Usage:
    python calc_mechanism.py --jobs 4
"""
from __future__ import annotations

import argparse
import oa_config   # run: data folder, mocsy constants, GFDL density (OA_RUN)
import traceback
import warnings
from multiprocessing import Pool

import netCDF4
import numpy as np

import calc_bias_adjusted as cb
import calc_omega_arag_obs as co
import mocsy
MOCSY_OPTB, MOCSY_OPTK1K2 = oa_config.OPTB, oa_config.OPTK1K2   # total boron, K1/K2 of the run (oa_config.py)

OUT = oa_config.DATA / 'mechanism'          # OceanSODA; other references: data/mechanism_<reference>
VARS = cb.VARS
WINDOW = oa_config.WINDOWS['OceanSODA']     # set to the reference's window in main()


def grid():
    with netCDF4.Dataset(cb.REF['omega']) as f:
        return np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')


def write(path, lat, lon, fields, attrs=None):
    g = cb.new_nc(path, lat, lon, attrs or {})
    for name, a in fields.items():
        x = g.createVariable(name, 'f4', ('lat', 'lon'), zlib=True, fill_value=cb.FILL)
        x[:] = np.ma.masked_invalid(a)
    g.close()


def time_mean(run, i0, i1, jy, ix, lon2, lat2):
    """Means of T, S, DIC, ALK over steps i0:i1 (NaN unless valid in every step)."""
    acc = {v: None for v in VARS}
    n = 0
    for a in range(i0, i1, cb.CHUNK):
        b = min(a + cb.CHUNK, i1)
        f = run.read(a, b, jy, ix, lon2, lat2)
        for v in VARS:
            s = f[v].sum(axis=0)
            acc[v] = s if acc[v] is None else acc[v] + s
        n += b - a
    return {v: acc[v] / n for v in VARS}


def revelle_omega(st, si, po4, lat2):
    t, s, dic, alk = (st[v] for v in VARS)
    ok = np.isfinite(t) & np.isfinite(s) & np.isfinite(dic) & np.isfinite(alk) \
        & np.isfinite(si) & np.isfinite(po4)
    n = int(ok.sum())
    r = mocsy.mvars(temp=co.cm.floor_temperature(t[ok], s[ok]), sal=s[ok], alk=alk[ok] * 1e-6, dic=dic[ok] * 1e-6,
                    sil=si[ok] * 1e-6, phos=po4[ok] * 1e-6, patm=np.ones(n),
                    depth=np.zeros(n), lat=lat2[ok], optcon='mol/kg', optt='Tpot', optp='m', optb=MOCSY_OPTB, optk1k2=MOCSY_OPTK1K2)
    R, om = np.full(t.shape, np.nan), np.full(t.shape, np.nan)
    good = co.cm.valid_carbonate(r[0], r[6])
    R[ok], om[ok] = np.where(good, np.asarray(r[8], float), np.nan), np.where(good, np.asarray(r[6], float), np.nan)
    return R, om


def process(model, member):
    lat, lon = grid()
    lon2, lat2 = np.meshgrid(lon, lat)
    si, po4 = co.load_nutrients(lat, lon)
    fields = {}
    for exp in ('historical', 'piControl'):
        run = cb.ModelRun(model, exp, member)
        jy = co.nearest_index(run.lat, lat)
        ix = co.nearest_index(run.lon, lon, periodic=True)
        if exp == 'historical':
            i0 = int(np.where(run.years == WINDOW[0])[0][0])
            i1 = int(np.where(run.years == WINDOW[1])[0][-1]) + 1
        else:
            i0, i1 = 0, run.nt
        st = time_mean(run, i0, i1, jy, ix, lon2, lat2)
        run.close()
        tag = 'hist' if exp == 'historical' else 'pi'
        for v in VARS:
            fields[f'{v}_{tag}'] = st[v]
        if exp == 'historical':
            fields['revelle_raw_hist'], fields['omega_raw_hist'] = revelle_omega(st, si, po4, lat2)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / f'state_{model}_{member}.nc', lat, lon, fields,
          dict(window=f'{WINDOW[0]}-{WINDOW[1]}', units='degC, psu, umol kg-1'))
    print(f'[done] {model} {member}', flush=True)


def obs_state():
    lat, lon = grid()
    obs = cb.obs_fields(*WINDOW, step=1)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / f'state_{cb.REF_NAME}.nc', lat, lon, {v: obs[v].mean(axis=0) for v in VARS},
          dict(window=f'{WINDOW[0]}-{WINDOW[1]}', note='mean of annual means'))
    print(f'[done] {cb.REF_NAME}', flush=True)


def run_one(model, member):
    try:
        process(model, member)
    except Exception:
        print(f'[FAILED] {model} {member}\n{traceback.format_exc()}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', default='OceanSODA', choices=list(cb.REFERENCES))
    ap.add_argument('--models', nargs='+')
    ap.add_argument('--jobs', type=int, default=1)
    a = ap.parse_args()
    global OUT
    cb.set_reference(a.reference)
    global WINDOW
    WINDOW = cb.REF.get('window', WINDOW)
    print(f'window {WINDOW[0]}-{WINDOW[1]}', flush=True)
    if a.reference != 'OceanSODA':
        OUT = oa_config.DATA / f'mechanism_{a.reference}'
    obs_state()
    runs = []
    for p in sorted((cb.OUT_DIR / 'piControl').glob(f'omega_arag_biasadj_Oyr_*_piControl_*_{cb.REF_NAME}grid.nc')):
        model, rest = p.name[len('omega_arag_biasadj_Oyr_'):].split('_piControl_')
        if not a.models or model in a.models:
            runs.append((model, rest.split('_')[0]))
    print(f'{len(runs)} runs', flush=True)
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            pool.starmap(run_one, runs)
    else:
        for r in runs:
            run_one(*r)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
