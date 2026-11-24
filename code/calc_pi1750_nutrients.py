#!/usr/bin/env python
"""
calc_pi1750_nutrients.py
========================
The 1750 Omega_arag of every model, exactly as calc_pi_xco2.py computes it (every month of the drift-corrected,
bias-adjusted piControl; DIC lowered until surface pCO2 has changed by (278.3 - 284.317) * (1 - pH2O); Omega_arag of
every month, averaged over all months), but with another nutrient climatology (silicate and phosphate) in every
mocsy call. Only the 1750 target, only Omega_arag, only the cells of a given constraint mask, so it is much faster
than calc_pi_xco2.py. Newton iteration warm-started from the time-mean DIC change of calc_pi_xco2.py
(ddic_1750_mean), iterated to the same tolerance (max |pCO2 - target| < 1e-4 uatm).

Nutrients:
    --nut woa23     World Ocean Atlas 2023 (data/<run>/obs/woa23_{si,po4}_surface_annual.nc, make_woa_nutrients.py)
    --nut glodap    GLODAPv2 as in the chain: must reproduce omega_arag_1750_mean of calc_pi_xco2.py (check)

Output: data/<run>/omega_arag_biasadj[_<reference>]/piControl/pi1750_<nut>_Oyr_<model>_piControl_<member>_<ref>grid.nc
    omega_arag_1750_mean   time mean of the monthly Omega_arag at 278.3 ppm (NaN where any month is missing)
    omega_arag_1750_chain  the same field of calc_pi_xco2.py (GLODAP), for comparison
Usage (code/):  OA_RUN=u74 python -u calc_pi1750_nutrients.py --reference OceanSODA --nut woa23 --jobs 6
                OA_RUN=u74 python -u calc_pi1750_nutrients.py --reference OceanSODA --nut glodap --models ACCESS-ESM1-5
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import datetime
import multiprocessing as mp
import pickle
import time
import traceback
import warnings

import netCDF4
import numpy as np
from pathlib import Path

import oa_config
import calc_bias_adjusted as cb
import calc_omega_arag_obs as co
import calc_pi_xco2 as cpx
from drift_tools import Drift
from nutrient_sensitivity import read_alt

X1750 = cpx.TARGETS['1750']
XMOD = cpx.XCO2_MODEL


def nutrients(nut, lat, lon):
    if nut == 'glodap':
        si, po4 = co.load_nutrients(lat, lon)
        return np.where(si < 0, 0.0, si), np.where(po4 < 0, 0.0, po4)
    d = Path(oa_config.DATA) / 'obs'
    return (read_alt((str(d / 'woa23_si_surface_annual.nc'), 'silicate'), lat, lon),
            read_alt((str(d / 'woa23_po4_surface_annual.nc'), 'phosphate'), lat, lon))


def omega_1750(fa, sel, si, po4, lat2, ddic, resid, iters):
    """Omega_arag at 1750 for the months of fa {var: (n, ny, nx)} on the cells sel (ny, nx); (n, ny, nx), NaN
    elsewhere. As calc_pi_xco2.target_states for one target, warm-started with the time-mean DIC change ddic."""
    t, s, dic, alk = (fa[v] for v in cb.VARS)
    n = t.shape[0]
    selb = np.broadcast_to(sel, t.shape)
    ok = (selb & np.isfinite(t) & np.isfinite(s) & np.isfinite(dic) & np.isfinite(alk)
          & (s > 0) & (dic > 0) & (alk > 0))
    out_om = np.full(t.shape, np.nan)
    if not ok.any():
        return out_om
    tt, ss = t[ok], s[ok]
    a, d0 = alk[ok] * 1e-6, dic[ok] * 1e-6
    sil = np.broadcast_to(si * 1e-6, t.shape)[ok]
    phos = np.broadcast_to(po4 * 1e-6, t.shape)[ok]
    la = np.broadcast_to(lat2, t.shape)[ok]
    out0 = cpx.mvars_all(tt, ss, d0, a, sil, phos, la)
    p0 = np.asarray(out0[1], float)
    target = p0 + (X1750 - XMOD) * (1.0 - cpx.water_vapour(tt, ss))
    d = d0 + np.broadcast_to(np.nan_to_num(ddic) * 1e-6, t.shape)[ok]          # warm start
    out = cpx.mvars_all(tt, ss, d, a, sil, phos, la)
    p, r = np.asarray(out[1], float), np.asarray(out[8], float)
    it = 0
    for it in range(1, cpx.N_ITER_MAX + 1):
        err = np.abs(p - target)[cpx.valid_omega(np.asarray(out[6], float)) & np.isfinite(p)]
        if err.size == 0 or float(err.max()) < cpx.PCO2_TOL:
            break
        with np.errstate(divide='ignore', invalid='ignore'):
            d = d + (target - p) * d / (r * p)
        out = cpx.mvars_all(tt, ss, d, a, sil, phos, la)
        p, r = np.asarray(out[1], float), np.asarray(out[8], float)
    iters[it] = iters.get(it, 0) + 1
    om = np.asarray(out[6], float)
    good = cpx.valid_omega(om) & np.isfinite(p)
    resid[0] = max(resid[0], float(np.nanmax(np.abs(p - target)[good])) if good.any() else 0.0)
    out_om[ok] = np.where(good, om, np.nan)
    return out_om


def process(model, member, nut, sel_mask):
    tag = f'{model} {member}'
    t0 = time.time()
    with netCDF4.Dataset(cpx.adj_path(model, member)) as f:
        olat = np.asarray(f['lat'][:], 'f8')
        olon = np.asarray(f['lon'][:], 'f8')
    with netCDF4.Dataset(cpx.xco2_path(model, member)) as f:
        ddic = np.ma.filled(f['ddic_1750_mean'][:].astype('f8'), np.nan)
        chain = np.ma.filled(f['omega_arag_1750_mean'][:].astype('f8'), np.nan)
    with netCDF4.Dataset(cb.bias_path(model, member)) as f:
        bias = {v: np.ma.filled(f[f'bias_{v}'][:].astype('f8'), np.nan) for v in cb.VARS}
        y0, y1 = (int(x) for x in f.overlap.split('-'))
    lon2, lat2 = np.meshgrid(olon, olat)
    si, po4 = nutrients(nut, olat, olon)
    sel = sel_mask & np.isfinite(si) & np.isfinite(po4)
    drift = Drift(model, member)
    run = cb.ModelRun(model, 'piControl', member)
    jy = co.nearest_index(run.lat, olat)
    ix = co.nearest_index(run.lon, olon, periodic=True)
    tsum = np.zeros(sel.shape)
    resid, iters = [0.0], {}
    try:
        for i0 in range(0, run.nt, cb.CHUNK):
            i1 = min(i0 + cb.CHUNK, run.nt)
            _, _, fa = cb.adjusted_fields(run, i0, i1, bias, drift, (y0, y1), jy, ix, lon2, lat2)
            tsum += omega_1750(fa, sel, si, po4, lat2, ddic, resid, iters).sum(axis=0)    # NaN propagates
    finally:
        run.close()
    field = np.where(sel, tsum / float(run.nt), np.nan)
    path = cb.OUT_DIR / 'piControl' / f'pi1750_{nut}_Oyr_{model}_piControl_{member}_{cb.REF_NAME}grid.nc'
    g = cb.new_nc(path, olat, olon, dict(
        title=f'1750 Omega_arag of the drift-corrected, bias-adjusted piControl with {nut} nutrients, {model}',
        method=('as calc_pi_xco2.py (target 1750 only, Omega_arag only, cells of the constraint mask): every month of '
                f'the adjusted piControl, DIC lowered until pCO2 changes by ({X1750} - {XMOD}) * (1 - pH2O), Newton '
                f'iteration warm-started from ddic_1750_mean, tolerance {cpx.PCO2_TOL} uatm; mean of the monthly '
                f'Omega_arag; Si and PO4: {nut}; ' + oa_config.CONSTANTS),
        history=f'{datetime.datetime.now():%Y-%m-%d %H:%M} calc_pi1750_nutrients.py, {oa_config.VERSION}'))
    for name, a, ln in (('omega_arag_1750_mean', field, f'time mean of the monthly Omega_arag at 1750, {nut}'),
                        ('omega_arag_1750_chain', chain, 'omega_arag_1750_mean of calc_pi_xco2.py (GLODAPv2)')):
        x = g.createVariable(name, 'f4', ('lat', 'lon'), zlib=True, fill_value=cb.FILL)
        x.long_name, x.units = ln, '1'
        x[:] = np.ma.masked_invalid(a)
    g.close()
    both = sel & np.isfinite(field) & np.isfinite(chain)
    w = oa_config.area_weights(olat, olon) * both
    gm = lambda a: float(np.sum(np.where(both, a, 0) * w) / w.sum())
    print(f'{tag}: {nut} {gm(field):.5f}, chain (GLODAP) {gm(chain):.5f}, difference {gm(field) - gm(chain):+.5f} '
          f'on {int(both.sum())} cells; max |pCO2 - target| {resid[0]:.1e}; Newton iterations per chunk '
          + ', '.join(f'{k}: {v}' for k, v in sorted(iters.items())) + f'; {time.time() - t0:.0f} s', flush=True)


def run_one(model, member, nut, sel_mask):
    try:
        process(model, member, nut, sel_mask)
    except Exception:
        print(f'[FAILED] {model} {member}\n{traceback.format_exc()}', flush=True)
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', default='OceanSODA', choices=list(cb.REFERENCES))
    ap.add_argument('--nut', default='woa23', choices=['woa23', 'glodap'])
    ap.add_argument('--models', nargs='+')
    ap.add_argument('--jobs', type=int, default=1)
    a = ap.parse_args()
    cb.set_reference(a.reference)
    with open(Path(oa_config.DATA) / 'constraint_results.pkl', 'rb') as f:
        R = pickle.load(f)['results'][a.reference]
    sel_mask = np.asarray(R['mask']).astype(bool)          # the constraint mask of the reference (native grid)
    names = list(R['names'])
    runs = [(m, r) for m, r in cpx.find_runs(a.models) if m in names]
    print(f'{a.reference}: {len(runs)} models, {a.nut}, {int(sel_mask.sum())} cells', flush=True)
    todo, running, status = list(runs), {}, {}
    ctx = mp.get_context('fork')
    while todo or running:
        while todo and len(running) < max(a.jobs, 1):
            m, r = todo.pop(0)
            p = ctx.Process(target=run_one, args=(m, r, a.nut, sel_mask))
            p.start()
            running[m] = p
        time.sleep(5)
        for m, p in list(running.items()):
            if p.exitcode is not None:
                p.join()
                status[m] = 'done' if p.exitcode == 0 else (f'KILLED (signal {-p.exitcode})' if p.exitcode < 0
                                                            else 'FAILED')
                del running[m]
    print('\nsummary:')
    for m, st in status.items():
        print(f'  {st:20s} {m}')


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
