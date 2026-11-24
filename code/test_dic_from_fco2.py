#!/usr/bin/env python
"""
test_dic_from_fco2.py  (does it matter that we take the DIC of the fCO2-products instead of their fCO2?)
=====================
The chain takes DIC and alkalinity (with T and S) of every fCO2-product and computes Omega_arag with the constants of
this study. The products derived their DIC from their mapped fCO2 and alkalinity with their own constants, so with our
constants this DIC does not give back their fCO2 exactly (UExP-FNN-U and CMEMS-LSCE-FFNN use the same K1, K2 and boron
as this study, OceanSODA-ETHZ and JMA-MLR others). The alternative: DIC solved with our constants so that the fCO2 of
the product is matched (fixed fCO2 and alkalinity), then everything else as before. Here, for each fCO2-product on the
common area of the paper, the decline 1750 -> 2024 with both:
    delta(x)    DIC(fixed fCO2) - DIC(product), per month and cell (mocsy, Newton with the Revelle factor until
                |fCO2 - target| < 1e-3 uatm, T floored at the freezing point as everywhere), averaged over the months
                of the bias window -> the change of the observed window state, and of 2024 -> the 2024 state
    2024        Omega_arag of the 2024 annual-mean state with DIC + delta(2024) minus with DIC, area mean on the common
                area, added to the paper's 2024 value of the product ('now'; the trend value moves alike)
    1750        every ESM: its bias-adjusted state at the piControl CO2 (xco2 file, before the CO2 of 1750) plus
                delta(window), DIC then solved to the CO2 of 1750 (278.3 ppm, calc_pi_xco2.target_states, as the chain),
                Omega_arag, area mean on the common area; the same without delta for the reference
    CUC         the observed CUC changes with the window DIC: <DIC/R> / pCO2 of the window-mean state with DIC + delta
                minus with DIC (mocsy, area mean on the common area), added to the observed CUC of the paper
    constraint  the regression of the 18 1750 values on dSST and CUC (as constraint_refs) at the observed predictors,
                before and after, and the decline 100 (1 - Omega_2024 / Omega_1750)
Printed per product: the fCO2 variable used, the mean and spread of delta, the change of the 2024 state, of the
constrained 1750 state, of the observed CUC and of the decline (percentage points).
Usage (code/):  OA_RUN=u74 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
                python -u test_dic_from_fco2.py 2>&1 | tee ../logs/u74/test_dic_from_fco2.log
"""
import glob
import pickle

import netCDF4
import numpy as np

import oa_config
import calc_bias_adjusted as cb
import calc_omega_arag_obs as co
import calc_pi_xco2 as px
from calc_omega_arag_products import Reader
from make_jma_ts import year_month

PRODUCTS = ('OceanSODA', 'UExP', 'JMA', 'CMEMS')
EVAL_YEAR = 2024
X1750 = 278.3
FCO2_TOL, N_ITER = 1e-3, 30
LAB = 'dT + CUC'


def onto(lat, lon, la, lo, a):
    jy = np.abs(lat[:, None] - la[None, :]).argmin(1)
    ix = np.abs(((lon[:, None] - lo[None, :]) + 180) % 360 - 180).argmin(1)
    return a[..., jy, :][..., ix]


def fco2_name(f):
    """the fCO2 variable of the product (pCO2 if no fCO2 is given): (name, mocsy index)"""
    names = {v.lower(): v for v in f.variables}
    bad = ('unc', 'err', 'std', 'atm', 'flux', 'fg', 'delta', 'sigma')
    for key, idx in (('fco2', 2), ('pco2', 1)):
        cand = [names[n] for n in names if key in n and not any(b in n for b in bad)]
        if cand:
            return sorted(cand, key=len)[0], idx
    raise SystemExit(f'no fCO2 or pCO2 variable in {f.filepath()}: {sorted(f.variables)}')


def dic_at_fco2(t, s, dic, alk, fco2, sil, phos, lat, idx):
    """DIC (mol/kg) that gives the target fCO2 (uatm) with our constants, Newton with the Revelle factor"""
    d = dic.copy()
    for _ in range(N_ITER):
        r = px.mvars_all(t, s, d, alk, sil, phos, lat)
        f, rv = np.asarray(r[idx], 'f8'), np.asarray(r[8], 'f8')
        with np.errstate(divide='ignore', invalid='ignore'):
            d = d + (fco2 - f) * d / (rv * f)
        good = np.isfinite(f)
        if not good.any() or np.nanmax(np.abs(f - fco2)[good]) < FCO2_TOL:
            break
    r = px.mvars_all(t, s, d, alk, sil, phos, lat)
    f = np.asarray(r[idx], 'f8')
    return np.where(np.abs(f - fco2) < 10 * FCO2_TOL, d, np.nan)


def annual_states(ref):
    """monthly T, S, DIC, TA and fCO2 of the product -> annual-mean states of the window years and of EVAL_YEAR, with
    DIC of the product and DIC at fixed fCO2 (only cells with all 12 months valid in both)"""
    cfg = cb.REFERENCES[ref]
    main = Reader(cfg['file'], cfg['order'])
    ts = Reader(cfg['ts_file'], 'time,lat,lon') if cfg.get('ts_file') else main
    fname, idx = fco2_name(main.f)
    print(f'  {ref}: {"fCO2" if idx == 2 else "pCO2"} from {fname} ({getattr(main.f[fname], "units", "?")})', flush=True)
    lat, lon = np.asarray(main.f['lat'][:], 'f8'), np.asarray(main.f['lon'][:], 'f8')
    ym = year_month(main.f['time'])
    si, po4 = co.load_nutrients(lat, lon)
    lat2 = np.broadcast_to(lat[:, None], (lat.size, lon.size))
    v = cfg['vars']
    w0, w1 = cfg['window']
    out = {}
    for y in list(range(w0, w1 + 1)) + [EVAL_YEAR]:
        ii = [k for k, (yy, mm) in enumerate(ym) if yy == y]
        assert len(ii) == 12 and ii == list(range(ii[0], ii[0] + 12)), f'{ref} {y}: {len(ii)} months'
        i0, i1 = ii[0], ii[-1] + 1
        t = ts(v['t'], i0, i1) - (273.15 if cfg['t_kelvin'] else 0.0)
        s = ts(v['s'], i0, i1)
        dic = main(v['dic'], i0, i1) * 1e-6
        alk = main(v['ta'], i0, i1) * 1e-6
        fc = main(fname, i0, i1)
        ok = (np.isfinite(t) & np.isfinite(s) & np.isfinite(dic) & np.isfinite(alk) & np.isfinite(fc)
              & (s > 0) & (dic > 0) & (alk > 0) & (fc > 0) & np.isfinite(si) & np.isfinite(po4))
        d2 = np.full(t.shape, np.nan)
        if ok.any():
            b = lambda a: np.broadcast_to(a, t.shape)[ok]
            d2[ok] = dic_at_fco2(t[ok], s[ok], dic[ok], alk[ok], fc[ok], b(si * 1e-6), b(po4 * 1e-6), b(lat2), idx)
        full = np.all(np.isfinite(d2), 0) & np.all(ok, 0)
        st = {k: np.where(full, a.mean(0), np.nan) for k, a in (('t', t), ('s', s), ('dic', dic), ('alk', alk), ('dic2', d2))}
        out[y] = st
    main.close()
    if ts is not main:
        ts.close()
    return out, lat, lon, si, po4, lat2


def chem(st, dic, si, po4, lat2):
    """Omega_arag and DIC / Revelle of an annual-mean state (fields in mol/kg)"""
    ok = np.isfinite(st['t']) & np.isfinite(dic) & np.isfinite(si) & np.isfinite(po4)
    om, cu = np.full(ok.shape, np.nan), np.full(ok.shape, np.nan)
    r = px.mvars_all(st['t'][ok], st['s'][ok], dic[ok], st['alk'][ok], si[ok] * 1e-6, po4[ok] * 1e-6, lat2[ok])
    o_, rv = np.asarray(r[6], 'f8'), np.asarray(r[8], 'f8')
    good = px.valid_omega(o_, np.asarray(r[0], 'f8'))
    om[ok], cu[ok] = np.where(good, o_, np.nan), np.where(good, dic[ok] * 1e6 / rv, np.nan)
    return om, cu


def fit(X, y, x0):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return x0 @ beta


def main():
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    print(f'weights {oa_config.WEIGHTS}; constants {oa_config.CONSTANTS}')
    res = {}
    for ref in PRODUCTS:
        print(f'\n=== {ref}', flush=True)
        r = pk['results_common'][ref]
        lat, lon, mask = np.asarray(r['lat'], 'f8'), np.asarray(r['lon'], 'f8'), np.asarray(r['mask'], bool)
        w = oa_config.area_weights(lat, lon)
        A, la, lo, si, po4, lat2 = annual_states(ref)
        on = lambda a: onto(lat, lon, la, lo, a)
        w0, w1 = cb.REFERENCES[ref]['window']
        win = [A[y] for y in range(w0, w1 + 1)]
        W_ = {k: np.mean([s_[k] for s_ in win], 0) for k in ('t', 's', 'dic', 'alk', 'dic2')}   # window-mean state
        dwin = on(W_['dic2'] - W_['dic']) * 1e6                                            # umol/kg
        S24 = A[EVAL_YEAR]
        o24a, _ = chem(S24, S24['dic'], si, po4, lat2)
        o24b, _ = chem(S24, S24['dic2'], si, po4, lat2)
        _, ca = chem(W_, W_['dic'], si, po4, lat2)
        _, cb_ = chem(W_, W_['dic2'], si, po4, lat2)
        d24, dcu = on(o24b - o24a), on(cb_ - ca)
        # 1750: every ESM, its adjusted state at the piControl CO2 with and without delta, DIC to 278.3 ppm
        adj = oa_config.DATA / ('omega_arag_biasadj' + ('' if ref == 'OceanSODA' else f'_{ref}'))
        si1, po41 = co.load_nutrients(lat, lon)
        la1 = np.broadcast_to(lat[:, None], mask.shape)
        y0, y1, cells = [], [], mask & np.isfinite(dwin) & np.isfinite(d24) & np.isfinite(dcu)
        for m in r['names']:
            fx = glob.glob(f'{adj}/piControl/omega_arag_biasadj_xco2_Oyr_{m}_piControl_*.nc')
            assert len(fx) == 1, f'{ref} {m}: {len(fx)} xco2 files'
            with netCDF4.Dataset(fx[0]) as f:
                g = {v: np.ma.filled(f[f'{v}_mean'][:].astype('f8'), np.nan) for v in px.VARS}
            pair = []
            for add in (0.0, np.nan_to_num(dwin)):
                fl = {v: (g[v] + (add if v == 'dissic' else 0.0))[None] for v in px.VARS}
                _, ind = next(px.target_states(fl, si1, po41, la1, {'1750': X1750}, [0.0]))
                pair.append(ind['omega_arag'][0])
            cells &= np.isfinite(pair[0]) & np.isfinite(pair[1])
            y0.append(pair[0]); y1.append(pair[1])
        am = lambda a: float(np.sum(a[cells] * w[cells]) / np.sum(w[cells]))
        Y0, Y1 = np.array([am(a) for a in y0]), np.array([am(a) for a in y1])
        dT, CUC = np.asarray(r['dT'], 'f8'), np.asarray(r['CUC'], 'f8')
        X = np.column_stack([np.ones(Y0.size), dT, CUC])
        cuc_o, dcu_m = float(r['CUC_obs']), am(dcu) / float(oa_config.WINDOW_PCO2[ref])
        x0 = np.array([1.0, float(r['dT_obs'][0]), cuc_o])
        x1 = np.array([1.0, float(r['dT_obs'][0]), cuc_o + dcu_m])
        c0, c1 = fit(X, Y0, x0), fit(X, Y1, x1)
        n0 = float(r['now']); n1 = n0 + am(d24)
        dec0, dec1 = 100 * (1 - n0 / c0), 100 * (1 - n1 / c1)
        frac = 100 * cells.sum() / mask.sum()
        print(f'  delta DIC (fixed fCO2 - product), window mean on the common area: {am(dwin):+.2f} umol/kg '
              f'(5-95 %: {np.nanpercentile(dwin[cells], 5):+.2f} to {np.nanpercentile(dwin[cells], 95):+.2f}); '
              f'cells {frac:.0f} % of the common area')
        print(f'  2024: Omega_arag {n0:.4f} -> {n1:.4f} ({am(d24):+.4f}); observed CUC {cuc_o:.5f} -> '
              f'{cuc_o + dcu_m:.5f} ({dcu_m:+.5f})')
        print(f'  1750: 18 ESMs {Y0.mean():.4f} -> {Y1.mean():.4f} ({(Y1 - Y0).mean():+.4f}, sd {(Y1 - Y0).std(ddof=1):.4f}); '
              f'constrained {c0:.4f} -> {c1:.4f} ({c1 - c0:+.4f}; check: paper {float(r["constrained"][LAB][0]):.4f})')
        print(f'  decline 1750 -> 2024: {dec0:.3f} % -> {dec1:.3f} % ({dec1 - dec0:+.3f} percentage points)', flush=True)
        res[ref] = (dec0, dec1, c1 - c0, n1 - n0)
    v = np.array([res[q][1] - res[q][0] for q in PRODUCTS])
    d0 = np.mean([res[q][0] for q in PRODUCTS]); d1 = np.mean([res[q][1] for q in PRODUCTS])
    print(f'\n=== mean of the four: decline {d0:.3f} % -> {d1:.3f} % ({d1 - d0:+.3f} points; per product '
          + ', '.join(f'{q} {x:+.3f}' for q, x in zip(PRODUCTS, v)) + ')')


if __name__ == '__main__':
    main()
