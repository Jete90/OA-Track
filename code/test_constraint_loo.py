#!/usr/bin/env python
"""
test_constraint_loo.py
======================
Perfect-model (leave-one-out) test of the emergent constraint with the real models, per carbonate system
indicator and per predictor set (dSST only, CUC only, both): does the constraint predict a model it has not seen
better than the multi-model mean of the others, and is its sigma right?

Each model in turn is the truth: the others are regressed on their predictors and the regression is evaluated
at the left-out model's own dSST and CUC (known exactly); the unconstrained estimate is the mean of the others.
Compared over all models:
    RMSE ratio   RMSE(constrained) / RMSE(unconstrained); < 1: the constraint helps
    coverage     fraction of left-out models within 1 sigma (0.68 expected): sigma of the constraint (residual
                 with leverage) and of the unconstrained mean (spread of the others x sqrt(1 + 1/(n-1)))
    R2, p        of the regression on all models (F-test)
    random       mean RMSE ratio with random predictors (the same number), i.e. what a constraint that works by
                 accident gives
Global: global means on the constraint mask of the reference (as the paper). Cells: the same per cell with the
local dSST and CUC (cell-by-cell constraint), area-weighted over the cells; also the fraction of the area where
the constraint helps.

Data: constraint_results.pkl (predictors per model and per cell, the Omega_arag of 1750) and, for the other
indicators, the xco2 files of calc_pi_xco2.py version 2 (<indicator>_<state>_mean per model). With an old
production folder only Omega_arag is available (from the pkl).

Usage:  python test_constraint_loo.py --data ../data/u74
        python test_constraint_loo.py --data .../OA_boundary_production_v2/data/u74 --reference OceanSODA UExP
        OA_RUN=u74 python test_constraint_loo.py --data .../OA_boundary_production/data/u74 --mechanism
            (all indicators now, from the mechanism files; run in the code folder of that production)
        options: --state 1750 (any state of the xco2 files), --no-cells (global only, fast), --nrand 200
        in a notebook: %run test_constraint_loo.py --data ...
"""
import argparse
import glob
import pickle
from pathlib import Path

import netCDF4
import numpy as np

import oa_config
import test_state_constraint as ts      # window_drift: the drift of the window as in the constraint

INDICATORS = ('omega_arag', 'omega_calc', 'ph', 'hplus', 'hplus_free', 'pco2', 'fco2', 'co2', 'hco3', 'co3',
              'revelle', 'dic')
SETS = {'dSST': ('dT',), 'CUC': ('CUC',), 'both': ('dT', 'CUC')}
# cells only: local dSST with the global CUC of each model (the same value in every cell)
CELL_SETS = {**SETS, 'dSST + global CUC': ('dT', 'gCUC')}


# ------------------------------------------------------------------------------------------------ data
def xco2_files(data, ref, names):
    """{model: path} of the xco2 files of a reference"""
    out = {}
    for m in names:
        hits = sorted(glob.glob(str(Path(data) / 'omega_arag_biasadj*' / 'piControl' /
                                    f'omega_arag_biasadj_xco2_Oyr_{m}_piControl_*_{ref}grid.nc')))
        if hits:
            out[m] = hits[0]
    return out


def model_values(R, data, ref, state):
    """{indicator: (global (n,), fields (n, ny, nx))} for the models of the reference; Omega_arag of 1750 from
    the pkl (exactly the paper's values), everything else from the xco2 files"""
    names = list(R['names'])
    out = {}
    if state == '1750':
        out['omega_arag'] = (np.asarray(R['y1750'], 'f8'), np.asarray(R['fields']['y1750'], 'f8'))
    files = xco2_files(data, ref, names)
    if len(files) != len(names):
        if files:
            print(f'  {ref}: xco2 files for {len(files)} of {len(names)} models only: other indicators skipped')
        return out
    lat = np.asarray(R['lat'], 'f8')
    mask = np.asarray(R['mask']).astype(bool)
    w = np.cos(np.deg2rad(lat))[:, None] * mask
    for ind in INDICATORS:
        F = []
        for m in names:
            with netCDF4.Dataset(files[m]) as f:
                v = f'{ind}_{state}_mean'
                if v not in f.variables:
                    F = None
                    break
                a = np.ma.filled(f[v][:].astype('f8'), np.nan)
                if a.shape != mask.shape:
                    raise ValueError(f'{files[m]}: grid {a.shape} differs from the constraint grid {mask.shape}')
                F.append(a)
        if F is None:
            continue
        F = np.array(F)
        valid = mask & np.all(np.isfinite(F), axis=0)
        g = np.array([np.sum(np.where(valid, a, 0) * w) / np.sum(w * valid) for a in F])
        if ind == 'omega_arag' and 'omega_arag' in out:
            d = np.max(np.abs(g - out['omega_arag'][0]))
            print(f'  {ref}: Omega_arag {state} from the xco2 files against the pkl: max |difference| {d:.1e}'
                  + ('' if d < 1e-3 else ' (expected for an old folder, whose xco2 files hold another quantity; the '
                     'pkl values are used, and the other indicators of these files are not trustworthy)'))
            continue
        out[ind] = (g, F)
    return out


# ------------------------------------------------------------------------------------------- mechanism mode
VARS = ('thetao', 'so', 'dissic', 'talk')
X_MODEL, X_1750 = 284.317, 278.3
MVARS_INDEX = {'ph': 0, 'pco2': 1, 'fco2': 2, 'co2': 3, 'hco3': 4, 'co3': 5, 'omega_arag': 6, 'omega_calc': 7,
               'revelle': 8}


def _total_to_free(t, s):
    """[H+]free / [H+]total (Dickson 1990 KS, Morris and Riley 1966 ST), as calc_pi_xco2.total_to_free"""
    tk = t + 273.15
    ion = 19.924 * s / (1000.0 - 1.005 * s)
    ln_ks = (-4276.1 / tk + 141.328 - 23.093 * np.log(tk)
             + (-13856.0 / tk + 324.57 - 47.986 * np.log(tk)) * np.sqrt(ion)
             + (35474.0 / tk - 771.54 + 114.723 * np.log(tk)) * ion
             - 2698.0 / tk * ion ** 1.5 + 1776.0 / tk * ion ** 2 + np.log(1.0 - 0.001005 * s))
    return 1.0 / (1.0 + 0.14 / 96.062 * s / 1.80655 / np.exp(ln_ks))


def mechanism_values(R, data, ref, changes=False):
    """{indicator: (global (n,), fields (n, ny, nx))} of an approximate 1750 state per model, built as the
    current production code builds its preindustrial Omega_arag (constraint_refs.py): the time-mean state
    'observed window state minus the model change' (historical window minus drift-corrected piControl, from the
    mechanism and drift files), then DIC lowered so that pCO2 drops by (284.317 - 278.3) ppm (1 - pH2O), all
    indicators from that state with mocsy and the constants of the run. Needs to run in the code folder
    (oa_config, OA_RUN, mocsy). With changes=True: the model changes themselves (historical window minus
    drift-corrected piControl) in T, S, DIC, ALK and DIC-ALK, in K, psu and umol/kg, to see which of them the
    predictors explain."""
    import oa_config
    import calc_omega_arag_obs as co
    import mocsy
    data = Path(data)
    mech = data / ('mechanism' if ref == 'OceanSODA' else f'mechanism_{ref}')
    with netCDF4.Dataset(mech / f'state_{ref}.nc') as f:
        obs = {v: np.ma.filled(f[v][:].astype('f8'), np.nan) for v in VARS}
        mlat, mlon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        w0, w1 = (int(x) for x in str(f.window).split('-'))
    lat, mask = np.asarray(R['lat'], 'f8'), np.asarray(R['mask']).astype(bool)
    if mlat.size != lat.size:
        raise ValueError(f'{mech}: grid differs from the constraint grid')
    si, po4 = (np.where(a < 0, 0.0, a) for a in co.load_nutrients(mlat, mlon))
    lat2 = np.broadcast_to(mlat[:, None], mask.shape)
    names = list(R['names'])
    out = {k: [] for k in list(MVARS_INDEX) + ['hplus', 'hplus_free', 'dic']}
    for m in names:
        hits = sorted(glob.glob(str(mech / f'state_{m}_*.nc')))
        if not hits:
            raise FileNotFoundError(f'{mech}: no state file for {m}')
        member = Path(hits[0]).stem.split(f'state_{m}_')[1]
        with netCDF4.Dataset(hits[0]) as f:
            st = {k: np.ma.filled(f[k][:].astype('f8'), np.nan) for k in f.variables if k not in ('lat', 'lon')}
        drift = ts.window_drift(data, m, member, (w0, w1), mlat, mlon)   # as in the constraint
        chg = {v: st[f'{v}_hist'] - st[f'{v}_pi'] - drift[v] for v in VARS}
        if changes:                                   # the model change itself, not the 1750 state
            chg['dissic-talk'] = chg['dissic'] - chg['talk']
            for v, a_ in chg.items():
                out.setdefault(v, []).append(np.where(mask, a_, np.nan))
            continue
        X = {v: obs[v] - chg[v] for v in VARS}
        ok = mask & np.all([np.isfinite(X[v]) for v in VARS], axis=0) & np.isfinite(si) & np.isfinite(po4)
        t, s_, a_ = X['thetao'][ok], X['so'][ok], X['talk'][ok] * 1e-6
        kw = dict(sil=si[ok] * 1e-6, phos=po4[ok] * 1e-6, patm=np.ones(t.size), depth=np.zeros(t.size), lat=lat2[ok],
                  optcon='mol/kg', optt='Tpot', optp='m', optb=oa_config.OPTB, optk1k2=oa_config.OPTK1K2)
        d = X['dissic'][ok] * 1e-6
        r = mocsy.mvars(temp=t, sal=s_, alk=a_, dic=d, **kw)
        p, rv = np.asarray(r[1], float), np.asarray(r[8], float)
        tk = t + 273.15
        ph2o = np.exp(24.4543 - 67.4509 * (100.0 / tk) - 4.8489 * np.log(tk / 100.0) - 0.000544 * s_)
        target = p + (X_1750 - X_MODEL) * (1.0 - ph2o)
        for _ in range(8):
            d = d + (target - p) * d / (rv * p)
            r = mocsy.mvars(temp=t, sal=s_, alk=a_, dic=d, **kw)
            p, rv = np.asarray(r[1], float), np.asarray(r[8], float)
            if np.nanmax(np.abs(p - target)) < 1e-4:
                break
        vals = {k: np.asarray(r[i], float) for k, i in MVARS_INDEX.items()}
        for k in ('co2', 'hco3', 'co3'):
            vals[k] = vals[k] * 1e6
        vals['hplus'] = 10.0 ** -vals['ph'] * 1e9
        vals['hplus_free'] = vals['hplus'] * _total_to_free(t, s_)
        vals['dic'] = d * 1e6
        for k, v in vals.items():
            f_ = np.full(mask.shape, np.nan)
            f_[ok] = v
            out[k].append(f_)
    w = np.cos(np.deg2rad(lat))[:, None] * mask
    res = {}
    if changes:
        for k, lab in (('thetao', 'dT (K)'), ('so', 'dS (psu)'), ('dissic', 'dDIC'), ('talk', 'dALK'), ('dissic-talk', 'dDIC-dALK')):
            F = np.array(out[k])
            valid = mask & np.all(np.isfinite(F), axis=0)
            res[lab] = (np.array([np.sum(np.where(valid, a, 0) * w) / np.sum(w * valid) for a in F]), F)
        g = res['dT (K)'][0]
        dT = np.asarray(R['dT'], 'f8')
        print(f'  {ref}: model dT of the state (window minus piControl) against the predictor dSST (window minus '
              f'1850-1900): mean difference {np.mean(g - dT):+.3f} K, correlation {np.corrcoef(g, dT)[0, 1]:.3f}')
        return res
    for k in INDICATORS:
        F = np.array(out[k])
        valid = mask & np.all(np.isfinite(F), axis=0)
        res[k] = (np.array([np.sum(np.where(valid, a, 0) * w) / np.sum(w * valid) for a in F]), F)
    y = np.asarray(R['y1750'], 'f8')
    g = res['omega_arag'][0]
    print(f'  {ref}: approximate 1750 state (time-mean, mechanism files): Omega_arag against the pkl: mean '
          f'{np.mean(g - y):+.4f}, max |difference| {np.max(np.abs(g - y)):.4f}, correlation across models '
          f'{np.corrcoef(g, y)[0, 1]:.4f}; RMSE ratio of the test uses these values for every indicator')
    return res


# ------------------------------------------------------------------------------------------------ global
def fit_predict(X, y, x0):
    """least squares on (X, y), value and sigma (residual with leverage) at x0"""
    n, p = X.shape
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ beta
    s2 = e @ e / (n - p)
    return x0 @ beta, np.sqrt(s2 * (1 + x0 @ np.linalg.solve(XtX, x0)))


def loo_global(y, P):
    """leave-one-out: (errors constrained, sigmas, errors unconstrained, sigmas) over the models"""
    n = y.size
    X = np.column_stack([np.ones(n)] + list(P))
    ec, sc, eu, su = (np.empty(n) for _ in range(4))
    for i in range(n):
        k = np.arange(n) != i
        c, s = fit_predict(X[k], y[k], X[i])
        ec[i], sc[i] = c - y[i], s
        eu[i] = y[k].mean() - y[i]
        su[i] = y[k].std(ddof=1) * np.sqrt(1 + 1 / (n - 1))
    return ec, sc, eu, su


def r2_p(y, P):
    """R2 and F-test p value of the regression on all models"""
    from scipy import stats
    n, p = y.size, len(P)
    X = np.column_stack([np.ones(n)] + list(P))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - e @ e / sst
    F = (r2 / p) / ((1 - r2) / (n - p - 1)) if r2 < 1 else np.inf
    return r2, float(stats.f.sf(F, p, n - p - 1))


# ------------------------------------------------------------------------------------------------ cells
def loo_cells(Y, P, w):
    """per-cell leave-one-out; Y (n, C), P list of (n, C), w (C,) area weights of the valid cells.
    Returns RMSE ratio (area-weighted), coverage of both, fraction of the area where the constraint helps, RMSE
    ratio of the area mean, and in the units of the indicator the area-weighted RMS error of the constrained and of
    the unconstrained prediction and the RMS sigma of the constrained prediction"""
    n, C = Y.shape
    X = np.stack([np.ones((n, C))] + list(P), -1).transpose(1, 0, 2)      # (C, n, p)
    p = X.shape[2]
    se_c, se_u, cov_c, cov_u, ss_c = np.zeros(C), np.zeros(C), np.zeros(C), np.zeros(C), np.zeros(C)
    gm_c, gm_u = np.zeros(n), np.zeros(n)                 # error of the area mean of the fields per left-out model
    for i in range(n):
        k = np.arange(n) != i
        Xk, yk = X[:, k], Y[k].T                                            # (C, n-1, p), (C, n-1)
        inv = np.linalg.inv(np.einsum('cmi,cmj->cij', Xk, Xk))
        beta = np.einsum('cij,cmj,cm->ci', inv, Xk, yk)
        e = yk - np.einsum('cmi,ci->cm', Xk, beta)
        s2 = (e ** 2).sum(1) / (n - 1 - p)
        x0 = X[:, i]
        c = np.einsum('ci,ci->c', beta, x0)
        sc = np.sqrt(s2 * (1 + np.einsum('ci,cij,cj->c', x0, inv, x0)))
        u = yk.mean(1)
        su = yk.std(1, ddof=1) * np.sqrt(1 + 1 / (n - 1))
        gm_c[i] = np.sum(w * (c - Y[i])) / w.sum()
        gm_u[i] = np.sum(w * (u - Y[i])) / w.sum()
        se_c += (c - Y[i]) ** 2
        se_u += (u - Y[i]) ** 2
        cov_c += np.abs(c - Y[i]) < sc
        ss_c += sc ** 2
        cov_u += np.abs(u - Y[i]) < su
    rmse_c, rmse_u = np.sqrt(se_c / n), np.sqrt(se_u / n)
    ratio = np.sqrt(np.sum(w * rmse_c ** 2) / np.sum(w * rmse_u ** 2))
    ratio_gm = np.sqrt(np.mean(gm_c ** 2) / np.mean(gm_u ** 2))
    wm = lambda x: float(np.sqrt(np.sum(w * x) / w.sum()))
    return (ratio, np.sum(w * cov_c / n) / w.sum(), np.sum(w * cov_u / n) / w.sum(), np.sum(w * (rmse_c < rmse_u)) / w.sum(),
            ratio_gm, wm(se_c / n), wm(se_u / n), wm(ss_c / n))


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='data/<run> folder with constraint_results.pkl')
    ap.add_argument('--reference', nargs='+', help='default: all in the pkl')
    ap.add_argument('--state', default='1750')
    ap.add_argument('--no-cells', action='store_true')
    ap.add_argument('--changes', action='store_true',
                    help='with --mechanism: regress the model changes (dT, dS, dDIC, dALK, dDIC-dALK) instead of '
                         'the indicators; spread printed in units, not in %%')
    ap.add_argument('--mechanism', action='store_true',
                    help='all indicators from an approximate 1750 state built from the mechanism and drift files '
                         '(for a folder without the v2 xco2 files; run in its code folder with OA_RUN set)')
    ap.add_argument('--nrand', type=int, default=200, help='realisations of the random-predictor test')
    a, _ = ap.parse_known_args()
    with open(Path(a.data) / 'constraint_results.pkl', 'rb') as f:
        PK = pickle.load(f)
    rng = np.random.default_rng(0)
    for ref in a.reference or list(PK['results']):
        R = PK['results'][ref]
        names = list(R['names'])
        n = len(names)
        V = mechanism_values(R, a.data, ref, a.changes) if a.mechanism else model_values(R, a.data, ref, a.state)
        print(f'\n===== {ref}: {n} models, state {a.state}, indicators {list(V)}')
        dT, CUC = np.asarray(R['dT'], 'f8'), np.asarray(R['CUC'], 'f8')
        print(f'  global predictors: dSST {dT.min():.2f}-{dT.max():.2f} K, CUC {CUC.min():.4f}-{CUC.max():.4f}; '
              f'correlation dSST-CUC {np.corrcoef(dT, CUC)[0, 1]:+.2f}')
        G = {'dT': dT, 'CUC': CUC}
        print('  GLOBAL (leave one model out): RMSE ratio constrained/unconstrained (< 1 helps), coverage of 1 sigma '
              '(0.68), R2 and p of the fit on all models; random = same with random predictors')
        sp_lab = 'spread' if a.changes else 'spread %'
        print(f'  {"indicator":11s} {sp_lab:>8s} | ' + ' | '.join(f'{s:^27s}' for s in SETS) + f' | {"random 1 / 2":>12s} '
              f'| {"cov. unc.":>9s}')
        print(f'  {"":11s} {"":8s} | ' + ' | '.join(f'{"ratio":>6s} {"cov":>5s} {"R2":>5s} {"p":>7s}' for _ in SETS) + ' |')
        if a.changes:
            print('  slopes of the fit on all models with both predictors, per 1 sd of the predictor (dSST sd '
                  f'{dT.std(ddof=1):.3f} K, CUC sd {CUC.std(ddof=1):.4f}), with the t value; correlation of the change with dS:')
            for ind, (y, _) in V.items():
                X = np.column_stack([np.ones(n), dT, CUC])
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                e = y - X @ beta
                cov = e @ e / (n - 3) * np.linalg.inv(X.T @ X)
                tval = beta / np.sqrt(np.diag(cov))
                cs = np.corrcoef(y, V['dS (psu)'][0])[0, 1] if ind != 'dS (psu)' else 1.0
                print(f'    {ind:11s} mean {y.mean():+9.4f}  dSST {beta[1] * dT.std(ddof=1):+9.4f} (t {tval[1]:+5.1f})  '
                      f'CUC {beta[2] * CUC.std(ddof=1):+9.4f} (t {tval[2]:+5.1f})  corr with dS {cs:+.2f}')
        for ind, (y, _) in V.items():
            cells = []
            for s, keys in SETS.items():
                P = [G[k] for k in keys]
                ec, sc, eu, su = loo_global(y, P)
                r2, p = r2_p(y, P)
                cells.append(f'{np.sqrt(np.mean(ec ** 2) / np.mean(eu ** 2)):6.2f} {np.mean(np.abs(ec) < sc):5.2f} '
                             f'{r2:5.2f} {p:7.1e}')
            rnd = []
            for npred in (1, 2):
                rr = []
                for _ in range(a.nrand):
                    ec, _, eu, _ = loo_global(y, list(rng.normal(size=(npred, n))))
                    rr.append(np.mean(ec ** 2) / np.mean(eu ** 2))
                rnd.append(np.sqrt(np.mean(rr)))
            spread = y.std(ddof=1) if a.changes else 100 * y.std(ddof=1) / abs(y.mean())
            print(f'  {ind:11s} {spread:8.3f} | ' + ' | '.join(cells) + f' | {rnd[0]:5.2f} {rnd[1]:5.2f} | '
                  f'{np.mean(np.abs(eu) < su):9.2f}')
        ABS_C = {}
        if not a.no_cells:
            Fd = R['fields']
            DTf, CUf = np.asarray(Fd['dT'], 'f8'), np.asarray(Fd['CUC'], 'f8')
            lat = np.asarray(R['lat'], 'f8')
            base = np.asarray(R['mask']).astype(bool) & np.all(np.isfinite(DTf), 0) & np.all(np.isfinite(CUf), 0)
            print('  CELLS (leave one model out per cell, local dSST and CUC; last: local dSST and global CUC): RMSE ratio (area-weighted), coverage of 1 '
                  'sigma constrained / unconstrained, fraction of the area where the constraint helps')
            print('  (last number of each set: RMSE ratio of the AREA MEAN of the cell-by-cell constrained fields, to be '
                  'compared with the global constraint above)')
            print(f'  {"indicator":11s} | ' + ' | '.join(f'{s:^29s}' for s in CELL_SETS))
            for ind, (_, Fy) in V.items():
                ok = base & np.all(np.isfinite(Fy), 0)
                iw = np.where(ok.ravel())[0]
                w = oa_config.area_weights(lat, np.asarray(R['lon'], 'f8')).ravel()[iw]
                sel = lambda A: A.reshape(n, -1)[:, iw]
                Pc = {'dT': sel(DTf), 'CUC': sel(CUf), 'gCUC': np.repeat(CUC[:, None], iw.size, axis=1)}
                cells = []
                for s, keys in CELL_SETS.items():
                    ratio, cc, cu, frac, rgm, ec_, eu_, sc_ = loo_cells(sel(Fy), [Pc[k] for k in keys], w)
                    cells.append(f'{ratio:5.2f} {cc:4.2f}/{cu:4.2f} {100 * frac:3.0f}% {rgm:5.2f}')
                    if s == 'both':
                        ABS_C[ind] = (ec_, eu_, sc_, cc)
                print(f'  {ind:11s} | ' + ' | '.join(f'{c:29s}' for c in cells))
        print('  in the units of the indicator, both predictors: RMS error of the prediction of the left-out model, '
              'constrained and unconstrained (multi-model mean of the others), RMS sigma of the constrained '
              'prediction, and the fraction of left-out models within it (0.68 expected)')
        for ind, (y, _) in V.items():
            ec, sc, eu, su = loo_global(y, [G['dT'], G['CUC']])
            line = (f'    {ind:11s} global: error {np.sqrt(np.mean(ec ** 2)):.4g} vs {np.sqrt(np.mean(eu ** 2)):.4g}, '
                    f'sigma {np.sqrt(np.mean(sc ** 2)):.4g}, within {np.sum(np.abs(ec) < sc)}/{n}')
            if ind in ABS_C:
                ec_, eu_, sc_, cc = ABS_C[ind]
                line += f';  cells: error {ec_:.4g} vs {eu_:.4g}, sigma {sc_:.4g}, within {cc:.2f}'
            print(line)


if __name__ == '__main__':
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
