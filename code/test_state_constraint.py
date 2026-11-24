#!/usr/bin/env python
"""
test_state_constraint.py
========================
One product, one state (1750): the constrained STATE against the constrained INDICATOR.

    direct     the paper: Omega_arag of every model's 1750 state, regressed on dSST and CUC (global means; and per
               cell with the local predictors)
    state A    every change (dT, dS, dDIC, dALK) regressed on dSST and CUC with the same weights, the 1750 state
               built from the constrained changes, Omega_arag computed from it
    state C    dT regressed on dSST, dDIC-dALK regressed on dSST and CUC, dS and dALK the multi-model mean;
               the state built from these, Omega_arag computed from it (the proposed method)

For each: the global value (global weights, Omega field, area mean on the constraint mask) and the area mean of
the cell-by-cell version (local weights per cell), the difference to the direct regression, and a leave-one-out
test (each model the truth, the others give the constraint) of the global value and of the cell fields.
The sigma of state C comes from the model residuals: every model's residual (its changes minus the fitted
changes) is added to the constrained changes, Omega_arag is recomputed from the 18 residual states and the
spread, times the leverage factor, is the sigma; the observed-predictor terms are added in quadrature. It is
compared with the sigma of the direct regression.

The states are the time-mean window states minus the time-mean model changes (mechanism and drift files), DIC
lowered so that pCO2 drops by (284.317 - 278.3) ppm (1 - pH2O), Omega_arag with mocsy and the constants of the
run (as test_constraint_loo.py --mechanism; these reproduce the paper's per-model values to 4e-4).

Saves, per product, the global means behind the figure of the mechanism (fig cell 'state_constraint'):
    <data>/state_constraint/state_<ref>.npz   per model: dSST, CUC (predictors), global mean changes dT, dS, dDIC, dALK
    (1750 to the window, drift corrected, on the valid cells), Omega_arag 1750; the observed predictors; direct and
    state-C Omega_1750 with their sigma; the global regression of dT on dSST and of dDIC - dALK on dSST and CUC.

Usage (in the code folder of the production):
    OA_RUN=u74 python test_state_constraint.py --data ../data/u74 [--reference OceanSODA] [--no-cells]
"""
import argparse
import glob
import pickle
from pathlib import Path

import netCDF4
import numpy as np

VARS = ('thetao', 'so', 'dissic', 'talk')
X_MODEL, X_1750 = 284.317, 278.3
CUC_SIG = 0.0          # the observed CUC uncertainty is counted with the mapping term (cuc_mapping_uncertainty.py);
                       # its value is saved for the figure (CUC_obs_sigma)


# ------------------------------------------------------------------------------------------------ data
def cuc_map_sigma(data, ref):
    """the mapping uncertainty of the observed CUC on the product's own area (cuc_mapping_uncertainty.py), for the
    figure; NaN if not computed yet"""
    p = Path(data) / 'product_uncertainty' / f'cuc_mapping_{ref}.npz'
    return float(np.load(p)['own_sigma_cuc_C2']) if p.exists() else np.nan


def window_drift(data, model, member, window, lat, lon):
    """{v: (ny, nx)} the piControl drift over the years parallel to the window, minus the piControl mean, on the grid
    lat, lon (nearest neighbour), from drift_tools.Drift.window_mean: the same function as the constraint
    (constraint_refs.drift_means) and the data product, so that a window needs no period stored in the drift file
    (the stored ones are only those of calc_drift.py; UExP-FNN-U has its own window 1980-2014 since 12 Nov 2026)"""
    import drift_tools as dt
    path = dt.drift_path(model, member)
    if path.resolve() != (Path(data) / 'drift' / path.name).resolve():
        raise ValueError(f'drift file {path} is not in {data}/drift: --data and OA_RUN/OA_BASE disagree')
    d = dt.Drift(model, member)
    jj = np.abs(lat[:, None] - d.lat[None]).argmin(1)
    ii = np.abs(((lon[:, None] - d.lon[None]) + 180) % 360 - 180).argmin(1)
    out = {v: d.window_mean(v, window)[np.ix_(jj, ii)] for v in VARS}
    del d
    return out


def read_changes(R, data, ref):
    """observed window state {v: (ny, nx)}, model changes {v: (n, ny, nx)} (window minus drift-corrected
    piControl), member names"""
    data = Path(data)
    mech = data / ('mechanism' if ref == 'OceanSODA' else f'mechanism_{ref}')
    with netCDF4.Dataset(mech / f'state_{ref}.nc') as f:
        obs = {v: np.ma.filled(f[v][:].astype('f8'), np.nan) for v in VARS}
        mlat, mlon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        w0, w1 = (int(x) for x in str(f.window).split('-'))
    chg = {v: [] for v in VARS}
    for m in R['names']:
        hits = sorted(glob.glob(str(mech / f'state_{m}_*.nc')))
        member = Path(hits[0]).stem.split(f'state_{m}_')[1]
        with netCDF4.Dataset(hits[0]) as f:
            st = {k: np.ma.filled(f[k][:].astype('f8'), np.nan) for k in f.variables if k not in ('lat', 'lon')}
        drift = window_drift(data, m, member, (w0, w1), mlat, mlon)
        for v in VARS:
            chg[v].append(st[f'{v}_hist'] - st[f'{v}_pi'] - drift[v])
    return obs, {v: np.array(a) for v, a in chg.items()}, mlat, mlon


class Chem:
    """Omega_arag of 1750 states on the valid cells: state at 284.317 ppm -> DIC lowered -> Omega"""

    def __init__(self, lat, lon, ok):
        import oa_config
        import calc_omega_arag_obs as co
        import mocsy
        self.mocsy, self.cfg = mocsy, oa_config
        si, po4 = (np.where(a < 0, 0.0, a) for a in co.load_nutrients(lat, lon))
        self.ok = ok & np.isfinite(si) & np.isfinite(po4)
        lat2 = np.broadcast_to(lat[:, None], ok.shape)
        n = int(self.ok.sum())
        self.kw = dict(sil=si[self.ok] * 1e-6, phos=po4[self.ok] * 1e-6, patm=np.ones(n), depth=np.zeros(n),
                       lat=lat2[self.ok], optcon='mol/kg', optt='Tpot', optp='m', optb=oa_config.OPTB,
                       optk1k2=oa_config.OPTK1K2)

    def omega(self, X):
        """X: {v: field (ny, nx)} at 284.317 ppm -> Omega_arag field at 278.3 ppm"""
        t, s, a = X['thetao'][self.ok], X['so'][self.ok], X['talk'][self.ok] * 1e-6
        d = X['dissic'][self.ok] * 1e-6
        r = self.mocsy.mvars(temp=t, sal=s, alk=a, dic=d, **self.kw)
        p, rv = np.asarray(r[1], float), np.asarray(r[8], float)
        tk = t + 273.15
        ph2o = np.exp(24.4543 - 67.4509 * (100.0 / tk) - 4.8489 * np.log(tk / 100.0) - 0.000544 * s)
        target = p + (X_1750 - X_MODEL) * (1.0 - ph2o)
        for _ in range(8):
            d = d + (target - p) * d / (rv * p)
            r = self.mocsy.mvars(temp=t, sal=s, alk=a, dic=d, **self.kw)
            p, rv = np.asarray(r[1], float), np.asarray(r[8], float)
            if np.nanmax(np.abs(p - target)) < 1e-4:
                break
        om = np.full(self.ok.shape, np.nan)
        om[self.ok] = np.asarray(r[6], float)
        return om


# ------------------------------------------------------------------------------------------------ regression
def weights(P, p0):
    """regression weights w (n,) such that the prediction at p0 is w @ y; P: list of predictor vectors (n,),
    p0: their observed values. Also the leverage and (X'X)^-1 for the sigma."""
    n = P[0].size
    X = np.column_stack([np.ones(n)] + list(P))
    inv = np.linalg.inv(X.T @ X)
    x0 = np.array([1.0] + list(p0))
    return x0 @ inv @ X.T, float(x0 @ inv @ x0), X, x0


def sigma_of(y, w, lev, X, x0, sp):
    """regression sigma of the prediction w @ y: residual with leverage and the observed predictors"""
    n, p = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    s2 = e @ e / (n - p)
    return float(np.sqrt(s2 * (1 + lev) + np.sum((beta[1:] * np.asarray(sp)) ** 2)))


def cell_weights(P, p0, iok):
    """per-cell regression weights: P list of (n, C) predictor fields, p0 list of (C,) observed; -> (C, n), lev (C,)"""
    n, C = P[0].shape
    X = np.stack([np.ones((n, C))] + list(P), -1).transpose(1, 0, 2)          # (C, n, p)
    inv = np.linalg.inv(np.einsum('cmi,cmj->cij', X, X))
    x0 = np.stack([np.ones(C)] + list(p0), -1)                                  # (C, p)
    H = np.einsum('cij,cmj->cim', inv, X)                                       # (C, p, n)
    return np.einsum('ci,cim->cm', x0, H), np.einsum('ci,cij,cj->c', x0, inv, x0)


# ------------------------------------------------------------------------------------------------ the states
def build_state(obs, chg, wT, wC, sel):
    """state C: dT with weights wT, dDIC-dALK with weights wC, dS and dALK the mean of the selected models;
    weights: (n,) global or (C, n) per cell applied on the flattened fields"""
    def apply(w, a):                                  # a (n, ny, nx) -> (ny, nx)
        A = a[sel].reshape(len(sel), -1)
        return (w @ A if w.ndim == 1 else np.einsum('cm,mc->c', w, A)).reshape(a.shape[1:])
    dT = apply(wT, chg['thetao'])
    dC = apply(wC, chg['dissic'] - chg['talk'])
    dS = chg['so'][sel].mean(0)
    dA = chg['talk'][sel].mean(0)
    return {'thetao': obs['thetao'] - dT, 'so': obs['so'] - dS, 'dissic': obs['dissic'] - (dC + dA), 'talk': obs['talk'] - dA}


def build_state_A(obs, chg, w, sel):
    def apply(a):
        A = a[sel].reshape(len(sel), -1)
        return (w @ A if w.ndim == 1 else np.einsum('cm,mc->c', w, A)).reshape(a.shape[1:])
    return {v: obs[v] - apply(chg[v]) for v in VARS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--reference', nargs='+')
    ap.add_argument('--no-cells', action='store_true')
    a, _ = ap.parse_known_args()
    with open(Path(a.data) / 'constraint_results.pkl', 'rb') as f:
        PK = pickle.load(f)
    for ref in a.reference or list(PK['results']):
        R = PK['results'][ref]
        names = list(R['names'])
        n = len(names)
        obs, chg, lat, lon = read_changes(R, a.data, ref)
        mask = np.asarray(R['mask']).astype(bool)
        ok = mask & np.all([np.isfinite(obs[v]) for v in VARS], 0) & np.all([np.all(np.isfinite(chg[v]), 0) for v in VARS], 0)
        chem = Chem(lat, lon, ok)
        ok = chem.ok
        w_area = np.cos(np.deg2rad(lat))[:, None] * np.ones(ok.shape) * ok
        gm = lambda f: float(np.nansum(np.where(ok, f, 0) * w_area) / w_area.sum())
        dT, CUC = np.asarray(R['dT'], 'f8'), np.asarray(R['CUC'], 'f8')
        dTo, dTs = float(R['dT_obs'][0]), float(R['dT_obs'][1])
        CUo = float(R['CUC_obs'])
        print(f'\n===== {ref}: {n} models, 1750 state, {int(ok.sum())} cells; observed dSST {dTo:.3f} +- {dTs:.3f} K, CUC {CUo:.4f}')

        # per-model Omega fields of the 1750 state (direct)
        OM = np.array([chem.omega({v: obs[v] - chg[v][m] for v in VARS}) for m in range(n)])
        y = np.array([gm(o) for o in OM])
        paper = R['constrained']['dT + CUC']
        print(f'  per-model global Omega_1750 against the pkl: max |difference| {np.max(np.abs(y - np.asarray(R["y1750"]))):.1e}')

        all_ = np.arange(n)
        wC, levC, XC, x0C = weights([dT, CUC], [dTo, CUo])
        wT, levT, XT, x0T = weights([dT], [dTo])
        direct = float(wC @ y)
        s_direct = sigma_of(y, wC, levC, XC, x0C, [dTs, CUC_SIG])
        omA = chem.omega(build_state_A(obs, chg, wC, all_))
        omC = chem.omega(build_state(obs, chg, wT, wC, all_))
        gA, gC = gm(omA), gm(omC)
        print(f'  GLOBAL Omega_1750: direct regression {direct:.4f} +- {s_direct:.4f} (paper {paper[0]:.4f} +- {paper[1]:.4f}); '
              f'state A {gA:.4f} ({gA - direct:+.1e}); state C {gC:.4f} ({gC - direct:+.1e}); unconstrained {y.mean():.4f}')
        # sigma of state C from the model residuals (consistent through the chemistry)
        fitT = XT @ np.linalg.lstsq(XT, chg['thetao'].reshape(n, -1), rcond=None)[0]
        fitC = XC @ np.linalg.lstsq(XC, (chg['dissic'] - chg['talk']).reshape(n, -1), rcond=None)[0]
        baseC = build_state(obs, chg, wT, wC, all_)
        om_res = []
        for m in range(n):
            rT = (chg['thetao'][m].ravel() - fitT[m]).reshape(ok.shape)
            rC = ((chg['dissic'] - chg['talk'])[m].ravel() - fitC[m]).reshape(ok.shape)
            rS = chg['so'][m] - chg['so'].mean(0)
            rA = chg['talk'][m] - chg['talk'].mean(0)
            X = {'thetao': baseC['thetao'] - rT, 'so': baseC['so'] - rS, 'dissic': baseC['dissic'] - (rC + rA), 'talk': baseC['talk'] - rA}
            om_res.append(gm(chem.omega(X)))
        om_res = np.array(om_res)
        # residual std with the degrees of freedom of the fits (n-3 for the regressed, n-1 for the means), leverage
        s_res = np.sqrt(np.sum((om_res - gC) ** 2) / (n - 3) * (1 + levC))
        # observed predictors: the state rebuilt at dSST_obs +- sigma and CUC_obs +- sigma (half differences)
        def gC_at(dTo_, CUo_):
            wC_, *_ = weights([dT, CUC], [dTo_, CUo_])
            wT_, *_ = weights([dT], [dTo_])
            return gm(chem.omega(build_state(obs, chg, wT_, wC_, all_)))
        s_pred = np.hypot(0.5 * (gC_at(dTo + dTs, CUo) - gC_at(dTo - dTs, CUo)),
                          0.5 * (gC_at(dTo, CUo + CUC_SIG) - gC_at(dTo, CUo - CUC_SIG)))
        print(f'  sigma of state C from the model residuals: {s_res:.4f} (residual states, leverage) + observed predictors '
              f'{s_pred:.4f} -> {np.hypot(s_res, s_pred):.4f}; direct regression sigma {s_direct:.4f}')
        # global means for the figure of the mechanism
        G = {v: np.array([gm(chg[v][m]) for m in range(n)]) for v in VARS}
        gdc = G['dissic'] - G['talk']
        bT = np.linalg.lstsq(XT, G['thetao'], rcond=None)[0]
        bC = np.linalg.lstsq(XC, gdc, rcond=None)[0]
        out = Path(a.data) / 'state_constraint'
        out.mkdir(exist_ok=True)
        np.savez(out / f'state_{ref}.npz', names=np.array(names), dSST=dT, CUC=CUC, dT=G['thetao'], dS=G['so'],
                 dDIC=G['dissic'], dALK=G['talk'], y1750=y, dSST_obs=dTo, dSST_obs_sigma=dTs, CUC_obs=CUo,
                 CUC_obs_sigma=cuc_map_sigma(a.data, ref), direct=direct, direct_sigma=s_direct, state=gC,
                 state_sigma=np.hypot(s_res, s_pred), state_sigma_models=s_res, state_sigma_predictors=s_pred,
                 unconstrained=y.mean(), unconstrained_sd=y.std(ddof=1), beta_dT=bT, beta_dC=bC,
                 dT_constrained=float(wT @ G['thetao']), dT_constrained_sigma=sigma_of(G['thetao'], wT, levT, XT, x0T, [dTs]),
                 dC_constrained=float(wC @ gdc), dC_constrained_sigma=sigma_of(gdc, wC, levC, XC, x0C, [dTs, CUC_SIG]))
        print(f'  global means: dT on dSST {bT[1]:.3f} degC/degC (R2 {1 - np.var(G["thetao"] - XT @ bT) / np.var(G["thetao"]):.2f}), '
              f'dDIC-dALK = {bC[0]:.1f} {bC[1]:+.2f} dSST {bC[2]:+.1f} CUC (R2 {1 - np.var(gdc - XC @ bC) / np.var(gdc):.2f}); '
              f'saved {out / f"state_{ref}.npz"}')

        # leave-one-out, global
        e = {k: [] for k in ('u', 'direct', 'A', 'C')}
        for i in range(n):
            k = all_[all_ != i]
            wCk, *_ = weights([dT[k], CUC[k]], [dT[i], CUC[i]])
            wTk, *_ = weights([dT[k]], [dT[i]])
            e['u'].append(y[k].mean() - y[i])
            e['direct'].append(wCk @ y[k] - y[i])
            e['A'].append(gm(chem.omega(build_state_A(obs, chg, wCk, k))) - y[i])
            e['C'].append(gm(chem.omega(build_state(obs, chg, wTk, wCk, k))) - y[i])
        rm = {k: np.sqrt(np.mean(np.square(v))) for k, v in e.items()}
        print(f'  leave-one-out, global: RMSE unconstrained {rm["u"]:.4f}; direct {rm["direct"]:.4f} (ratio {rm["direct"] / rm["u"]:.2f}), '
              f'state A {rm["A"]:.4f} ({rm["A"] / rm["u"]:.2f}), state C {rm["C"]:.4f} ({rm["C"] / rm["u"]:.2f})')
        if a.no_cells:
            continue

        # cells: local predictors
        F = R['fields']
        DTf, CUf = np.asarray(F['dT'], 'f8'), np.asarray(F['CUC'], 'f8')
        dTo_f = np.mean([np.asarray(F['dsst_obs'][k], 'f8') for k in F['dsst_obs']], 0)
        CUo_f = np.asarray(F['CUC_obs'], 'f8')
        okc = ok & np.all(np.isfinite(DTf), 0) & np.all(np.isfinite(CUf), 0) & np.isfinite(dTo_f) & np.isfinite(CUo_f)
        iw = np.where(okc.ravel())[0]
        wa = w_area.ravel()[iw]
        am = lambda f: float(np.nansum(f.ravel()[iw] * wa) / wa.sum())
        sel = lambda A: A.reshape(A.shape[0], -1)[:, iw]

        def local(k, x_dT=dTo_f, x_CU=CUo_f):
            """cell fields of direct, A, C with the models k and the local predictors, evaluated at the
            predictor fields x_dT, x_CU (the observations, or the left-out model's own); weights on the valid cells"""
            wCc, _ = cell_weights([sel(DTf[k]), sel(CUf[k])], [x_dT.ravel()[iw], x_CU.ravel()[iw]], iw)
            wTc, _ = cell_weights([sel(DTf[k])], [x_dT.ravel()[iw]], iw)
            full = lambda w: np.full((ok.size, len(k)), np.nan)
            WC, WT = full(wCc), full(wTc)
            WC[iw], WT[iw] = wCc, wTc
            WC, WT = np.nan_to_num(WC), np.nan_to_num(WT)
            direct_f = np.einsum('cm,mc->c', WC, OM[k].reshape(len(k), -1)).reshape(ok.shape)
            direct_f[~okc] = np.nan
            A_f = chem.omega(build_state_A(obs, chg, WC, k)); A_f[~okc] = np.nan
            C_f = chem.omega(build_state(obs, chg, WT, WC, k)); C_f[~okc] = np.nan
            return direct_f, A_f, C_f
        d_f, A_f, C_f = local(all_)
        print(f'  CELLS (local dSST and CUC), area mean of the fields: direct {am(d_f):.4f}, state A {am(A_f):.4f} '
              f'({am(A_f) - am(d_f):+.1e}), state C {am(C_f):.4f} ({am(C_f) - am(d_f):+.1e}); max |state C - direct| per cell '
              f'{np.nanmax(np.abs(C_f - d_f)):.4f}, rms {np.sqrt(np.nanmean((C_f - d_f) ** 2)):.4f}')
        # leave-one-out, cells
        se = {k: np.zeros(ok.shape) for k in ('u', 'direct', 'A', 'C')}
        for i in range(n):
            k = all_[all_ != i]
            d_f, A_f, C_f = local(k, DTf[i], CUf[i])
            se['u'] += (OM[k].mean(0) - OM[i]) ** 2
            se['direct'] += (d_f - OM[i]) ** 2
            se['A'] += (A_f - OM[i]) ** 2
            se['C'] += (C_f - OM[i]) ** 2
        rms = {k: np.sqrt(am(v / n)) for k, v in se.items()}
        print(f'  leave-one-out, cells (area-weighted RMSE): unconstrained {rms["u"]:.4f}; direct {rms["direct"] / rms["u"]:.2f}, '
              f'state A {rms["A"] / rms["u"]:.2f}, state C {rms["C"] / rms["u"]:.2f}')


if __name__ == '__main__':
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
