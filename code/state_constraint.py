"""
state_constraint.py
===================
The emergent constraint applied to the model STATE (the method of the data product):

For a set of n models with the predictors dSST (warming of the reference window relative to 1850-1900) and CUC
(carbon uptake capacity of the window state), and the adjusted state X~_m = (T, S, DIC, ALK) of every model for
one month (preindustrial climatology, or a smoothed historical or scenario month), the constrained state is
    T        the regression of the models' T on dSST, evaluated at the observed dSST
    DIC-ALK  the regression of the models' DIC-ALK on dSST and CUC, evaluated at the observed values
    S, ALK   the multi-model mean (no observation predicts them)
    DIC      = (DIC-ALK) + ALK
globally (one regression on the global-mean predictors, applied to every cell) and regionally (the regression
in every cell with the local predictors; cells without local predictors take the multi-model mean). The
unconstrained state is the multi-model mean. All indicators are computed from these states with mocsy
(calc_pi_xco2.target_states), so that every indicator belongs to one chemically consistent state.

Uncertainty (1 sigma), consistently through the chemistry, from a set of states computed together:
    fit         every model's residual (its state minus its fitted value) is added to the constrained state;
                the indicators of these n residual states scatter around the constrained indicators with the
                variance sigma^2 (n - p)/n; the prediction error at the observed predictors is
                sqrt(sum of squares / (n - p) * (1 + leverage)), with p = 3 and the leverage of the DIC-ALK
                regression (conservative for T, S and ALK)
    predictors  the constrained state rebuilt at dSST_obs -/+ sigma and CUC_obs -/+ sigma: half differences
    CO2         (the caller) the state lowered or raised to the CO2 -/+ its sigma: half differences
    unconstrained   the spread (std, ddof 1) of the indicators of the n model states
The states of one month are stacked and sent to mocsy in one call (STACK below gives the order).

Regression weights: the prediction of a least-squares line at x0 is w @ y with w = x0 (X'X)^-1 X', the
fitted values are P y with P = X (X'X)^-1 X' (hat matrix), the residuals (I - P) y. Both depend on the
predictors only, so one set of matrices serves every variable, month and year.
"""
from __future__ import annotations

import warnings

import numpy as np

import calc_omega_arag as cm       # the freezing-point floor
import oa_config                   # the area weights (OA_WEIGHTS)

CUC_SIG = 0.0                         # default only: the product passes the mapping uncertainty of the observed CUC
                                      # (cuc_mapping_uncertainty.py, global value and field; calc_product_states.py)
P_FIT = 3                             # parameters of the DIC-ALK regression (the sigma uses its dof and leverage)
VARS = ('thetao', 'so', 'dissic', 'talk')


# ---------------------------------------------------------------------------------------------------- algebra
def design(preds):
    """design matrix (n, p) from a list of predictor vectors (n,)"""
    n = preds[0].size
    return np.column_stack([np.ones(n)] + [np.asarray(p, 'f8') for p in preds])


class GlobalFit:
    """one regression for all cells: weights at x0 (n,), hat matrix (n, n), leverage at x0"""

    def __init__(self, preds, x0):
        X = design(preds)
        self.inv = np.linalg.inv(X.T @ X)
        self.X = X
        self.P = X @ self.inv @ X.T
        self.at(x0)

    def at(self, x0):
        self.x0 = np.array([1.0] + list(x0))
        self.w = self.x0 @ self.inv @ self.X.T
        self.lev = float(self.x0 @ self.inv @ self.x0)
        return self

    def weights(self, x0):
        x = np.array([1.0] + list(x0))
        return x @ self.inv @ self.X.T

    def r2(self, y):
        """R2 of the fit to y (n,)"""
        e = y - self.P @ y
        ss = np.sum((y - y.mean()) ** 2)
        return float(1 - e @ e / ss) if ss > 0 else np.nan


class CellFit:
    """the same regression in every cell c (C cells): preds list of (n, C), x0 list of (C,)"""

    def __init__(self, preds, x0):
        n, C = preds[0].shape
        X = np.stack([np.ones((n, C))] + [np.asarray(p, 'f8') for p in preds], -1).transpose(1, 0, 2)  # (C, n, p)
        self.X = X
        self.inv = np.linalg.inv(np.einsum('cmi,cmj->cij', X, X))                    # (C, p, p)
        self.P = np.einsum('cmi,cij,ckj->cmk', X, self.inv, X)                        # (C, n, n)
        self.at(x0)

    def at(self, x0):
        self.x0 = np.stack([np.ones(x0[0].size)] + [np.asarray(a, 'f8') for a in x0], -1)   # (C, p)
        self.w = np.einsum('ci,cij,ckj->ck', self.x0, self.inv, self.X)                  # (C, n)
        self.lev = np.einsum('ci,cij,cj->c', self.x0, self.inv, self.x0)                  # (C,)
        return self

    def weights(self, x0):
        x = np.stack([np.ones(x0[0].size)] + [np.asarray(a, 'f8') for a in x0], -1)
        return np.einsum('ci,cij,ckj->ck', x, self.inv, self.X)

    def predict(self, Y, w=None):
        """Y (n, C) -> prediction at x0 (C,)"""
        return np.einsum('cm,mc->c', self.w if w is None else w, Y)

    def fitted(self, Y):
        """Y (n, C) -> fitted values (n, C)"""
        return np.einsum('cmk,kc->mc', self.P, Y)


class Constraint:
    """The constraint of one model set on the cells of a grid.

    names: models; dT, CUC: global predictors (n,); dTo, dTs, CUo: observed (mean, sigma of dSST; CUC);
    DTf, CUf: local predictors (n, ny, nx); dTo_f, dTs_f, CUo_f: observed local (ny, nx);
    base: (ny, nx) bool, cells of the area (constraint mask). Local fits exist where base and all local
    predictors are finite (ok); elsewhere (flag 2) the regional state is the multi-model mean."""

    def __init__(self, names, dT, CUC, dTo, dTs, CUo, DTf, CUf, dTo_f, dTs_f, CUo_f, base, CUs=CUC_SIG, CUs_f=None):
        self.names = list(names)
        self.n = len(names)
        dT, CUC = np.asarray(dT, 'f8'), np.asarray(CUC, 'f8')
        self.dT, self.CUC, self.dTo, self.dTs, self.CUo = dT, CUC, float(dTo), float(dTs), float(CUo)
        self.CUs = float(CUs)                     # 1 sigma of the observed CUC (global); per cell CUs_f
        self.gT = GlobalFit([dT], [dTo])
        self.gC = GlobalFit([dT, CUC], [dTo, CUo])
        self.gT_pm = {'dT-': self.gT.weights([dTo - dTs]), 'dT+': self.gT.weights([dTo + dTs])}
        self.gC_pm = {'dT-': self.gC.weights([dTo - dTs, CUo]), 'dT+': self.gC.weights([dTo + dTs, CUo]),
                      'CU-': self.gC.weights([dTo, CUo - self.CUs]), 'CU+': self.gC.weights([dTo, CUo + self.CUs])}
        self.shape = base.shape
        self.base = np.asarray(base).astype(bool)
        DTf, CUf = np.asarray(DTf, 'f8'), np.asarray(CUf, 'f8')
        self.ok = (self.base & np.all(np.isfinite(DTf), 0) & np.all(np.isfinite(CUf), 0)
                   & np.isfinite(dTo_f) & np.isfinite(dTs_f) & np.isfinite(CUo_f))
        # cells whose predictors have no spread across the models cannot be fitted (singular X'X)
        okr = self.ok.ravel()
        sd_T = np.std(DTf.reshape(self.n, -1), axis=0, ddof=1)
        sd_C = np.std(CUf.reshape(self.n, -1), axis=0, ddof=1)
        okr &= np.nan_to_num(sd_T) > 1e-6
        okr &= np.nan_to_num(sd_C) > 1e-8
        self.ok = okr.reshape(self.shape)
        self.iok = np.where(okr)[0]
        sel = lambda a: a.reshape(a.shape[0], -1)[:, self.iok]
        s1 = lambda a: np.asarray(a, 'f8').ravel()[self.iok]
        cus = s1(CUs_f) if CUs_f is not None else np.full(self.iok.size, self.CUs)
        self.cT = CellFit([sel(DTf)], [s1(dTo_f)])
        self.cC = CellFit([sel(DTf), sel(CUf)], [s1(dTo_f), s1(CUo_f)])
        self.cT_pm = {'dT-': self.cT.weights([s1(dTo_f) - s1(dTs_f)]), 'dT+': self.cT.weights([s1(dTo_f) + s1(dTs_f)])}
        self.cC_pm = {'dT-': self.cC.weights([s1(dTo_f) - s1(dTs_f), s1(CUo_f)]),
                      'dT+': self.cC.weights([s1(dTo_f) + s1(dTs_f), s1(CUo_f)]),
                      'CU-': self.cC.weights([s1(dTo_f), s1(CUo_f) - cus]),
                      'CU+': self.cC.weights([s1(dTo_f), s1(CUo_f) + cus])}
        self.flag = np.where(self.ok, 1.0, np.where(self.base, 2.0, np.nan))
        # order of the stacked states sent to the chemistry
        n = self.n
        self.stack = (['loc', 'glob', 'mean'] + [f'rloc{m}' for m in range(n)] + [f'rglob{m}' for m in range(n)]
                      + [f'mod{m}' for m in range(n)]
                      + [f'loc_{k}' for k in ('dT-', 'dT+', 'CU-', 'CU+')] + [f'glob_{k}' for k in ('dT-', 'dT+', 'CU-', 'CU+')])
        self.index = {k: i for i, k in enumerate(self.stack)}
        # with anomalies (states(X, A)): the regional constrained state plus every model's internal variability
        self.stack_iv = self.stack + [f'iv{m}' for m in range(n)]
        self.index_iv = {k: i for i, k in enumerate(self.stack_iv)}

    # ------------------------------------------------------------------------------------------ states
    def states(self, X, A=None):
        """X: {var: (n, ny, nx)} the adjusted states of the n models for one month -> {var: (n_states, ny, nx)}
        in the order of self.stack. NaN cells of any model propagate to the fits (a cell needs every model).
        A: {var: (n, ny, nx)} the models' internal variability of the month (raw minus smoothed): appends the
        states 'iv<m>' = regional constrained state + anomaly of model m (order self.stack_iv)"""
        n, shape = self.n, self.shape
        C = shape[0] * shape[1]
        Y = {v: np.asarray(X[v], 'f8').reshape(n, C) for v in VARS}
        Y['dmc'] = Y['dissic'] - Y['talk']                                   # DIC - ALK
        out = {v: np.full((len(self.stack), C), np.nan) for v in VARS}
        mean = {v: Y[v].mean(0) for v in Y}
        # unconstrained mean and the models themselves
        for v in VARS:
            out[v][self.index['mean']] = mean[v]
            for m in range(n):
                out[v][self.index[f'mod{m}']] = Y[v][m]

        # global weights (applied in every cell): T, DIC-ALK regressed; S, ALK the mean
        def assemble(T, dmc, S=mean['so'], ALK=mean['talk']):
            return {'thetao': T, 'so': S, 'dissic': dmc + ALK, 'talk': ALK}

        gT = self.gT.w @ Y['thetao']
        gC = self.gC.w @ Y['dmc']
        def put(key, st):
            for v in VARS:
                out[v][self.index[key]] = st[v]

        put('glob', assemble(gT, gC))
        for k, w in self.gT_pm.items():
            put(f'glob_{k}', assemble(w @ Y['thetao'], self.gC_pm[k] @ Y['dmc']))
        for k in ('CU-', 'CU+'):
            put(f'glob_{k}', assemble(gT, self.gC_pm[k] @ Y['dmc']))
        fitT, fitC = self.gT.P @ Y['thetao'], self.gC.P @ Y['dmc']           # fitted values per model (n, C)
        for m in range(n):
            put(f'rglob{m}', assemble(gT + (Y['thetao'][m] - fitT[m]), gC + (Y['dmc'][m] - fitC[m]),
                                      mean['so'] + (Y['so'][m] - mean['so']), mean['talk'] + (Y['talk'][m] - mean['talk'])))
        # local weights on the ok cells; the mean elsewhere (flag 2)
        iok = self.iok
        loc = {v: mean[v].copy() for v in VARS}
        Yt, Yd = Y['thetao'][:, iok], Y['dmc'][:, iok]
        lT, lC = self.cT.predict(Yt), self.cC.predict(Yd)
        loc['thetao'][iok], loc['dissic'][iok] = lT, lC + mean['talk'][iok]
        put('loc', loc)
        for k in ('dT-', 'dT+'):
            st = {v: mean[v].copy() for v in VARS}
            st['thetao'][iok] = self.cT.predict(Yt, self.cT_pm[k])
            st['dissic'][iok] = self.cC.predict(Yd, self.cC_pm[k]) + mean['talk'][iok]
            put(f'loc_{k}', st)
        for k in ('CU-', 'CU+'):
            st = {v: mean[v].copy() for v in VARS}
            st['thetao'][iok] = lT
            st['dissic'][iok] = self.cC.predict(Yd, self.cC_pm[k]) + mean['talk'][iok]
            put(f'loc_{k}', st)
        fT, fC = self.cT.fitted(Yt), self.cC.fitted(Yd)
        for m in range(n):
            st = {v: Y[v][m].copy() for v in VARS}                        # flag-2 cells: the model itself
            st['thetao'][iok] = lT + (Yt[m] - fT[m])
            st['dissic'][iok] = lC + (Yd[m] - fC[m]) + mean['talk'][iok] + (Y['talk'][m, iok] - mean['talk'][iok])
            st['so'][iok] = mean['so'][iok] + (Y['so'][m, iok] - mean['so'][iok])
            st['talk'][iok] = mean['talk'][iok] + (Y['talk'][m, iok] - mean['talk'][iok])
            put(f'rloc{m}', st)
        out = {v: a.reshape((len(self.stack),) + shape) for v, a in out.items()}
        if A is not None:
            iloc = self.index['loc']
            iv = {v: out[v][iloc][None] + np.asarray(A[v], 'f8').reshape((n,) + shape) for v in VARS}
            out = {v: np.concatenate([out[v], iv[v]], axis=0) for v in VARS}
        # no state colder than the freezing point of its salinity (the regressions and their perturbations can take
        # an ice-covered cell below it): the state written and the state the chemistry sees are the same
        out['thetao'] = cm.floor_temperature(out['thetao'], out['so'])
        return out

    # ------------------------------------------------------------------------------------------ sigma
    def sigma(self, I, which):
        """I: (n_states, ...) annual means of one indicator for the stacked states; which 'loc' or 'glob'.
        Returns (value, sigma_fit, sigma_pred_dT, sigma_pred_CU, spread_unconstrained, mean_unconstrained)."""
        ix = self.index
        c = I[ix[which]]
        res = np.array([I[ix[f'r{which}{m}']] for m in range(self.n)])
        # a residual state whose chemistry mocsy did not solve (NaN) drops out of the sum, n reduced; fewer than
        # P_FIT + 1 valid states: NaN
        okr = np.isfinite(res)
        n_ok = okr.sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            ss = np.where(n_ok > P_FIT, np.nansum(np.where(okr, (res - c) ** 2, 0.0), axis=0) / np.maximum(n_ok - P_FIT, 1), np.nan)
        if which == 'glob':
            lev = self.gC.lev
        elif np.ndim(c) == 0:                        # an area mean of the regional fields: mean local leverage
            lev = float(np.mean(self.cC.lev)) if self.iok.size else 0.0
        else:                                        # maps: the leverage of the cell (0 on flag-2 cells)
            lev = np.zeros(c.shape)
            lev.reshape(-1)[self.iok] = self.cC.lev
        s_fit = np.sqrt(ss * (1.0 + lev))
        s_dT = 0.5 * np.abs(I[ix[f'{which}_dT+']] - I[ix[f'{which}_dT-']])
        s_CU = 0.5 * np.abs(I[ix[f'{which}_CU+']] - I[ix[f'{which}_CU-']])
        mods = np.array([I[ix[f'mod{m}']] for m in range(self.n)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)          # all models NaN in a cell
            return c, s_fit, s_dT, s_CU, np.nanstd(mods, axis=0, ddof=1), np.nanmean(mods, axis=0)

    def internal(self, I):
        """I: (n_states_iv, ...) annual means of one indicator including the 'iv<m>' states: the sigma of internal
        variability, the root mean square over the models of (indicator of constrained + anomaly) - (constrained)"""
        ix = self.index_iv
        d = np.array([I[ix[f'iv{m}']] - I[ix['loc']] for m in range(self.n)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.sqrt(np.nanmean(d ** 2, axis=0))

    def models(self, I):
        ix = self.index
        return np.array([I[ix[f'mod{m}']] for m in range(self.n)])

    def direct(self, y):
        """the direct regression of a quantity y (n,) on dSST and CUC: (value, sigma, R2)"""
        gc = self.gC
        y = np.asarray(y, 'f8')
        beta = gc.inv @ gc.X.T @ y
        c = float(gc.x0 @ beta)
        e = y - gc.X @ beta
        s2 = e @ e / (self.n - P_FIT)
        s_obs = np.sum((beta[1:] * np.array([self.dTs, self.CUs])) ** 2)
        return c, float(np.sqrt(s2 * (1 + gc.lev) + s_obs)), gc.r2(y)

    def direct_cells(self, Y):
        """the direct regression per cell of fields Y (n, ny, nx): the field of the cell predictions (NaN off ok)"""
        out = np.full(self.shape, np.nan)
        out.reshape(-1)[self.iok] = self.cC.predict(np.asarray(Y, 'f8').reshape(self.n, -1)[:, self.iok])
        return out


def area_mean(field, lat, lon, mask):
    """area-weighted mean of a (..., ny, nx) field over mask & finite (oa_config.area_weights: cell area, or cell area x
    ocean fraction)"""
    w = oa_config.area_weights(lat, lon)
    m = mask & np.isfinite(field)
    wm = w * m
    den = wm.sum(axis=(-2, -1))
    return np.nansum(np.where(m, field, 0.0) * w, axis=(-2, -1)) / np.where(den > 0, den, np.nan)
