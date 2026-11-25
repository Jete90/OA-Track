#!/usr/bin/env python
"""
fit_sigma.py
============
The 1 sigma of the forced state estimated from annual values (the 'sigma_fit' of Extended Data Table 2), for many
series at once and every year of the record: the same estimate as make_oatrack_simple._trend, which takes one series
and one year at a time.

For annual values y (n years) and every year j of the record:
    se(j)    the standard error of the quadratic fit in time, evaluated in year j, from the residual variance and
             their lag-1 autocorrelation r (AR(1)), times the factor by which this estimate underestimates the true
             error (Monte Carlo with AR(1) noise, as make_oatrack_simple._ar1_calibration: rho_c is the autocorrelation
             whose residuals give r as median estimate, the factor the true over the median estimated error at rho_c)
    range(j) the range of the eight estimators of the state of year j: quadratic fit in time and against CO2, Enting
             splines of 15, 20 and 25 years in time and against CO2 (cut-offs converted with the mean growth rate)
    sigma_fit(j) = sqrt(se(j)^2 + (range(j) / 2)^2)
Every estimator is linear in y (a smoother matrix), and the AR(1) error is a sum over the lags of r^lag times fixed
weights, so that all series and years are computed with matrix products. The calibration factor is tabulated over
rho_c (step RHO_STEP) for every year of the record and interpolated; with the same random numbers as
_ar1_calibration (seed 1: the 33 simulations of the median grid, then the one at rho_c), the result equals _trend to
the interpolation (checked by check()).
"""
import numpy as np

import make_oatrack_simple as ms

N_SIM = 20000
GRID = np.linspace(0.0, 0.8, 33)          # as _ar1_calibration
RHO_STEP = 0.0025


def _ar1(e, rho):
    x = np.empty_like(e)
    x[:, 0] = e[:, 0]
    for i in range(1, e.shape[1]):
        x[:, i] = rho * x[:, i - 1] + np.sqrt(1 - rho ** 2) * e[:, i]
    return x


def _lag1(r):
    """the lag-1 autocorrelation of each row as np.corrcoef(r[:-1], r[1:]), clipped to [0, 0.99]"""
    a, b = r[..., :-1] - r[..., :-1].mean(-1, keepdims=True), r[..., 1:] - r[..., 1:].mean(-1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.clip((a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1)), 0.0, 0.99)


class FitSigma:
    def __init__(self, years, xco2):
        """years: the n annual years (consecutive), xco2: their atmospheric CO2"""
        yrs = np.asarray(years, 'f8')
        xc = np.asarray(xco2, 'f8')
        n = self.n = yrs.size
        t = yrs - 2000.0
        gr = (xc[-1] - xc[0]) / (yrs[-1] - yrs[0])
        X = np.vander(t, 3)
        self.Hq = X @ np.linalg.solve(X.T @ X, X.T)                 # row j: the weights wq of the fit in year j
        Xc = np.vander(xc - 400.0, 3)
        H = [self.Hq, Xc @ np.linalg.solve(Xc.T @ Xc, Xc.T)]
        for co in ms.ENT_CUTOFFS:
            H += [ms._smoother(yrs, co), ms._smoother(xc, co * gr)]
        self.H = np.array(H)                                        # (8, n, n)
        # AR(1) error: wq C wq = s2 sum_d r^d G[d, j], G[d, j] = sum over |i - k| = d of wq_j[i] wq_j[k]
        lag = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        self.G = np.array([[(np.outer(self.Hq[j], self.Hq[j]) * (lag == d)).sum() for j in range(n)] for d in range(n)])
        # the calibration of _ar1_calibration, with its random numbers
        rng = np.random.default_rng(1)
        Hq0 = np.vander(np.arange(n, dtype='f8'), 3)
        Hq0 = Hq0 @ np.linalg.pinv(Hq0)
        med = []
        for rho in GRID:
            x = _ar1(rng.standard_normal((N_SIM, n)), rho)
            med.append(np.median(_lag1(x - x @ Hq0.T)))
        self.med = np.array(med)
        e_last = rng.standard_normal((N_SIM, n))                    # the draws of the simulation at rho_c
        self.rho_tab = np.arange(0.0, 0.8 + RHO_STEP / 2, RHO_STEP)
        fac = np.empty((self.rho_tab.size, n))
        for i, rho in enumerate(self.rho_tab):
            x = _ar1(e_last, rho)
            r_ = x - x @ Hq0.T
            rh = _lag1(r_)
            s2 = (r_ ** 2).sum(1) / (n - 3)
            k = slice(0, 2000)
            se = np.sqrt(s2[k, None] * (rh[k, None] ** np.arange(n)[None]) @ self.G)    # (2000, n)
            fac[i] = (x @ self.Hq.T).std(0) / np.median(se, 0)
        self.fac = fac

    def rho_c(self, r_obs):
        return np.interp(r_obs, self.med, GRID)

    def __call__(self, Y):
        """Y (n, ...): annual values; returns sigma_fit (n, ...), missing where a series is incomplete"""
        Y = np.asarray(Y, 'f8')
        sh = Y.shape
        Yf = Y.reshape(self.n, -1)
        ok = np.all(np.isfinite(Yf), 0)
        Z = np.where(ok, Yf, 0.0)
        F = np.einsum('eij,jm->eim', self.H, Z)                    # every estimator in every year
        rng_ = F.max(0) - F.min(0)
        res = Z - self.Hq @ Z
        s2 = (res ** 2).sum(0) / (self.n - 3)
        r1 = np.where(ok, _lag1(res.T), 0.0)                       # incomplete series: missing below
        se_u = np.sqrt(s2[None] * ((r1[:, None] ** np.arange(self.n)[None]) @ self.G).T)    # (n, m)
        rc = self.rho_c(r1)
        x = np.clip((rc - self.rho_tab[0]) / RHO_STEP, 0, self.rho_tab.size - 1.000001)
        i0 = np.floor(x).astype(int)
        w = x - i0
        fac = (1 - w)[None] * self.fac[i0].T + w[None] * self.fac[i0 + 1].T                # (n, m)
        out = np.hypot(se_u * fac, rng_ / 2)
        return np.where(ok[None], out, np.nan).reshape(sh)


def check(years, xco2, Y):
    """sigma_fit of FitSigma against make_oatrack_simple._trend for the series Y (n, m): the largest relative difference"""
    ms.XCO2.update({int(y): float(x) for y, x in zip(years, xco2)})
    fs = FitSigma(years, xco2)
    a = fs(Y)
    worst = 0.0
    for m in range(Y.shape[1]):
        for j, y in enumerate(years):
            e = ms._trend(None, series=(np.asarray(years, 'f8'), Y[:, m]), year=int(y))
            b = np.hypot(e['se'], e['range'] / 2)
            worst = max(worst, abs(a[j, m] - b) / b)
    return worst
