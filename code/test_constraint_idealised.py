#!/usr/bin/env python
"""
test_constraint_idealised.py
============================
Idealised (perfect-knowledge) test of the emergent constraint for every carbonate system indicator: does the
regression on dSST and CUC recover the true 1750 value, and what does it do to the uncertainty?

Synthetic world, one surface regime at a time (tropical, mid-latitude, polar), repeated NREP times:
  truth   the real ocean: window state (T_o, ALK_o, DIC_o with pCO2 = P_WIN - lag); true 1750 state at
          T_o - dT_true, ALK_o, in equilibrium with 278.3 ppm
  models  N models, each with a present-day bias in T (sd 0.8 K) and ALK (sd 40 umol/kg), its own warming dT_m
          (window minus preindustrial, mean 0.75, sd 0.15 K) and disequilibrium lag (mean 8, sd 4 uatm);
          piControl state in equilibrium with 284.3 ppm at T_w - dT_m; window state with pCO2 = P_WIN - lag_m at T_w
  chain   as in the paper: adjusted piControl = piControl + (obs - model window) in T, ALK and DIC; DIC then lowered
          so that pCO2 drops by 284.3 - 278.3 ppm; every indicator from that state (mocsy with the constants of the
          run, oa_config.py, as the chain; DIC for a given pCO2 by Newton iteration with the Revelle factor, as
          calc_pi_xco2.py)
  predictors  dSST_m = dT_m (observed: dT_true + noise, sd 0.05 K), CUC_m = DIC/(R pCO2_atm) of the raw model
          window state (observed: of the obs window, sd 0.001)
  constraint  y_m = a + b dSST_m + c CUC_m, evaluated at the observed predictors; sigma as in the chain (residual
          with leverage, observed predictors)
  null test   the same with two random predictors unrelated to the models (a constraint "by accident")

Printed per regime and indicator (over the NREP realisations, relative to the true 1750 value):
  RMSE of the unconstrained mean and of the constrained value, mean sigma of both, coverage (fraction of truths
  within 1 sigma, 0.68 expected), mean R2, and the same for the null test.

Usage:  python test_constraint_idealised.py [--nrep 400] [--nmod 18]
"""
import argparse

import numpy as np

import oa_config
import calc_omega_arag_obs as co     # mocsy path
import calc_omega_arag as cm         # freezing-point floor
import calc_pi_xco2 as cpx           # total to free scale
import mocsy

P_WIN, X_PI, X_1750 = 360.0, 284.317, 278.3
S = 35.0
REGIMES = {'tropical': (27.0, 2300.0), 'mid-latitude': (15.0, 2300.0), 'polar': (0.0, 2290.0)}
SIL, PHOS = 5.0, 0.5                 # umol/kg
N_ITER_MAX, PCO2_TOL = 30, 1e-6      # Newton iterations for DIC at a given pCO2 (uatm)
NAMES = ('omega_arag', 'omega_calc', 'ph', 'hplus', 'hplus_free', 'pco2', 'co2', 'hco3', 'co3', 'revelle', 'dic')


def _mvars(alk, dic, t):
    """mocsy as in the chain (surface, 1 atm, potential temperature, constants of the run); alk, dic in umol/kg,
    any shape; returns the mvars output list (concentrations in mol/kg) and the shape"""
    alk, dic, t = np.broadcast_arrays(np.asarray(alk, 'f8'), np.asarray(dic, 'f8'), np.asarray(t, 'f8'))
    n = alk.size
    s = np.full(n, S)
    r = mocsy.mvars(temp=cm.floor_temperature(t.ravel(), s, count=False), sal=s, alk=alk.ravel() * 1e-6,
                    dic=dic.ravel() * 1e-6, sil=np.full(n, SIL * 1e-6), phos=np.full(n, PHOS * 1e-6),
                    patm=np.ones(n), depth=np.zeros(n), lat=np.zeros(n), optcon='mol/kg', optt='Tpot', optp='m',
                    optb=oa_config.OPTB, optk1k2=oa_config.OPTK1K2)
    return [np.asarray(x, 'f8').reshape(alk.shape) for x in r], alk.shape


def dic_from(alk, pco2, t):
    """DIC (umol/kg) with surface pCO2 = pco2 (uatm)"""
    alk, pco2, t = np.broadcast_arrays(np.asarray(alk, 'f8'), np.asarray(pco2, 'f8'), np.asarray(t, 'f8'))
    d = 0.9 * alk
    for _ in range(N_ITER_MAX):
        r, _ = _mvars(alk, d, t)
        p, rv = r[1], r[8]
        if np.max(np.abs(p - pco2)) < PCO2_TOL:
            break
        d = d + (pco2 - p) * d / (rv * p)                     # Newton with the Revelle factor (as calc_pi_xco2.py)
    else:
        raise RuntimeError(f'DIC not converged: max |pCO2 - target| {np.max(np.abs(p - pco2)):.2e} uatm')
    return d


def chem(alk, dic, t):
    """all indicators (NAMES) of the state, concentrations in umol/kg, [H+] in nmol/kg"""
    r, _ = _mvars(alk, dic, t)
    t_ = np.broadcast_to(np.asarray(t, 'f8'), r[0].shape)
    h = 10 ** -r[0] * 1e9                                     # [H+] total scale
    return dict(omega_arag=r[6], omega_calc=r[7], ph=r[0], hplus=h, hplus_free=h * cpx.total_to_free(t_, S),
                pco2=r[1], co2=r[3] * 1e6, hco3=r[4] * 1e6, co3=r[5] * 1e6, revelle=r[8],
                dic=np.broadcast_to(np.asarray(dic, 'f8'), r[0].shape))


def fit(X, y, x0, sx0):
    """regression of the chain: value at x0 and sigma (residual with leverage, observed predictors)"""
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    s2 = e @ e / (n - p)
    lev = x0 @ np.linalg.inv(X.T @ X) @ x0
    sig = np.sqrt(s2 * (1 + lev) + np.sum((beta[1:] * sx0) ** 2))
    r2 = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    return x0 @ beta, sig, r2


def run(regime, nrep, nmod, rng, lag_sd=4.0, alk_sd=40.0):
    T0, A0 = REGIMES[regime]
    # ---- truth, one per realisation
    dT_true = rng.normal(0.85, 0.10, nrep)
    lag_true = rng.normal(8.0, lag_sd, nrep)
    T_o = np.full(nrep, T0)
    A_o = np.full(nrep, A0)
    D_o = dic_from(A_o, P_WIN - lag_true, T_o)
    truth = chem(A_o, dic_from(A_o, np.full(nrep, X_1750), T_o - dT_true), T_o - dT_true)
    r_o = chem(A_o, D_o, T_o)
    cuc_o = D_o / (r_o['revelle'] * P_WIN)
    # ---- models (nrep x nmod)
    sh = (nrep, nmod)
    T_w = T0 + rng.normal(0, 0.8, sh)
    A_m = A0 + rng.normal(0, alk_sd, sh)
    dT = rng.normal(0.75, 0.15, sh)
    lag = rng.normal(8.0, lag_sd, sh)
    D_pi = dic_from(A_m, np.full(sh, X_PI), T_w - dT)
    D_w = dic_from(A_m, P_WIN - lag, T_w)
    r_w = chem(A_m, D_w, T_w)
    cuc = D_w / (r_w['revelle'] * P_WIN)
    # ---- chain: bias adjustment of the piControl state, then DIC lowered to 1750
    T_a = (T_w - dT) + (T_o[:, None] - T_w)
    A_a = A_m + (A_o[:, None] - A_m)
    D_a = D_pi + (D_o[:, None] - D_w)
    p_a = chem(A_a, D_a, T_a)['pco2']
    D_1750 = dic_from(A_a, p_a - (X_PI - X_1750), T_a)
    Y = chem(A_a, D_1750, T_a)
    # ---- observed predictors with their errors
    dT_obs = dT_true + rng.normal(0, 0.05, nrep)
    cuc_obs = cuc_o + rng.normal(0, 0.001, nrep)
    sets = {'dSST': ([dT], [dT_obs], [0.05]), 'CUC': ([cuc], [cuc_obs], [0.001]),
            'both': ([dT, cuc], [dT_obs, cuc_obs], [0.05, 0.001])}
    nul = rng.normal(0, 1, (nrep, nmod, 2)), rng.normal(0, 1, (nrep, 2))      # unrelated predictors
    out = {}
    for k in NAMES:
        y, tr = Y[k], truth[k]
        rel = 100.0 / np.abs(tr)
        u, su = y.mean(1), y.std(1, ddof=1)
        o = dict(truth=np.mean(tr), rmse_u=np.sqrt(np.mean(((u - tr) * rel) ** 2)), sig_u=np.mean(su * rel),
                 cov_u=np.mean(np.abs(u - tr) < su))
        for lab, (P, Po, sp) in {**sets, 'null': ([nul[0][..., 0], nul[0][..., 1]], [nul[1][:, 0], nul[1][:, 1]],
                                                  [0.3, 0.3])}.items():
            c, sc, r2 = np.empty(nrep), np.empty(nrep), np.empty(nrep)
            for i in range(nrep):
                X = np.column_stack([np.ones(nmod)] + [p_[i] for p_ in P])
                c[i], sc[i], r2[i] = fit(X, y[i], np.array([1.0] + [q[i] for q in Po]), np.array(sp))
            o[lab] = dict(rmse=np.sqrt(np.mean(((c - tr) * rel) ** 2)), sig=np.mean(sc * rel),
                          cov=np.mean(np.abs(c - tr) < sc), r2=np.mean(r2))
        out[k] = o
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nrep', type=int, default=400)
    ap.add_argument('--nmod', type=int, default=18)
    ap.add_argument('--lag-sd', type=float, default=4.0, help='sd of the air-sea disequilibrium (uatm)')
    ap.add_argument('--alk-sd', type=float, default=40.0, help='sd of the model ALK bias (umol/kg)')
    ap.add_argument('--regimes', nargs='+', default=list(REGIMES))
    a, _ = ap.parse_known_args()
    rng = np.random.default_rng(1750)
    for regime in a.regimes:
        o = run(regime, a.nrep, a.nmod, rng, a.lag_sd, a.alk_sd)
        print(f'\n{regime} ({REGIMES[regime][0]:.0f} C), lag sd {a.lag_sd} uatm, ALK bias sd {a.alk_sd}, {a.nmod} models, {a.nrep} realisations; RMSE and sigma in % of the '
              f'true 1750 value, coverage = fraction of truths within 1 sigma')
        print('  RMSE of the constrained value relative to the unconstrained one (< 1: the constraint helps), '
              'mean R2 in brackets (random predictors: about p/(n-1))')
        print(f'  {"indicator":11s} {"unconstr. RMSE %":>16s} | {"dSST only":>12s} {"CUC only":>12s} {"both":>12s} | '
              f'{"random (2)":>12s} | coverage both')
        for k, v in o.items():
            cells = ' '.join(f'{v[l]["rmse"] / v["rmse_u"]:5.2f} ({v[l]["r2"]:.2f})' for l in ('dSST', 'CUC', 'both'))
            print(f'  {k:11s} {v["rmse_u"]:16.3f} | {cells} | {v["null"]["rmse"] / v["rmse_u"]:5.2f} ({v["null"]["r2"]:.2f}) '
                  f'| {v["both"]["cov"]:.2f}')

if __name__ == '__main__':
    main()
