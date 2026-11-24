#!/usr/bin/env python
"""
test_alk_residuals.py
=====================
Is the alkalinity change of the models already in the residual spread of the constraint?
For each product (common area): the 18 per-model values of constraint_results.pkl (y1750 = the model's 1750 state
anchored to the product, dT, CUC), the ordinary least-squares fit y1750 ~ dT + CUC across the models, its
residuals, and their regression on the model's salinity-normalized TA change (drift-corrected,
test_alk_diseq_models.py and test_alk_models_trend.py, values copied from the logs of 6 November 2026).
Expected physically: about -0.0024 in Omega_arag per umol/kg (test C of test_disequilibrium_alk.py).
Also: the same with dALK where the pkl has it per model, and the residual standard deviation with and without nTA.

Usage (code/):  OA_RUN=u74 python -u test_alk_residuals.py
"""
import pickle

import numpy as np

import oa_config

# nTA change (umol/kg): 1850-1869 -> 1995-2014 (log 12:02) and -> 2015-2024 (log 12:37, no MRI)
NTA_2014 = {'ACCESS-ESM1-5': 0.27, 'CESM2-WACCM': -0.35, 'CESM2': -0.46, 'CMCC-ESM2': -1.68, 'CNRM-ESM2-1': 2.70,
            'CanESM5-CanOE': -0.87, 'CanESM5': 1.01, 'EC-Earth3-CC': -0.34, 'GFDL-CM4': 4.25, 'GFDL-ESM4': 4.47,
            'MIROC-ES2L': 0.18, 'MPI-ESM1-2-HR': 1.81, 'MPI-ESM1-2-LR': -0.36, 'MRI-ESM2-0': 0.59, 'NorESM2-LM': -0.13,
            'NorESM2-MM': 0.46, 'UKESM1-0-LL': 4.92}
NTA_2024 = {'ACCESS-ESM1-5': 0.13, 'CESM2-WACCM': -0.94, 'CESM2': -1.34, 'CMCC-ESM2': 1.35, 'CNRM-ESM2-1': 2.97,
            'CanESM5-CanOE': -1.43, 'CanESM5': 1.41, 'EC-Earth3-CC': -0.29, 'GFDL-CM4': 6.25, 'GFDL-ESM4': 5.42,
            'IPSL-CM6A-LR': 0.01, 'MIROC-ES2L': 0.03, 'MPI-ESM1-2-HR': 2.45, 'MPI-ESM1-2-LR': -0.14,
            'NorESM2-LM': 0.10, 'NorESM2-MM': 0.17, 'UKESM1-0-LL': 7.54}


def ols(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return b, y - X1 @ b


def main():
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    for ref, r in pk['results_common'].items():
        names = list(r['names'])
        y, dT, cuc = (np.asarray(r[k], 'f8') for k in ('y1750', 'dT', 'CUC'))
        b, res = ols(np.column_stack([dT, cuc]), y)
        print(f'\n===== {ref}: y1750 = {b[0]:.3f} {b[1]:+.4f} dT {b[2]:+.4f} CUC; residual sd {res.std(ddof=3):.4f} '
              f'(constrained sigma {r["constrained"]["dT + CUC"][1]:.4f})')
        if 'dA' in r:
            dA = np.asarray(r['dA'], 'f8')
            print(f'  dA per model: mean {dA.mean():+.4g}, sd {dA.std(ddof=1):.4g}, corr with residual '
                  f'{np.corrcoef(dA, res)[0, 1]:+.2f}')
        for lab, N in (('1850s -> 1995-2014', NTA_2014), ('1850s -> 2015-2024', NTA_2024)):
            sel = np.array([n in N for n in names])
            x = np.array([N[n] for n in np.array(names)[sel]])
            rr = res[sel]
            s, _ = ols(x[:, None], rr)
            c = np.corrcoef(x, rr)[0, 1]
            b2, res2 = ols(np.column_stack([dT[sel], cuc[sel], x]), y[sel])
            print(f'  nTA {lab} ({sel.sum()} models): corr with residual {c:+.2f}, slope {s[1]:+.5f} per umol/kg '
                  f'(physical about -0.0024); residual sd without/with nTA {res[sel].std(ddof=3):.4f} / '
                  f'{res2.std(ddof=4):.4f}; nTA coefficient in the joint fit {b2[3]:+.5f}')
        fb = np.array([n in ('GFDL-CM4', 'GFDL-ESM4', 'UKESM1-0-LL') for n in names])
        print(f'  residuals of the feedback models: {", ".join(f"{n} {v:+.4f}" for n, v in zip(np.array(names)[fb], res[fb]))}; '
              f'mean of the others {res[~fb].mean():+.4f}')


if __name__ == '__main__':
    main()
