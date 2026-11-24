#!/usr/bin/env python
"""
cuc_mapping_uncertainty.py
==========================
The mapping uncertainty of the observed CUC (the predictor of the emergent constraint) and its correlation with the
mapping error of the observed Omega_arag, per product, on its own area and on the common area.

The observed CUC is <DIC / R> / pCO2_atm of the product's window state (constraint_refs.py; windows of
oa_config.WINDOWS), R the Revelle factor. The models' CUC comes from their own states, so a mapping error
of the product moves only the observed value. The same errors also move the observed Omega_arag of the window, and
through the bias adjustment every model's 1750 state. Both are computed from the same error fields, so that their
correlation is kept:

    the error fields  the window mean of the product's state errors per mapping component c (fCO2, TA, SST, SSS
                      with their decorrelation lengths L_c), as stored by calc_product_obs.py in
                      data/<run>/product/obs_<ref>.nc (level_dx_*, level_uncertainty.py); the same pattern in all
                      months (C2, conservative; C1, months independent, is reported for comparison)
    per cell          e_CUC = dDIC / R - DIC / R^2 * dR,  e_Omega = dOmega_arag, with dR and dOmega_arag from the
                      derivatives of mocsy at the window state (level_uncertainty.jacobian)
    area means        quadratic form of the box sums with exp(-d / L_c) between the box centres (BoxCov)

    var(CUC), var(Omega), cov(CUC, Omega): sums over the components (independent of each other)

The constrained 1750 value moves by dOmega_1750 = dOmega (bias adjustment) + beta_CUC dCUC (the constraint), with
beta_CUC the coefficient of CUC in the regression of the models' 1750 Omega_arag on dSST and CUC. The script reports
var(Omega_1750) = var(Omega) + beta^2 var(CUC) + 2 beta cov(CUC, Omega) and the correlation, which the notebook
(Fig. 4 cell, Tables S1 and S3) combines with the 2024 mapping uncertainty (product_uncertainty_mc_C.py).

The set value of constraint_refs.CUC_SIG (0.001, the difference between two products on the common mask, already
contained in the spread across the products) is printed for comparison.

Per cell (fields, for the cell-by-cell constraint): the same errors without the area mean (inside a cell the errors
of a component are fully correlated, components independent): the local sigma of CUC and of the window Omega_arag
and their correlation; the local CUC coefficient beta_CUC of the cell-by-cell regression (local dSST and CUC, as
regional_common.local_constraint) and the shift beta_CUC * sigma_CUC; the current local constraint sigma (with the
set value 0.001) and the same with the local mapping sigma of CUC in its place.

Needs: constraint_results.pkl, the mechanism window states (state_<ref>.nc), data/<run>/product/obs_<ref>.nc with the
level errors (calc_product_obs.py).
Output: data/<run>/product_uncertainty/cuc_mapping_<ref>.npz (global values and, per area, the fields)
Usage (code folder):  OA_RUN=u74 python -u cuc_mapping_uncertainty.py [--reference OceanSODA ...]
"""
import argparse
import pickle

import netCDF4
import numpy as np

import oa_config
import calc_omega_arag_obs as co
import level_uncertainty as lu
import regional_common as rc

# window and mean atmospheric pCO2 of the window per reference, as in constraint_refs.py (not imported: it reads the
# SST products on import); the recomputed observed CUC is checked against constraint_results.pkl, which catches any
# difference
VARS = ('thetao', 'so', 'dissic', 'talk')
CUC_SIG = 0.001                       # constraint_refs.CUC_SIG
_D = str(oa_config.DATA)
REFS = {'OceanSODA': dict(mech=f'{_D}/mechanism', win=oa_config.WINDOWS['OceanSODA'], pco2=oa_config.WINDOW_PCO2['OceanSODA']),
        'UExP': dict(mech=f'{_D}/mechanism_UExP', win=oa_config.WINDOWS['UExP'], pco2=oa_config.WINDOW_PCO2['UExP']),
        'JMA': dict(mech=f'{_D}/mechanism_JMA', win=oa_config.WINDOWS['JMA'], pco2=oa_config.WINDOW_PCO2['JMA']),
        'CMEMS': dict(mech=f'{_D}/mechanism_CMEMS', win=oa_config.WINDOWS['CMEMS'], pco2=oa_config.WINDOW_PCO2['CMEMS']),
        'Jiang': dict(mech=f'{_D}/mechanism_Jiang', win=oa_config.WINDOWS['Jiang'], pco2=oa_config.WINDOW_PCO2['Jiang'])}   # constraint_results_jiang.pkl


def read_state(path):
    with netCDF4.Dataset(path) as f:
        lat, lon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        return lat, lon, {v: np.ma.filled(f[v][:].astype('f8'), np.nan) for v in VARS}


def read_errors(path, lat, lon):
    """{component: (L, {x: annual C2 error field}, {x: (12, ny, nx) monthly})} from the product's obs file"""
    got = lu.read_dx(path, lat, lon)
    if got is None:
        raise SystemExit(f'{path}: no level errors (level_dx_*); run calc_product_obs.py for this reference first')
    names, dX = got
    with netCDF4.Dataset(path) as f:
        L = dict(zip([str(s) for s in f['component'][:]], np.asarray(f['component_length'][:], 'f8')))
        window = getattr(f, 'level_window', '')
    out = {}
    for c in names:
        mon = {x: dX[c][x] for x in lu.DX}
        ann = {x: np.where(np.isfinite(a).any(0), np.nanmean(a, 0), np.nan) for x, a in mon.items()}
        out[c] = (float(L[c]), ann, mon)
    return out, window


def per_cell(J, base, dx):
    """the errors of DIC/R and of Omega_arag in a cell for state errors dx {x: field}"""
    dR = sum(J['revelle'][x] * np.nan_to_num(dx[x]) for x in lu.DX)
    dO = sum(J['omega_arag'][x] * np.nan_to_num(dx[x]) for x in lu.DX)
    R, D = base['revelle'], base['dic']
    return np.nan_to_num(dx['dissic']) / R - D / R ** 2 * dR, dO


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', nargs='+', default=None)
    a = ap.parse_args()
    data = oa_config.DATA
    with open(data / 'constraint_results.pkl', 'rb') as f:
        PK = pickle.load(f)
    if (data / 'constraint_results_jiang.pkl').exists():           # Jiang et al. (2023): own and common area
        with open(data / 'constraint_results_jiang.pkl', 'rb') as f:
            J = pickle.load(f)
        for key, jk in (('results', 'own'), ('results_common', 'common')):
            if 'Jiang' in J.get(jk, {}):
                PK[key]['Jiang'] = J[jk]['Jiang']
    refs = a.reference or [q for q in REFS if q in PK['results'] or q in PK['results_common']]
    out_dir = data / 'product_uncertainty'
    out_dir.mkdir(parents=True, exist_ok=True)
    for ref in refs:
        cfg = REFS[ref]
        lat, lon, obs = read_state(f"{cfg['mech']}/state_{ref}.nc")
        lat2 = np.broadcast_to(lat[:, None], (lat.size, lon.size))
        si, po4 = co.load_nutrients(lat, lon)
        si, po4 = np.where(si < 0, 0.0, si), np.where(po4 < 0, 0.0, po4)
        base, J = lu.jacobian(obs, si, po4, lat2)
        errs, window = read_errors(data / 'product' / f'obs_{ref}.nc', lat, lon)
        print(f'\n===== {ref}: window {cfg["win"][0]}-{cfg["win"][1]} (level errors: window {window}), '
              f'{len(errs)} mapping components: ' + ', '.join(f'{c} (L {v[0]:.0f} km)' for c, v in errs.items()), flush=True)
        if window and window != f'{cfg["win"][0]}-{cfg["win"][1]}':
            print(f'  NOTE: the level errors are the window mean of {window}, the constraint uses {cfg["win"]}')
        res = {}
        for area, key in (('own', 'results'), ('common', 'results_common')):
            if ref not in PK[key]:
                continue
            R = PK[key][ref]
            if not (np.allclose(np.asarray(R['lat'], 'f8'), lat) and np.allclose(np.asarray(R['lon'], 'f8'), lon)):
                raise SystemExit(f'{ref} {area}: grid of constraint_results.pkl differs from state_{ref}.nc')
            mask = np.asarray(R['mask']).astype(bool)
            ok = mask & np.isfinite(base['revelle']) & np.isfinite(base['omega_arag']) & np.isfinite(base['dic'])
            w = oa_config.area_weights(lat, lon) * ok
            W = float(w.sum())
            cuc_obs = float(np.sum(np.where(ok, base['dic'] / base['revelle'], 0) * w) / W) / cfg['pco2']
            # beta of CUC (and dSST) in the regression of the models' 1750 Omega_arag
            X = np.column_stack([np.ones(len(R['names'])), np.asarray(R['dT'], 'f8'), np.asarray(R['CUC'], 'f8')])
            beta = np.linalg.lstsq(X, np.asarray(R['y1750'], 'f8'), rcond=None)[0]
            vC = {'C1': 0.0, 'C2': 0.0}
            vO = {'C1': 0.0, 'C2': 0.0}
            cv = {'C1': 0.0, 'C2': 0.0}
            for c, (L, ann, mon) in errs.items():
                bc = lu.BoxCov(lat, lon, L, ok)
                op = bc.operator(w)
                eC, eO = per_cell(J, base, ann)
                EC, EO = op @ np.where(ok, eC, 0).ravel(), op @ np.where(ok, eO, 0).ravel()
                vC['C2'] += float(EC @ bc.K @ EC)
                vO['C2'] += float(EO @ bc.K @ EO)
                cv['C2'] += float(EC @ bc.K @ EO)
                for m in range(12):                       # C1: months independent (annual mean of 12 draws)
                    eCm, eOm = per_cell(J, base, {x: mon[x][m] for x in lu.DX})
                    ECm, EOm = op @ np.where(ok, eCm, 0).ravel(), op @ np.where(ok, eOm, 0).ravel()
                    vC['C1'] += float(ECm @ bc.K @ ECm) / 144.0
                    vO['C1'] += float(EOm @ bc.K @ EOm) / 144.0
                    cv['C1'] += float(ECm @ bc.K @ EOm) / 144.0
            r = {}
            for v in ('C1', 'C2'):
                sC = np.sqrt(vC[v]) / W / cfg['pco2']
                sO = np.sqrt(vO[v]) / W
                cov = cv[v] / W ** 2 / cfg['pco2']
                rho = cov / (sC * sO) if sC > 0 and sO > 0 else np.nan
                s1750 = np.sqrt(max(sO ** 2 + beta[2] ** 2 * sC ** 2 + 2 * beta[2] * cov, 0.0))
                r[v] = dict(sigma_cuc=sC, sigma_omega_window=sO, cov=cov, rho=rho, sigma_omega_1750=s1750)
            # per cell (C2): components independent, fully correlated within a cell
            sC2 = np.zeros(ok.shape)
            sO2 = np.zeros(ok.shape)
            cvc = np.zeros(ok.shape)
            for c, (L, ann, mon) in errs.items():
                eC, eO = per_cell(J, base, ann)
                sC2 += eC ** 2
                sO2 += eO ** 2
                cvc += eC * eO
            sC_f = np.where(ok, np.sqrt(sC2) / cfg['pco2'], np.nan)
            sO_f = np.where(ok, np.sqrt(sO2), np.nan)
            rho_f = np.where(ok & (sC2 > 0) & (sO2 > 0), cvc / np.sqrt(np.maximum(sC2 * sO2, 1e-300)), np.nan)
            F = R['fields']
            Yf, DTf, CUf = (np.asarray(F[k], 'f8') for k in ('y1750', 'dT', 'CUC'))
            dso = np.array([np.asarray(v, 'f8') for v in F['dsst_obs'].values()])
            dTo_f, dTs_f = dso.mean(0), dso.std(0, ddof=1)
            CUo_f = np.asarray(F['CUC_obs'], 'f8')
            okc = ok & np.all(np.isfinite(Yf), 0) & np.all(np.isfinite(DTf), 0) & np.all(np.isfinite(CUf), 0) \
                & np.isfinite(dTo_f) & np.isfinite(dTs_f) & np.isfinite(CUo_f)
            iw = np.where(okc.ravel())[0]
            flat = lambda A: A.reshape(A.shape[0], -1)[:, iw]
            yc, st, b = rc.local_constraint(flat(Yf), flat(DTf), flat(CUf), dTo_f.ravel()[iw], dTs_f.ravel()[iw],
                                            CUo_f.ravel()[iw], None, return_beta=True)
            def to_field(v):
                o = np.full(okc.size, np.nan)
                o[iw] = v
                return o.reshape(okc.shape)
            beta_f = to_field(b[:, 2])
            st_old = to_field(st)
            sC_c = sC_f.ravel()[iw]
            st_new = to_field(np.sqrt(np.maximum(st ** 2 - (b[:, 2] * rc.CUC_SIG) ** 2, 0) + (b[:, 2] * sC_c) ** 2))
            shift = np.abs(beta_f) * sC_f
            smap1750 = np.sqrt(np.maximum(sO_f ** 2 + beta_f ** 2 * sC_f ** 2 + 2 * beta_f * rho_f * sO_f * sC_f, 0))
            wc = oa_config.area_weights(lat, lon) * okc
            pct = lambda a, q: float(np.nanpercentile(a[okc], q))
            awm = lambda a: float(np.nansum(np.where(okc, a, 0) * wc) / wc[np.isfinite(a) & okc].sum())
            rel = st_new / st_old - 1
            print(f'    cells ({int(okc.sum())}): local sigma CUC median {pct(sC_f, 50):.4f} (5-95 % {pct(sC_f, 5):.4f}-'
                  f'{pct(sC_f, 95):.4f}); local beta_CUC median {pct(beta_f, 50):.2f} ({pct(beta_f, 5):.2f} to {pct(beta_f, 95):.2f}); '
                  f'shift |beta| sigma median {pct(shift, 50):.4f}, 95 % {pct(shift, 95):.4f}')
            print(f'    local constraint sigma: without observed CUC (regional_common.CUC_SIG {rc.CUC_SIG}) area mean {awm(st_old):.4f}, with the local mapping '
                  f'sigma of CUC {awm(st_new):.4f}; increase median {100 * pct(rel, 50):.1f} %, 95 % {100 * pct(rel, 95):.1f} %, '
                  f'area with more than 10 %: {100 * float(np.sum(wc * (rel > 0.10)) / wc.sum()):.1f} %; '
                  f'correlation CUC-Omega median {pct(rho_f, 50):+.2f}')
            fields = dict(sigma_cuc=sC_f, sigma_omega_window=sO_f, rho=rho_f, beta_cuc=beta_f, shift=shift,
                          sigma_constraint_set=st_old, sigma_constraint_mapping=st_new, sigma_map_1750=smap1750,
                          mask=okc, lat=lat, lon=lon)
            res[area] = dict(cuc_obs=cuc_obs, cuc_obs_pkl=float(R['CUC_obs']), beta_cuc=float(beta[2]),
                             beta_dsst=float(beta[1]), **{f'{k}_{v}': x for v in r for k, x in r[v].items()},
                             **{f'field_{k}': v for k, v in fields.items()})
            if abs(cuc_obs - float(R['CUC_obs'])) > 1e-4:
                raise SystemExit(f'{ref} {area}: observed CUC {cuc_obs:.5f} differs from the pkl {float(R["CUC_obs"]):.5f}: '
                                 'window state, mask or pCO2 not as in constraint_refs.py')
            print(f'  {area:6s} CUC_obs {cuc_obs:.5f} (pkl {float(R["CUC_obs"]):.5f}); beta_CUC {beta[2]:.3f} per unit CUC')
            for v in ('C2', 'C1'):
                x = r[v]
                print(f'    {v}: sigma CUC {x["sigma_cuc"]:.5f} (set value {CUC_SIG}), -> 1750 Omega {abs(beta[2]) * x["sigma_cuc"]:.4f}; '
                      f'sigma Omega window {x["sigma_omega_window"]:.4f}; correlation {x["rho"]:+.2f}; '
                      f'together in the 1750 value {x["sigma_omega_1750"]:.4f}')
        path = out_dir / f'cuc_mapping_{ref}.npz'
        np.savez(path, **{f'{area}_{k}': v for area, d in res.items() for k, v in d.items()},
                 window=str(window), cuc_sig_set=CUC_SIG, pco2=cfg['pco2'])
        print(f'  saved {path}', flush=True)


if __name__ == '__main__':
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
