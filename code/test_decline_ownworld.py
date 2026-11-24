#!/usr/bin/env python
"""
test_decline_ownworld.py  (the relative decline 1850 -> 2024 of every ESM in its own world, without bias adjustment,
========================   compared with the chain)
Each model's own states as in the data product: the smoothed, drift-removed series of calc_smooth_states.py
(data/smoothed/smoothed_<model>_<member>.nc) plus the drift level of 1850-1900 (as test_polar_shift_models.py),
annual mean of the monthly slots, Omega_arag with mocsy (constants of the chain, GLODAP Si and PO4, temperature
floored at the freezing point, validity check). Area means on the common area of OceanSODA-ETHZ (results_common of
constraint_results.pkl, mapped to the model grid by nearest neighbour), the same cells in all years.
End state: the year in which the scenario's CO2 equals the observed CO2 of 2024 (NOAA global mean, co2_history.py),
the scenario CO2 rises faster than observed (2024: SSP1-2.6 +2.3, SSP2-4.5 +3.3 ppm); between two years linear in
Omega_arag. Done for SSP1-2.6 and SSP2-4.5, and the plain year 2024 of SSP2-4.5 for comparison.
Decline 1850 -> end state of every model, then regressions on the chain's model dSST and CUC (OceanSODA, common area),
evaluated at the observed values; unconstrained = model mean.
Chain for comparison: the bias-adjusted 1850 state of every model (y_states['1850']) constrained the same way, as
decline to the observed forced 2024 state of OceanSODA (quadratic fit, results_common 'now'); and the best estimate
1750 -> 2024 of the chain. Only for us, no figures.
Usage (code/):  OA_RUN=u74 python -u test_decline_ownworld.py
"""
import pickle
import warnings

import netCDF4
import numpy as np

import oa_config
import calc_omega_arag_obs as co            # puts mocsy on the path
import calc_pi_xco2 as cpx
import calc_smooth_states as css
import co2_history
import drift_tools as dt
import mocsy  # noqa: E402
from feedback_pattern import DRIFT_WINDOW as WINDOW    # (1850, 1900)

VARS = ('thetao', 'so', 'dissic', 'talk')
SCEN = ('ssp126', 'ssp245')
warnings.filterwarnings('ignore', message='Mean of empty slice')

pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
r = pk['results_common']['OceanSODA']
clat, clon, cmask = np.asarray(r['lat'], 'f8'), np.asarray(r['lon'], 'f8'), np.asarray(r['mask'], bool)
cnames = list(r['names'])
CUCm, dTm = np.asarray(r['CUC'], 'f8'), np.asarray(r['dT'], 'f8')
CUC_obs, DT_obs = float(r['CUC_obs']), float(r['dT_obs'][0])

co2 = co2_history.CO2History()
TARGET = co2.noaa[2024]


def end_year(s):
    """fractional year in which the CO2 of scenario s equals TARGET"""
    yy = np.arange(2015, 2031)
    c = np.array([co2.ssp[s][y] for y in yy])
    return float(np.interp(TARGET, c, yy))


T_END = {s: end_year(s) for s in SCEN}
print(f'observed CO2 2024 (NOAA) {TARGET:.2f} ppm; reached in ' + ', '.join(f'{s} {T_END[s]:.2f}' for s in SCEN)
      + f'; observed dSST {DT_obs:.3f} degC, CUC {CUC_obs:.4f}')


def omega(st, m, si, po4, la):
    t = co.cm.floor_temperature(st['thetao'][m], st['so'][m], count=False)
    n = int(m.sum())
    o = mocsy.mvars(temp=t, sal=st['so'][m], alk=st['talk'][m] * 1e-6, dic=st['dissic'][m] * 1e-6, sil=si[m] * 1e-6,
                    phos=po4[m] * 1e-6, patm=np.ones(n), depth=np.zeros(n), lat=la[m], optcon='mol/kg',
                    optt='Tpot', optp='m', optb=oa_config.OPTB, optk1k2=oa_config.OPTK1K2)
    ph, om = np.asarray(o[0], 'f8'), np.asarray(o[6], 'f8')
    out = np.full(m.shape, np.nan)
    out[m] = np.where(co.cm.valid_carbonate(ph, om), om, np.nan)
    return out


def model_levels(model, member):
    """area means of Omega_arag on the common area: 1850, and per scenario the years around the end state"""
    d = dt.Drift(model, member)
    with netCDF4.Dataset(css.out_path(model, member)) as f:
        lat, lon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        yh, ys = np.asarray(f['year_hist'][:]).astype(int), np.asarray(f['year_scen'][:]).astype(int)
        off = {v: d.window_mean(v, WINDOW) for v in VARS}
        need = {('hist', 1850)}
        for s in SCEN:
            if f'thetao_{s}' in f.variables:
                need |= {(s, int(np.floor(T_END[s]))), (s, int(np.floor(T_END[s])) + 1)}
        if 'thetao_ssp245' in f.variables:
            need |= {('ssp245', 2024)}
        st = {}
        for k, y in need:
            yrs = yh if k == 'hist' else ys
            st[(k, y)] = {v: np.ma.filled(f[f'{v}_{k}'][int(np.searchsorted(yrs, y))], np.nan).astype('f8').mean(axis=0)
                          + off[v] for v in VARS}
    jy = np.abs(lat[:, None] - clat[None, :]).argmin(1)
    ix = np.abs(((lon[:, None] - clon[None, :]) + 180) % 360 - 180).argmin(1)
    mc = cmask[np.ix_(jy, ix)]
    si, po4 = co.load_nutrients(lat, lon)
    la = np.broadcast_to(lat[:, None], mc.shape)
    w = oa_config.area_weights(lat, lon)
    ok = mc & np.isfinite(si) & np.isfinite(po4) & np.all(
        [np.isfinite(s_[v]) & (s_[v] != 0 if v != 'thetao' else True) for s_ in st.values() for v in VARS], axis=0)
    om = {k: omega(s_, ok, si, po4, la) for k, s_ in st.items()}
    ok &= np.all([np.isfinite(o) for o in om.values()], axis=0)
    am = {k: float(np.sum(o[ok] * w[ok]) / np.sum(w[ok])) for k, o in om.items()}
    out = {'1850': am[('hist', 1850)], 'area': float(np.sum(w[ok]) / np.sum(w[mc]))}
    for s in SCEN:
        y0 = int(np.floor(T_END[s]))
        if (s, y0) in am:
            out[s] = am[(s, y0)] + (T_END[s] - y0) * (am[(s, y0 + 1)] - am[(s, y0)])
    if ('ssp245', 2024) in am:
        out['2024'] = am[('ssp245', 2024)]
    return out


cols = ('ssp126', 'ssp245', '2024')
rows = {}
print(f'\n{"model":16s} {"1850":>6s}  decline to: {"SSP1-2.6@CO2":>12s} {"SSP2-4.5@CO2":>12s} {"SSP2-4.5 2024":>13s}  area')
for model, member in cpx.find_runs(None):
    if model not in cnames:
        continue
    try:
        L = model_levels(model, member)
    except Exception as e:
        print(f'{model:16s} failed: {type(e).__name__}: {e}', flush=True)
        continue
    dec = {c: 100 * (1 - L[c] / L['1850']) if c in L else np.nan for c in cols}
    rows[model] = dec
    print(f'{model:16s} {L["1850"]:6.3f}  {"":11s} ' + ' '.join(f'{dec[c]:12.2f}' for c in cols[:2])
          + f' {dec["2024"]:13.2f}  {100 * L["area"]:5.1f}%', flush=True)


def con(y, xs, x0):
    X = np.column_stack([np.ones(len(y))] + xs)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    return float(np.r_[1.0, x0] @ b), 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))


print('\nown world, decline 1850 -> end state (%): unconstrained, then constrained (R2)')
for c in cols:
    ms = [m for m in rows if np.isfinite(rows[m][c])]
    y = np.array([rows[m][c] for m in ms])
    i = [cnames.index(m) for m in ms]
    cu, t = CUCm[i], dTm[i]
    out = [f'{len(ms)} models, mean {y.mean():5.2f} +- {y.std(ddof=1):.2f}']
    for lab, xs, x0 in (('dSST', [t], [DT_obs]), ('CUC', [cu], [CUC_obs]), ('dSST+CUC', [t, cu], [DT_obs, CUC_obs])):
        v, r2 = con(y, xs, x0)
        out.append(f'{lab} {v:5.2f} (R2 {r2:.2f})')
    print(f'  {c:7s}: ' + '; '.join(out))

now = float(r['now'])
print(f'\nchain (bias-adjusted, OceanSODA, common area), decline to the observed forced 2024 state {now:.3f}:')
ys_ = r['y_states']
if '1850' in ys_:
    y = np.asarray(ys_['1850'], 'f8')
    out = [f'unconstrained {100 * (1 - now / y.mean()):5.2f}']
    for lab, xs, x0 in (('dSST', [dTm], [DT_obs]), ('CUC', [CUCm], [CUC_obs]), ('dSST+CUC', [dTm, CUCm], [DT_obs, CUC_obs])):
        v, r2 = con(y, xs, x0)
        out.append(f'{lab} {100 * (1 - now / v):5.2f} (R2 {r2:.2f})')
    print('  1850 -> 2024: ' + '; '.join(out))
else:
    print(f'  no 1850 state in the pickle (states {list(ys_)})')
yc = r['constrained']['dT + CUC'][0]
print(f'  1750 -> 2024, best estimate of the chain: {100 * (1 - now / yc):.2f} (1750 state {yc:.3f})')
