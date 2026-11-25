#!/usr/bin/env python
"""
Uncertainty of the global-mean surface Omega_arag from the equilibrium constants (Orr et al. 2018),
for the UExP-FNN-U v2026-1 2024 state, and its effect on the relative decline since 1750.

mocsy.merrors (method of moments, numerical derivatives) is called once per constant with only that
constant's uncertainty non-zero (pK0 0.002, pK1 0.0075, pK2 0.015, pKb 0.01, pKw 0.01, pKspa 0.02,
pKspc 0.02; total boron 2%). The constants are the same everywhere, so each constant's error is
fully correlated: its global-mean 1 sigma is the area-weighted mean of the cell values. Constants
combined in quadrature (independent of each other).

Effect on the relative decline D = 1 - O2024/O1750, with the 1750 states of the paper: for every
model, one piControl year (12 months) is drift-corrected, bias-adjusted and DIC lowered to the 1750 CO2 exactly
as in calc_pi_xco2.py (adjusted_fields, target_states); the constant errors are computed on that state and
averaged over the models. For each constant, dD = R * (s2024/O2024 - s1750/O1750), R = O2024/O1750.
For comparison, the same is done for an approximate 1750 state built from the 2024 state.

Usage:  python -u product_constants_uncertainty.py
"""
import numpy as np
import netCDF4 as nc

import oa_config
import calc_omega_arag_obs as co
import mocsy
import calc_pi_xco2 as px
import calc_bias_adjusted as cb
from drift_tools import Drift
from pi_states import TARGETS

FN = str(oa_config.OBS_DIR / 'UExP-FNN-U/Fordetal_UExP-FNN-U_surface-carbonate-system_v2026-1.nc')
YEAR = 2024
XCO2_2024, XCO2_1750 = 422.8, 278.3        # ppm, NOAA global annual mean 2024; AR6 1750
OPT = dict(optcon='mol/kg', optt='Tpot', optp='m', optb=oa_config.OPTB, optk1k2=oa_config.OPTK1K2)
EPK = np.array([0.002, 0.0075, 0.015, 0.01, 0.01, 0.02, 0.02])
NAMES = ['K0', 'K1', 'K2', 'KB', 'KW', 'Ksp(arag)', 'Ksp(calc)']

ds = nc.Dataset(FN)
t = ds['time'][:].astype('f8')
d = np.datetime64('1970-01-15') + t.astype('i8').astype('timedelta64[D]')
it = np.where(d.astype('datetime64[Y]').astype(int) + 1970 == YEAR)[0]
rd = lambda k: np.ma.filled(ds[k][:, :, it[0]:it[-1] + 1].astype('f8'), np.nan)   # (lon, lat, 12)
lat = ds['lat'][:].astype('f8')
lon = ds['lon'][:].astype('f8') % 360.
area = np.ma.filled(ds['area'][:].astype('f8'), np.nan)
F = {k: rd(k) for k in ['ta', 'dic', 'subskin_temp', 'subskin_salinity', 'saturation_aragonite_dissocation_unc']}
ds.close()
sil2d, po42d = co.load_nutrients(lat, lon)

ok = np.isfinite(area)[:, :, None] & np.ones(F['ta'].shape, bool)
for k in ['ta', 'dic', 'subskin_temp', 'subskin_salinity']:
    ok &= np.isfinite(F[k])
IX, IY, IM = np.nonzero(ok)
n = IX.size
W = oa_config.area_weights(lat, lon).T[IX, IY]      # the weights of the run (OA_WEIGHTS), not the file's whole-cell area; (lat, lon) -> (lon, lat) like the file
S = F['subskin_salinity'][ok]
T = co.cm.floor_temperature(F['subskin_temp'][ok] - 273.15, S)      # the product's own T, floored at the freezing point
TA, DIC = F['ta'][ok] * 1e-6, F['dic'][ok] * 1e-6
SIL = np.clip(np.nan_to_num(sil2d.T[IX, IY]), 0, None) * 1e-6      # GLODAP has small negative values
PHOS = np.clip(np.nan_to_num(po42d.T[IX, IY]), 0, None) * 1e-6
LAT = lat[IY]
Z, P1 = np.zeros(n), np.ones(n)
N0 = n
print(f'{n} valid cell-months')


def gmean(x):
    """annual mean of monthly area-weighted means"""
    out = []
    for k in range(12):
        s = (IM == k) & np.isfinite(x)
        out.append(np.sum(x[s] * W[s]) / np.sum(W[s]))
    return np.mean(out)


def state(dic):
    r = mocsy.mvars(temp=T, sal=S, alk=TA, dic=dic, sil=SIL, phos=PHOS, patm=P1, depth=Z, lat=LAT, **OPT)
    om = np.asarray(r[6], float)
    return np.asarray(r[2], float), np.where(co.cm.valid_carbonate(r[0], om), om, np.nan)   # fCO2, OmegaA


def errs(dic, epk, ebt, X=None):
    t_, s_, ta_, sil_, ph_, la_ = (T, S, TA, SIL, PHOS, LAT) if X is None else X
    z_ = np.zeros(t_.size)
    r = mocsy.merrors(t_, s_, ta_, dic, sil_, ph_, np.ones(t_.size), z_, la_, z_, z_, z_, z_, z_, z_,
                      OPT['optcon'], OPT['optt'], OPT['optp'], optb=OPT['optb'], optk1k2=OPT['optk1k2'],
                      epk=epk, ebt=ebt)
    e = np.asarray(r[6], float)                                     # 1 sigma of OmegaA
    return np.where(np.isfinite(e) & (e < 20), e, np.nan) if X is not None else np.where(bad, np.nan, e)


# 2024 state and approximate 1750 state (fCO2 lowered by the atmospheric rise, same TA, T, S)
F24, O24 = state(DIC)
target = F24 - (XCO2_2024 - XCO2_1750) * np.mean(F24) / XCO2_2024   # scale ppm -> uatm
dic = DIC.copy()
with np.errstate(divide='ignore', invalid='ignore'):
    for _ in range(5):
        f0, _ = state(dic)
        f1, _ = state(dic + 1e-6)
        der = (f1 - f0) / 1e-6
        step = np.where(np.isfinite(der) & (der > 0), (target - f0) / der, 0.)
        dic = dic + np.clip(np.nan_to_num(step), -5e-4, 5e-4)
F17, O17 = state(dic)
bad = ~(np.isfinite(O24) & np.isfinite(O17) & (np.abs(F17 - target) < 0.5))
O24, O17 = np.where(bad, np.nan, O24), np.where(bad, np.nan, O17)
print(f'cells excluded (mocsy failure or no convergence): {bad.sum()} of {n}')
DIC17 = dic
g24, g17 = gmean(O24), gmean(O17)
R = g24 / g17
print(f'global Omega_arag: 2024 {g24:.3f}, approx. 1750 {g17:.3f}, decline {100 * (1 - R):.1f}%')
print(f'  (file dissociation component, 2 sigma, area mean: {gmean(F["saturation_aragonite_dissocation_unc"][ok]):.3f})')

print(f'\n{"constant":12s} {"1s 2024":>8s} {"1s 1750":>8s} {"rel 2024":>9s} {"rel 1750":>9s} {"d decline":>10s}')
rows = []
for i, name in enumerate(NAMES + ['B total']):
    epk = np.zeros(7)
    ebt = 0.0
    if i < 7:
        epk[i] = EPK[i]
    else:
        ebt = 0.02
    s24, s17 = gmean(errs(DIC, epk, ebt)), gmean(errs(DIC17, epk, ebt))
    dd = 100 * R * abs(s24 / g24 - s17 / g17)                        # percentage points
    rows.append((s24, s17, dd))
    print(f'{name:12s} {s24:8.4f} {s17:8.4f} {100 * s24 / g24:8.2f}% {100 * s17 / g17:8.2f}% {dd:9.3f} pp')
s = np.array(rows)
tot24, tot17, totd = np.sqrt((s ** 2).sum(0))
print(f'{"total":12s} {tot24:8.4f} {tot17:8.4f} {100 * tot24 / g24:8.2f}% {100 * tot17 / g17:8.2f}% {totd:9.3f} pp')

allk = gmean(errs(DIC, EPK, 0.02))
print(f'\ncheck: all constants in one call (cell-wise quadrature, then mean) 2024: {allk:.4f}')

# ------------------------------------------------------------------ 1750 states of the paper (per model)
print('\n1750 states of the paper (bias-adjusted piControl, DIC lowered to 1750 CO2), first piControl year')
CASES = NAMES + ['B total']
per_model = []
for model, member in px.find_runs():
    with nc.Dataset(px.adj_path(model, member)) as g:
        mask = np.asarray(g['mask'][:]).astype(bool)
        olat = np.asarray(g['lat'][:], 'f8')
        olon = np.asarray(g['lon'][:], 'f8')
    with nc.Dataset(cb.bias_path(model, member)) as g:
        bias = {v: np.ma.filled(g[f'bias_{v}'][:].astype('f8'), np.nan) for v in px.VARS}
        y0, y1 = (int(x) for x in g.overlap.split('-'))
    lon2, lat2 = np.meshgrid(olon, olat)
    si, po4 = co.load_nutrients(olat, olon)
    run = cb.ModelRun(model, 'piControl', member)
    try:
        jy = co.nearest_index(run.lat, olat)
        ix = co.nearest_index(run.lon, olon, periodic=True)
        _, _, fa = cb.adjusted_fields(run, 0, run.step, bias, Drift(model, member), (y0, y1), jy, ix, lon2, lat2)
    finally:
        run.close()
    res = dict(px.target_states(fa, si, po4, lat2, {'1750': TARGETS['1750'], 'pi284': px.XCO2_MODEL}, [0.0]))
    om = {'1750': res['1750']['omega_arag']}
    dd = res['1750']['dic'] - res['pi284']['dic']                     # umol/kg, 284.317 -> 278.3 ppm
    t_, s_, d_, a_ = (fa[v] for v in px.VARS)
    good = np.isfinite(om['1750']) & np.isfinite(dd) & mask[None]
    im, iy, ixx = np.nonzero(good)
    X = (t_[good], s_[good], a_[good] * 1e-6, np.clip(si[iy, ixx], 0, None) * 1e-6,
         np.clip(po4[iy, ixx], 0, None) * 1e-6, olat[iy])
    d17 = (d_[good] + dd[good]) * 1e-6
    w = oa_config.area_weights(olat, olon)[iy, ixx]

    def gm(x):
        # mean over the time slots present: 12 months, or 1 for a model with annual output (fa has run.step slots)
        return np.mean([np.nansum((x * w)[im == k]) / np.sum(w[(im == k) & np.isfinite(x)]) for k in np.unique(im)])

    o17 = gm(om['1750'][good])
    sk = []
    for i in range(len(CASES)):
        epk = np.zeros(7); ebt = 0.0
        if i < 7:
            epk[i] = EPK[i]
        else:
            ebt = 0.02
        sk.append(gm(errs(d17, epk, ebt, X)))
    if not (np.isfinite(o17) and np.all(np.isfinite(sk))):
        raise ValueError(f'{model}: 1750 Omega_arag {o17} or a constant error not finite: {sk}')
    per_model.append((o17, sk))
    print(f'  {model:16s} Omega 1750 {o17:.3f}  Ksp(arag) {sk[5]:.4f}  K2 {sk[2]:.4f}  K1 {sk[1]:.4f}', flush=True)

o17m = np.mean([p[0] for p in per_model])
s17m = np.mean([p[1] for p in per_model], axis=0)
R = g24 / o17m
print(f'\nmulti-model 1750 Omega_arag {o17m:.3f}; 2024 (UExP) {g24:.3f}; decline {100 * (1 - R):.1f}%')
print(f'{"constant":12s} {"1s 2024":>8s} {"1s 1750":>8s} {"rel 2024":>9s} {"rel 1750":>9s} {"d decline":>10s}')
dds = []
for i, name in enumerate(CASES):
    s24 = rows[i][0]
    dd = 100 * R * abs(s24 / g24 - s17m[i] / o17m)
    dds.append(dd)
    print(f'{name:12s} {s24:8.4f} {s17m[i]:8.4f} {100 * s24 / g24:8.2f}% {100 * s17m[i] / o17m:8.2f}% {dd:9.3f} pp')
print(f'{"total":12s} {np.sqrt(sum(r[0] ** 2 for r in rows)):8.4f} {np.sqrt((s17m ** 2).sum()):8.4f} '
      f'{"":9s} {"":9s} {np.sqrt(np.sum(np.square(dds))):9.3f} pp')

# relative 1 sigma (fully correlated, all constants) and effect on the decline, for the notebook (Extended Data Table 2)
OUTD = oa_config.DATA / 'product_uncertainty'
OUTD.mkdir(parents=True, exist_ok=True)
OUTF = OUTD / f'constants_UExP_{YEAR}.npz'
np.savez(OUTF, names=np.array(CASES),
         s24=np.array([r[0] for r in rows]), s17=s17m, dd=np.array(dds), o24=g24, o17=o17m,
         rel24=np.sqrt(sum(r[0] ** 2 for r in rows)) / g24, rel17=np.sqrt((s17m ** 2).sum()) / o17m,
         totd=np.sqrt(np.sum(np.square(dds))))
print(f'saved {OUTF}')
