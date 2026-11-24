#!/usr/bin/env python
"""
test_diseq_residual.py  (18 November 2026)
======================
Does the third control of the simulated carbon uptake, how closely surface waters follow the rising atmospheric CO2,
explain the part of the model spread that CUC and dSST leave unexplained?

For each of the 18 ESMs, from the mechanism states that also give the change of DIC minus alkalinity of Extended Data
Fig. 3d (data/mechanism/state_<model>_<member>.nc, calc_mechanism.py; OceanSODA-ETHZ window 1983-2014 and the mean of
the piControl run, umol/kg, degC, psu, on the OceanSODA grid):
  surface pCO2 of both mean states with mocsy (constants of the run, GLODAP Si and PO4, as in the chain),
  the change of the air-sea disequilibrium  dD = dpCO2_ocean - dpCO2_air  (uatm),
  with pCO2_air = xCO2 (1 - pH2O) at 1 atm, xCO2 the CMIP6 history (1983-2014 mean; the piControl at 1850),
  and the ratio  dpCO2_ocean / dpCO2_air  (1 = the surface follows the atmosphere fully),
area mean on the common area with the weights of the run (as test_r2_mechanism.py).
Then, across the models, with y = bias-adjusted 1750 Omega_arag and d(DIC-ALK) of constraint_results.pkl / the
mechanism files:
  R2 of y ~ CUC + dSST, of y ~ CUC + dSST + dD, and of y ~ dD alone; the same for d(DIC-ALK);
  the correlation of dD with the residual of y ~ CUC + dSST (how much of the unexplained part it carries),
  and its correlation with CUC and dSST (how much of it the two predictors already contain).
Also (18 Nov, second version): the levels of the disequilibrium, present-day (1983-2014) D_hist and piControl D_pi,
with their spread across the models, and the same regressions with D_hist instead of dD (could the present-day
disequilibrium, which the fCO2-products observe, stand in for the change, i.e. is D_pi similar across the models?),
and the observed D_hist of OceanSODA-ETHZ (its 1983-2014 mean state, data/mechanism/state_OceanSODA.nc) on the
same area.
Reads only, writes nothing.
Usage (code/):  OA_RUN=u74 python -u test_diseq_residual.py
"""
import glob
import pickle

import netCDF4
import numpy as np

import calc_omega_arag_obs as co
import co2_history
import mocsy
import oa_config
from test_alk_diseq_models import ph2o, series

REF, WIN = 'OceanSODA', (1983, 2014)


def r2(y, *xs):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in xs])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    return 1.0 - (res @ res) / ((y - y.mean()) @ (y - y.mean())), res


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def pco2(t, s, dic, ta, si, po4, la):
    n = t.size
    r = mocsy.mvars(temp=t, sal=s, alk=ta * 1e-6, dic=dic * 1e-6, sil=si * 1e-6, phos=po4 * 1e-6, patm=np.ones(n),
                    depth=np.zeros(n), lat=la, optcon='mol/kg', optt='Tpot', optp='m', optb=co.MOCSY_OPTB,
                    optk1k2=co.MOCSY_OPTK1K2)
    return np.asarray(r[1], 'f8')


yh, xh = series(co2_history.historical())
atm = dict(zip(yh, xh))
X_HIST = float(np.mean([atm[y] for y in range(WIN[0], WIN[1] + 1)]))
X_PI = float(atm[1850])
print(f'run {oa_config.DATA}; {REF} window {WIN[0]}-{WIN[1]}: xCO2 {X_HIST:.1f} ppm, piControl {X_PI:.1f} ppm; '
      f'area weights: {oa_config.WEIGHTS_TEXT}')

with open(oa_config.DATA / 'constraint_results.pkl', 'rb') as f:
    r = pickle.load(f)['results_common'][REF]
names = list(r['names'])
y = np.asarray(r['y1750'], float)
cuc, dT = np.asarray(r['CUC'], float), np.asarray(r['dT'], float)
lat, lon = np.asarray(r['lat'], float), np.asarray(r['lon'], float)
mask = np.asarray(r['mask']).astype(bool)
w = oa_config.area_weights(lat, lon)

dD, ratio, dda, Dh, Dp = (np.full(len(names), np.nan) for _ in range(5))
nut = {}
print(f'{"model":18s} {"dpCO2 ocean":>11s} {"dpCO2 air":>9s} {"dD":>7s} {"ratio":>6s} {"d(DIC-ALK)":>10s} {"D_hist":>7s} {"D_pi":>7s}')
for i, m in enumerate(names):
    hits = sorted(glob.glob(str(oa_config.DATA / 'mechanism' / f'state_{m}_*.nc')))
    if not hits:
        print(f'{m:18s} no mechanism file')
        continue
    with netCDF4.Dataset(hits[0]) as g:
        mlat, mlon = np.asarray(g['lat'][:], float), np.asarray(g['lon'][:], float)
        get = lambda k: np.ma.filled(g[k][:].astype('f8'), np.nan)
        F = {tag: {v: get(f'{v}_{tag}') for v in ('thetao', 'so', 'dissic', 'talk')} for tag in ('hist', 'pi')}
    key = (mlat.size, mlon.size, float(mlon[0]))
    if key not in nut:
        nut[key] = co.load_nutrients(mlat, mlon)
    si, po4 = nut[key]
    la = np.broadcast_to(mlat[:, None], si.shape)
    ok = np.isfinite(si) & np.isfinite(po4)
    for tag in ('hist', 'pi'):
        ok &= np.all([np.isfinite(F[tag][v]) for v in F[tag]], axis=0)
    p = {}
    for tag in ('hist', 'pi'):
        A = F[tag]
        p[tag] = np.full(si.shape, np.nan)
        p[tag][ok] = pco2(A['thetao'][ok], A['so'][ok], A['dissic'][ok], A['talk'][ok], si[ok], po4[ok], la[ok])
    a_h, a_p = (np.full(si.shape, np.nan) for _ in range(2))
    a_h[ok] = X_HIST * (1 - ph2o(F['hist']['thetao'][ok], F['hist']['so'][ok]))
    a_p[ok] = X_PI * (1 - ph2o(F['pi']['thetao'][ok], F['pi']['so'][ok]))
    air = a_h - a_p
    d_oc = p['hist'] - p['pi']
    lev_h, lev_p = p['hist'] - a_h, p['pi'] - a_p
    d_da = (F['hist']['dissic'] - F['hist']['talk']) - (F['pi']['dissic'] - F['pi']['talk'])
    if mlon.max() > 180 and lon.min() < 0:            # mechanism grid 0..360, product grid -180..180
        sh = -int(np.argmin(np.abs(mlon - 180.5)))
        d_oc, air, d_da, lev_h, lev_p = (np.roll(a, sh, axis=1) for a in (d_oc, air, d_da, lev_h, lev_p))
    good = mask & np.isfinite(d_oc) & np.isfinite(air) & np.isfinite(d_da)
    mean = lambda a: float(np.sum(a[good] * w[good]) / np.sum(w[good]))
    o_, a_ = mean(d_oc), mean(air)
    dD[i], ratio[i], dda[i], Dh[i], Dp[i] = o_ - a_, o_ / a_, mean(d_da), mean(lev_h), mean(lev_p)
    print(f'{m:18s} {o_:11.1f} {a_:9.1f} {dD[i]:+7.1f} {ratio[i]:6.3f} {dda[i]:10.1f} {Dh[i]:+7.1f} {Dp[i]:+7.1f}', flush=True)

g = np.isfinite(dD)
print(f'\n{int(g.sum())} of {len(names)} models; dD {np.nanmean(dD):+.1f} +- {np.nanstd(dD, ddof=1):.1f} uatm, '
      f'ratio {np.nanmean(ratio):.3f} +- {np.nanstd(ratio, ddof=1):.3f}')
for lab, yy in (('1750 Omega_arag', y), ('d(DIC-ALK)', dda)):
    yy = yy[g]
    r2_2, res = r2(yy, cuc[g], dT[g])
    r2_3, _ = r2(yy, cuc[g], dT[g], dD[g])
    print(f'  {lab:16s} R2: CUC + dSST {r2_2:.2f}; CUC + dSST + dD {r2_3:.2f}; dD alone {r2(yy, dD[g])[0]:.2f}; '
          f'correlation of dD with the residual of CUC + dSST {corr(res, dD[g]):+.2f}')
print(f'\nlevels: D_hist {np.nanmean(Dh):+.1f} +- {np.nanstd(Dh, ddof=1):.1f} uatm, D_pi {np.nanmean(Dp):+.1f} +- '
      f'{np.nanstd(Dp, ddof=1):.1f} uatm (dD {np.nanmean(dD):+.1f} +- {np.nanstd(dD, ddof=1):.1f}); r(D_hist, dD) '
      f'{corr(Dh[g], dD[g]):+.2f}, r(D_pi, dD) {corr(Dp[g], dD[g]):+.2f}')
for lab, yy in (('1750 Omega_arag', y), ('d(DIC-ALK)', dda)):
    yy = yy[g]
    r2_2, res = r2(yy, cuc[g], dT[g])
    print(f'  {lab:16s} with D_hist: R2 CUC + dSST + D_hist {r2(yy, cuc[g], dT[g], Dh[g])[0]:.2f} (with dD '
          f'{r2(yy, cuc[g], dT[g], dD[g])[0]:.2f}); correlation of D_hist with the residual of CUC + dSST {corr(res, Dh[g]):+.2f}')
# observed: OceanSODA-ETHZ 1983-2014 mean state on the same area
obs = sorted(glob.glob(str(oa_config.DATA / 'mechanism' / f'state_{REF}*.nc')))
obs = [o for o in obs if '_'.join(o.split('/')[-1][:-3].split('_')[1:]) in (REF, f'{REF}-ETHZ')] or obs[:1]
if obs:
    with netCDF4.Dataset(obs[0]) as g_:
        olat, olon = np.asarray(g_['lat'][:], float), np.asarray(g_['lon'][:], float)
        O = {v: np.ma.filled(g_[v][:].astype('f8'), np.nan) for v in ('thetao', 'so', 'dissic', 'talk')}
    si, po4 = co.load_nutrients(olat, olon)
    la = np.broadcast_to(olat[:, None], si.shape)
    ok = np.isfinite(si) & np.isfinite(po4) & np.all([np.isfinite(O[v]) for v in O], axis=0)
    lev = np.full(si.shape, np.nan)
    lev[ok] = pco2(O['thetao'][ok], O['so'][ok], O['dissic'][ok], O['talk'][ok], si[ok], po4[ok], la[ok]) \
        - X_HIST * (1 - ph2o(O['thetao'][ok], O['so'][ok]))
    if olon.max() > 180 and lon.min() < 0:
        lev = np.roll(lev, -int(np.argmin(np.abs(olon - 180.5))), axis=1)
    gd = mask & np.isfinite(lev)
    print(f'  observed D_hist ({obs[0].split("/")[-1]}, {100 * w[gd].sum() / w[mask].sum():.0f} % of the area): '
          f'{np.sum(lev[gd] * w[gd]) / np.sum(w[gd]):+.1f} uatm (models {np.nanmean(Dh):+.1f} +- {np.nanstd(Dh, ddof=1):.1f})')
else:
    print('  no observed state file state_OceanSODA*.nc in data/mechanism')
print(f'  dD against the predictors: r(dD, CUC) {corr(dD[g], cuc[g]):+.2f}, r(dD, dSST) {corr(dD[g], dT[g]):+.2f}, '
      f'R2(dD ~ CUC + dSST) {r2(dD[g], cuc[g], dT[g])[0]:.2f}')
