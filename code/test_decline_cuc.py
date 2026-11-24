#!/usr/bin/env python
"""
test_decline_cuc.py  (what does a constraint of the relative decline itself give, compared with the best estimate?)
===================
Variant B of test_baseline_sensitivity.py showed that the relative decline hardly depends on the present-day map when
the rise of CO2 is consistent. So each model's own relative decline can be applied to the observed state directly.
Here, for OceanSODA-ETHZ on the common area (constraint_results.pkl, results_common):
  d_m   relative decline of Omega_arag of every ESM from its own piControl to its own historical 1983-2014 state
        (unadjusted, annual-mean states of data/mechanism/state_<model>_<member>.nc, mocsy as in the chain, the same
        cells for both states), in percent
  regressions of d_m on CUC alone, on dSST alone and on both (the model CUC and dSST of the chain), evaluated at
  the observed CUC and dSST; unconstrained = model mean
  for comparison, the chain itself: the bias-adjusted piControl state (pi284) of every model, its constrained value
  with dSST + CUC (best estimate) and with CUC alone, each expressed as the decline to the observed 1983-2014 mean
  (annual series of the chain on the same mask)
The state implied by a constrained decline is Omega_obs / (1 - d). Only for us, no figures.
Usage (code/):  OA_RUN=u74 python -u test_decline_cuc.py
"""
import glob
import pickle

import netCDF4
import numpy as np

import calc_omega_arag_obs as co            # puts mocsy on the path (calc_omega_arag.MOCSY_PATH)
import oa_config
import mocsy  # noqa: E402

MECH = oa_config.DATA / 'mechanism'
VARS = ('thetao', 'so', 'dissic', 'talk')

with open(oa_config.DATA / 'constraint_results.pkl', 'rb') as f:
    P = pickle.load(f)
r = P['results_common']['OceanSODA']
lat, lon = np.asarray(r['lat'], 'f8'), np.asarray(r['lon'], 'f8')
mask = np.asarray(r['mask']).astype(bool)
names = list(r['names'])
CUC, dT = np.asarray(r['CUC'], 'f8'), np.asarray(r['dT'], 'f8')
CUC_obs, DT_obs = float(r['CUC_obs']), float(r['dT_obs'][0])
si, po4 = co.load_nutrients(lat, lon)
w = oa_config.area_weights(lat, lon)
lat2 = np.meshgrid(lon, lat)[1]


def omega(st, m):
    t, s = st['thetao'][m], st['so'][m]
    t = co.cm.floor_temperature(t, s, count=False)
    n = int(m.sum())
    out = mocsy.mvars(temp=t, sal=s, alk=st['talk'][m] * 1e-6, dic=st['dissic'][m] * 1e-6, sil=si[m] * 1e-6,
                      phos=po4[m] * 1e-6, patm=np.ones(n), depth=np.zeros(n), lat=lat2[m], optcon='mol/kg',
                      optt='Tpot', optp='m', optb=oa_config.OPTB, optk1k2=oa_config.OPTK1K2)
    ph, om = np.asarray(out[0], 'f8'), np.asarray(out[6], 'f8')
    f = np.full(lat2.shape, np.nan)
    f[m] = np.where(co.cm.valid_carbonate(ph, om), om, np.nan)
    return f


am = lambda x, m: float(np.sum(x[m] * w[m]) / np.sum(w[m]))

d = np.full(len(names), np.nan)
print(f'{len(names)} models, OceanSODA common area; observed CUC {CUC_obs:.4f}, dSST {DT_obs:.3f} degC')
for i, mn in enumerate(names):
    hits = sorted(glob.glob(str(MECH / f'state_{mn}_*.nc')))
    if not hits:
        print(f'  {mn}: no mechanism file')
        continue
    with netCDF4.Dataset(hits[0]) as g:
        mlat, mlon = np.asarray(g['lat'][:], 'f8'), np.asarray(g['lon'][:], 'f8')
        st = {k: {v: np.ma.filled(g[f'{v}_{k}'][:].astype('f8'), np.nan) for v in VARS} for k in ('hist', 'pi')}
    assert np.allclose(mlat, lat) and np.allclose(mlon, lon), f'{mn}: grid differs from the constraint grid'
    m = mask & np.isfinite(si) & np.isfinite(po4) & np.all([np.isfinite(st[k][v]) for k in st for v in VARS], axis=0)
    oh, op = omega(st['hist'], m), omega(st['pi'], m)
    mm = m & np.isfinite(oh) & np.isfinite(op)
    d[i] = 100 * (1 - am(oh, mm) / am(op, mm))
    print(f'  {mn:16s} CUC {CUC[i]:.4f}  dSST {dT[i]:.3f}  piControl {am(op, mm):.3f} -> 1983-2014 {am(oh, mm):.3f}: '
          f'decline {d[i]:5.2f}%  ({100 * w[mm].sum() / w[mask].sum():.1f}% of the area)')


def con(y, cols, x0):
    """least squares on the columns, prediction at x0, R2"""
    X = np.column_stack([np.ones(len(y))] + cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    return float(np.r_[1.0, x0] @ b), 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean())), b


g = np.isfinite(d)
yrs = np.asarray(r['year'])
ogm = np.asarray(r['ogm'], 'f8')
sel = (yrs >= 1983) & (yrs <= 2014) & np.isfinite(ogm)
om_obs = float(ogm[sel].mean())
print(f'\nobserved Omega_arag 1983-2014 (annual series of the chain, same mask): {om_obs:.4f} ({int(sel.sum())} years)')

print('\n(1) relative decline of each model from its own states, piControl -> 1983-2014 (variant B)')
print(f'  unconstrained (model mean)  {d[g].mean():5.2f}% +- {d[g].std(ddof=1):.2f}  -> piControl state {om_obs / (1 - d[g].mean() / 100):.3f}')
for lab, cols, x0 in (('CUC alone', [CUC[g]], [CUC_obs]), ('dSST alone', [dT[g]], [DT_obs]),
                      ('dSST + CUC', [dT[g], CUC[g]], [DT_obs, CUC_obs])):
    yc, r2, b = con(d[g], cols, x0)
    print(f'  {lab:12s} R2 {r2:.2f}  constrained {yc:5.2f}%  -> piControl state {om_obs / (1 - yc / 100):.3f}   '
          f'coefs ' + ' '.join(f'{x:+.3g}' for x in b))

print('\n(2) the chain: bias-adjusted piControl state (pi284) of every model, constrained, as decline to the observed mean')
ypi = np.asarray(r['y_states']['pi284'], 'f8')
print(f'  unconstrained (model mean)  state {ypi.mean():.3f}  decline {100 * (1 - om_obs / ypi.mean()):5.2f}%')
for lab, cols, x0 in (('CUC alone', [CUC], [CUC_obs]), ('dSST alone', [dT], [DT_obs]),
                      ('dSST + CUC', [dT, CUC], [DT_obs, CUC_obs])):
    yc, r2, _ = con(ypi, cols, x0)
    print(f'  {lab:12s} R2 {r2:.2f}  state {yc:.3f}  decline {100 * (1 - om_obs / yc):5.2f}%')
sk = r['states_by_lab']['dT + CUC']['pi284']
print(f'  (stored best estimate of the chain, pi284 with dSST + CUC: {sk["constrained"]:.3f} +- {sk["constrained_sigma"]:.3f})')
