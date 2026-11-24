#!/usr/bin/env python
"""
test_baseline_sensitivity.py  (does the decline depend on the present-day map the simulated change is applied to?)
============================
The chain computes the preindustrial state as baseline minus the simulated change. Here the same change is applied
to very different baselines, and the relative decline from the preindustrial state to the baseline is compared.
  change:    multi-model mean of (historical 1983-2014 minus piControl) of T, S, DIC and TA per cell, from the
             mechanism files of step 1 (data/mechanism/state_<model>_<member>.nc); the CO2 of the piControl (284 ppm)
             is kept, the test is about the baseline, not the reference year
  baselines: (a) OceanSODA-ETHZ 1983-2014 (state_OceanSODA.nc)
             (b) each of the 18 ESMs' own unadjusted 1983-2014 state
             (c) OceanSODA with T +-5 degC, S +-1, TA +-100 umol/kg and combinations; in (c) DIC is recomputed so that
                 the pCO2 of every cell stays that of OceanSODA (the baseline stays consistent with the atmosphere)
For every baseline, two ways of applying the change: A, the same changes of T, S, DIC and TA (the preindustrial pCO2
then depends on the buffer capacity of the baseline, it is printed); B, the same changes of T, S and TA, and DIC of the
preindustrial state recomputed so that its pCO2 lies below the baseline's by the same amount per cell as for
OceanSODA (the same rise of CO2, the uptake follows from the chemistry of the baseline). Area means on the common area
of the main analysis (constraint_results.pkl): Omega_arag of the baseline and of the preindustrial state, the decline
in percent and in absolute terms, and the area-mean pCO2 of the baseline and of the preindustrial state. For each
baseline also its CUC as in the chain, <DIC/R>/pCO2_atm (constraint_refs.py), and the DIC uptake it predicts for the
rise of pCO2 (CUC_cell x dpCO2_cell, area mean) against the uptake of variant B; at the end, across all baselines,
the regressions of the decline of variant B on the baseline CUC and on its Omega_arag. Temperatures floored at the freezing point, as in the chain; GLODAP nutrients; constants of the run.
Usage (code/):  OA_RUN=u74 python -u test_baseline_sensitivity.py
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


def read(path, tag=''):
    with netCDF4.Dataset(path) as g:
        lat, lon = np.asarray(g['lat'][:], 'f8'), np.asarray(g['lon'][:], 'f8')
        return lat, lon, {v: np.ma.filled(g[f'{v}{tag}'][:].astype('f8'), np.nan) for v in VARS}


lat, lon, obs = read(MECH / 'state_OceanSODA.nc')
with open(oa_config.DATA / 'constraint_results.pkl', 'rb') as f:
    P = pickle.load(f)
r = P['results_common']['OceanSODA']
assert np.allclose(np.asarray(r['lat'], 'f8'), lat) and np.allclose(np.asarray(r['lon'], 'f8'), lon), 'grids differ'
mask = np.asarray(r['mask']).astype(bool)
si, po4 = co.load_nutrients(lat, lon)
w = oa_config.area_weights(lat, lon)
lat2 = np.meshgrid(lon, lat)[1]

files = sorted(glob.glob(str(MECH / 'state_*_*.nc')))
models, hist, delta = [], [], []
for p in files:
    _, _, h = read(p, '_hist')
    _, _, q = read(p, '_pi')
    models.append(p.split('state_')[1].rsplit('_', 1)[0])
    hist.append(h)
    delta.append({v: h[v] - q[v] for v in VARS})
D = {v: np.nanmean(np.stack([d[v] for d in delta]), axis=0) for v in VARS}
print(f'{len(models)} models; mean change on the common area: ' + ', '.join(
    f'{v} {np.nansum(np.where(mask, D[v], 0) * w) / np.sum(w * mask):+.2f}' for v in VARS))


def chem(st, m):
    """pCO2, Omega_arag, Revelle factor of the state st on the cells m"""
    t, s = st['thetao'][m], st['so'][m]
    t = co.cm.floor_temperature(t, s, count=False)
    n = int(m.sum())
    out = mocsy.mvars(temp=t, sal=s, alk=st['talk'][m] * 1e-6, dic=st['dissic'][m] * 1e-6, sil=si[m] * 1e-6,
                      phos=po4[m] * 1e-6, patm=np.ones(n), depth=np.zeros(n), lat=lat2[m], optcon='mol/kg',
                      optt='Tpot', optp='m', optb=oa_config.OPTB, optk1k2=oa_config.OPTK1K2)
    ph, pco2, om, rev = (np.asarray(out[i], 'f8') for i in (0, 1, 6, 8))
    ok = co.cm.valid_carbonate(ph, om)
    return np.where(ok, pco2, np.nan), np.where(ok, om, np.nan), np.where(ok, rev, np.nan)


def dic_for_pco2(st, m, target, it=8):
    """DIC (umol/kg) on the cells m so that pCO2 equals target, Newton steps with the Revelle factor"""
    st = dict(st)
    dic = st['dissic'].copy()
    for _ in range(it):
        st['dissic'] = dic
        p, _, R = chem(st, m)
        step = dic[m] / R * (target - p) / p
        dic[m] = dic[m] + np.where(np.isfinite(step), step, 0.0)
    return dic


m0 = mask & np.all([np.isfinite(obs[v]) & np.isfinite(D[v]) for v in VARS], axis=0) & np.isfinite(si) & np.isfinite(po4)
am = lambda x, m: float(np.nansum(np.where(m, x, 0) * w) / np.sum(w * (m & np.isfinite(x))))


def field(vals, m):
    out = np.full(lat2.shape, np.nan)
    out[m] = vals
    return out


# the rise of pCO2 per cell for OceanSODA under the mean change (the reference for variant B)
p_obs, _, _ = chem(obs, m0)
p_obs_pre, _, _ = chem({v: obs[v] - D[v] for v in VARS}, m0)
DP = field(p_obs - p_obs_pre, m0)
TARGET = field(p_obs, m0)


PCO2_ATM = 344.0                     # only scales CUC (as in constraint_refs.py: mean atmospheric pCO2 of 1983-2014)
ROWS = []


def case(name, base):
    m = m0 & np.all([np.isfinite(base[v]) for v in VARS], axis=0)
    p_b, o_b, R_b = chem(base, m)
    cuc = field(base['dissic'][m] / R_b / PCO2_ATM, m)
    pred = field(base['dissic'][m] / R_b / p_b, m) * DP           # dDIC = DIC/(R pCO2) dpCO2, per cell
    preA = {v: base[v] - D[v] for v in VARS}
    p_a, o_a, _ = chem(preA, m)
    preB = dict(preA)
    preB['dissic'] = dic_for_pco2(preA, m, (field(p_b, m) - DP)[m])
    p_bb, o_bb, _ = chem(preB, m)
    ob, oa, obb = field(o_b, m), field(o_a, m), field(o_bb, m)
    pb, pa, pbb = field(p_b, m), field(p_a, m), field(p_bb, m)
    both = m & np.isfinite(ob) & np.isfinite(oa) & np.isfinite(obb)
    Ob, Oa, Obb = am(ob, both), am(oa, both), am(obb, both)
    dicd = am(field(preB['dissic'][m], m) - field(preA['dissic'][m], m), both)
    upB = am(field(base['dissic'][m] - preB['dissic'][m], m), both)
    print(f'  {name:22s} Omega {Ob:5.3f} | A: pre {Oa:5.3f}, decline {100 * (1 - Ob / Oa):5.2f}% ({Oa - Ob:.3f}), '
          f'pCO2 {am(pb, both):5.1f} -> {am(pa, both):5.1f} | B: pre {Obb:5.3f}, decline {100 * (1 - Ob / Obb):5.2f}% '
          f'({Obb - Ob:.3f}), pCO2 -> {am(pbb, both):5.1f}, DIC uptake {-dicd:+.1f} vs A | CUC {am(cuc, both):.4f}, '
          f'uptake B {upB:.1f}, predicted by CUC {am(pred, both):.1f} umol/kg')
    ROWS.append((name, am(cuc, both), Ob, 100 * (1 - Ob / Obb), 100 * (1 - Ob / Oa)))
    return 100 * (1 - Ob / Oa), 100 * (1 - Ob / Obb)


print('\n(a) OceanSODA-ETHZ 1983-2014')
ref = case('OceanSODA', obs)

print('\n(b) unadjusted 1983-2014 state of each ESM as baseline (same mean change)')
dec = np.array([case(mn, h) for mn, h in zip(models, hist)])
print(f'  -> decline A {dec[:, 0].min():.2f} to {dec[:, 0].max():.2f}%, B {dec[:, 1].min():.2f} to {dec[:, 1].max():.2f}% '
      f'(OceanSODA {ref[0]:.2f} / {ref[1]:.2f})')

print('\n(c) OceanSODA perturbed, DIC recomputed for the pCO2 of OceanSODA (same mean change)')
pert = [('T +5', dict(thetao=5)), ('T -5', dict(thetao=-5)), ('S +1', dict(so=1)), ('S -1', dict(so=-1)),
        ('TA +100', dict(talk=100)), ('TA -100', dict(talk=-100)),
        ('T +5, TA +100', dict(thetao=5, talk=100)), ('T -5, TA -100', dict(thetao=-5, talk=-100)),
        ('T +5, S +1, TA +100', dict(thetao=5, so=1, talk=100)), ('T -5, S -1, TA -100', dict(thetao=-5, so=-1, talk=-100))]
dec = []
for name, d in pert:
    b = {v: obs[v] + d.get(v, 0.0) for v in VARS}
    b['dissic'] = dic_for_pco2(b, m0, TARGET[m0])
    dec.append(case(name, b))
dec = np.array(dec)
print(f'  -> decline A {dec[:, 0].min():.2f} to {dec[:, 0].max():.2f}%, B {dec[:, 1].min():.2f} to {dec[:, 1].max():.2f}% '
      f'(OceanSODA {ref[0]:.2f} / {ref[1]:.2f})')


print('\nacross all baselines (a, b, c): regressions of the decline (variant B, and A) on the baseline CUC and Omega_arag')
nm, cu, om, dB, dA = (np.array(x) for x in zip(*ROWS))
for lab, x in (('CUC', cu), ('Omega', om)):
    for vl, y in (('B', dB), ('A', dA)):
        b = np.polyfit(x, y, 1)
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        print(f'  decline {vl} ~ {lab:5s}: slope {b[0]:+.3g} per unit, R2 {r2:.2f}')
real = np.array([n not in [p[0] for p in pert] for n in nm])
for lab, x in (('CUC', cu), ('Omega', om)):
    b = np.polyfit(x[real], dB[real], 1)
    print(f'  realistic baselines only (a, b): decline B ~ {lab}: slope {b[0]:+.3g}, R2 {np.corrcoef(x[real], dB[real])[0, 1] ** 2:.2f}')
