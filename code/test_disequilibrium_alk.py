#!/usr/bin/env python
"""
test_disequilibrium_alk.py
==========================
How much of the 1750 surface state, and with it the decline 1750-2024, rests on the models, and how far idealised
extremes of the air-sea disequilibrium and of alkalinity move the decline. For each of the four products, with its
observed 2024 annual state (single year) on the common area of the paper:

  ours   the constrained 1750 state of the chain (states_<ref>_pi.nc, state 1750) against the observed 2024 state
  model  what the models assume: the global mean change of the disequilibrium (pCO2 ocean - pCO2 air) from 1750 to
         2024 implied by the constrained 1750 pCO2, and the 1750-2024 changes of T, S and TA
  A      model-free: 1750 = the observed 2024 state with ocean pCO2 lowered by exactly the atmospheric rise
         (disequilibrium unchanged), T, S and TA unchanged
  A_mod  as A, but with the models' 1750 T, S and TA: the difference to ours is the disequilibrium change alone
  B      A with the 1750 disequilibrium shifted by -20 ... +20 uatm (positive: 1750 ocean pCO2 higher)
  C      A with 1750 TA shifted by -10 ... +10 umol/kg at the pCO2 of A
  D      A with 1750 T lower by 1.0 degC everywhere (a uniform warming of 1 degC since 1750)
  E      observed tracking: trend 1990-2024 of the common-area mean of ocean pCO2 minus atmospheric pCO2 (NOAA),
         uatm per decade

Same chemistry as the chain (mocsy 2.4.0, constants of oa_config, GLODAP Si and PO4, cos(latitude)). DIC for a
given pCO2 and TA is found by iteration (Revelle-type update, converged to < 0.01 uatm).

Usage (code/):  OA_RUN=u74 python -u test_disequilibrium_alk.py
"""
import pickle

import netCDF4
import numpy as np

import oa_config
import calc_omega_arag_obs as co
import co2_history
import mocsy

OUT = oa_config.DATA / 'product'
PRODUCTS = ('OceanSODA', 'UExP', 'JMA', 'CMEMS')
YEAR = 2024
DQ = (-20, -10, -5, 5, 10, 20)              # uatm
DTA = (-10, -5, 5, 10)                       # umol/kg
DT = 1.0                                     # degC


def noaa_series():
    r = co2_history.noaa()
    if isinstance(r, dict):
        y = np.array(sorted(r)); x = np.array([r[k] for k in y], 'f8')
    elif isinstance(r, tuple) and len(r) == 2:
        y, x = (np.asarray(a, 'f8') for a in r)
    else:
        a = np.asarray(r, 'f8'); y, x = a[:, 0], a[:, 1]
    return y.astype(int), x


def ph2o(t, s):
    """water vapour pressure at the surface in atm (Weiss and Price 1980)"""
    tk = t + 273.15
    return np.exp(24.4543 - 67.4509 * (100 / tk) - 4.8489 * np.log(tk / 100) - 0.000544 * s)


def mv(t, s, ta, dic, si, po4, lat):
    """mocsy, all 1-D, TA and DIC in mol/kg; returns pco2 (uatm), omega_arag"""
    n = t.size
    r = mocsy.mvars(temp=t, sal=s, alk=ta, dic=dic, sil=si, phos=po4, patm=np.ones(n), depth=np.zeros(n), lat=lat,
                    optcon='mol/kg', optt='Tpot', optp='m', optb=co.MOCSY_OPTB, optk1k2=co.MOCSY_OPTK1K2)
    return np.asarray(r[1], 'f8'), np.asarray(r[6], 'f8')


def dic_for_pco2(t, s, ta, p_target, dic0, si, po4, lat):
    dic = dic0.copy()
    for _ in range(30):
        p, _o = mv(t, s, ta, dic, si, po4, lat)
        dic = dic * (p_target / p) ** (1 / 10.0)          # Revelle-type update
        if np.nanmax(np.abs(p - p_target)) < 0.01:
            break
    p, om = mv(t, s, ta, dic, si, po4, lat)
    return dic, om, float(np.nanmax(np.abs(p - p_target)))


def gm(a, w):
    m = np.isfinite(a)
    return float(np.sum(a[m] * w[m]) / w[m].sum())


def main():
    pdir = OUT
    yn, xn = noaa_series()
    res_pkl = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    print(f'products from {pdir}; NOAA {yn[0]}-{yn[-1]}, {YEAR}: {xn[yn == YEAR][0]:.2f} ppm')
    summary = {}
    for ref in PRODUCTS:
        with netCDF4.Dataset(pdir / f'obs_{ref}.nc') as f:
            yrs = np.asarray(f['year'][:]).astype(int)
            lat = np.asarray(f['lat'][:], 'f8'); lon = np.asarray(f['lon'][:], 'f8')
            i = int(np.where(yrs == YEAR)[0][0])
            g = {k: np.ma.filled(f[k][i].astype('f8'), np.nan) for k in ('thetao', 'so', 'talk', 'dic', 'pco2', 'omega_arag')}
            pco2_all = np.ma.filled(f['pco2'][:].astype('f8'), np.nan)
            t_all, s_all = (np.ma.filled(f[k][:].astype('f8'), np.nan) for k in ('thetao', 'so'))
        with netCDF4.Dataset(pdir / f'states_{ref}_pi.nc') as f:
            names = [str(x) for x in f['state'][:]]
            k = next(j for j, nm in enumerate(names) if '1750' in nm)
            x1750 = float(f['state_xco2'][k])
            plat = np.asarray(f['lat'][:], 'f8'); plon = np.asarray(f['lon'][:], 'f8')
            P = {q: np.ma.filled(f[f'{q}_state'][k].astype('f8'), np.nan) for q in ('omega_arag', 'pco2', 'thetao', 'so', 'talk')}
        # same grid: map the states onto the obs grid by nearest coordinate (they may differ in the longitude origin)
        jy = np.abs(lat[:, None] - plat[None, :]).argmin(1)
        ix = np.abs(((lon[:, None] - plon[None, :]) + 180) % 360 - 180).argmin(1)
        P = {q: a[np.ix_(jy, ix)] for q, a in P.items()}
        # the common area of the paper (constraint_results.pkl), else every cell valid in both
        mk = res_pkl['results_common'][ref].get('mask') if 'results_common' in res_pkl else None
        mk = np.asarray(mk, bool) if mk is not None else None
        if mk is None or mk.shape != g['omega_arag'].shape:
            print(f'  {ref}: common mask of the pickle not usable ({None if mk is None else mk.shape}), cells valid in both used')
            mk = np.ones(g['omega_arag'].shape, bool)
        m = mk & np.isfinite(g['omega_arag']) & np.isfinite(P['omega_arag']) & np.isfinite(g['dic'])
        cw = np.cos(np.deg2rad(lat))[:, None] * np.ones(m.shape)
        print(f'\n===== {ref}: {m.sum()} cells, {100 * (cw * m).sum() / (cw * np.isfinite(g["omega_arag"])).sum():.1f} % of '
              f'its 2024 area; state "{names[k]}" xCO2 {x1750:.1f} ppm')
        u = 1e-6 if np.nanmedian(g['talk']) > 10 else 1.0       # umol/kg -> mol/kg
        t, s, ta, dic = g['thetao'][m], g['so'][m], g['talk'][m] * u, g['dic'][m] * u
        si, po4 = (a[m] * 1e-6 for a in co.load_nutrients(lat, lon))
        la = np.broadcast_to(lat[:, None], m.shape)[m]
        w = cw[m]
        p24, om24 = mv(t, s, ta, dic, si, po4, la)
        print(f'  2024 recomputed: Omega {gm(om24, w):.4f} (file {gm(g["omega_arag"][m], w):.4f}), pCO2 {gm(p24, w):.1f} uatm '
              f'(file {gm(g["pco2"][m], w):.1f})')
        dec = lambda om_pi: 100 * (1 - gm(om24, w) / gm(om_pi, w))
        x24 = float(xn[yn == YEAR][0])
        datm24 = x24 * (1 - ph2o(t, s)); datm50 = x1750 * (1 - ph2o(t, s))
        r = {}
        r['ours'] = dec(P['omega_arag'][m])
        # what the models assume
        dis24 = p24 - datm24
        dis50 = P['pco2'][m] - x1750 * (1 - ph2o(P['thetao'][m], P['so'][m]))
        print(f'  ours (constrained 1750 vs observed 2024, single year): decline {r["ours"]:.2f} %')
        print(f'  models: disequilibrium 1750 {gm(dis50, w):+.1f} uatm, 2024 (observed) {gm(dis24, w):+.1f} uatm, change '
              f'{gm(dis24 - dis50, w):+.1f} uatm; T change {gm(t - P["thetao"][m], w):+.2f} degC, S {gm(s - P["so"][m], w):+.3f}, '
              f'TA {gm((ta / u - P["talk"][m]), w):+.2f} umol/kg')
        # A: constant disequilibrium, T, S, TA of 2024
        pA = p24 - (datm24 - datm50)
        dA, omA, err = dic_for_pco2(t, s, ta, pA, dic, si, po4, la)
        r['A'] = dec(omA)
        print(f'  A  model-free (constant disequilibrium, 2024 T, S, TA): decline {r["A"]:.2f} %  (ours - A {r["ours"] - r["A"]:+.2f}; '
              f'solver {err:.3f} uatm)')
        # A_mod: constant disequilibrium with the models' 1750 T, S, TA
        tm, sm, tam = P['thetao'][m], P['so'][m], P['talk'][m] * u
        pAm = (p24 - datm24) + x1750 * (1 - ph2o(tm, sm))
        _, omAm, err = dic_for_pco2(tm, sm, tam, pAm, dic, si, po4, la)
        r['A_mod'] = dec(omAm)
        print(f'  A_mod (constant disequilibrium, models\' 1750 T, S, TA): decline {r["A_mod"]:.2f} %  '
              f'(ours - A_mod = the disequilibrium change: {r["ours"] - r["A_mod"]:+.2f} points)')
        # B: disequilibrium shifts
        for d in DQ:
            _, om, _e = dic_for_pco2(t, s, ta, pA + d, dA, si, po4, la)
            r[f'B{d:+d}'] = dec(om)
        print('  B  1750 disequilibrium shifted (uatm): ' + ', '.join(f'{d:+d}: {r[f"B{d:+d}"] - r["A"]:+.2f}' for d in DQ)
              + f' points; per uatm {(r["B+10"] - r["B-10"]) / 20:+.3f}')
        # C: TA shifts at the pCO2 of A
        for d in DTA:
            _, om, _e = dic_for_pco2(t, s, ta + d * 1e-6, pA, dA, si, po4, la)
            r[f'C{d:+d}'] = dec(om)
        per = (r['C+10'] - r['C-10']) / 20
        print('  C  1750 TA shifted (umol/kg): ' + ', '.join(f'{d:+d}: {r[f"C{d:+d}"] - r["A"]:+.2f}' for d in DTA)
              + f' points; per umol/kg {per:+.3f}; a shift of {0.48 / abs(per):.1f} umol/kg moves the decline by 0.48 points')
        # D: uniform warming
        pD = (p24 - datm24) + x1750 * (1 - ph2o(t - DT, s))
        _, om, _e = dic_for_pco2(t - DT, s, ta, pD, dA, si, po4, la)
        r['D'] = dec(om)
        print(f'  D  1750 T lower by {DT} degC (else A): {r["D"] - r["A"]:+.2f} points')
        # E: observed tracking
        ok = (yrs >= 1990) & (yrs <= YEAR)
        ser = []
        for j in np.where(ok)[0]:
            xa = float(xn[yn == yrs[j]][0])
            pa = xa * (1 - ph2o(t_all[j][m], s_all[j][m]))
            ser.append(gm(pco2_all[j][m] - pa, w))
        ser = np.array(ser)
        b = np.polyfit(yrs[ok], ser, 1)[0]
        print(f'  E  observed disequilibrium (ocean - air) on this area: {yrs[ok][0]} {ser[0]:+.1f}, {YEAR} {ser[-1]:+.1f} uatm; '
              f'trend {10 * b:+.2f} uatm per decade; atmosphere {10 * np.polyfit(yrs[ok], [xn[yn == y][0] for y in yrs[ok]], 1)[0]:+.1f} ppm per decade')
        summary[ref] = r
    print('\n===== mean of the four products (points of decline)')
    keys = list(next(iter(summary.values())))
    for k_ in keys:
        v = np.mean([summary[q][k_] for q in summary])
        print(f'  {k_:8s} {v:6.2f}' + ('' if k_ in ('ours', 'A', 'A_mod') else f'  ({v - np.mean([summary[q]["A"] for q in summary]):+.2f} vs A)'))


if __name__ == '__main__':
    main()
