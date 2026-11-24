#!/usr/bin/env python
"""
test_alk_diseq_models.py
========================
Two checks of what the 1750 state takes from the models (see test_disequilibrium_alk.py), on the common area of
the paper, cos(latitude):

 1  Surface alkalinity in each of the 18 ESMs: change from 1850-1869 to 1995-2014, raw and salinity-normalized
    (nTA = TA x 35 / S, per year), and salinity, drift-corrected with the parallel piControl years (linear fit
    over the parallel years). TA in umol/kg with a density of 1025 kg m-3 (changes only). The models with
    Omega-dependent calcification (UKESM1-0-LL, GFDL-ESM4, GFDL-CM4) are marked.
 2  Air-sea disequilibrium (ocean pCO2 - atmospheric pCO2, moist air at 1 atm) of the constrained states of each
    product (states_<ref>_pi/hist.nc: the 1750 state and 1850-2014) against the observed disequilibrium of the
    product (obs_<ref>.nc, NOAA CO2): levels and linear trends over the common years 1990-2014.
    The bias adjustment matches levels, not trends: the trends are the test.

Usage (code/):  OA_RUN=u74 python -u test_alk_diseq_models.py
"""
import pickle

import netCDF4
import numpy as np

import oa_config
import calc_omega_arag as cm
import calc_pi_xco2 as cpx
import co2_history
import drift_tools as dt

OUT = oa_config.DATA / 'product'
PRODUCTS = ('OceanSODA', 'UExP', 'JMA', 'CMEMS')
CALC_OMEGA = ('UKESM1-0-LL', 'GFDL-ESM4', 'GFDL-CM4')
W0, W1 = (1850, 1869), (1995, 2014)
TR = (1990, 2014)
RHO = 1025.0


def series(r):
    """(years, values) from a co2_history return value (dict, tuple or 2-column array)"""
    if isinstance(r, dict):
        y = np.array(sorted(r)); return y.astype(int), np.array([r[k] for k in y], 'f8')
    if isinstance(r, tuple) and len(r) == 2:
        return np.asarray(r[0]).astype(int), np.asarray(r[1], 'f8')
    a = np.asarray(r, 'f8'); return a[:, 0].astype(int), a[:, 1]


def ph2o(t, s):
    tk = t + 273.15
    return np.exp(24.4543 - 67.4509 * (100 / tk) - 4.8489 * np.log(tk / 100) - 0.000544 * s)


def years_of(f):
    t = f['time']
    d = netCDF4.num2date(t[:], t.units, getattr(t, 'calendar', 'standard'))
    return np.array([x.year for x in d])


def surface(path, var):
    with netCDF4.Dataset(path) as f:
        v = f[var]
        a = v[:, 0] if v.ndim == 4 else v[:]
        y = years_of(f)
        lat, lon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
    return y, np.ma.filled(np.asarray(a, 'f8') if not np.ma.isMA(a) else a.astype('f8'), np.nan), lat, lon


def common_mask_on(lat, lon):
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    m = np.asarray(pk['results_common']['UExP']['mask'], bool)
    with netCDF4.Dataset(OUT / 'obs_UExP.nc') as f:
        olat, olon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
    jy = np.abs(lat[:, None] - olat[None, :]).argmin(1)
    ix = np.abs(((lon[:, None] - olon[None, :]) + 180) % 360 - 180).argmin(1)
    return m[np.ix_(jy, ix)]


def gm(a, m, lat):
    w = np.cos(np.deg2rad(lat))[:, None] * m
    a = np.where(m, a, 0.0)
    ok = np.isfinite(a)
    return float(np.sum(np.where(ok, a, 0) * w) / np.sum(w * ok))


def part1():
    print('\n===== 1  surface TA change 1850-1869 -> 1995-2014, drift-corrected (umol/kg; S in psu)')
    print(f'{"model":16s} {"dTA":>7s} {"dnTA":>7s} {"dS":>7s}  drift dTA  calcification')
    rows = []
    for model, member in cpx.find_runs(None):
        try:
            ph = {v: cm.input_path(v, model, 'historical', member) for v in ('talk', 'so')}
            pc = {v: cm.input_path(v, model, 'piControl', member) for v in ('talk', 'so')}
            if None in ph.values() or None in pc.values():
                print(f'{model:16s} files missing: {ph} {pc}'); continue
            yh, ta, lat, lon = surface(ph['talk'], 'talk')
            _, so, _, _ = surface(ph['so'], 'so')
            yc, tac, _, _ = surface(pc['talk'], 'talk')
            _, soc, _, _ = surface(pc['so'], 'so')
            m = common_mask_on(lat, lon)
            d = dt.Drift(model, member)
            par = d.by + (yh - 1850)                                   # parallel control year of each hist year
            ic = np.array([np.argmin(np.abs(yc - p)) for p in par])
            k = 1e6 / RHO                                              # mol m-3 -> umol/kg
            g = lambda a: np.array([gm(a[i], m, lat) for i in range(a.shape[0])])
            gh = {'TA': g(ta) * k, 'nTA': g(ta * 35 / so) * k, 'S': g(so)}
            gc = {'TA': g(tac[ic]) * k, 'nTA': g(tac[ic] * 35 / soc[ic]) * k, 'S': g(soc[ic])}
            w0 = (yh >= W0[0]) & (yh <= W0[1]); w1 = (yh >= W1[0]) & (yh <= W1[1])
            out = {}
            for q in gh:
                b = np.polyfit(yh, gc[q], 1)                           # linear drift over the parallel years
                drift = np.polyval(b, yh)
                out[q] = (gh[q][w1].mean() - gh[q][w0].mean()) - (drift[w1].mean() - drift[w0].mean())
                out[q + '_drift'] = drift[w1].mean() - drift[w0].mean()
            rows.append((model, out))
            print(f'{model:16s} {out["TA"]:+7.2f} {out["nTA"]:+7.2f} {out["S"]:+7.3f}  {out["TA_drift"]:+8.2f}   '
                  f'{"Omega-dependent" if model in CALC_OMEGA else ""}', flush=True)
        except Exception as e:                                         # one model failing does not stop the rest
            print(f'{model:16s} failed: {type(e).__name__}: {e}', flush=True)
    if rows:
        for q in ('TA', 'nTA', 'S'):
            a = np.array([r[1][q] for r in rows])
            print(f'  {q}: mean {a.mean():+.3f}, range {a.min():+.3f} to {a.max():+.3f} ({len(a)} models)')


def part2():
    print(f'\n===== 2  disequilibrium (ocean - air pCO2, uatm) of the constrained states against the products')
    yn, xn = series(co2_history.noaa())
    yh_, xh_ = series(co2_history.historical())
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    for ref in PRODUCTS:
        m = np.asarray(pk['results_common'][ref]['mask'], bool)
        with netCDF4.Dataset(OUT / f'obs_{ref}.nc') as f:
            yo = np.asarray(f['year'][:]).astype(int); lat = np.asarray(f['lat'][:], 'f8')
            lon = np.asarray(f['lon'][:], 'f8')
            po, to, so = (np.ma.filled(f[k][:].astype('f8'), np.nan) for k in ('pco2', 'thetao', 'so'))
        with netCDF4.Dataset(OUT / f'states_{ref}_hist.nc') as f:
            ym = np.asarray(f['year'][:]).astype(int)
            pm, tm, sm = (np.ma.filled(f[k][:].astype('f8'), np.nan) for k in ('pco2', 'thetao', 'so'))
            hlat, hlon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        with netCDF4.Dataset(OUT / f'states_{ref}_pi.nc') as f:
            names = [str(x) for x in f['state'][:]]
            k = next(j for j, nm in enumerate(names) if '1750' in nm)
            x50 = float(f['state_xco2'][k])
            p50, t50, s50 = (np.ma.filled(f[f'{q}_state'][k].astype('f8'), np.nan) for q in ('pco2', 'thetao', 'so'))
            plat, plon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        # states onto the grid of the obs file (the longitude origin may differ)
        rg = lambda la, lo: (np.abs(lat[:, None] - la[None, :]).argmin(1),
                             np.abs(((lon[:, None] - lo[None, :]) + 180) % 360 - 180).argmin(1))
        jy, ix = rg(hlat, hlon)
        pm, tm, sm = (a[:, jy][:, :, ix] for a in (pm, tm, sm))
        jy, ix = rg(plat, plon)
        p50, t50, s50 = (a[np.ix_(jy, ix)] for a in (p50, t50, s50))
        dis_o = np.array([gm(po[i] - xn[yn == y][0] * (1 - ph2o(to[i], so[i])), m, lat) if y in yn else np.nan
                          for i, y in enumerate(yo)])
        dis_m = np.array([gm(pm[i] - xh_[yh_ == y][0] * (1 - ph2o(tm[i], sm[i])), m, lat) for i, y in enumerate(ym)])
        dis50 = gm(p50 - x50 * (1 - ph2o(t50, s50)), m, lat)
        io = (yo >= TR[0]) & (yo <= TR[1]) & np.isfinite(dis_o); im = (ym >= TR[0]) & (ym <= TR[1])
        bo = 10 * np.polyfit(yo[io], dis_o[io], 1)[0]; bm = 10 * np.polyfit(ym[im], dis_m[im], 1)[0]
        a = lambda y0, y1, yy, dd: np.nanmean(dd[(yy >= y0) & (yy <= y1)])
        print(f'  {ref:9s} 1750 {dis50:+5.1f}; models 1850-1869 {a(1850, 1869, ym, dis_m):+5.1f}, 1950-1969 '
              f'{a(1950, 1969, ym, dis_m):+5.1f}, 1990-1999 {a(1990, 1999, ym, dis_m):+5.1f}, 2005-2014 '
              f'{a(2005, 2014, ym, dis_m):+5.1f}; products 1990-1999 {a(1990, 1999, yo, dis_o):+5.1f}, 2005-2014 '
              f'{a(2005, 2014, yo, dis_o):+5.1f}; trend {TR[0]}-{TR[1]} models {bm:+.2f}, products {bo:+.2f} uatm per decade',
              flush=True)


if __name__ == '__main__':
    part2()
    part1()
