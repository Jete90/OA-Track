#!/usr/bin/env python
"""
test_sampling_polar.py
======================
Two of the structural uncertainties, on the data of the chain.

A  Sampling effect on the products' trends. The decline uses the product's 2024 state relative to its bias window
   (OceanSODA, UExP 1983-2014; CMEMS 1985-2014; JMA 1990-2014). Any error in how the product evolves from its
   window to 2024 enters directly. Compared here for the air-sea disequilibrium (ocean pCO2 minus atmospheric pCO2,
   moist air, common area of the paper, cos(latitude)):
     product: obs_<ref>.nc with NOAA CO2
     models:  the constrained states of the same product (states_<ref>_hist.nc to 2014, states_<ref>_ssp245.nc
              2015-2024) with the CMIP6 CO2 history and SSP2-4.5
   E = (product 2024 - product window mean) - (models 2024 - models window mean), 2024 from a linear fit over the
   last 10 years (2015-2024) of each series. Effect on the decline: -0.17 x E points (0.17 points per uatm,
   test_disequilibrium_alk.py): a product lag growing faster than the models' (E < 0) makes the product's 2024
   Omega_arag too high relative to its window and the decline too small, if the models are right.
   Also the linear trends 1990-2024 of both.

B  The polar shift. For each product with a larger own area than the common area: the decline 1750-2024 on its own
   area and on the common area (constraint_results.pkl), the decline it implies for the part outside the common
   area, and the decline it implies for the whole ocean (the common area plus the rest at that product's decline
   outside). The spread of these whole-ocean estimates is the uncertainty of the shift. Declines from area means
   of Omega_arag (decline = 1 - mean 2024 / mean 1750), the weights of the two parts from their area and 1750 mean.

Usage (code/):  OA_RUN=u74 python -u test_sampling_polar.py
"""
import pickle

import netCDF4
import numpy as np

import oa_config
import co2_history
import regional_common as rc
from test_alk_diseq_models import series, ph2o

OUT = oa_config.DATA / 'product'
PRODUCTS = ('OceanSODA', 'UExP', 'JMA', 'CMEMS')
WINDOW = {'OceanSODA': (1983, 2014), 'UExP': (1983, 2014), 'JMA': (1990, 2014), 'CMEMS': (1985, 2014)}
SENS = 0.17
END = (2015, 2024)


def onto(lat, lon, la, lo, a):
    jy = np.abs(lat[:, None] - la[None, :]).argmin(1)
    ix = np.abs(((lon[:, None] - lo[None, :]) + 180) % 360 - 180).argmin(1)
    return a[..., jy, :][..., ix]


def gm(a, m, lat):
    w = np.cos(np.deg2rad(lat))[:, None] * m
    ok = np.isfinite(a) & (w > 0)
    return float(np.sum(np.where(ok, a, 0) * w) / np.sum(w * ok))


def model_series(ref, lat, lon, m, atm):
    ys, ds = [], []
    for part in ('hist', 'ssp245'):
        with netCDF4.Dataset(OUT / f'states_{ref}_{part}.nc') as f:
            yy = np.asarray(f['year'][:]).astype(int)
            la, lo = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
            keep = np.where((yy <= 2014) if part == 'hist' else ((yy >= 2015) & (yy <= 2024)))[0]
            for i in keep:
                p, t, s = (onto(lat, lon, la, lo, np.ma.filled(f[k][i].astype('f8'), np.nan)) for k in ('pco2', 'thetao', 'so'))
                ys.append(yy[i]); ds.append(gm(p - atm[yy[i]] * (1 - ph2o(t, s)), m, lat))
    return np.array(ys), np.array(ds)


def part_a(pk):
    print('\n===== A  sampling: product against constrained models, disequilibrium on the common area (uatm)')
    yn, xn = series(co2_history.noaa())
    yh, xh = series(co2_history.historical()); ys, xs = series(co2_history.scenario('ssp245'))
    atm = dict(zip(yh, xh)); atm.update({y: x for y, x in zip(ys, xs) if y not in atm})
    noaa = dict(zip(yn, xn))
    effects = []
    for ref in PRODUCTS:
        m = np.asarray(pk['results_common'][ref]['mask'], bool)
        with netCDF4.Dataset(OUT / f'obs_{ref}.nc') as f:
            yo = np.asarray(f['year'][:]).astype(int)
            lat, lon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
            po, to, so = (np.ma.filled(f[k][:].astype('f8'), np.nan) for k in ('pco2', 'thetao', 'so'))
        okp = np.array([y in noaa and y <= 2024 for y in yo])
        dp = np.array([gm(po[i] - noaa[y] * (1 - ph2o(to[i], so[i])), m, lat) for i, y in enumerate(yo) if okp[i]])
        yp = yo[okp]
        ym, dm = model_series(ref, lat, lon, m, atm)
        w0, w1 = WINDOW[ref]
        fit24 = lambda y, d: np.polyval(np.polyfit(y[(y >= END[0]) & (y <= END[1])], d[(y >= END[0]) & (y <= END[1])], 1), 2024)
        wm = lambda y, d: d[(y >= w0) & (y <= w1)].mean()
        cp = fit24(yp, dp) - wm(yp, dp); cm_ = fit24(ym, dm) - wm(ym, dm)
        E = cp - cm_
        tr = lambda y, d: 10 * np.polyfit(y[(y >= 1990) & (y <= 2024)], d[(y >= 1990) & (y <= 2024)], 1)[0]
        effects.append(-SENS * E)
        print(f'  {ref:9s} window {w0}-{w1}: change window mean -> 2024 product {cp:+.1f}, models {cm_:+.1f}, E {E:+.1f} uatm '
              f'-> effect on the decline {-SENS * E:+.2f} points; trend 1990-2024 product {tr(yp, dp):+.2f}, models '
              f'{tr(ym, dm):+.2f} uatm per decade', flush=True)
    e = np.array(effects)
    print(f'  mean of the four products {e.mean():+.2f} points (the paper uses the mean of the four), range '
          f'{e.min():+.2f} to {e.max():+.2f}')


def part_b(pk):
    print('\n===== B  polar shift: decline on own area, common area, outside, and implied for the whole ocean')
    out = []
    for ref in PRODUCTS + ('Jiang',):
        if ref not in pk['results'] or ref not in pk['results_common']:
            continue
        ro, rcm = pk['results'][ref], pk['results_common'][ref]
        lat, lon = np.asarray(rcm['lat'], 'f8'), np.asarray(rcm['lon'], 'f8')
        w = np.cos(np.deg2rad(lat))[:, None] * np.ones((lat.size, lon.size))
        ocean = rc.ocean_on(lat, lon).astype(bool)
        A = lambda m: float((w * m).sum() / (w * ocean).sum())
        mo, mc = np.asarray(ro['mask'], bool), np.asarray(rcm['mask'], bool)
        k = list(ro['constrained'])[0]
        P_o, P_c = float(ro['constrained'][k][0]), float(rcm['constrained'][k][0])
        N_o, N_c = float(ro['now']), float(rcm['now'])
        a_o, a_c = A(mo & ocean), A(mc & ocean)
        D_o, D_c = 100 * (1 - N_o / P_o), 100 * (1 - N_c / P_c)
        if a_o - a_c < 0.01:
            print(f'  {ref:9s} own area {100 * a_o:.1f} %, common {100 * a_c:.1f} % of the ocean: no area outside to use')
            continue
        # the part outside: area means from the two areas
        P_x = (P_o * a_o - P_c * a_c) / (a_o - a_c); N_x = (N_o * a_o - N_c * a_c) / (a_o - a_c)
        D_x = 100 * (1 - N_x / P_x)
        # whole ocean: the common area plus the rest of the ocean at this product's outside state
        a_r = 1 - a_c
        D_g = 100 * (1 - (N_c * a_c + N_x * a_r) / (P_c * a_c + P_x * a_r))
        out.append(D_g - D_c)
        print(f'  {ref:9s} own {100 * a_o:.1f} %, common {100 * a_c:.1f} % of the ocean; decline own {D_o:.2f}, common '
              f'{D_c:.2f}, outside {D_x:.2f} % (1750 {P_x:.2f}, now {N_x:.2f}); whole ocean {D_g:.2f} %, shift '
              f'{D_g - D_c:+.2f} points', flush=True)
    if out:
        o = np.array(out)
        print(f'  shift to the whole ocean: mean {o.mean():+.2f}, sd {o.std(ddof=1) if o.size > 1 else np.nan:.2f}, range '
              f'{o.min():+.2f} to {o.max():+.2f} points ({o.size} estimates)')


if __name__ == '__main__':
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    part_b(pk)
    part_a(pk)
