"""
projection.py
=============
The projections of the surface carbonate system that the figures use, computed once per run and cached in the
run's data folder (delete a cache file to recompute it). Every figure cell calls these functions instead of
reading the cache directly, so the cells can run in any order; the preparation cell of the notebook calls them
first so that the slow part happens there.

    global_projection()     per product: global mean Omega_arag on the common area from the forced 2024 state (in every
                            cell the quadratic fit of the product's annual means up to 2024, evaluated in 2024, for T, S,
                            DIC and ALK, as the 2024 maps of the data product), surface pCO2 raised with the atmosphere at fixed ALK and S (DIC by Newton
                            iteration with mocsy, as in calc_pi_xco2.py), with fixed T ('fixed') and with T rising
                            with CO2 at the observed rate of 1990-2024 ('scaled'); x in ppm
                            -> fig_crossing_projection.pkl  (Figs. 1 and 4, the crossing against CO2)
    regional_projection()   the same per cell, T rising with CO2 -> fig_regional_crossing_projection.pkl (common area);
                            regional_projection('results'): each product on its own area -> ..._own.pkl (Fig. 4 maps)
    noaa_co2()              {year: ppm} of the NOAA global annual mean (downloaded once to co2_annmean_gl.txt)

"""
import os
import pickle
import urllib.request

import numpy as np

import regional_common as rc
import calc_bias_adjusted as cb
import calc_omega_arag_obs as co
import calc_pi_xco2 as cpx
import make_product as mp            # quad_fit_year: the fitted 2024 state, the same function as the data product
import oa_config
_WTAG = ('' if oa_config.WEIGHTS == 'ocean' else f'_{oa_config.WEIGHTS}') + '_fit2024'   # caches keyed by the weights and the start

STATE_YEAR = 2024                    # the projections start from the forced (fitted) state of this year
STATE_OLD = (2020, 2024)             # the 2020-2024 mean state as the start: printed for comparison only
TREND = (1990, 2024)                 # period for the observed warming per ppm (k_T)
XMAX = 480.0                         # right end of the global projection (ppm)
XS = np.arange(0.0, XMAX - 300.0, 2.0)   # global: ppm above the state
DX = np.arange(0.0, 202.0, 2.0)          # per cell: ppm above the state (to about 620 ppm)
VARS = cb.VARS
NOAA_URL = 'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_gl.txt'


def noaa_co2():
    path = f'{rc.ROOT}/co2_annmean_gl.txt'
    if not os.path.exists(path):
        urllib.request.urlretrieve(NOAA_URL, path)
    co2 = np.array([[float(v) for v in l.split()[:2]] for l in open(path) if l.strip() and not l.startswith('#')])
    return dict(zip(co2[:, 0].astype(int), co2[:, 1]))


# CO2 growth rate for converting ppm into years: 2.6 ppm/yr, a round value for constant emissions at the 2024 level.
# Global Carbon Budget 2025 (Friedlingstein et al. 2025): atmospheric growth regressed linearly on fossil + land-use
# emissions, 1990-2024, gives 2.62 +- 0.13 ppm/yr at the 2024 emissions (a quadratic fit gives the same, 2.61); the
# mean airborne fraction of 2015-2024 (0.49) also gives 2.6; NOAA global growth 2016-2025 2.6 ppm/yr. A constant
# airborne fraction since 1990 (0.44) would give 2.4 ppm/yr.
CO2_GROWTH = 2.6


def co2_growth(verbose=True):
    """(growth ppm/yr,) for constant emissions at the 2024 level (CO2_GROWTH)"""
    if verbose:
        print(f'CO2 growth for constant emissions: {CO2_GROWTH:.1f} ppm/yr')
    return (CO2_GROWTH,)


def _results(which='results_common'):
    """which: 'results_common' (the common area of the four products) or 'results' (each product on its own area)"""
    with open(f'{rc.ROOT}/constraint_results.pkl', 'rb') as f:
        return pickle.load(f)[which]


def _state(r, ref, xco2):
    """the forced 2024 state of a product on the mask of r (common or own area): in every cell the quadratic fit of the
    annual means of the product's record up to STATE_YEAR, evaluated in STATE_YEAR (mp.quad_fit_year, as the 2024 maps
    of the data product, make_product.py); its warming per ppm over TREND; and, for comparison, the
    STATE_OLD mean state"""
    cb.set_reference(ref)
    lat, lon, m = np.asarray(r['lat'], 'f8'), np.asarray(r['lon'], 'f8'), np.asarray(r['mask']).astype(bool)
    lat2 = np.broadcast_to(lat[:, None], m.shape)
    w = oa_config.area_weights(lat, lon) * m
    gm = lambda a: np.nansum(np.where(m & np.isfinite(a), a, 0) * w) / w[np.isfinite(a) & m].sum()
    si, po4 = co.load_nutrients(lat, lon)
    ry, og = np.asarray(r['year']).astype(int), np.asarray(r['ogm'], 'f8')
    first = int(ry[np.isfinite(og)][0])                       # the first complete year of the product's record
    ann = cb.obs_fields(first, STATE_YEAR, step=1)            # annual means, degC psu umol/kg
    yrs = np.arange(first, STATE_YEAR + 1)
    st = {v: mp.quad_fit_year(yrs, ann[v], STATE_YEAR) for v in VARS}
    old = (yrs >= STATE_OLD[0]) & (yrs <= STATE_OLD[1])
    st_old = {v: ann[v][old].mean(axis=0) for v in VARS}
    x_state = float(xco2[STATE_YEAR])
    tr = yrs >= TREND[0]
    k_T = np.polyfit([xco2[y] for y in yrs[tr]], [gm(ann['thetao'][i]) for i in np.nonzero(tr)[0]], 1)[0]
    ok = m & np.isfinite(si) & np.isfinite(po4)
    for v in VARS:
        ok &= np.isfinite(st[v]) & np.isfinite(st_old[v])
    pack = lambda S_: dict(t0=S_['thetao'][ok], s0=S_['so'][ok], d0=S_['dissic'][ok] * 1e-6, a0=S_['talk'][ok] * 1e-6,
                           sil=si[ok] * 1e-6, phos=po4[ok] * 1e-6, la=lat2[ok])
    return dict(lat=lat, lon=lon, m=m, ok=ok, w=w, x_state=x_state, k_T=k_T, cells=pack(st), cells_old=pack(st_old),
                first=first)


def _omega(c, p0, dx, kT):
    """Omega_arag of the cells with pCO2 raised by dx ppm (water-vapour corrected) and T by kT * dx"""
    t = c['t0'] + kT * dx
    target = p0 + dx * (1.0 - cpx.water_vapour(t, c['s0']))
    d = c['d0'].copy()
    p, om, rv = cpx.mvars_cells(t, c['s0'], d, c['a0'], c['sil'], c['phos'], c['la'])
    for _ in range(6):
        d = d + (target - p) * d / (rv * p)
        p, om, rv = cpx.mvars_cells(t, c['s0'], d, c['a0'], c['sil'], c['phos'], c['la'])
    return om


def global_projection():
    cache = f'{rc.ROOT}/fig_crossing_projection{_WTAG}.pkl'
    proj = pickle.load(open(cache, 'rb')) if os.path.exists(cache) else {}
    RES = _results()
    missing = [q for q in RES if q not in proj]
    if not missing:
        return proj
    xco2 = noaa_co2()
    for ref in missing:
        r = RES[ref]
        S = _state(r, ref, xco2)
        c = S['cells']
        p0, om0, _ = cpx.mvars_cells(c['t0'], c['s0'], c['d0'], c['a0'], c['sil'], c['phos'], c['la'])
        wk = S['w'][S['ok']]

        def gmean(om):
            good = np.isfinite(om)
            return np.sum(om[good] * wk[good]) / wk[good].sum()

        om_state = np.sum(om0 * wk) / wk.sum()
        now = rc.fit_now(r, STATE_YEAR)       # the fitted global mean of STATE_YEAR (annual global means, quadratic)
        shift = now - om_state                # global mean of the fitted cells against the fit of the global mean
        proj[ref] = dict(x=S['x_state'] + XS, x_state=S['x_state'], k_T=S['k_T'],
                         fixed=np.array([gmean(_omega(c, p0, dx, 0.0)) for dx in XS]) + shift,
                         scaled=np.array([gmean(_omega(c, p0, dx, S['k_T'])) for dx in XS]) + shift)
        # comparison with the STATE_OLD mean state as the start: decline per ppm over the first 10 ppm
        co_ = S['cells_old']
        p0o, om0o, _ = cpx.mvars_cells(co_['t0'], co_['s0'], co_['d0'], co_['a0'], co_['sil'], co_['phos'], co_['la'])
        rate = lambda c_, p_: (gmean(_omega(c_, p_, 0.0, S['k_T'])) - gmean(_omega(c_, p_, 10.0, S['k_T']))) / 10.0
        r_new, r_old = rate(c, p0), rate(co_, p0o)
        print(f'global projection {ref}: start = fitted {STATE_YEAR} state (record from {S["first"]}) at {S["x_state"]:.1f} ppm, '
              f'k_T {100 * S["k_T"]:.2f} °C per 100 ppm; Omega of the fitted cells {om_state:.4f}, fit of the global mean '
              f'{now:.4f} (shift {shift:+.4f}); decline per ppm over the first 10 ppm {r_new:.5f}, from the '
              f'{STATE_OLD[0]}-{STATE_OLD[1]} mean state {r_old:.5f} ({100 * (r_new / r_old - 1):+.2f} %)', flush=True)
        pickle.dump(proj, open(cache, 'wb'))
    return proj


def regional_projection(which='results_common'):
    """per cell, T rising with CO2; which: 'results_common' (common area) or 'results'
    (each product on its own area: the best-estimate maps of Fig. 4, mean of the products that cover each cell)"""
    tag = '' if which == 'results_common' else '_own'
    cache = f'{rc.ROOT}/fig_regional_crossing_projection{tag}{_WTAG}.pkl'
    proj = pickle.load(open(cache, 'rb')) if os.path.exists(cache) else {}
    RES = _results(which)
    missing = [q for q in RES if q not in proj]
    if not missing:
        return proj
    xco2 = noaa_co2()
    for ref in missing:
        S = _state(RES[ref], ref, xco2)
        c, m, ok = S['cells'], S['m'], S['ok']
        p0, _, _ = cpx.mvars_cells(c['t0'], c['s0'], c['d0'], c['a0'], c['sil'], c['phos'], c['la'])
        curves = np.full((DX.size,) + m.shape, np.nan, 'f4')
        for i, dx in enumerate(DX):
            f2 = np.full(m.shape, np.nan)
            f2[ok] = _omega(c, p0, dx, S['k_T'])
            curves[i] = rc.to_180(S['lon'], f2)[1]                 # same longitude order as rc.load()
        proj[ref] = dict(x=S['x_state'] + DX, x_state=S['x_state'], k_T=S['k_T'], curves=curves)
        pickle.dump(proj, open(cache, 'wb'))
        print(f'regional projection {ref} ({which}): done (state {S["x_state"]:.1f} ppm, k_T {100 * S["k_T"]:.2f} °C per 100 ppm)',
              flush=True)
    return proj
