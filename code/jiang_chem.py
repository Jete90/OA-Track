#!/usr/bin/env python
"""
jiang_chem.py
=============
The surface states of Jiang et al. (2023, NCEI 0259391, median of 14 ESMs) recomputed with mocsy, as every other
carbonate-system calculation of this study.

Their states are defined by surface fCO2 and alkalinity (with temperature and salinity): fCO2 follows the
atmospheric CO2 of the year, DIC is derived. omega() therefore solves DIC with mocsy until the surface fCO2 equals
theirs (shifted by dx where asked) for the constants given, and returns Omega_arag of that state. This is the same
step as in calc_pi_xco2.py (DIC to a target pCO2) and in make_product_alkalinity.py (DIC at constant pCO2): DIC first,
then Omega_arag from DIC and alkalinity, both with the same constants.

Used for
  - the conversion of their published values to the constants of this study: omega() with our constants divided by
    omega() with theirs (K1, K2 of Lueker et al. 2000, total boron of Lee et al. 2010), and
  - their 1750 and 1850 states at other CO2 values, with their method: their 1850 fields (CMIP6 forcing, 284.59 ppm
    in 1850-1854) with fCO2 shifted by the difference of the target CO2; their own 1750 state is the shift by the
    ice-core difference 1852 - 1752 (288.57 - 276.39 ppm), i.e. 272.41 ppm.

Set-up as calc_pi_xco2.py: mocsy 2.4.0, optt='Tpot', optp='m', depth 0, patm 1 atm, GLODAPv2 surface Si and PO4
(calc_omega_arag_obs.load_nutrients), temperature floored at the freezing point; DIC by Newton iteration with the
Revelle factor, starting from their DIC of the same year, until max |fCO2 - target| < FCO2_TOL.

Input:  data/<run>/jiang2023/{fCO2,TA,DIC,Temperature,Salinity,Aragonite}_median_historical.nc
Usage (code/, checks against their published fields and prints the factors used by the notebook):
        OA_RUN=u74 python jiang_chem.py
"""
import netCDF4
import numpy as np

import oa_config
import calc_omega_arag_obs as co     # mocsy path, nutrients
import calc_omega_arag as cm         # freezing-point floor, validity of the chemistry
import mocsy

DIR = oa_config.DATA / 'jiang2023'
JIANG = dict(optb='l10', optk1k2='l')                               # the constants of Jiang et al. (2023)
OURS = dict(optb=oa_config.OPTB, optk1k2=oa_config.OPTK1K2)         # the constants of this run
X1850_CMIP6, X1752, X1852 = 284.59, 276.39, 288.57                  # ppm (CMIP6 forcing 1850-1854, ice cores)
X_J1750 = X1850_CMIP6 - (X1852 - X1752)                             # 272.41 ppm, the 1750 state of Jiang et al.
N_ITER_MAX = 20
FCO2_TOL = 1e-4                                                     # uatm

_CACHE = {}


def _read(var, year):
    """field of Jiang et al. for year (their time axis: the first year of the decade), longitudes -180..180 sorted"""
    with netCDF4.Dataset(DIR / f'{var}_median_historical.nc') as f:
        name = [k for k, v in f.variables.items() if v.ndim == 3][0]
        t = np.asarray(f['time'][:]).astype(int)
        lat = np.asarray(f['latitude'][:], 'f8')
        lon = np.asarray(f['longitude'][:], 'f8')
        x = np.ma.filled(f[name][list(t).index(year)].astype('f8'), np.nan)
    lat = lat[:, 0] if lat.ndim == 2 else lat
    lon = (((lon[0] if lon.ndim == 2 else lon) + 180.0) % 360.0) - 180.0
    o = np.argsort(lon)
    x = x[:, o]
    x[~np.isfinite(x) | (np.abs(x) > 1e5)] = np.nan
    return lat, lon[o], x


def grid():
    """lat, lon (-180..180, sorted) of the fields of Jiang et al."""
    lat, lon, _ = _read('fCO2', 1850)
    return lat, lon


def field(var, year):
    return _read(var, year)[2]


def _mvars(t, s, dic, alk, sil, phos, lat, optb, optk1k2):
    n = t.size
    return mocsy.mvars(temp=cm.floor_temperature(t, s, count=False), sal=s, alk=alk, dic=dic, sil=sil, phos=phos,
                       patm=np.ones(n), depth=np.zeros(n), lat=lat,
                       optcon='mol/kg', optt='Tpot', optp='m', optb=optb, optk1k2=optk1k2)


def omega(year, dx=0.0, optb=JIANG['optb'], optk1k2=JIANG['optk1k2']):
    """Omega_arag (lat, lon) of the state of Jiang et al. of year with surface fCO2 + dx (ppm, taken as uatm), DIC
    solved with mocsy for the given constants; NaN where any input is missing or the chemistry fails"""
    key = (year, round(float(dx), 6), optb, optk1k2)
    if key in _CACHE:
        return _CACHE[key]
    lat, lon, fc = _read('fCO2', year)
    ta, dic0, t, s = (field(v, year) for v in ('TA', 'DIC', 'Temperature', 'Salinity'))
    if np.nanmean(t) > 100:
        t = t - 273.15
    si, po4 = co.load_nutrients(lat, lon)
    la = np.broadcast_to(lat[:, None], fc.shape)
    ok = np.isfinite(fc) & np.isfinite(ta) & np.isfinite(dic0) & np.isfinite(t) & np.isfinite(s) & (s > 0) \
        & np.isfinite(si) & np.isfinite(po4)
    target = fc[ok] + dx
    args = (t[ok], s[ok])
    a, sil, phos = ta[ok] * 1e-6, si[ok] * 1e-6, po4[ok] * 1e-6
    d = dic0[ok] * 1e-6
    r = _mvars(*args, d, a, sil, phos, la[ok], optb, optk1k2)
    for it in range(N_ITER_MAX):
        f_, rv = np.asarray(r[2], 'f8'), np.asarray(r[8], 'f8')
        err = np.abs(f_ - target)
        if np.nanmax(err) < FCO2_TOL:
            break
        with np.errstate(divide='ignore', invalid='ignore'):
            d = d + (target - f_) * d / (rv * f_)                         # Newton with the Revelle factor
        r = _mvars(*args, d, a, sil, phos, la[ok], optb, optk1k2)
    f_, om_ = np.asarray(r[2], 'f8'), np.asarray(r[6], 'f8')
    good = cm.valid_carbonate(np.asarray(r[0], 'f8'), om_) & (np.abs(f_ - target) < 10 * FCO2_TOL)
    out = np.full(fc.shape, np.nan)
    out[ok] = np.where(good, om_, np.nan)
    _CACHE[key] = out
    return out


def main():
    lat, lon = grid()
    w = oa_config.area_weights(lat, lon)
    pub = {y: field('Aragonite', y) for y in (1750, 1850, 2010)}
    gm = lambda x, m: float(np.sum(x[m] * w[m]) / w[m].sum())
    print(f'Jiang et al. (2023) recomputed with mocsy (DIC solved for their fCO2); weights {oa_config.WEIGHTS}')
    for y, dx, lab in ((1850, 0.0, '1850 (284.59 ppm)'), (1850, X_J1750 - X1850_CMIP6, '1750 (272.41 ppm, their method)'),
                       (2010, 0.0, '2010s')):
        p = pub[1750 if lab.startswith('1750') else y]
        om_j = omega(y, dx, **JIANG)
        m = np.isfinite(om_j) & np.isfinite(p)
        print(f'  {lab:32s} published {gm(p, m):.4f}, recomputed with their constants {gm(om_j, m):.4f} '
              f'(cells: mean diff {np.mean((om_j - p)[m]):+.4f}, max |diff| {np.max(np.abs(om_j - p)[m]):.4f}); '
              f'with ours ({OURS["optb"]}, {OURS["optk1k2"]}): factor {gm(omega(y, dx, **OURS), m) / gm(om_j, m):.4f}')
    b = omega(1850, X_J1750 - X1850_CMIP6, **JIANG)
    for x in (278.3, 285.5):
        o_ = omega(1850, x - X1850_CMIP6, **JIANG)
        m = np.isfinite(o_) & np.isfinite(b)
        ref = b if x < 280 else omega(1850, 0.0, **JIANG)
        lab = 'their 1750 (272.41 ppm)' if x < 280 else 'their 1850 (284.59 ppm)'
        print(f'  {x:.1f} ppm relative to {lab}: {gm(o_, m) / gm(ref, m):.4f} (their constants)')


if __name__ == '__main__':
    main()
