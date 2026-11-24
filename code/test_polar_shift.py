#!/usr/bin/env python
"""
test_polar_shift.py
===================
How much larger is the decline 1750-2024 over the whole ocean than over the common area of the paper, and how well
is that shift known? Cell by cell, so that each part of the ocean outside the common area is estimated by the
products that actually cover it.

Per product and cell: the constrained 1750 Omega_arag (states_<ref>_pi.nc, state 1750) and the 2024 Omega_arag
from a quadratic fit of the product's years up to 2024, evaluated in 2024 (as the chain does for the area means).
  common area:   the paper's mask (constraint_results.pkl)
  outside:       the ocean (regional_common.ocean_on) minus the common area; each cell gets the mean of the products
                 that cover it; cells covered by none get the mean of their region (Arctic north of 60 N, Southern
                 Ocean south of 55 S, other) or are left out (variant)
Declines from area means of Omega_arag, cos(latitude). Central: mean of the products in every cell. Uncertainty:
each product alone (its own cells; elsewhere outside the common area the mean of the others), the spread of the
four shifts.

Usage (code/):  OA_RUN=u74 python -u test_polar_shift.py
"""
import pickle

import netCDF4
import numpy as np

import oa_config
import regional_common as rc

OUT = oa_config.DATA / 'product'
PRODUCTS = ('OceanSODA', 'UExP', 'JMA', 'CMEMS')
YEAR = 2024


def onto(lat, lon, la, lo, a):
    jy = np.abs(lat[:, None] - la[None, :]).argmin(1)
    ix = np.abs(((lon[:, None] - lo[None, :]) + 180) % 360 - 180).argmin(1)
    return a[..., jy, :][..., ix]


def fields(ref, lat, lon):
    with netCDF4.Dataset(OUT / f'states_{ref}_pi.nc') as f:
        names = [str(x) for x in f['state'][:]]
        k = next(j for j, nm in enumerate(names) if '1750' in nm)
        P = onto(lat, lon, np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8'),
                 np.ma.filled(f['omega_arag_state'][k].astype('f8'), np.nan))
    with netCDF4.Dataset(OUT / f'obs_{ref}.nc') as f:
        yo = np.asarray(f['year'][:]).astype(int)
        om = np.ma.filled(f['omega_arag'][:].astype('f8'), np.nan)
        om = onto(lat, lon, np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8'), om)
    sel = yo <= YEAR
    y, a = yo[sel] - YEAR, om[sel]
    ok = np.all(np.isfinite(a), axis=0)
    N = np.full(ok.shape, np.nan)
    A = a[:, ok]
    X = np.column_stack([np.ones(y.size), y, y ** 2])
    b, *_ = np.linalg.lstsq(X, A, rcond=None)
    N[ok] = b[0]                                         # the fit evaluated in 2024 (x = 0)
    good = ok & np.isfinite(P)
    return np.where(good, P, np.nan), np.where(good, N, np.nan)


def decline(P, N, m, w):
    mm = m & np.isfinite(P) & np.isfinite(N)
    return 100 * (1 - np.sum(N[mm] * w[mm]) / np.sum(P[mm] * w[mm])), mm


def main():
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    rcm = pk['results_common']['UExP']
    lat, lon = np.asarray(rcm['lat'], 'f8'), np.asarray(rcm['lon'], 'f8')
    common = np.asarray(rcm['mask'], bool)
    ocean = rc.ocean_on(lat, lon).astype(bool)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones(ocean.shape)
    outside = ocean & ~common
    la2 = np.broadcast_to(lat[:, None], ocean.shape)
    region = np.where(la2 > 60, 'Arctic', np.where(la2 < -55, 'Southern', 'other'))
    F = {r: fields(r, lat, lon) for r in PRODUCTS}
    Ps = np.array([F[r][0] for r in PRODUCTS]); Ns = np.array([F[r][1] for r in PRODUCTS])
    cov = np.isfinite(Ps) & np.isfinite(Ns)
    n_cov = cov.sum(axis=0)
    A = lambda m: 100 * (w * m).sum() / (w * ocean).sum()
    print(f'ocean {A(ocean):.1f} %, common {A(common & ocean):.1f} %, outside {A(outside):.1f} % of the ocean; outside covered '
          f'by at least one product {A(outside & (n_cov > 0)):.1f} %, by none {A(outside & (n_cov == 0)):.1f} %')
    for g in ('Arctic', 'Southern', 'other'):
        mg = outside & (region == g)
        print(f'  outside, {g:8s}: {A(mg):.1f} % of the ocean, covered {A(mg & (n_cov > 0)):.1f} %; products covering: '
              + ', '.join(f'{r} {A(mg & cov[i]):.1f}' for i, r in enumerate(PRODUCTS)))

    def complete(P, N, fill=True):
        """outside the common area: cells without a value get the mean of their region (or stay empty)"""
        P, N = P.copy(), N.copy()
        if fill:
            for g in ('Arctic', 'Southern', 'other'):
                mg = outside & (region == g)
                have = mg & np.isfinite(P) & np.isfinite(N)
                miss = mg & ~(np.isfinite(P) & np.isfinite(N))
                if have.any() and miss.any():
                    P[miss] = np.sum(P[have] * w[have]) / w[have].sum()
                    N[miss] = np.sum(N[have] * w[have]) / w[have].sum()
        return P, N

    with np.errstate(invalid='ignore'):
        Pm, Nm = np.nanmean(Ps, axis=0), np.nanmean(Ns, axis=0)
    print('\ncentral (mean of the products in every cell):')
    for fill in (True, False):
        P, N = complete(Pm, Nm, fill)
        dc, _ = decline(P, N, common, w)
        dg, mm = decline(P, N, ocean, w)
        print(f'  {"uncovered cells filled with the regional mean" if fill else "uncovered cells left out":46s}: common {dc:.2f} %, '
              f'whole ocean {dg:.2f} % ({A(mm):.1f} % of the ocean), shift {dg - dc:+.2f} points')
        if fill:
            for g in ('Arctic', 'Southern', 'other'):
                mg = outside & (region == g)
                d, _ = decline(P, N, mg, w)
                print(f'      outside {g:8s}: decline {d:.2f} %')
    print('\neach product alone (its cells; elsewhere outside the common area the mean of the others):')
    sh = []
    for i, r in enumerate(PRODUCTS):
        with np.errstate(invalid='ignore'):
            Po = np.nanmean(np.delete(Ps, i, axis=0), axis=0); No = np.nanmean(np.delete(Ns, i, axis=0), axis=0)
        P = np.where(cov[i], Ps[i], np.where(outside, Po, np.nan))
        N = np.where(cov[i], Ns[i], np.where(outside, No, np.nan))
        P, N = complete(P, N, True)
        dc, _ = decline(P, N, common, w)
        dg, _ = decline(P, N, ocean, w)
        sh.append(dg - dc)
        print(f'  {r:9s}: common {dc:.2f} %, whole ocean {dg:.2f} %, shift {dg - dc:+.2f} points')
    sh = np.array(sh)
    print(f'shift: products {sh.mean():+.2f} +- {sh.std(ddof=1):.2f} (1 sigma of the four), range {sh.min():+.2f} to {sh.max():+.2f}')


if __name__ == '__main__':
    main()
