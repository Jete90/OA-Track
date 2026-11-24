#!/usr/bin/env python
"""
check_annual_state_omega.py
===========================
The models enter the chain as annual means (step 1: one value per year of T, S, DIC, TA), so their Omega_arag
(1750 and historical) is the saturation state of the annual-mean state. The observed Omega_arag of OceanSODA-ETHZ
(and so the 2024 value of the decline) is the annual mean of the monthly saturation states. This script measures
the difference between the two on the observations themselves, per year:

    J(year) = global mean of [ mean of the 12 monthly Omega_arag ] - global mean of [ Omega_arag(annual-mean state) ]

over the OceanSODA constraint mask (results[OceanSODA]['mask'] of constraint_results.pkl), GLODAP nutrients, the
chemistry of the run. If J is not negligible, the decline mixes two definitions (2024: mean of monthly; 1750: annual
state). A J that is constant in time would still bias the decline by about 100 * J / Omega_1750 percentage points.

Usage (code/):  OA_RUN=u74 python -u check_annual_state_omega.py [--years 1985 2000 2014 2024]
"""
import argparse
import pickle
from pathlib import Path

import netCDF4
import numpy as np

import oa_config
import calc_omega_arag_obs as co


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=[1985, 1995, 2005, 2014, 2024])
    a = ap.parse_args()
    with open(Path(oa_config.DATA) / 'constraint_results.pkl', 'rb') as f:
        R = pickle.load(f)['results']['OceanSODA']
    mask = np.asarray(R['mask']).astype(bool)
    pi = float(R['constrained'][list(R['constrained'])[0]][0])
    with netCDF4.Dataset(co.SRC) as f:
        lat = np.asarray(f['lat'][:], 'f8')
        lon = np.asarray(f['lon'][:], 'f8')
        tv = f['time']
        dates = netCDF4.num2date(tv[:], tv.units, getattr(tv, 'calendar', 'standard'))
        yrs = np.array([d.year for d in dates])
        si, po4 = co.load_nutrients(lat, lon)
        lat2 = np.broadcast_to(lat[:, None], mask.shape)
        assert mask.shape == (lat.size, lon.size), 'mask and product grid differ'
        w = np.cos(np.deg2rad(lat2))
        print(f'OceanSODA constraint mask: {int(mask.sum())} cells; constrained 1750 {pi:.4f}')
        for y in a.years:
            idx = np.where(yrs == y)[0]
            if idx.size != 12:
                print(f'{y}: {idx.size} months, skipped')
                continue
            i0, i1 = idx[0], idx[-1] + 1
            X = {k: co.read(f[v], i0, i1) for k, v in (('t', 'temperature'), ('s', 'salinity'), ('dic', 'dic'),
                                                        ('alk', 'talk'))}
            ok = mask & np.isfinite(si) & np.isfinite(po4)
            for k in X:
                ok &= np.all(np.isfinite(X[k]), axis=0)
            ok &= np.all(X['s'] > 0, axis=0) & np.all(X['dic'] > 0, axis=0) & np.all(X['alk'] > 0, axis=0)
            n = int(ok.sum())
            sil, phos, la = si[ok] * 1e-6, po4[ok] * 1e-6, lat2[ok]
            om_m = np.array([co.omega_cells(X['t'][m][ok], X['s'][m][ok], X['dic'][m][ok] * 1e-6,
                                            X['alk'][m][ok] * 1e-6, sil, phos, la) for m in range(12)])
            om_mean_monthly = om_m.mean(axis=0)
            mean_state = {k: X[k].mean(axis=0)[ok] for k in X}
            om_annual_state = co.omega_cells(mean_state['t'], mean_state['s'], mean_state['dic'] * 1e-6,
                                             mean_state['alk'] * 1e-6, sil, phos, la)
            good = np.isfinite(om_mean_monthly) & np.isfinite(om_annual_state)
            ww = w[ok][good]
            g1 = float(np.sum(om_mean_monthly[good] * ww) / ww.sum())
            g2 = float(np.sum(om_annual_state[good] * ww) / ww.sum())
            d = om_mean_monthly[good] - om_annual_state[good]
            print(f'{y}: mean of monthly {g1:.4f}, annual-mean state {g2:.4f}, J = {g1 - g2:+.4f} '
                  f'({100 * (g1 - g2) / pi:+.2f} points of decline relative to 1750); local 5-95 % '
                  f'{np.percentile(d, 5):+.4f} to {np.percentile(d, 95):+.4f}; {n} cells')


if __name__ == '__main__':
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
