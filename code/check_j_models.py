#!/usr/bin/env python
"""
check_j_models.py
=================
Does the offset J = (mean of the monthly Omega_arag) - (Omega_arag of the annual-mean state) change between an
early, near-preindustrial period and the present in the models? If J were the same at both ends, using
monthly values throughout would change the decline only by J (Omega_pi - Omega_now) / Omega_pi^2, i.e. about
0.01 percentage points. If J changes with CO2 (larger seasonal cycle of Omega_arag at lower buffer capacity),
the two definitions differ by more. This script measures it on the raw monthly historical runs that are on disk
(no drift correction or bias adjustment is needed: J depends only on the seasonal cycle within each year).

Per model with monthly T, S, DIC and TA of the historical run (CMIP6 names, original_grid/<model>/, regridded
'_r' files preferred, otherwise any monthly file on a regular lat-lon grid):
    early   1850-1869   (CO2 about 285-288 ppm, close to the preindustrial state)
    late    1995-2014   (CO2 about 360-397 ppm)
every year of both periods: monthly Omega_arag and Omega_arag of the annual-mean state (mocsy, same set-up and
constants as the chain, GLODAPv2 Si and PO4, top level, units converted as in calc_bias_adjusted.ModelRun.read).
Cells must have all 12 months valid in every year of both periods (fixed mask per model). Area-weighted means.

Printed per model: Omega_arag (annual state) early and late, J early and late, and the decline early -> late in
percent with both definitions and its difference (percentage points). Multi-model mean at the end.

Usage (code/):  OA_RUN=u74 python -u check_j_models.py [--models ...] [--early 1850 1869] [--late 1995 2014]
"""
import argparse
import glob
import os
import re
import warnings
from collections import defaultdict

import netCDF4
import numpy as np

import calc_omega_arag as cm
import calc_omega_arag_obs as co

import oa_config
ROOT = str(oa_config.MODEL_DIR / 'CMIP_DATA/CMIP6_DATA/original_grid')
MEMBERS = {'ACCESS-ESM1-5': 'r1i1p1f1', 'CanESM5-CanOE': 'r1i1p2f1', 'CanESM5': 'r1i1p1f1', 'CESM2': 'r1i1p1f1',
           'CESM2-WACCM': 'r1i1p1f1', 'CMCC-ESM2': 'r1i1p1f1', 'CNRM-ESM2-1': 'r1i1p1f2', 'EC-Earth3-CC': 'r1i1p1f1',
           'GFDL-CM4': 'r1i1p1f1', 'GFDL-ESM4': 'r1i1p1f1', 'IPSL-CM6A-LR': 'r1i1p1f1', 'MIROC-ES2L': 'r1i1p1f2',
           'MPI-ESM1-2-HR': 'r1i1p1f1', 'MPI-ESM1-2-LR': 'r1i1p1f1', 'MRI-ESM2-0': 'r1i2p1f1',
           'NorESM2-LM': 'r1i1p1f1', 'NorESM2-MM': 'r1i1p1f1', 'UKESM1-0-LL': 'r1i1p1f2'}
NAMES = {'thetao': ('thetao', 'tos'), 'so': ('so', 'sos'), 'dissic': ('dissic', 'dissicos'), 'talk': ('talk', 'talkos')}


class Monthly:
    """month-by-month access to one variable over several files: field(year, month) -> top level (ny, nx)"""

    def __init__(self, files, name):
        self.idx, self.ds = {}, []
        for p in files:
            d = netCDF4.Dataset(p)
            t = d['time']
            ys, ms = cm.year_month(t[:], t.units, getattr(t, 'calendar', 'standard'))
            per = defaultdict(int)
            for y in ys:
                per[int(y)] += 1
            if max(per.values()) < 12:              # annual file
                d.close()
                continue
            self.ds.append(d)
            for i, (y, m) in enumerate(zip(ys, ms)):
                self.idx.setdefault((int(y), int(m)), (d, i))
        self.name = name
        self.units = getattr(self.ds[0][name], 'units', '') if self.ds else ''

    def has(self, years):
        return all((y, m) in self.idx for y in years for m in range(1, 13))

    def field(self, y, m):
        d, i = self.idx[(y, m)]
        return cm.read(d[self.name], i, i + 1)[0]

    def grid(self):
        d = self.ds[0]
        return np.asarray(d['lat'][:], 'f8'), np.asarray(d['lon'][:], 'f8')

    def close(self):
        for d in self.ds:
            d.close()


def open_model(model, member, years_needed):
    """{var: Monthly} for the historical run, or None (with the reason)"""
    out = {}
    for v, names in NAMES.items():
        best = None
        for name in names:
            fs = sorted(glob.glob(os.path.join(ROOT, model, f'{name}_Omon_{model}_historical_{member}_*.nc')))
            reg = [f for f in fs if f.endswith('_r.nc')]
            for cand in (reg, [f for f in fs if not f.endswith('_r.nc')]):
                if not cand:
                    continue
                M = Monthly(cand, name)
                if M.ds and M.has(years_needed):
                    try:
                        lat, lon = M.grid()
                        if lat.ndim == 1 and lon.ndim == 1:
                            best = M
                            break
                    except (IndexError, KeyError):
                        pass
                M.close()
            if best:
                break
        if best is None:
            for M in out.values():
                M.close()
            return None, f'no monthly {v} for all years needed on a regular grid'
        out[v] = best
    return out, ''


def year_fields(F, model, y, lon2, lat2):
    """12 months of T (degC), S, DIC and TA (umol kg-1), as calc_bias_adjusted.ModelRun.read"""
    t = np.array([F['thetao'].field(y, m) for m in range(1, 13)])
    s = np.array([F['so'].field(y, m) for m in range(1, 13)])
    if np.nanmedian(t) > 100:
        t = t - 273.15
    if np.nanmedian(s) < 1:
        s = s * 1000.0
    rho = cm.surface_density(t, s, lon2, lat2)
    rho_m = cm.model_density(model, t, s, lon2, lat2)
    out = {'t': t, 's': s}
    for v in ('dissic', 'talk'):
        a = np.array([F[v].field(y, m) for m in range(1, 13)])
        out[v] = cm.to_mol_m3(a, F[v].units, rho) / rho_m * 1e6
    return out


def omega_year(X, si, po4, lat2):
    """monthly Omega (12, ny, nx) and Omega of the annual-mean state (ny, nx); NaN unless all 12 months valid"""
    t, s, dic, alk = X['t'], X['s'], X['dissic'] * 1e-6, X['talk'] * 1e-6
    ok = (np.isfinite(t) & np.isfinite(s) & np.isfinite(dic) & np.isfinite(alk) & (s > 0) & (dic > 0) & (alk > 0)
          & np.isfinite(si) & np.isfinite(po4))
    om = np.full(t.shape, np.nan)
    if ok.any():
        om[ok] = co.omega_cells(t[ok], s[ok], dic[ok], alk[ok], np.broadcast_to(si * 1e-6, t.shape)[ok],
                                np.broadcast_to(po4 * 1e-6, t.shape)[ok], np.broadcast_to(lat2, t.shape)[ok])
    st, mm = co.annual_state_omega(t, s, dic, alk, om, si, po4, lat2)
    return st[0], mm[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=list(MEMBERS))
    ap.add_argument('--early', nargs=2, type=int, default=[1850, 1869])
    ap.add_argument('--late', nargs=2, type=int, default=[1995, 2014])
    a = ap.parse_args()
    periods = {'early': range(a.early[0], a.early[1] + 1), 'late': range(a.late[0], a.late[1] + 1)}
    need = list(periods['early']) + list(periods['late'])
    rows = []
    print(f'early {a.early[0]}-{a.early[1]}, late {a.late[0]}-{a.late[1]}; J = mean of monthly - annual state\n',
          flush=True)
    for model in a.models:
        F, why = open_model(model, MEMBERS[model], need)
        if F is None:
            print(f'{model:16s} skipped: {why}', flush=True)
            continue
        lat, lon = F['dissic'].grid()
        lon2, lat2 = np.meshgrid(lon, lat)
        si, po4 = co.load_nutrients(lat, lon)
        res = {}
        for p, yrs in periods.items():
            st, mm = [], []
            for y in yrs:
                s_, m_ = omega_year(year_fields(F, model, y, lon2, lat2), si, po4, lat2)
                st.append(s_)
                mm.append(m_)
            res[p] = (np.array(st), np.array(mm))
        for M in F.values():
            M.close()
        mask = np.ones(lat2.shape, bool)
        for st, mm in res.values():
            mask &= np.all(np.isfinite(st), axis=0) & np.all(np.isfinite(mm), axis=0)
        w = np.cos(np.deg2rad(lat2)) * mask
        gm = lambda f: float(np.sum(np.where(mask, f.mean(axis=0), 0) * w) / w.sum())
        S = {p: gm(res[p][0]) for p in periods}
        M_ = {p: gm(res[p][1]) for p in periods}
        d_state = 100 * (1 - S['late'] / S['early'])
        d_mm = 100 * (1 - M_['late'] / M_['early'])
        J = {p: M_[p] - S[p] for p in periods}
        rows.append((J['early'], J['late'], d_state, d_mm))
        print(f'{model:16s} Omega (annual state) {S["early"]:.3f} -> {S["late"]:.3f}; J {J["early"]:+.4f} -> '
              f'{J["late"]:+.4f}; decline annual state {d_state:.3f} %, mean of monthly {d_mm:.3f} %, '
              f'difference {d_mm - d_state:+.3f} points; {100 * w.sum() / np.cos(np.deg2rad(lat2)).sum():.0f} % '
              f'of the sphere', flush=True)
    if rows:
        r = np.array(rows)
        print(f'\n{len(rows)} models: J early {r[:, 0].mean():+.4f} (range {r[:, 0].min():+.4f} to {r[:, 0].max():+.4f}), '
              f'J late {r[:, 1].mean():+.4f} (range {r[:, 1].min():+.4f} to {r[:, 1].max():+.4f}); difference of the '
              f'declines (mean of monthly minus annual state) {np.mean(r[:, 3] - r[:, 2]):+.3f} points '
              f'(range {np.min(r[:, 3] - r[:, 2]):+.3f} to {np.max(r[:, 3] - r[:, 2]):+.3f})')


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
