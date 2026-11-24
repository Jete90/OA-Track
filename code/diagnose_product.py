#!/usr/bin/env python
"""
diagnose_product.py  (data product: where are the irregular cells, and why)
===================
The summaries of the product files (summarize_product.py, 6 November 2026) show a few irregular cells. This script
locates them in the product files and in the states files behind them and prints what is there, so that every one
can be fixed at its source. Nothing is changed. Sections:
  A  cells inside the mask without a constrained state in model years (S, DIC and all indicators missing while T
     and alkalinity are present): which cells and years, and what the states files hold there
  B  cells with the value 0 in the combined files (make_product.py, combine() before 7 November 2026: a reference
     without a value in a cell entered the mean as zero; fixed) and whether the cells of A explain them
  C  extreme constraint uncertainties (omega_arag_sigma > 1, ph_sigma > 1, pco2_sigma > 2000 uatm): the cells, the
     constrained state there and the sigma components (fit, dSST, CUC, CO2) of the states files
  D  unphysical multi-model mean scenario states (ph_<ssp>_unconstrained < 7): cells and years; with --models the
     model whose smoothed state is responsible (its T, S, DIC, alkalinity at the cell)
  E  temperatures below the freezing point of seawater (UNESCO 1983 at the surface), best estimate, named states
     and scenarios
  F  Jiang et al.: cells with a value but without the constraint sigma in its observed years (2005-2014)
  G  the alkalinity adjustment m(t) around the switch year (make_product_alkalinity.py: the model formula before,
     the product formula from the switch year; the two differ there)
  H  the 2100 step of the scenario series (models without a 2100 state), from a linear fit of 2090 to 2099
Output: the listing, and data/<run>/product/diagnose_product.json with the counts and the first cells of every section.
Usage (code/):  OA_RUN=u74 python -u diagnose_product.py                      # all references and the combined files
                OA_RUN=u74 python -u diagnose_product.py --reference UExP --max-cells 30 --models
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

import netCDF4
import numpy as np

import oa_config

OUT = oa_config.DATA / 'product'
REFS = ['OceanSODA', 'UExP', 'JMA', 'CMEMS', 'Jiang']
SCEN = ('ssp126', 'ssp245', 'ssp370', 'ssp585')
LIMITS = {'omega_arag': 1.0, 'ph': 1.0, 'pco2': 2000.0}        # section C: a sigma above these is not a plausible uncertainty
PH_MIN = 7.0                                                   # section D
REPORT = {}


def rd(f, name, i=None):
    """variable name (index i along the first dimension) as float64 with NaN for missing values"""
    v = f[name]
    a = v[:] if i is None else v[i]
    return np.ma.filled(np.ma.asarray(a).astype('f8'), np.nan)


def freezing_point(S):
    """freezing temperature of seawater at the surface (UNESCO 1983), degC"""
    S = np.clip(S, 0.0, 50.0)
    return -0.0575 * S + 1.710523e-3 * S ** 1.5 - 2.154996e-4 * S ** 2


def compact(years):
    """1850-1853, 1990 for a sorted list of years"""
    years = sorted(int(y) for y in years)
    if not years:
        return '-'
    out, a, b = [], years[0], years[0]
    for y in years[1:] + [None]:
        if y is not None and y == b + 1:
            b = y
            continue
        out.append(f'{a}' if a == b else f'{a}-{b}')
        if y is not None:
            a = b = y
    return ', '.join(out)


class States:
    """the states files of a reference and the index map from the product grid (lon -180..180) to their grid"""

    def __init__(self, ref, lon_product):
        self.files = {}
        for part in ['pi', 'hist'] + list(SCEN):
            p = OUT / f'states_{ref}_{part}.nc'
            if p.exists():
                self.files[part] = netCDF4.Dataset(p)
        f0 = next(iter(self.files.values()), None)
        self.ok = f0 is not None
        if self.ok:
            lon0 = np.asarray(f0['lon'][:], 'f8')
            l = np.where(lon0 > 180, lon0 - 360, lon0)
            o = np.argsort(l)
            assert np.allclose(l[o], lon_product), 'product and states grids differ'
            self.o = o                                   # product index i -> states index o[i]
            self.years = {k: np.asarray(f['year'][:]).astype(int) for k, f in self.files.items()}

    def part(self, year, scen=None):
        if scen and scen in self.files:
            return scen
        return 'pi' if year < 1850 else 'hist'

    def at(self, name, year, j, i, scen=None):
        """value of states variable name in year at product cell (j, i); NaN if absent"""
        part = self.part(year, scen)
        if part not in self.files or name not in self.files[part].variables:
            return np.nan
        f, ys = self.files[part], self.years[part]
        k = np.searchsorted(ys, year)
        if k >= ys.size or ys[k] != year:
            return np.nan
        v = f[name]
        a = v[k, j, self.o[i]] if v.ndim == 3 else v[j, self.o[i]]
        return float(np.ma.filled(np.ma.asarray(a).astype('f8'), np.nan))

    def close(self):
        for f in self.files.values():
            f.close()


def section(title):
    print(f'\n{"=" * 100}\n{title}\n{"=" * 100}', flush=True)


# ================================================================================================ A
def sec_a(refs, max_cells):
    section('A  cells inside the mask without a constrained state in model years')
    found = {}
    for ref in refs:
        p = OUT / f'carbonate_system_{ref}.nc'
        if not p.exists():
            continue
        with netCDF4.Dataset(p) as f:
            lat, lon = rd(f, 'lat'), rd(f, 'lon')
            years = np.asarray(f['year'][:]).astype(int)
            mask = rd(f, 'mask') > 0
            model = rd(f, 'source') != 4
            T, O, S, A = (rd(f, v) for v in ('thetao', 'omega_arag', 'so', 'talk'))
            flag = rd(f, 'constrained_flag')
        bad = mask[None] & model[:, None, None] & np.isfinite(T) & ~np.isfinite(O)
        bad_s = bad & ~np.isfinite(S)
        bad_a = bad & ~np.isfinite(A)
        n_cy = int(bad.sum())
        anyc = bad.any(0)
        cells = list(zip(*np.where(anyc)))
        print(f'\n{ref}: {n_cy} cell-years ({100 * n_cy / max(1, (mask[None] & model[:, None, None]).sum()):.4f} % of the '
              f'model cell-years inside the mask), {len(cells)} distinct cells; of these cell-years {int(bad_s.sum())} without S, '
              f'{int(bad_a.sum())} without alkalinity')
        per_year = bad.sum(axis=(1, 2))
        yrs = years[per_year > 0]
        if yrs.size:
            print(f'  years with missing cells: {compact(yrs)}; most in {years[np.argmax(per_year)]} ({per_year.max()} cells)')
        st = States(ref, lon)
        rows = []
        for (j, i) in cells[:max_cells]:
            kk = np.where(bad[:, j, i])[0]
            ys = years[kk]
            y0 = int(ys[0])
            row = dict(lat=float(lat[j]), lon=float(lon[i]), years=compact(ys), n=int(ys.size), flag=float(flag[j, i]),
                       T=float(T[kk[0], j, i]), talk=float(A[kk[0], j, i]))
            if st.ok:
                row.update({f'states_{v}': st.at(v, y0, j, i) for v in ('so', 'dic', 'talk', 'thetao', 'omega_arag')})
                row['states_so_sigma_fit'] = st.at('so_sigma_fit', y0, j, i)
                row['states_dic_sigma_fit'] = st.at('dic_sigma_fit', y0, j, i)
            rows.append(row)
            print(f'  {row["lat"]:7.2f} {row["lon"]:8.2f}  flag {row["flag"]:.0f}  missing {row["years"]} ({row["n"]} years); '
                  f'product T {row["T"]:.2f}, TA {row["talk"]:.1f}' +
                  (f'; states file {y0}: S {row["states_so"]:.3f}, DIC {row["states_dic"]:.1f}, TA {row["states_talk"]:.1f}, '
                   f'T {row["states_thetao"]:.2f}, Omega {row["states_omega_arag"]:.3f}' if st.ok else ''))
        st.close()
        found[ref] = dict(cell_years=n_cy, cells=len(cells), years=compact(yrs), examples=rows, all_cells=[(int(j), int(i)) for j, i in cells])
    REPORT['A'] = {k: {kk: vv for kk, vv in v.items() if kk != 'all_cells'} for k, v in found.items()}
    return found


# ================================================================================================ B
def sec_b(found_a, max_cells):
    section('B  cells with the value 0 in the combined files')
    union = set()
    for v in found_a.values():
        union |= set(map(tuple, v['all_cells']))
    out = {}
    for kind in ('common', 'all'):
        p = OUT / f'carbonate_system_{kind}.nc'
        if not p.exists():
            continue
        with netCDF4.Dataset(p) as f:
            lat, lon = rd(f, 'lat'), rd(f, 'lon')
            years = np.asarray(f['year'][:]).astype(int)
            mask = rd(f, 'mask') > 0
            for v in ('switch1990', 'switchfirst'):
                O = rd(f, f'omega_arag_{v}')
                S = rd(f, f'so_{v}')
                zero = mask[None] & (O == 0)
                n = int(zero.sum())
                cells = set(zip(*np.where(zero.any(0))))
                inter = len(cells & union)
                per_year = zero.sum(axis=(1, 2))
                print(f'\n{kind} {v}: {n} cell-years with Omega_arag = 0 in {len(cells)} cells; years {compact(years[per_year > 0])}; '
                      f'{inter} of the cells are cells of section A (a reference without a state there)')
                for (j, i) in sorted(cells)[:max_cells]:
                    print(f'  {lat[j]:7.2f} {lon[i]:8.2f}  years {compact(years[zero[:, j, i]])}')
                out[f'{kind}_{v}'] = dict(cell_years=n, cells=len(cells), in_A=inter, years=compact(years[per_year > 0]))
    REPORT['B'] = out


# ================================================================================================ C
def sec_c(refs, max_cells):
    section('C  extreme constraint uncertainties')
    out = {}
    for ref in refs:
        p = OUT / f'carbonate_system_{ref}.nc'
        if not p.exists():
            continue
        with netCDF4.Dataset(p) as f:
            lat, lon = rd(f, 'lat'), rd(f, 'lon')
            years = np.asarray(f['year'][:]).astype(int)
            years_ssp = np.asarray(f['year_ssp'][:]).astype(int) if 'year_ssp' in f.dimensions else None
            states = [str(s) for s in f['state'][:]] if 'state' in f.dimensions else []
            mask = rd(f, 'mask') > 0
            st = States(ref, lon)
            rows = []
            for q, lim in LIMITS.items():
                for suffix, yrs, scen in [('', years, None), ('_state', np.arange(len(states)), None)] + [(f'_{s}', years_ssp, s) for s in SCEN]:
                    name = f'{q}{suffix}_sigma' if suffix != '_state' else f'{q}_sigma_state'
                    if name not in f.variables or yrs is None:
                        continue
                    X = rd(f, name)
                    bad = mask[None] & (X > lim)
                    n = int(bad.sum())
                    if n == 0:
                        continue
                    cells = list(zip(*np.where(bad.any(0))))
                    k = np.unravel_index(np.nanargmax(np.where(bad, X, -np.inf)), X.shape)
                    yl = states[k[0]] if suffix == '_state' else int(yrs[k[0]])
                    print(f'\n{ref} {name} > {lim:g}: {n} cell-years in {len(cells)} cells; max {X[k]:.4g} at {lat[k[1]]:.2f}, {lon[k[2]]:.2f} ({yl})')
                    base = f'{q}{suffix}' if suffix != '_state' else f'{q}_state'
                    for (j, i) in cells[:max_cells]:
                        kk = np.where(bad[:, j, i])[0]
                        y0 = kk[int(np.argmax(X[kk, j, i]))]
                        ylab = states[y0] if suffix == '_state' else int(yrs[y0])
                        sv = {v: float(rd(f, f'{v}{suffix}' if suffix != '_state' else f'{v}_state', y0)[j, i])
                              for v in ('thetao', 'so', 'talk', 'dic') if (f'{v}{suffix}' if suffix != '_state' else f'{v}_state') in f.variables}
                        val = float(rd(f, base, y0)[j, i]) if base in f.variables else np.nan
                        comp = ''
                        if st.ok and suffix != '_state':
                            c = {k_: st.at(f'{q}_sigma_{k_}', int(yrs[y0]), j, i, scen) for k_ in ('fit', 'dsst', 'cuc', 'co2')}
                            comp = '; components fit {fit:.4g}, dSST {dsst:.4g}, CUC {cuc:.4g}, CO2 {co2:.4g}'.format(**c)
                        print(f'  {lat[j]:7.2f} {lon[i]:8.2f}  {ylab}: sigma {X[y0, j, i]:.4g} (value {val:.4g}); years {compact(yrs[kk]) if suffix != "_state" else len(kk)}'
                              f'; state T {sv.get("thetao", np.nan):.2f}, S {sv.get("so", np.nan):.2f}, TA {sv.get("talk", np.nan):.1f}, '
                              f'DIC {sv.get("dic", np.nan):.1f}, TA-DIC {sv.get("talk", np.nan) - sv.get("dic", np.nan):.1f}{comp}')
                        rows.append(dict(var=name, lat=float(lat[j]), lon=float(lon[i]), year=str(ylab), sigma=float(X[y0, j, i]), state=sv))
            st.close()
            out[ref] = rows
            if not rows:
                print(f'\n{ref}: none')
    REPORT['C'] = out


# ================================================================================================ D
def sec_d(refs, max_cells, models):
    section('D  unphysical multi-model mean scenario states (pH < 7)')
    out = {}
    for ref in refs:
        p = OUT / f'carbonate_system_{ref}.nc'
        if not p.exists():
            continue
        rows = []
        with netCDF4.Dataset(p) as f:
            lat, lon = rd(f, 'lat'), rd(f, 'lon')
            if 'year_ssp' not in f.dimensions:
                continue
            years_ssp = np.asarray(f['year_ssp'][:]).astype(int)
            mask = rd(f, 'mask') > 0
            for s in SCEN:
                name = f'ph_{s}_unconstrained'
                if name not in f.variables:
                    continue
                P = rd(f, name)
                bad = mask[None] & (P < PH_MIN)
                n = int(bad.sum())
                if n == 0:
                    continue
                cells = list(zip(*np.where(bad.any(0))))
                print(f'\n{ref} {s}: {n} cell-years with multi-model mean pH < {PH_MIN} in {len(cells)} cells, years {compact(years_ssp[bad.any(axis=(1, 2))])}')
                for (j, i) in cells[:max_cells]:
                    kk = np.where(bad[:, j, i])[0]
                    y0 = kk[int(np.argmin(P[kk, j, i]))]
                    sv = {v: float(rd(f, f'{v}_{s}_unconstrained', y0)[j, i]) for v in ('thetao', 'so', 'talk', 'dic') if f'{v}_{s}_unconstrained' in f.variables}
                    cst = float(rd(f, f'ph_{s}', y0)[j, i])
                    print(f'  {lat[j]:7.2f} {lon[i]:8.2f}  {years_ssp[y0]}: multi-model mean pH {P[y0, j, i]:.3f} (constrained {cst:.3f}); years {compact(years_ssp[kk])}; '
                          f'mean state T {sv.get("thetao", np.nan):.2f}, S {sv.get("so", np.nan):.2f}, TA {sv.get("talk", np.nan):.1f}, DIC {sv.get("dic", np.nan):.1f}')
                    rows.append(dict(scenario=s, lat=float(lat[j]), lon=float(lon[i]), year=int(years_ssp[y0]), ph=float(P[y0, j, i]), years=compact(years_ssp[kk])))
        if rows and models:
            attribute_models(ref, rows[:3])
        out[ref] = rows
        if not rows:
            print(f'\n{ref}: none')
    REPORT['D'] = out


def attribute_models(ref, rows):
    """the smoothed, adjusted state of every model of the scenario at the cells (annual means): which model is off"""
    import calc_product_states as cps
    try:
        R = cps.Reference(ref)
    except Exception as e:
        print(f'  model attribution not possible ({e})')
        return
    lon0 = np.asarray(R.lon, 'f8')
    l = np.where(lon0 > 180, lon0 - 360, lon0)
    o = np.argsort(l)
    for row in rows:
        s, year = row['scenario'], row['year']
        j = int(np.argmin(np.abs(np.asarray(R.lat, 'f8') - row['lat'])))
        i = o[int(np.argmin(np.abs(l[o] - row['lon'])))]
        names = cps.scenario_models(R.names, R.members, s)
        print(f'  models of {s} at {row["lat"]:.2f}, {row["lon"]:.2f} in {year} (adjusted smoothed states, annual means):')
        for m in names:
            try:
                sm = cps.Smoothed(R, m)
                X = sm.state(year, s if year >= 1970 else 'hist')
                sm.close()
                vals = {v: float(np.nanmean(X[v][:, j, i])) for v in X}
                print(f'    {m:22s} T {vals.get("thetao", np.nan):7.2f}  S {vals.get("so", np.nan):7.2f}  DIC {vals.get("dissic", np.nan):8.1f}  '
                      f'TA {vals.get("talk", np.nan):8.1f}  TA-DIC {vals.get("talk", np.nan) - vals.get("dissic", np.nan):7.1f}')
            except Exception as e:
                print(f'    {m:22s} not read ({e})')


# ================================================================================================ E
def sec_e(refs, max_cells):
    section('E  temperatures below the freezing point of seawater')
    out = {}
    for ref in refs:
        p = OUT / f'carbonate_system_{ref}.nc'
        if not p.exists():
            continue
        rows = []
        with netCDF4.Dataset(p) as f:
            lat, lon = rd(f, 'lat'), rd(f, 'lon')
            mask = rd(f, 'mask') > 0
            flag = rd(f, 'constrained_flag')
            labels = {'thetao': np.asarray(f['year'][:]).astype(int)}
            if 'state' in f.dimensions:
                labels['thetao_state'] = np.array([str(s) for s in f['state'][:]])
            if 'year_ssp' in f.dimensions:
                for s in SCEN:
                    if f'thetao_{s}' in f.variables:
                        labels[f'thetao_{s}'] = np.asarray(f['year_ssp'][:]).astype(int)
            for name, lab in labels.items():
                T = rd(f, name)
                sname = name.replace('thetao', 'so')
                S = rd(f, sname) if sname in f.variables else np.full_like(T, 35.0)
                tf = freezing_point(np.where(np.isfinite(S), S, 35.0))
                bad = mask[None] & (T < tf - 0.1)
                n = int(bad.sum())
                if n == 0:
                    print(f'\n{ref} {name}: none')
                    continue
                cells = list(zip(*np.where(bad.any(0))))
                k = np.unravel_index(np.nanargmin(np.where(bad, T, np.inf)), T.shape)
                print(f'\n{ref} {name}: {n} cell-years below the freezing point (by more than 0.1 K) in {len(cells)} cells; '
                      f'coldest {T[k]:.2f} degC (S {S[k]:.2f}, freezing point {tf[k]:.2f}) at {lat[k[1]]:.2f}, {lon[k[2]]:.2f} ({lab[k[0]]})')
                for (j, i) in cells[:max_cells]:
                    kk = np.where(bad[:, j, i])[0]
                    y0 = kk[int(np.argmin(T[kk, j, i]))]
                    print(f'  {lat[j]:7.2f} {lon[i]:8.2f}  flag {flag[j, i]:.0f}  coldest {T[y0, j, i]:.2f} degC (S {S[y0, j, i]:.2f}, '
                          f'freezing {tf[y0, j, i]:.2f}) in {lab[y0]}; {kk.size} of {T.shape[0]} years/states')
                    rows.append(dict(var=name, lat=float(lat[j]), lon=float(lon[i]), t_min=float(T[y0, j, i]), s=float(S[y0, j, i]), n=int(kk.size)))
        out[ref] = rows
    REPORT['E'] = out


# ================================================================================================ F
def sec_f(max_cells):
    section('F  Jiang et al.: value without constraint sigma in its observed years (2005-2014)')
    p = OUT / 'carbonate_system_Jiang.nc'
    if not p.exists():
        print('no Jiang file')
        return
    rows = []
    with netCDF4.Dataset(p) as f:
        lat, lon = rd(f, 'lat'), rd(f, 'lon')
        years = np.asarray(f['year'][:]).astype(int)
        mask = rd(f, 'mask') > 0
        flag = rd(f, 'constrained_flag')
        O, SG, TS = rd(f, 'omega_arag'), rd(f, 'omega_arag_sigma'), rd(f, 'thetao_sigma')
        bad = mask[None] & np.isfinite(O) & ~np.isfinite(SG)
        n = int(bad.sum())
        cells = list(zip(*np.where(bad.any(0))))
        print(f'\n{n} cell-years with Omega_arag but no omega_arag_sigma in {len(cells)} cells; years {compact(years[bad.any(axis=(1, 2))])}; '
              f'thetao_sigma present in {int((bad & np.isfinite(TS)).sum())} of them')
        st = States('Jiang', lon)
        for (j, i) in cells[:max_cells]:
            kk = np.where(bad[:, j, i])[0]
            y0 = int(years[kk[0]])
            comp = {k_: st.at(f'omega_arag_sigma_{k_}', y0, j, i) for k_ in ('fit', 'dsst', 'cuc', 'co2')}
            dsig = {k_: st.at(f'dic_sigma_{k_}', y0, j, i) for k_ in ('fit', 'dsst', 'cuc', 'co2')}
            print(f'  {lat[j]:7.2f} {lon[i]:8.2f}  flag {flag[j, i]:.0f}  years {compact(years[kk])}; states {y0}: Omega sigma components '
                  f'fit {comp["fit"]:.4g}, dSST {comp["dsst"]:.4g}, CUC {comp["cuc"]:.4g}, CO2 {comp["co2"]:.4g}; DIC sigma components '
                  f'fit {dsig["fit"]:.4g}, dSST {dsig["dsst"]:.4g}, CUC {dsig["cuc"]:.4g}, CO2 {dsig["co2"]:.4g}')
            rows.append(dict(lat=float(lat[j]), lon=float(lon[i]), years=compact(years[kk]), components=comp))
        st.close()
    REPORT['F'] = dict(cell_years=n, cells=len(cells), examples=rows)


# ================================================================================================ G
def sec_g(refs):
    section('G  the alkalinity adjustment m(t) around the switch year')
    out = {}
    for ref in refs:
        p = OUT / f'carbonate_system_{ref}_alkalinity.nc'
        if not p.exists():
            continue
        with netCDF4.Dataset(p) as f:
            years = np.asarray(f['year'][:]).astype(int)
            src = rd(f, 'source')
            m = rd(f, 'talk_missing')
            om = rd(f, 'omega_arag_gm_common_alk')
            tc = float(f.t_c)
        prod = np.where(src == 4)[0]
        if prod.size == 0:
            continue
        k = prod[0]
        sel = range(max(0, k - 3), min(years.size, k + 3))
        print(f'\n{ref}: switch {years[k]}, t_c {tc:.1f}; ' + '; '.join(f'{years[t]}: m {m[t]:+.3f} umol/kg, dOmega {om[t]:+.5f}' for t in sel))
        if k > 0:
            print(f'  step at the switch: m {m[k] - m[k - 1]:+.3f} umol/kg, Omega_arag change {om[k] - om[k - 1]:+.5f}'
                  f' (the model formula would continue with about {m[k - 1] + (m[k - 1] - m[k - 2]) if k > 1 else np.nan:+.3f})')
        out[ref] = dict(switch=int(years[k]), step_m=float(m[k] - m[k - 1]) if k > 0 else None, step_omega=float(om[k] - om[k - 1]) if k > 0 else None)
    p = OUT / 'carbonate_system_common_alkalinity.nc'
    if p.exists():
        with netCDF4.Dataset(p) as f:
            years = np.asarray(f['year'][:]).astype(int)
            for v in ('switch1990', 'switchfirst'):
                if f'talk_gm_{v}_alk' in f.variables:
                    m = rd(f, f'talk_gm_{v}_alk')
                    om = rd(f, f'omega_arag_gm_{v}_alk')
                    sel = [t for t in range(years.size) if 1978 <= years[t] <= 1992]
                    print(f'\ncommon {v}: ' + '; '.join(f'{years[t]}: m {m[t]:+.3f}, dOmega {om[t]:+.5f}' for t in sel))
    REPORT['G'] = out


# ================================================================================================ H
def sec_h(refs):
    section('H  the 2100 step of the scenario series (models without 2100)')
    out = {}
    for ref in refs + ['common']:
        p = OUT / f'carbonate_system_{ref}.nc'
        if not p.exists():
            continue
        with netCDF4.Dataset(p) as f:
            if 'year_ssp' not in f.dimensions:
                continue
            ys = np.asarray(f['year_ssp'][:]).astype(int)
            res = {}
            for s in SCEN:
                name = f'omega_arag_gm_common_{s}_global' if ref != 'common' else f'omega_arag_gm_{s}_global'
                if name not in f.variables:
                    continue
                v = rd(f, name)
                sel = (ys >= 2090) & (ys <= 2099)
                c = np.polyfit(ys[sel] - 2090.0, v[sel], 1)
                ext = np.polyval(c, 2100 - 2090.0)
                step = float(v[ys == 2100][0] - ext)
                nm = ''
                if f'n_models_{s}' in f.variables:
                    n = np.asarray(f[f'n_models_{s}'][:]).astype(int)
                    nm = f', models {n[ys == 2099][0]} in 2099 and {n[ys == 2100][0]} in 2100'
                res[s] = step
                print(f'{ref:10s} {s}: 2100 value {v[ys == 2100][0]:.4f}, linear fit 2090-2099 extrapolated {ext:.4f}, step {step:+.4f}{nm}')
            out[ref] = res
    REPORT['H'] = out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', nargs='+', choices=REFS)
    ap.add_argument('--max-cells', type=int, default=20)
    ap.add_argument('--models', action='store_true', help='section D: read the smoothed states of every model at the cells')
    ap.add_argument('--sections', default='ABCDEFGH')
    a = ap.parse_args()
    refs = a.reference or REFS
    print(f'product files in {OUT}; references {refs}; sections {a.sections}', flush=True)
    found_a = {}
    steps = {'A': lambda: sec_a(refs, a.max_cells), 'B': lambda: sec_b(found_a, a.max_cells), 'C': lambda: sec_c(refs, a.max_cells),
             'D': lambda: sec_d(refs, a.max_cells, a.models), 'E': lambda: sec_e(refs, a.max_cells), 'F': lambda: sec_f(a.max_cells),
             'G': lambda: sec_g(refs), 'H': lambda: sec_h(refs)}
    for s in a.sections:
        try:
            r = steps[s]()
            if s == 'A' and r:
                found_a.update(r)
        except Exception:
            print(f'\nsection {s} failed:\n{traceback.format_exc()}', flush=True)
    out = OUT / 'diagnose_product.json'
    with open(out, 'w') as f:
        json.dump(REPORT, f, indent=1, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f'\nwritten {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
