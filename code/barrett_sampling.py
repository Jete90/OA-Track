#!/usr/bin/env python
"""
barrett_sampling.py
===================
The sampling of the observed surface TA trend of Barrett et al. (2025, GBC), on the 1 degree grid of the products,
written to data/<run>/structural/barrett_sampling.npz. structural_adjustments.py uses it to scale the feedback
pattern: Barrett et al. fit one regression through all their measurements, so their rate is the mean of the trend
over their samples, each measurement with the same weight, not an area mean over the common area.

Their selection (section 2.1), applied to the GLODAPv2.2023 merged master file:
  cruises associated with GO-SHIP, WOCE/JGOFS, CLIVAR or SOCCOM, on GO-SHIP transects (201 cruises): from the CCHDO
    cruise list (https://cchdo.ucsd.edu/api/v1/cruise/all, saved as cchdo_cruises.json; --refresh downloads it
    again), matched to the GLODAP expocodes (exact, else without a suffix after '_', else the same ship code and a
    start date within MATCH_DAYS: CCHDO keeps many WOCE-era cruises under their old expocodes); three definitions:
      programs        the CCHDO programs name one of PROGRAMS (tags are incomplete for international cruises)
      lines           the CCHDO line designations name a hydrographic section (A, P, I or S, optionally R, then a
                      number: A16N, AR07W, P16S, I08S, SR03, S04P; and ARC01), time series excluded (TIMESERIES
                      programs, group Timeseries; repeat stations such as ARS01 or PRS02 do not match)
      programs+lines  either (default, --selection)
  measured TA with GLODAP flag 2 (calculated values have flag 0), salinity 30-37, depth < 25 m,
  1 September 1991 to 4 October 2021
  not applied: their trimming to the stations on the transects (test casts)
Printed: programs of the CCHDO list; per definition the GLODAP cruises of 1991-2021 (all depths, compare 201), surface
measurements, polar shares; with feedback_pattern.npz: p per definition, the cruises with the largest effect on p, and
the range of p if the cruises beyond Barrett's 201 are removed so as to raise or lower p most (all selected cruises in
barrett_cruises.csv); the programs, groups and lines of the GLODAP cruises not selected.

Saved: n_points (measurements per cell, the selection), n_points_<definition>, n_points_all (all GLODAP cruises),
n_cruises, lat, lon, the totals per filter step.

Usage (code/):  OA_RUN=u74 python -u barrett_sampling.py [--glodap <csv>] [--refresh] [--selection programs+lines]
"""
import argparse
import collections
import datetime
import json
import pickle
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import oa_config

DEST = oa_config.DATA / 'structural'
GLODAP = oa_config.ESM2M_DIR / 'GLODAP' / 'GLODAPv2.2023_Merged_Master_File_with_canth_model.csv'
CCHDO_URL = 'https://cchdo.ucsd.edu/api/v1/cruise/all'
PROGRAMS = ('GO-SHIP', 'WOCE', 'JGOFS', 'CLIVAR', 'SOCCOM')     # Barrett et al. section 2.1
TIMESERIES = ('Hydrostation S', 'BATS', 'HOT', 'Line P', 'Line-P', 'Station P', 'Station-P')
SECTION = re.compile(r'^([APIS]R?\d|ARC\d)', re.I)
MATCH_DAYS = 7
DEFINITIONS = ('programs', 'lines', 'programs+lines')
N_BARRETT = 201
START, END = (1991, 9, 1), (2021, 10, 4)
S_RANGE = (30.0, 37.0)
DEPTH_MAX = 25.0
NEED = ('expocode', 'year', 'month', 'day', 'latitude', 'longitude', 'depth', 'salinity', 'talk', 'talkf')


def columns(path):
    """the GLODAP column names (with or without the G2 prefix) of the variables in NEED"""
    head = list(pd.read_csv(path, nrows=0).columns)
    low = {c.lower(): c for c in head}
    out = {}
    for k in NEED:
        c = low.get('g2' + k) or low.get(k)
        if c is None:
            raise KeyError(f'{path}: no column for {k}; columns: {head}')
        out[k] = c
    return out


def edges(c):
    d = np.diff(c).mean()
    return np.r_[c[0] - d / 2, c + d / 2]


def cells(lat, lon, la, lo):
    """indices (j, i) of the grid cells that contain the points (la, lo); lo in any convention, the grid in its own
    (-179.5..179.5 or 0.5..359.5): the longitude is wrapped onto the grid, not clipped to its last column"""
    d = float(np.diff(lon).mean())
    assert np.allclose(np.diff(lon), d) and np.isclose(lon.size * d, 360.0), 'longitudes: not a regular global grid'
    j = np.clip(np.searchsorted(edges(lat), la, side='right') - 1, 0, lat.size - 1)
    i = np.minimum(np.floor(((lo - (lon[0] - d / 2)) % 360.0) / d).astype(int), lon.size - 1)
    return j, i


def base(e):
    return str(e).strip().upper().split('_')[0]


def gdate(e):
    """start date in a 12-character expocode (ship code + YYYYMMDD), else None"""
    try:
        return datetime.date(int(e[4:8]), int(e[8:10]), int(e[10:12]))
    except (ValueError, IndexError):
        return None


def cchdo(refresh):
    """CCHDO cruises: expocode -> (programs, groups, lines), keyed by the exact and the base expocode, and by ship
    code -> [(start date, record)]"""
    f = DEST / 'cchdo_cruises.json'
    if refresh or not f.exists():
        DEST.mkdir(parents=True, exist_ok=True)
        print(f'  downloading {CCHDO_URL}', flush=True)
        urllib.request.urlretrieve(CCHDO_URL, f)
    data = json.load(open(f))
    exact, bases, ships, names = {}, {}, collections.defaultdict(list), collections.Counter()
    for c in data:
        col = c.get('collections') or {}
        rec = tuple([str(x) for x in (col.get(k) or [])] for k in ('programs', 'groups', 'woce_lines'))
        e = str(c.get('expocode', '')).strip().upper()
        exact[e] = rec
        bases.setdefault(base(e), rec)
        try:
            ships[e[:4]].append((datetime.date.fromisoformat(str(c.get('startDate'))[:10]), rec))
        except ValueError:
            pass
        names.update(rec[0])
    print(f'  CCHDO list ({f.name}): {len(data)} cruises; programs (cruises): '
          + ', '.join(f'{n} ({k})' for n, k in names.most_common(30)))
    return exact, bases, ships


def selected(rec, how):
    progs, groups, lines = rec
    by_prog = any(k.lower() in p.lower() for p in progs for k in PROGRAMS)
    ts = any(p in TIMESERIES for p in progs) or any('timeseries' in g.lower().replace(' ', '') for g in groups)
    by_line = (not ts) and any(SECTION.match(l.strip()) for l in lines)
    return {'programs': by_prog, 'lines': by_line, 'programs+lines': by_prog or by_line}[how]


def cruise_table(d, ex, surf, sel, rec, found_by, pat, lat, lon):
    """per selected cruise: surface measurements, pattern mean, contribution to p (barrett_cruises.csv); the range of p
    when the cruises beyond Barrett's 201 are removed so as to raise or lower p most"""
    m = surf & np.isin(ex, list(sel))
    j, i = cells(lat, lon, d['latitude'].to_numpy()[m], d['longitude'].to_numpy()[m])
    pv = pat[j, i]
    t = pd.DataFrame({'expocode': ex[m], 'pattern': pv, 'lat': d['latitude'].to_numpy()[m], 'year': d['year'].to_numpy()[m]})
    t = t[np.isfinite(t['pattern'])]
    g = t.groupby('expocode').agg(n=('pattern', 'size'), sum_pattern=('pattern', 'sum'), lat_min=('lat', 'min'),
                                  lat_max=('lat', 'max'), year=('year', 'min'))
    P, N = g['sum_pattern'].sum(), g['n'].sum()
    p = P / N
    g['mean_pattern'] = g['sum_pattern'] / g['n']
    g['contribution_to_p'] = g['n'] * (g['mean_pattern'] - p) / N
    g['found_by'] = [found_by.get(e, '') for e in g.index]
    for k, lab in enumerate(('programs', 'groups', 'lines')):
        g[lab] = [';'.join(rec[e][k]) for e in g.index]
    g = g.sort_values('contribution_to_p')
    g.drop(columns='sum_pattern').to_csv(DEST / 'barrett_cruises.csv', float_format='%.4f')
    print(f'  {len(g)} cruises with surface data, p = {p:.3f}; largest effects on p (barrett_cruises.csv):')
    for e, r in pd.concat([g.head(5), g.tail(5)]).iterrows():
        print(f'    {e:14s} {r["year"]:.0f} {r["lines"] or r["programs"] or r["groups"]:18.18s} n {r["n"]:5.0f}  lat {r["lat_min"]:+5.0f}..{r["lat_max"]:+4.0f}'
              f'  pattern {r["mean_pattern"]:.2f}  effect on p {r["contribution_to_p"]:+.4f}')
    n_drop = len(g) - N_BARRETT
    if n_drop > 0:
        rng = []
        for sign in (+1, -1):
            s_, n_, left, gone = P, N, set(g.index), []
            for _ in range(n_drop):
                e = max(left, key=lambda c: sign * (s_ - g.at[c, 'sum_pattern']) / (n_ - g.at[c, 'n']))
                s_ -= g.at[e, 'sum_pattern']; n_ -= g.at[e, 'n']; left.remove(e); gone.append(e)
            rng.append((s_ / n_, gone))
        print(f'  removing the {n_drop} cruises beyond Barrett et al.\'s 201 that change p most: p {rng[1][0]:.3f} to {rng[0][0]:.3f}')
        for lab, (pp, gone) in zip(('raise', 'lower'), rng):
            print(f'    to {lab} p, remove: ' + ', '.join(f'{e} ({g.at[e, "lines"] or g.at[e, "programs"]})' for e in gone))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--glodap', default=str(GLODAP))
    ap.add_argument('--refresh', action='store_true', help='download the CCHDO cruise list again')
    ap.add_argument('--selection', default='programs+lines', choices=DEFINITIONS)
    a = ap.parse_args()
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    r0 = pk['results_common']['UExP']
    lat, lon = np.asarray(r0['lat'], 'f8'), np.asarray(r0['lon'], 'f8')
    exact, bases, ships = cchdo(a.refresh)

    col = columns(a.glodap)
    d = pd.read_csv(a.glodap, usecols=list(col.values()), na_values=[-9999, -999, '-9999', 'NaN'], low_memory=False)
    d = d.rename(columns={v: k for k, v in col.items()})
    steps = [('all rows', len(d))]
    d = d[d['talkf'] == 2]; steps.append(('measured TA, flag 2', len(d)))
    d = d[d['talk'].notna()]; steps.append(('TA value present', len(d)))
    d = d[(d['salinity'] >= S_RANGE[0]) & (d['salinity'] <= S_RANGE[1])]; steps.append((f'salinity {S_RANGE[0]:g}-{S_RANGE[1]:g}', len(d)))
    date = d['year'] * 10000 + d['month'] * 100 + d['day']
    d = d[(date >= START[0] * 10000 + START[1] * 100 + START[2]) & (date <= END[0] * 10000 + END[1] * 100 + END[2])]
    steps.append(('1 Sep 1991 to 4 Oct 2021', len(d)))
    surf = (d['depth'] < DEPTH_MAX).to_numpy()
    steps.append((f'depth < {DEPTH_MAX:g} m', int(surf.sum())))
    for s_, n_ in steps:
        print(f'  {s_:28s} {n_:9d}')

    ex = d['expocode'].astype(str).str.strip().str.upper().to_numpy()
    cruises = sorted(set(ex))
    def by_date(e):
        g = gdate(e)
        if g is None or not ships.get(e[:4]):
            return None
        dd, r = min(((abs((g - s_).days), r_) for s_, r_ in ships[e[:4]]), key=lambda t: t[0])
        return r if dd <= MATCH_DAYS else None
    rec, how_found, found_by = {}, collections.Counter(), {}
    for e in cruises:
        r = exact.get(e); lab = 'exact'
        if r is None:
            r = bases.get(base(e)); lab = 'without suffix'
        if r is None:
            r = by_date(e); lab = 'ship and date'
        rec[e] = r
        if r is not None:
            how_found[lab] += 1; found_by[e] = lab
    found = [e for e in cruises if rec[e] is not None]
    print(f'  GLODAP cruises 1991-2021 (all depths, filters above): {len(cruises)}; in the CCHDO list {len(found)} ('
          + ', '.join(f'{k} {v}' for k, v in how_found.items()) + ')')

    def grid(m):
        j, i = cells(lat, lon, d['latitude'].to_numpy()[m], d['longitude'].to_numpy()[m])
        n = np.zeros((lat.size, lon.size)); np.add.at(n, (j, i), 1)
        cc = pd.DataFrame({'j': j, 'i': i, 'c': ex[m]}).drop_duplicates()
        nc = np.zeros((lat.size, lon.size)); np.add.at(nc, (cc['j'].to_numpy(), cc['i'].to_numpy()), 1)
        la = d['latitude'].to_numpy()[m]
        return n, nc, (f'{int(m.sum())} surface measurements, {len(set(ex[m]))} cruises with surface data, '
                       f'{int((n > 0).sum())} cells; south of 60S {100 * np.mean(la < -60):.1f} %, north of 60N {100 * np.mean(la > 60):.1f} %')

    out, sel_of = {}, {}
    for how in DEFINITIONS:
        sel = {e for e in found if selected(rec[e], how)}
        sel_of[how] = sel
        m = surf & np.isin(ex, list(sel))
        n, nc, txt = grid(m)
        out[how] = (n, nc)
        print(f'  {how:15s} {len(sel):4d} cruises (Barrett et al. 201); {txt}')
    n_all, _, txt = grid(surf)
    print(f'  {"all GLODAP":15s} {len(cruises):4d} cruises; {txt}')
    fp = DEST / 'feedback_pattern.npz'
    if fp.exists():
        z = np.load(fp)
        key = 'pattern' if 'pattern' in z.files else 'pattern_cmip6'
        pat = np.asarray(z[key], 'f8')
        pm = lambda n: float(np.sum(np.where(np.isfinite(pat), n * pat, 0)) / np.sum(np.where(np.isfinite(pat), n, 0)))
        print(f'  p with {key} (mean of the pattern over the measurements): '
              + ', '.join(f'{how} {pm(out[how][0]):.3f}' for how in DEFINITIONS) + f', all GLODAP {pm(n_all):.3f}')
        cruise_table(d, ex, surf, sel_of[a.selection], rec, found_by, pat, lat, lon)

    rest = [e for e in found if not selected(rec[e], a.selection)]
    for k, lab in enumerate(('programs', 'groups', 'lines')):
        c = collections.Counter(x for e in rest for x in (rec[e][k] or ['(none)']))
        print(f'  not selected, in CCHDO ({len(rest)} cruises), {lab}: ' + ', '.join(f'{n} ({v})' for n, v in c.most_common(25)))
    print(f'  not in CCHDO ({len(cruises) - len(found)}): ' + ', '.join([e for e in cruises if rec[e] is None][:15]))

    res = dict(n_points=out[a.selection][0], n_cruises=out[a.selection][1], n_points_all=n_all, lat=lat, lon=lon,
               selection=a.selection, steps=np.array([s_ for s_, _ in steps]), counts=np.array([n_ for _, n_ in steps]),
               programs=np.array(PROGRAMS), source=str(a.glodap), created=str(datetime.datetime.now()))
    for how in DEFINITIONS:
        res['n_points_' + how.replace('+', '_')] = out[how][0]
    np.savez(DEST / 'barrett_sampling.npz', **res)
    print(f'written {DEST / "barrett_sampling.npz"} (selection: {a.selection})')


if __name__ == '__main__':
    main()
