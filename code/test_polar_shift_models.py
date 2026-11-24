#!/usr/bin/env python
"""
test_polar_shift_models.py
==========================
The whole ocean against the common area of the paper in each of the 18 ESMs: the shift of the decline and the gap
of the level of Omega_arag. structural_adjustments.py takes the uncertainty of the whole-ocean adjustment from the
spread of both across the models (polar_models.npz, written here).

States as in the data product: the smoothed, drift-removed states of calc_smooth_states.py (30-year Enting spline of
X(t) - s(k(t)), piControl years in front, the historical run continued with the scenario of the file, SSP2-4.5 where
the model has it), plus the mean of the drift spline over the piControl years parallel to WINDOW (as
calc_product_states.py does with the window of a reference; here without the bias adjustment, the model's own state).
Omega_arag with mocsy (constants of the chain, GLODAP Si and PO4) of the annual-mean state (mean of the monthly
slots) in 1850, 2014 and 2024 (from the series continued with the scenario; the scenarios differ little by 2024), and
in 1750, which no model simulates: the 1850 state with surface pCO2 lowered from 284.317 to 278.3 ppm at constant
alkalinity, temperature and salinity (calc_pi_xco2.target_states), as the 1750 states of the chain (since 16 Nov
2026; before, the shift 1850 -> 2024 was scaled to the decline from 1750, which gave 0.005 points less).
Masks: common area (constraint_results.pkl), the ocean of the paper (regional_common.ocean_on: every cell any product
covers, without inland seas) and every ocean cell of the model; the same cells in all years.
Printed per model: area means of the common area and the ocean, the relative gap of the level ocean / common - 1 in
1750 and 2024, the declines 1750 -> 2024 and the shift ocean minus common area (points, relative to the decline);
1750 -> 1950, 1750 -> 2000 and 1750 -> 2014 (historical only) and 1850 -> 2024 as checks. Then mean and sd across the models.
Since 20 Nov 2026 also levels_filled: every model's Omega_arag of each year put onto the grid of the products
(nearest cell) and filled to the ocean of the paper where the model has no value (fill_tools, as the products and
Jiang et al. in structural_adjustments.py), area means of the common area and the ocean with the weights of the
products' grid. structural_adjustments.py takes the models' shift and the spread of the level gap from these, so
that the three estimates of the whole-ocean adjustment rest on the same ocean. The means on the model's own cells
(levels) stay for comparison.

Output: data/<run>/structural/polar_models.npz, levels(model, mask, year) of the area means on the model's own
cells, levels_filled(model, mask, year) with the model filled to the ocean of the paper, and the metadata.
Usage (code/):  OA_RUN=u74 python -u test_polar_shift_models.py 2>&1 | tee ../logs/u74/test_polar_shift_models.log
"""
import datetime
import pickle
import warnings

import netCDF4
import numpy as np

import oa_config
import calc_omega_arag_obs as co
import calc_pi_xco2 as cpx
import calc_smooth_states as css
import drift_tools as dt
import mocsy
import regional_common as rc
from feedback_pattern import DRIFT_WINDOW as WINDOW    # (1850, 1900): the level of the drift-corrected state
from fill_tools import fill_to, regions                # the filling of the ocean of the paper (as the products)

YEARS = (1850, 1950, 2000, 2014, 2024)                 # years read from the smoothed files (1950, 2000: checks)
X1750 = 278.3                                          # ppm, AR6 (Gulev et al. 2021): the 1750 state of the paper
LEVEL_YEARS = (1750,) + YEARS                          # years of the levels (1750 from the 1850 state)
BASE = 1750                                            # start of the central decline
YEAR = 2024                                            # end of the central decline; 2014: check
MASKS = ('common', 'ocean', 'all')
VARS = ('thetao', 'so', 'dissic', 'talk')
DEST = oa_config.DATA / 'structural'


def annual_states(model, member):
    """the drift-corrected annual-mean state {year: {var: (ny, nx)}} of the smoothed series in YEARS, the grid, the
    scenario that continues the historical series and the cutoff of the smoothing"""
    path = css.out_path(model, member)
    if not path.exists():
        raise FileNotFoundError(f'{path}: run calc_smooth_states.py first')
    d = dt.Drift(model, member)
    st = {y: {} for y in YEARS}
    with netCDF4.Dataset(path) as f:
        lat, lon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        cont, cutoff = str(f.continuation), float(f.cutoff_years)
        yh = np.asarray(f['year_hist'][:]).astype(int)
        ys = np.asarray(f['year_scen'][:]).astype(int)
        for v in VARS:
            off = d.window_mean(v, WINDOW)                                    # mean of s over W' (piControl years)
            assert off.shape == (lat.size, lon.size), f'{model}: drift grid {off.shape}, smoothed {lat.size, lon.size}'
            for y in YEARS:
                if y in yh:
                    a = f[f'{v}_hist'][int(np.searchsorted(yh, y))]
                elif cont != 'none' and y in ys:
                    a = f[f'{v}_{cont}'][int(np.searchsorted(ys, y))]
                else:
                    raise ValueError(f'{model}: no {y} in the smoothed file (continuation {cont})')
                st[y][v] = np.ma.filled(a, np.nan).astype('f8').mean(axis=0) + off      # (month, ny, nx) -> annual
    return st, lat, lon, cont, cutoff


def omega(st, si, po4, la):
    """Omega_arag of a state; NaN where a variable is missing (land cells stored as 0, as CanESM5, excluded); as in the
    chain, the temperature floored at the freezing point and a carbonate system outside calc_omega_arag.valid_carbonate
    NaN (Omega_arag >= 20: e.g. the Red Sea of ACCESS-ESM1-5, an isolated basin with TA and DIC above 10000 umol/kg
    and Omega_arag up to 100, which otherwise dominates its whole-ocean mean)"""
    ok = (np.all([np.isfinite(st[v]) for v in VARS], axis=0) & (st['so'] > 0) & (st['dissic'] > 0) & (st['talk'] > 0))
    n = int(ok.sum())
    t = co.cm.floor_temperature(st['thetao'][ok], st['so'][ok], count=False)
    r = mocsy.mvars(temp=t, sal=st['so'][ok], alk=st['talk'][ok] * 1e-6, dic=st['dissic'][ok] * 1e-6,
                    sil=si[ok] * 1e-6, phos=po4[ok] * 1e-6, patm=np.ones(n), depth=np.zeros(n), lat=la[ok],
                    optcon='mol/kg', optt='Tpot', optp='m', optb=co.MOCSY_OPTB, optk1k2=co.MOCSY_OPTK1K2)
    om = np.full(ok.shape, np.nan)
    om[ok] = np.where(co.cm.valid_carbonate(r[0], r[6]), np.asarray(r[6], 'f8'), np.nan)
    return om


def onto(lat, lon, la, lo, a):
    """a (..., la, lo) on the grid lat, lon by the nearest cell (as structural_adjustments.onto)"""
    jy = np.abs(lat[:, None] - la[None, :]).argmin(1)
    ix = np.abs(((lon[:, None] - lo[None, :]) + 180) % 360 - 180).argmin(1)
    return a[..., jy, :][..., ix]


def filled_levels(om, lat, lon, clat, clon, cmask_c, wc, ocean_c, region_c):
    """the area means (common area, ocean of the paper) of the model's Omega_arag of every year of LEVEL_YEARS, put
    onto the products' grid and filled where the model has no value in any year; returns (2, years) and the share of
    the ocean the model covers itself"""
    A = np.array([onto(clat, clon, lat, lon, om[y]) for y in LEVEL_YEARS])
    have = np.all(np.isfinite(A), axis=0) & ocean_c
    A = np.where(have, A, np.nan)
    F, _ = fill_to(list(A), have, ocean_c & ~have, region_c, wc)
    F = np.array(F)
    lev = np.array([[np.sum(F[j][m] * wc[m]) / np.sum(wc[m]) for j in range(len(LEVEL_YEARS))] for m in (cmask_c & ocean_c, ocean_c)])
    return lev, float(wc[have].sum() / wc[ocean_c].sum())


def summary(L):
    """declines, shifts and relative level gaps from levels (model, mask, year of LEVEL_YEARS); g50: the gap of the
    base year (1750)"""
    i0, i1, i2, i5, i3, i4 = (LEVEL_YEARS.index(y) for y in (BASE, 2014, YEAR, 1850, 2000, 1950))
    c, o = L[:, 0], L[:, 1]
    dec = lambda a, j, k=i0: 100 * (1 - a[:, j] / a[:, k])
    return dict(dc=dec(c, i2), do=dec(o, i2), da=dec(L[:, 2], i2), dc14=dec(c, i1), do14=dec(o, i1),
                dc50=dec(c, i2, i5), do50=dec(o, i2, i5), dc00=dec(c, i3), do00=dec(o, i3), dc195=dec(c, i4), do195=dec(o, i4), g50=o[:, i0] / c[:, i0] - 1, g24=o[:, i2] / c[:, i2] - 1)


def main():
    warnings.filterwarnings('ignore', message='Mean of empty slice')      # land cells
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    rcm = pk['results_common']['UExP']
    clat, clon, cmask = np.asarray(rcm['lat'], 'f8'), np.asarray(rcm['lon'], 'f8'), np.asarray(rcm['mask'], bool)
    wc = oa_config.area_weights(clat, clon)                                 # the products' grid: the filled levels
    ocean_c = rc.ocean_on(clat, clon).astype(bool)
    region_c = regions(clat, ocean_c.shape)
    print(f'smoothed states of calc_smooth_states.py ({css.OUT}), drift level of {WINDOW[0]}-{WINDOW[1]}; Omega_arag '
          f'of the annual-mean state, 1750 the 1850 state at {X1750} ppm; gap: ocean / common - 1 (%); decline from {BASE} (%)')
    print(f'{"model":16s} {"cont.":7s} {"common":>7s} {"ocean":>7s} {"gap":>6s} {"common":>7s} {"ocean":>7s} {"gap":>6s}   '
          f'{"common":>7s} {"ocean":>7s} {"all":>7s} {"shift":>7s} {"rel %":>6s}   | to 2014: {"shift":>7s} {"rel %":>6s}')
    print(f'{"":16s} {"":7s} {"Omega_arag " + str(BASE):>22s} {"Omega_arag " + str(YEAR):>22s}   {"decline to " + str(YEAR):>23s}')
    names, members, conts, levels, cutoffs = [], [], [], [], set()
    levels_f, coverages = [], []
    for model, member in cpx.find_runs(None):
        try:
            st, lat, lon, cont, cutoff = annual_states(model, member)
            jy = np.abs(lat[:, None] - clat[None, :]).argmin(1)
            ix = np.abs(((lon[:, None] - clon[None, :]) + 180) % 360 - 180).argmin(1)
            m_c = cmask[np.ix_(jy, ix)]
            m_o = rc.ocean_on(lat, lon).astype(bool)
            si, po4 = co.load_nutrients(lat, lon)
            la = np.broadcast_to(lat[:, None], m_c.shape)
            w = oa_config.area_weights(lat, lon)
            om = {y: omega(st[y], si, po4, la) for y in YEARS}
            _, res = next(cpx.target_states({v: st[1850][v][None] for v in VARS}, si, po4, la, {'1750': X1750}, [0.0]))
            om[1750] = res['omega_arag'][0]                                     # the 1850 state at 278.3 ppm
            m_all = np.all([np.isfinite(om[y]) for y in LEVEL_YEARS], axis=0)     # the same cells in every year
            lev = np.array([[np.sum(om[y][m & m_all] * w[m & m_all]) / np.sum(w[m & m_all]) for y in LEVEL_YEARS]
                            for m in (m_c, m_o, np.ones_like(m_all))])
            levf, covered = filled_levels(om, lat, lon, clat, clon, cmask, wc, ocean_c, region_c)
        except Exception as e:
            print(f'{model:16s} failed: {type(e).__name__}: {e}', flush=True)
            continue
        names.append(model); members.append(member); conts.append(cont); levels.append(lev); cutoffs.add(cutoff)
        levels_f.append(levf); coverages.append(covered)
        s = {k: v[0] for k, v in summary(lev[None]).items()}
        i0, i2 = LEVEL_YEARS.index(BASE), LEVEL_YEARS.index(YEAR)
        print(f'{model:16s} {cont:7s} {lev[0, i0]:7.3f} {lev[1, i0]:7.3f} {100 * s["g50"]:+6.2f} {lev[0, i2]:7.3f} '
              f'{lev[1, i2]:7.3f} {100 * s["g24"]:+6.2f}   {s["dc"]:7.2f} {s["do"]:7.2f} {s["da"]:7.2f} '
              f'{s["do"] - s["dc"]:+7.2f} {100 * (s["do"] - s["dc"]) / s["dc"]:+6.2f}   |          '
              f'{s["do14"] - s["dc14"]:+7.2f} {100 * (s["do14"] - s["dc14"]) / s["dc14"]:+6.2f}', flush=True)
        dfc, dfo = 100 * (1 - levf[0, i2] / levf[0, i0]), 100 * (1 - levf[1, i2] / levf[1, i0])
        print(f'{"":16s} filled to the ocean of the paper ({100 * covered:.1f} % covered by the model): common {dfc:6.2f} %, '
              f'ocean {dfo:6.2f} %, shift {dfo - dfc:+.3f} points ({100 * (dfo - dfc) / dfc:+.2f} %); level gap 1750 '
              f'{100 * (levf[1, i0] / levf[0, i0] - 1):+.2f} %, {YEAR} {100 * (levf[1, i2] / levf[0, i2] - 1):+.2f} %', flush=True)
    if not names:
        raise SystemExit('no model')
    L = np.array(levels)
    s = summary(L)
    sd = lambda a: a.std(ddof=1) if a.size > 1 else np.nan
    rel, rel14 = (s['do'] - s['dc']) / s['dc'], (s['do14'] - s['dc14']) / s['dc14']
    rel50 = (s['do50'] - s['dc50']) / s['dc50']
    rel00 = (s['do00'] - s['dc00']) / s['dc00']
    rel195 = (s['do195'] - s['dc195']) / s['dc195']
    print(f'  to {YEAR} ({len(names)} models): shift common -> ocean mean {(s["do"] - s["dc"]).mean():+.2f}, sd '
          f'{sd(s["do"] - s["dc"]):.2f} points; relative {100 * rel.mean():+.2f} % (sd {100 * sd(rel):.2f} %); common -> all '
          f'model cells {(s["da"] - s["dc"]).mean():+.2f}')
    print(f'  to 2014 ({len(names)} models, check): shift mean {(s["do14"] - s["dc14"]).mean():+.2f}, sd '
          f'{sd(s["do14"] - s["dc14"]):.2f} points; relative {100 * rel14.mean():+.2f} % (sd {100 * sd(rel14):.2f} %)')
    print(f'  {BASE} -> 1950 (check, historical only): relative {100 * rel195.mean():+.2f} % (sd {100 * sd(rel195):.2f} %)')
    print(f'  {BASE} -> 2000 (check, historical only): relative {100 * rel00.mean():+.2f} % (sd {100 * sd(rel00):.2f} %)')
    print(f'  1850 -> {YEAR} (check, the former central value): relative {100 * rel50.mean():+.2f} % (sd {100 * sd(rel50):.2f} %)')
    i0, i2 = LEVEL_YEARS.index(BASE), LEVEL_YEARS.index(YEAR)
    for lab, g, j in ((str(BASE), s['g50'], i0), (str(YEAR), s['g24'], i2)):
        gap = L[:, 1, j] - L[:, 0, j]
        print(f'  level gap ocean / common - 1 in {lab}: mean {100 * g.mean():+.2f} %, sd {100 * sd(g):.2f} % '
              f'(absolute: mean {gap.mean():+.3f}, sd {sd(gap):.3f}; common area {L[:, 0, j].mean():.3f}, sd {sd(L[:, 0, j]):.3f})')
    print(f'  correlation of the gaps in {BASE} and {YEAR}: {np.corrcoef(s["g50"], s["g24"])[0, 1]:+.2f}')
    # the same with every model filled to the ocean of the paper (the estimate structural_adjustments.py uses)
    LF = np.array(levels_f)
    i0, i2 = LEVEL_YEARS.index(BASE), LEVEL_YEARS.index(YEAR)
    dfc, dfo = 100 * (1 - LF[:, 0, i2] / LF[:, 0, i0]), 100 * (1 - LF[:, 1, i2] / LF[:, 1, i0])
    relf = (dfo - dfc) / dfc
    gf50, gf24 = LF[:, 1, i0] / LF[:, 0, i0] - 1, LF[:, 1, i2] / LF[:, 0, i2] - 1
    print(f'  filled to the ocean of the paper ({len(names)} models; covered by the models themselves {100 * min(coverages):.1f} to '
          f'{100 * max(coverages):.1f} %): shift common -> ocean mean {(dfo - dfc).mean():+.3f}, sd {sd(dfo - dfc):.3f} points; '
          f'relative {100 * relf.mean():+.2f} % (sd {100 * sd(relf):.2f} %); level gap {BASE} {100 * gf50.mean():+.2f} % '
          f'(sd {100 * sd(gf50):.2f} %), {YEAR} {100 * gf24.mean():+.2f} % (sd {100 * sd(gf24):.2f} %), correlation '
          f'{np.corrcoef(gf50, gf24)[0, 1]:+.2f}')
    for lab, ja, jb in (('1850 -> ' + str(YEAR), LEVEL_YEARS.index(1850), i2), (f'{BASE} -> 2000', i0, LEVEL_YEARS.index(2000))):
        dc_, do_ = 100 * (1 - LF[:, 0, jb] / LF[:, 0, ja]), 100 * (1 - LF[:, 1, jb] / LF[:, 1, ja])
        r_ = (do_ - dc_) / dc_
        print(f'    filled, {lab} (check): shift {(do_ - dc_).mean():+.3f} points, relative {100 * r_.mean():+.2f} % (sd {100 * sd(r_):.2f} %)')
    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / 'polar_models.npz'
    np.savez(out, models=np.array(names), members=np.array(members), continuation=np.array(conts), levels=L,
             levels_filled=LF, covered_by_model=np.array(coverages),
             levels_filled_source='every model on the products\' grid (nearest cell), filled to the ocean of the paper where it has no '
                                  'value (fill_tools.fill_to: 5 x 5 neighbours in passes, then the regional mean), area means with the '
                                  'weights of the products\' grid (20 Nov 2026)',
             years=np.array(LEVEL_YEARS), masks=np.array(MASKS), year=YEAR, base=BASE, x1750=X1750, window=np.array(WINDOW),
             source=f'smoothed states of calc_smooth_states.py ({css.OUT}), cutoff {", ".join(f"{c:g}" for c in sorted(cutoffs))} years, drift level of '
                    f'{WINDOW[0]}-{WINDOW[1]}; Omega_arag of the annual-mean state (mocsy); 1750: the 1850 state at {X1750} ppm',
             created=str(datetime.datetime.now()))
    print(f'written {out}')


if __name__ == '__main__':
    main()
