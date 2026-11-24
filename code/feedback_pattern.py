#!/usr/bin/env python
"""
feedback_pattern.py
===================
The spatial pattern of the calcification feedback in surface alkalinity, for the alkalinity adjustment of
structural_adjustments.py, written to data/<run>/structural/feedback_pattern.npz.

Method: annual-mean surface TA and S, each drift-corrected as in the chain (CMIP6: drift_tools.Drift, the Enting
spline with a 50-year cutoff of the whole piControl at the parallel years, calc_drift.py; GFDL-ESM2M-COBALTv2: the same
spline, drift_tools.smoother, fitted here to its piControl 1700-2099), then nTA = TA x 35 / S (umol/kg; only for
S 20-45 and TA 500-5000 umol/kg), mapped to the 1 degree grid of the products. The forced pattern of a model is the
regression slope of every cell on its own common-area mean nTA over all years (CMIP6 1850-2014, GFDL-ESM2M-COBALTv2
1860-2014): the common-area mean of a slope map is 1, internal variability that is not coherent with the large-scale
change drops out, and the start year hardly matters. Then: moving mean over (2 SMOOTH + 1)^2 cells of ocean, gaps
filled with the zonal mean, normalized to a common-area mean of 1. Negative values are kept (calcification can also
increase locally, and TA anomalies are redistributed by the circulation).

  pattern              the pattern used: mean of the slope maps of four models with Omega-dependent calcification,
                       each with the same weight: GFDL-CM4, GFDL-ESM4, UKESM1-0-LL (CMIP6) and GFDL-ESM2M-COBALTv2
                       (ensemble mean of five members, coupled)
  pattern_s0, _s2, _s10  the same with half-widths 0, 2 and 10 cells of the moving mean (sensitivity)
  pattern_cmip6        the three CMIP6 models only (sensitivity)
  pattern_<model>      each of the three CMIP6 feedback models alone (sensitivity; ESM2M alone: pattern_esm2m)
  GFDL-ESM2M-COBALTv2  five members (historical to 2005, then RCP2.6; perturbed initial conditions; one piControl,
                       1700-2100, on the same calendar years; before 1860 atmospheric CO2 at the 1860 level and only
                       land use changing, so 1860-2014 is the forced period), three experiments: coupled; noRAD
                       (radiation at the 1860 CO2, carbon chemistry sees the rising CO2: the feedback without climate
                       change); noBGC (the reverse: climate change, freshwater and circulation, without the feedback).
    pattern_esm2m      coupled, all forcings
    pattern_esm2m_fb   noRAD: the feedback alone, carried by the circulation, without freshwater effects
    pattern_esm2m_cmb  coupled minus noBGC: the feedback with the circulation changes
                       (the experiments are regressed on the coupled ensemble-mean series, so their maps add up)

Time series (added 7 November 2026, for the time course of the alkalinity adjustment in structural_adjustments.py):
the common-area mean of the drift-corrected annual surface nTA of every feedback model, umol/kg: the CMIP6 models
1850-2014 (historical) and 2015-2100 for every scenario they have (series_<model>_historical, series_<model>_<ssp>,
with years_historical, years_<model>_<ssp>), and GFDL-ESM2M-COBALTv2 1860-2099 (historical to 2005, then RCP2.6) as the ensemble
mean of the five members for coupled, noRAD and noBGC (series_esm2m_<exp>, years_esm2m). The patterns are unchanged
(slopes over 1850-2014 and 1860-2014).

Printed: the slope maps by latitude band, the shares of the experiments, the patterns by band, area-weighted spatial
correlations, the series' changes.

Usage (code/):  OA_RUN=u74 python -u feedback_pattern.py
"""
import argparse
import datetime
import pickle
import warnings
from pathlib import Path

import netCDF4
import numpy as np

import oa_config
import calc_bias_adjusted as cb
import calc_pi_xco2 as cpx
import calc_product_members as pm
import drift_tools as dt
import regional_common as rc

DEST = oa_config.DATA / 'structural'
FEEDBACK = ('GFDL-CM4', 'GFDL-ESM4', 'UKESM1-0-LL')    # constraint_refs.ALK_MODELS
PERIOD_CMIP = (1850, 2014)
ESM2M_DIR = oa_config.ESM2M_DIR
ESM2M_EXPS = ('coupled', 'noRAD', 'noBGC')
ESM2M_SCEN = 'climate_change_RCP26'
ESM2M_MEMBERS = ('ens2', 'ens3', 'ens4', 'ens5', 'ens6')
ESM2M_NAME = 'GFDL-ESM2M-COBALTv2'
PERIOD_ESM2M = (1860, 2099)     # the files run to 2100 (RCP2.6 from 2006); the pattern uses 1860-2014
PATTERN_LAST = 2014             # last year of the slopes (patterns)
SCENARIOS = ('ssp126', 'ssp245', 'ssp370', 'ssp585')
CONTROL_ESM2M = (1700, 2099)
RHO_ESM2M = 1035.0         # kg m-3, GFDL reference density (as oa_config.REF_DENSITY for GFDL-CM4 and GFDL-ESM4)
DRIFT_WINDOW = (1850, 1900)  # reference window of Drift.term: only a constant offset, no effect on the slopes
SMOOTH = 5                 # half-width in cells of the moving mean (11 x 11 cells, about 1000 km)
SMOOTH_TESTS = (0, 2, 10)
S_RANGE = (20.0, 45.0)     # psu: nTA only where the normalization is meaningful (river TA dominates fresher water;
                           # near-zero S at coastal cells after regridding makes TA x 35 / S explode)
TA_RANGE = (500.0, 5000.0)   # umol/kg: physical surface TA (also removes unmasked fill values)
BANDS = ((-90, -60), (-60, -30), (-30, 30), (30, 60), (60, 90))


# ------------------------------------------------------------------------------------------------ helpers
def onto(lat, lon, la, lo, a):
    jy = np.abs(lat[:, None] - la[None, :]).argmin(1)
    ix = np.abs(((lon[:, None] - lo[None, :]) + 180) % 360 - 180).argmin(1)
    return a[..., jy, :][..., ix]


def amean(a, m, w):
    k = m & np.isfinite(a)
    return float(np.sum(a[k] * w[k]) / np.sum(w[k]))


def nta(ta, s):
    """TA x 35 / S from TA in umol/kg; NaN outside S_RANGE and TA_RANGE (missing, fill values, fresh coastal water).
    A cell outside in any year has no slope (slope_on needs all years) and is filled by the smoothing."""
    with np.errstate(invalid='ignore', divide='ignore'):
        ok = (s >= S_RANGE[0]) & (s <= S_RANGE[1]) & (ta >= TA_RANGE[0]) & (ta <= TA_RANGE[1])
        return np.where(ok, ta * 35 / s, np.nan)


def screen(S, ocean, w, lat, lon):
    """share of the ocean (and of the ocean north of 60 N) without a slope; cells with |slope| > 20 and the largest"""
    la2 = np.broadcast_to(lat[:, None], S.shape)
    miss = ocean & ~np.isfinite(S)
    arc = ocean & (la2 >= 60)
    big = ocean & np.isfinite(S) & (np.abs(S) > 20)
    txt = (f'{100 * (w * miss).sum() / (w * ocean).sum():.1f} % of the ocean without values '
           f'({100 * (w * (miss & arc)).sum() / (w * arc).sum():.1f} % north of 60N); {int(big.sum())} cells |slope| > 20')
    if big.any():
        j, i = np.unravel_index(np.nanargmax(np.where(ocean, np.abs(S), np.nan)), S.shape)
        txt += f' (largest {S[j, i]:+.3g} at {lat[j]:.1f}N {lon[i]:.1f}E)'
    return txt


def spline_anomaly(a):
    """Enting spline (drift_tools.smoother, 50-year cutoff) of a (years, lat, lon) minus its mean, per cell; NaN where
    a cell is missing in any year"""
    n = a.shape[0]
    flat = a.reshape(n, -1)
    ok = np.all(np.isfinite(flat), axis=0)
    out = np.full(flat.shape, np.nan)
    out[:, ok] = dt.smoother(n) @ flat[:, ok]
    out[:, ok] -= flat[:, ok].mean(axis=0)
    return out.reshape(a.shape)


def area_series(Y, cmask, w):
    """common-area mean of Y (years, lat, lon) for every year, over the cells with values in all years"""
    m = cmask & np.all(np.isfinite(Y), axis=0)
    return np.array([np.sum(Y[i][m] * w[m]) / np.sum(w[m]) for i in range(Y.shape[0])])


def slope_on(x, Y):
    """regression slope of every cell of Y (years, lat, lon) on the series x"""
    ok = np.all(np.isfinite(Y), axis=0)
    xa = x - x.mean()
    S = np.full(ok.shape, np.nan)
    S[ok] = (xa[:, None] * (Y[:, ok] - Y[:, ok].mean(axis=0))).sum(axis=0) / (xa ** 2).sum()
    return S


def slope_map(years, Y, cmask, w):
    """regression slope of every cell on the common-area mean of Y itself"""
    x = area_series(Y, cmask, w)
    return slope_on(x, Y), x


def finish(raw, ocean, cmask, w, lat, k=SMOOTH):
    """moving mean over (2k + 1)^2 cells of ocean, zonal-mean fill, common-area mean 1 (negative values kept)"""
    raw = np.where(ocean, raw, np.nan)
    pad = np.pad(raw, ((k, k), (0, 0)), constant_values=np.nan)
    pad = np.concatenate([pad[:, raw.shape[1] - k:], pad, pad[:, :k]], axis=1) if k else pad
    num = np.zeros_like(raw); den = np.zeros_like(raw)
    for dy in range(2 * k + 1):
        for dx in range(2 * k + 1):
            sl = pad[dy:dy + raw.shape[0], dx:dx + raw.shape[1]]
            ok = np.isfinite(sl); num += np.where(ok, sl, 0); den += ok
    sm = np.where(ocean, np.where(den > 0, num / np.maximum(den, 1), np.nan), np.nan)
    with np.errstate(invalid='ignore'):
        zon = np.nanmean(sm, axis=1)
    zon = np.interp(lat, lat[np.isfinite(zon)], zon[np.isfinite(zon)])
    sm = np.where(ocean & ~np.isfinite(sm), zon[:, None] * np.ones(sm.shape), sm)
    neg = float((w * (ocean & (sm < 0))).sum() / (w * ocean).sum())
    return sm / amean(sm, cmask, w), neg


def bands(p, ocean, w, lat):
    la2 = np.broadcast_to(lat[:, None], p.shape)
    return [amean(p, ocean & (la2 >= a_) & (la2 < b_), w) for a_, b_ in BANDS]


def corr(a, b, m, w):
    k = m & np.isfinite(a) & np.isfinite(b)
    aa, bb, ww = a[k] - np.sum(a[k] * w[k]) / w[k].sum(), b[k] - np.sum(b[k] * w[k]) / w[k].sum(), w[k]
    return float(np.sum(ww * aa * bb) / np.sqrt(np.sum(ww * aa ** 2) * np.sum(ww * bb ** 2)))


# ------------------------------------------------------------------------------------------------ CMIP6
def cmip_model(model, member, lat, lon):
    """annual-mean surface nTA 1850-2014 of the historical run, TA and S read and drift-corrected as in the chain
    (calc_bias_adjusted.ModelRun, drift_tools.Drift), on the grid lat, lon"""
    run = cb.ModelRun(model, 'historical', member)
    years = run.years[::run.step]
    idx = np.where((years >= PERIOD_CMIP[0]) & (years <= PERIOD_CMIP[1]))[0]
    lon2, lat2 = np.meshgrid(run.lon, run.lat)
    jy, ix = np.arange(run.lat.size), np.arange(run.lon.size)
    ta, so = [], []
    for k in idx:
        f = run.read(k * run.step, (k + 1) * run.step, jy, ix, lon2, lat2)
        ta.append(f['talk'].mean(axis=0)); so.append(f['so'].mean(axis=0))
    run.close()
    y = years[idx]
    d = dt.Drift(model, member)
    ta = np.array(ta) - d.term('talk', 'historical', y, DRIFT_WINDOW)
    so = np.array(so) - d.term('so', 'historical', y, DRIFT_WINDOW)
    return y, onto(lat, lon, run.lat, run.lon, nta(ta, so))


def cmip_control_field(model, member, lat, lon):
    """surface nTA of the piControl averaged over the years parallel to DRIFT_WINDOW, on the grid lat, lon: the level
    the drift-corrected series refer to (drift_tools.Drift.term re-centres the spline on these years), so that
    series - its common-area mean is the anomaly to the control; and the years used"""
    d = dt.Drift(model, member)
    q0, q1 = dt.parallel_years(d.by, d.hist0, *DRIFT_WINDOW)
    run = cb.ModelRun(model, 'piControl', member)
    years = run.years[::run.step][:run.nt // run.step]
    idx = np.where((years >= q0) & (years <= q1))[0]
    assert idx.size > 0, f'{model}: no piControl years {q0}-{q1}'
    lon2, lat2 = np.meshgrid(run.lon, run.lat)
    jy, ix = np.arange(run.lat.size), np.arange(run.lon.size)
    ta, so = 0.0, 0.0
    for k in idx:
        f = run.read(k * run.step, (k + 1) * run.step, jy, ix, lon2, lat2)
        ta = ta + f['talk'].mean(axis=0) / idx.size
        so = so + f['so'].mean(axis=0) / idx.size
    run.close()
    return onto(lat, lon, run.lat, run.lon, nta(ta, so)), (int(years[idx][0]), int(years[idx][-1]))


def series_mask(Y, ctl, cmask):
    """the cells of one model's series: common-area cells with nTA in every historical year and in the control mean;
    the historical series, the scenario series and the control level are all means over these cells"""
    return cmask & np.all(np.isfinite(Y), axis=0) & np.isfinite(ctl)


def cmip_series(model, member, exp, lat, lon, mask, w, drift_member):
    """mean over the cells of mask of the drift-corrected annual surface nTA of one run (umol/kg), year by year, and
    its years; TA and S read and drift-corrected as in the chain, the drift that of the historical member
    (drift_member, as calc_smooth_states does for a substitute scenario member). A cell of mask without a value in a
    year (S or TA outside the ranges of nta) is left out of that year's mean; the number of such cell-years is
    returned."""
    run = cb.ModelRun(model, exp, member)
    years = run.years[::run.step][:run.nt // run.step]
    lon2, lat2 = np.meshgrid(run.lon, run.lat)
    jy, ix = np.arange(run.lat.size), np.arange(run.lon.size)
    d = dt.Drift(model, drift_member)
    out, dropped = [], 0
    for k, y_ in enumerate(years):
        f = run.read(k * run.step, (k + 1) * run.step, jy, ix, lon2, lat2)
        ta = f['talk'].mean(axis=0) - d.term('talk', exp, [y_], DRIFT_WINDOW)[0]
        so = f['so'].mean(axis=0) - d.term('so', exp, [y_], DRIFT_WINDOW)[0]
        a = onto(lat, lon, run.lat, run.lon, nta(ta, so))
        dropped += int((mask & ~np.isfinite(a)).sum())
        out.append(amean(a, mask, w))
    run.close()
    return years.astype(int), np.array(out), dropped


# ------------------------------------------------------------------------------------------------ GFDL-ESM2M-COBALTv2
def esm2m_annual(folder, var, y0, y1):
    """annual means of the surface field var from the decade files in folder"""
    years, out = [], []
    for dec in range(y0 // 10 * 10, y1 + 1, 10):
        f_ = folder / f'OA_{dec}.01.01_to_{dec + 10}.01.01_r360_180.nc'
        with netCDF4.Dataset(f_) as f:
            a = np.ma.filled(f[var][:, 0].astype('f8'), np.nan)
            la, lo = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        assert a.shape[0] == 120, f'{f_}: {a.shape[0]} months'
        a = a.reshape(10, 12, *a.shape[1:]).mean(axis=1)
        for k in range(10):
            if y0 <= dec + k <= y1:
                years.append(dec + k); out.append(a[k])
    return np.array(years), np.array(out), la, lo


def esm2m_ta_s(folder, y0, y1):
    """annual surface TA (umol/kg with RHO_ESM2M) and S"""
    y, ta, la, lo = esm2m_annual(folder, 'talk', y0, y1)
    _, so, _, _ = esm2m_annual(folder, 'salt', y0, y1)
    return y, ta * 1e6 / RHO_ESM2M, so, la, lo


def esm2m_control(lat=None, lon=None):
    """Enting-spline drift of TA and S of the piControl (all of 1700-2099), s(k) - mean, at PERIOD_ESM2M; with the
    grid also the nTA of the control mean on it (the level the series refer to)"""
    y, ta, so, la, lo = esm2m_ta_s(ESM2M_DIR / 'pi_control', *CONTROL_ESM2M)
    sel = (y >= PERIOD_ESM2M[0]) & (y <= PERIOD_ESM2M[1])
    ctl = None
    if lat is not None:
        ctl = onto(lat, lon, la, lo, nta(ta.mean(axis=0), so.mean(axis=0)))
    return y[sel], spline_anomaly(ta)[sel], spline_anomaly(so)[sel], ctl


def esm2m_experiment(exp, ctl, lat, lon, cmask, w):
    """ensemble mean of the drift-corrected nTA of an experiment, and the common-area series of every member"""
    yc, dta, dso = ctl
    acc, ser = None, {}
    for mem in ESM2M_MEMBERS:
        folder = ESM2M_DIR / exp / ESM2M_SCEN / mem / 'OA_variables'
        if not folder.is_dir():
            print(f'  {exp} {mem}: no folder {folder}', flush=True)
            continue
        y, ta, so, la, lo = esm2m_ta_s(folder, *PERIOD_ESM2M)
        assert np.array_equal(y, yc), f'{exp} {mem}: years differ from the control'
        a = onto(lat, lon, la, lo, nta(ta - dta, so - dso))
        ser[mem] = area_series(a, cmask, w)
        acc = a if acc is None else acc + a
    return (None if acc is None else acc / len(ser)), ser


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    warnings.filterwarnings('ignore', message='Mean of empty slice')
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    r0 = pk['results_common']['UExP']
    lat, lon = np.asarray(r0['lat'], 'f8'), np.asarray(r0['lon'], 'f8')
    cmask = np.asarray(r0['mask'], bool)
    ocean = rc.ocean_on(lat, lon).astype(bool)
    w = oa_config.area_weights(lat, lon)
    fmt = lambda b: ', '.join(f'{x_[0]}..{x_[1]}: {v_:.2f}' for x_, v_ in zip(BANDS, b))

    print('===== CMIP6 feedback models (1850-2014; scenarios to 2100)')
    raws, series = {}, {}
    for model, member in cpx.find_runs(None):
        if model not in FEEDBACK:
            continue
        y, Y = cmip_model(model, member, lat, lon)
        S_, x_ = slope_map(y, Y, cmask, w)                      # the pattern: slopes on the common-area mean
        raws[model] = S_
        # the series: historical, scenarios and control level all as means over the same cells of the model
        ctl_, yrs_c = cmip_control_field(model, member, lat, lon)
        mask_m = series_mask(Y, ctl_, cmask)
        x_ = area_series(Y, mask_m, w)
        lev_ = amean(ctl_, mask_m, w)
        series[f'series_{model}_historical'] = x_
        series['years_historical'] = y
        series[f'control_{model}'] = lev_
        series[f'cells_{model}'] = int(mask_m.sum())
        print(f'               series over {int(mask_m.sum())} of {int(cmask.sum())} common-area cells; control level {lev_:.2f} umol/kg '
              f'(piControl years {yrs_c[0]}-{yrs_c[1]}); historical anomaly to it {x_[y <= 1859].mean() - lev_:+.2f} (1850-59), '
              f'{x_[y >= 1995].mean() - lev_:+.2f} (1995-2014)', flush=True)
        for s_ in SCENARIOS:
            mem_s, note = pm.scenario_member(model, s_, member)
            if mem_s is None:
                continue
            ys_, xs_, dropped = cmip_series(model, mem_s, s_, lat, lon, mask_m, w, member)
            keep = (ys_ > int(y[-1])) & (ys_ <= 2100)
            series[f'series_{model}_{s_}'] = xs_[keep]
            series[f'years_{model}_{s_}'] = ys_[keep]
            print(f'               {s_} ({mem_s}{", " + note if note else ""}): nTA {xs_[keep][-1] - x_[(y >= 1995)].mean():+.2f} umol/kg '
                  f'(1995-2014 -> {ys_[keep][-1]}); {dropped} cell-years of the series cells without a value', flush=True)
        print(f'  {model:12s} common-area nTA {x_[(y >= 1995)].mean() - x_[y <= 1869].mean():+.2f} umol/kg (1850-69 -> 1995-2014); '
              f'slopes by latitude {fmt(bands(S_, ocean, w, lat))}\n               {screen(S_, ocean, w, lat, lon)}', flush=True)
        del Y
    assert set(raws) == set(FEEDBACK), f'missing: {set(FEEDBACK) - set(raws)}'

    print(f'===== GFDL-ESM2M-COBALTv2 ({PERIOD_ESM2M[0]}-{PERIOD_ESM2M[1]}, five members; slopes 1860-{PATTERN_LAST} on the coupled '
          'ensemble-mean common-area nTA)')
    *ctl, esm2m_ctl = esm2m_control(lat, lon)
    ctl = tuple(ctl)
    yrs = ctl[0]
    pat_sel = yrs <= PATTERN_LAST
    chg = lambda x_: x_[(yrs >= 1995) & pat_sel].mean() - x_[yrs <= 1879].mean()
    E = {}
    for exp in ESM2M_EXPS:
        Ym, ser = esm2m_experiment(exp, ctl, lat, lon, cmask, w)
        if Ym is None:
            print(f'  {exp}: no members, left out', flush=True)
            continue
        E[exp] = Ym
        print(f'  {exp:8s} common-area nTA change 1860-79 -> 1995-2014: members '
              + ', '.join(f'{v_:+.2f}' for v_ in map(chg, ser.values())) + ' umol/kg', flush=True)
    assert 'coupled' in E, 'coupled ensemble missing'
    # the series: every experiment and the control level as means over the same cells (those of the coupled ensemble)
    mask_e = series_mask(E['coupled'], esm2m_ctl, cmask)
    esm2m_level = amean(esm2m_ctl, mask_e, w)
    series['control_esm2m'] = esm2m_level
    series['cells_esm2m'] = int(mask_e.sum())
    series['years_esm2m'] = yrs
    print(f'  series over {int(mask_e.sum())} of {int(cmask.sum())} common-area cells; control level {esm2m_level:.2f} umol/kg '
          f'(nTA of the control mean 1700-2099)', flush=True)
    for exp, Ym in E.items():
        series[f'series_esm2m_{exp}'] = np.array([amean(Ym[i], mask_e, w) for i in range(Ym.shape[0])])
        print(f'  {exp:8s} ensemble-mean series: anomaly to the control {series[f"series_esm2m_{exp}"][yrs <= 1879].mean() - esm2m_level:+.2f} '
              f'(1860-79), {series[f"series_esm2m_{exp}"][-1] - esm2m_level:+.2f} ({yrs[-1]}) umol/kg', flush=True)
    x = area_series(E['coupled'][pat_sel], cmask, w)
    S = {exp: slope_on(x, Y[pat_sel]) for exp, Y in E.items()}
    del E
    yp = yrs[pat_sel]                      # x covers the pattern years only (1860 to PATTERN_LAST), not all of yrs
    print(f'  coupled ensemble mean: {x[yp >= 1995].mean() - x[yp <= 1879].mean():+.2f} umol/kg; slopes per unit of it '
          '(common-area mean = share; by latitude):')
    for exp, S_ in S.items():
        print(f'    {exp:8s} {amean(S_, cmask, w):+.2f}; {fmt(bands(S_, ocean, w, lat))}; {screen(S_, ocean, w, lat, lon)}')
    if 'noRAD' in S and 'noBGC' in S:
        sm_ = S['noRAD'] + S['noBGC']
        print(f'    noRAD + noBGC {amean(sm_, cmask, w):+.2f}; {fmt(bands(sm_, ocean, w, lat))}  (linearity: close to coupled)')
    raws[ESM2M_NAME] = S['coupled']
    members4 = FEEDBACK + (ESM2M_NAME,)

    # the patterns (all with negative values kept)
    raw4 = np.nanmean([raws[m] for m in members4], axis=0)
    raw3 = np.nanmean([raws[m] for m in FEEDBACK], axis=0)
    pats = {'pattern': ('four models (used)', raw4, SMOOTH)}
    for k_ in SMOOTH_TESTS:
        pats[f'pattern_s{k_}'] = (f'four models, half-width {k_}', raw4, k_)
    pats['pattern_cmip6'] = ('three CMIP6 models', raw3, SMOOTH)
    pats['pattern_esm2m'] = ('ESM2M coupled, all forcings', S['coupled'], SMOOTH)
    for m_ in FEEDBACK:                    # each CMIP6 feedback model alone (sensitivity, 16 Nov 2026)
        pats[f'pattern_{m_}'] = (f'{m_} alone', raws[m_], SMOOTH)
    if 'noRAD' in S:
        pats['pattern_esm2m_fb'] = ('ESM2M noRAD, feedback alone', S['noRAD'], SMOOTH)
    if 'noBGC' in S:
        pats['pattern_esm2m_cmb'] = ('ESM2M coupled - noBGC', S['coupled'] - S['noBGC'], SMOOTH)
    out = dict(lat=lat, lon=lon, models=np.array(members4), members=np.array(ESM2M_MEMBERS), smooth=SMOOTH,
               created=str(datetime.datetime.now()))
    print('  patterns (common-area mean 1; share of the ocean negative; by latitude):')
    for key, (lab, raw, k_) in pats.items():
        p_, neg_ = finish(raw, ocean, cmask, w, lat, k_)
        out[key] = p_
        print(f'    {key:18s} {lab:30s} {100 * neg_:4.1f} %  {fmt(bands(p_, ocean, w, lat))}')
    out['raw'] = raw4
    for m_ in members4:
        out['raw_' + m_] = raws[m_]
    out.update(series)
    out['series_note'] = ('common-area mean of the drift-corrected annual surface nTA (umol/kg): series_<model>_historical '
                          '(years_historical), series_<model>_<ssp> (years_<model>_<ssp>), series_esm2m_<exp> (years_esm2m; historical '
                          'to 2005, then RCP2.6; ensemble mean of the members); control_<model>, control_esm2m: the level of the '
                          'piControl the series refer to (series - control = anomaly to the control); every series of a '
                          'model and its control level are means over the same cells (cells_<model>, cells_esm2m: common-area '
                          'cells with nTA in every historical year and in the control mean)')

    print('\n===== comparison (area-weighted correlation over the ocean; over the common area)')
    for key in [k for k in pats if k != 'pattern']:
        print(f'  pattern vs {key:18s} {corr(out["pattern"], out[key], ocean, w):+.2f}; {corr(out["pattern"], out[key], cmask, w):+.2f}')
    for m_ in members4:
        oth = np.nanmean([raws[n] for n in members4 if n != m_], axis=0)
        print(f'  {m_:20s} vs the mean of the other three (unsmoothed): {corr(raws[m_], oth, ocean, w):+.2f}')
    DEST.mkdir(parents=True, exist_ok=True)
    np.savez(DEST / 'feedback_pattern.npz', **out)
    print(f'written {DEST / "feedback_pattern.npz"}')


if __name__ == '__main__':
    main()
