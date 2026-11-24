#!/usr/bin/env python
"""
test_whole_ocean_coverage.py  (18 November 2026)
============================
The three estimates of the whole-ocean adjustment (structural_adjustments.py, step 2) rest on different cells: the
mean of the products on the ocean of the paper with 2 % filled, each ESM on its own valid cells of that ocean, Jiang
et al. on their coverage shared with all 18 ESMs. Here all estimates on the same cells, three times:
  1  minimum overlap: the cells where every estimate (and every ESM) has a value, no filling
  2  the products' maximum coverage: the ocean cells any product covers in every year (98 %); the other estimates
     filled from their neighbours where they have no value
  3  the whole ocean of the paper (regional_common.ocean_on); every estimate filled from its neighbours
     (fill_neighbours of structural_adjustments.py, cells it cannot reach: the mean of their region)
Estimates (decline 1750 -> 2024 unless noted; both states always on the same cells):
  multi-product: in each cell the mean of the products that cover it (constrained 1750 state, 2024 quadratic fit),
     as structural_adjustments.py
  UExP-FNN-U alone (its own constrained 1750 state and 2024 fit)
  ESMs: each model's own drift-corrected states, not bias-adjusted (1750 = the 1850 state at 278.3 ppm), as
     test_polar_shift_models.py; mean and s.d. across the models
  Jiang et al.: their constrained 1750 state (states_Jiang_pi.nc) and their 2005-2014 state (obs_Jiang.nc); decline
     1750 -> 2005-2014
Printed per estimate and area: the decline on the common area, on the area, the shift in points and relative to the
decline on the common area; the relative shift times the paper's decline (mean of the four products on the common
area) is the shift as structural_adjustments.py scales the models and Jiang et al.
Second version (18 Nov 2026, evening): the products and Jiang et al. also without the constraint, their 1750
state the multi-model mean of the bias-adjusted ESMs in each cell (omega_arag_unconstrained_state of
states_<ref>_pi.nc), as the ESMs, which are never constrained.
Reads only, writes nothing.
Usage (code/):  OA_RUN=u74 python -u test_whole_ocean_coverage.py
"""
import pickle
import warnings

import netCDF4
import numpy as np

import oa_config
import regional_common as rc
import structural_adjustments as sa
import test_polar_shift_models as tps

warnings.filterwarnings('ignore', message='Mean of empty slice')
pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
r0 = pk['results_common']['UExP']
lat, lon = np.asarray(r0['lat'], 'f8'), np.asarray(r0['lon'], 'f8')
cmask = np.asarray(r0['mask'], bool)
w = oa_config.area_weights(lat, lon)
ocean = rc.ocean_on(lat, lon).astype(bool)
la2 = np.broadcast_to(lat[:, None], ocean.shape)
region = np.where(la2 > 60, 'Arctic', np.where(la2 < -55, 'Southern', 'other'))
dm = float(np.mean([100 * (1 - pk['results_common'][q]['now'] / pk['results_common'][q]['constrained']['dT + CUC'][0])
                    for q in sa.PRODUCTS]))
pct = lambda m: 100 * w[m & ocean].sum() / w[ocean].sum()
print(f'run {oa_config.DATA}; area weights: {oa_config.WEIGHTS_TEXT}; decline of the paper on the common area {dm:.2f} %')

# ---------------------------------------------------------------- the fields (P = 1750, N = present) on the obs grid
EST = {}
st = {q: sa.product_states(q, lat, lon) for q in sa.PRODUCTS}
Ps = np.array([st[q][0]['omega_arag'] for q in sa.PRODUCTS])
Ns = np.array([st[q][1]['omega_arag'] for q in sa.PRODUCTS])
good = np.isfinite(Ps) & np.isfinite(Ns)
EST['multi-product'] = [(np.nanmean(np.where(good, Ps, np.nan), axis=0), np.nanmean(np.where(good, Ns, np.nan), axis=0))]
iu = sa.PRODUCTS.index('UExP')
EST['UExP-FNN-U'] = [(np.where(good[iu], Ps[iu], np.nan), np.where(good[iu], Ns[iu], np.nan))]
prod_max = ocean & np.any(good, axis=0)


def pi_unconstrained(ref):
    """the unconstrained 1750 Omega_arag of a reference (multi-model mean of the bias-adjusted ESMs) on the obs grid"""
    with netCDF4.Dataset(sa.OUT / f'states_{ref}_pi.nc') as f:
        nm = [str(x) for x in f['state'][:]]
        k_ = next(j for j, x in enumerate(nm) if '1750' in x)
        return sa.onto(lat, lon, np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8'),
                       np.ma.filled(f['omega_arag_unconstrained_state'][k_].astype('f8'), np.nan))


PsU = np.array([pi_unconstrained(q) for q in sa.PRODUCTS])
goodU = np.isfinite(PsU) & np.isfinite(Ns)
EST['multi-product, unconstr.'] = [(np.nanmean(np.where(goodU, PsU, np.nan), axis=0),
                                    np.nanmean(np.where(goodU, Ns, np.nan), axis=0))]
EST['UExP-FNN-U, unconstr.'] = [(np.where(goodU[iu], PsU[iu], np.nan), np.where(goodU[iu], Ns[iu], np.nan))]

try:
    with netCDF4.Dataset(sa.OUT / 'states_Jiang_pi.nc') as f:
        names = [str(x) for x in f['state'][:]]
        k = next(j for j, nm in enumerate(names) if '1750' in nm)
        PJ = sa.onto(lat, lon, np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8'),
                     np.ma.filled(f['omega_arag_state'][k].astype('f8'), np.nan))
    with netCDF4.Dataset(sa.OUT / 'obs_Jiang.nc') as f:
        yj = np.asarray(f['year'][:]).astype(int)
        NJ = sa.onto(lat, lon, np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8'),
                     np.ma.filled(f['omega_arag'][:].astype('f8'), np.nan))
        NJ = NJ[0] if NJ.ndim == 3 else NJ
    print(f'Jiang et al.: present state of the year(s) {list(yj)} (2005-2014), 1750 state "{names[k]}"')
    okj = np.isfinite(PJ) & np.isfinite(NJ)
    EST['Jiang et al.'] = [(np.where(okj, PJ, np.nan), np.where(okj, NJ, np.nan))]
    PJU = pi_unconstrained('Jiang')
    okju = np.isfinite(PJU) & np.isfinite(NJ)
    EST['Jiang et al., unconstr.'] = [(np.where(okju, PJU, np.nan), np.where(okju, NJ, np.nan))]
except (FileNotFoundError, OSError, StopIteration) as e:
    print(f'Jiang et al.: not available ({type(e).__name__}: {e}); left out')

models = []
for model, member in tps.cpx.find_runs(None):
    try:
        s_, la_m, lo_m, _, _ = tps.annual_states(model, member)
        si, po4 = tps.co.load_nutrients(la_m, lo_m)
        lam = np.broadcast_to(la_m[:, None], si.shape)
        o24 = tps.omega(s_[tps.YEAR], si, po4, lam)
        _, res = next(tps.cpx.target_states({v: s_[1850][v][None] for v in tps.VARS}, si, po4, lam,
                                            {'1750': tps.X1750}, [0.0]))
        o50 = res['omega_arag'][0]
        P_, N_ = sa.onto(lat, lon, la_m, lo_m, o50), sa.onto(lat, lon, la_m, lo_m, o24)
        ok_ = np.isfinite(P_) & np.isfinite(N_)
        models.append((np.where(ok_, P_, np.nan), np.where(ok_, N_, np.nan)))
        print(f'  {model:16s} valid on {pct(ok_):.2f} % of the ocean', flush=True)
    except Exception as e:
        print(f'  {model:16s} failed: {type(e).__name__}: {e}', flush=True)
EST['ESMs'] = models
for lab, fl in EST.items():
    if lab != 'ESMs':
        print(f'{lab}: valid on {pct(np.isfinite(fl[0][0]) & np.isfinite(fl[0][1])):.2f} % of the ocean')
print(f'products, covered in every year by at least one: {pct(prod_max):.2f} % of the ocean')


def fill(P, N, have, need):
    """P, N filled in `need` from `have` (fill_neighbours; cells it cannot reach: the regional mean)"""
    (P, N), step = sa.fill_neighbours([P, N], have, need)
    rest = (step == -1) & need
    if rest.any():
        P, N = sa.regional_fill([P, N], need & ~rest & np.isfinite(P), rest, region, w)
    return P, N


def decline(P, N, m):
    k = m & np.isfinite(P) & np.isfinite(N)
    return 100 * (1 - sa.amean(N, k, w) / sa.amean(P, k, w)), pct(k)


# ---------------------------------------------------------------- the three areas
valid = lambda fl: np.isfinite(fl[0]) & np.isfinite(fl[1])
overlap = ocean.copy()
for fls in EST.values():
    for fl in fls:
        overlap &= valid(fl)
AREAS = {'1 minimum overlap': (overlap, False), '2 products, maximum coverage': (prod_max, True),
         '3 whole ocean': (ocean, True)}
print(f'\ncommon area {pct(cmask):.2f} % of the ocean; minimum overlap {pct(overlap):.2f} %, products maximum '
      f'{pct(prod_max):.2f} %, whole ocean 100 %')
for alab, (A, do_fill) in AREAS.items():
    print(f'\n===== {alab} ({pct(A):.2f} % of the ocean)')
    for lab, fls in EST.items():
        sh, rel, dcs, das = [], [], [], []
        for P, N in fls:
            if do_fill:
                need = A & ~valid((P, N))
                P, N = fill(P, N, valid((P, N)) & ocean, need)
            dc, _ = decline(P, N, cmask & A)
            da, cov = decline(P, N, A)
            sh.append(da - dc); rel.append((da - dc) / dc); dcs.append(dc); das.append(da)
        sh, rel = np.array(sh), np.array(rel)
        if len(fls) == 1:
            print(f'  {lab:25s} common {dcs[0]:6.2f} %, area {das[0]:6.2f} % (filled to {cov:.2f} %): shift {sh[0]:+.3f} '
                  f'points, {100 * rel[0]:+.2f} % of the decline -> scaled to the paper {rel[0] * dm:+.3f} points')
        else:
            print(f'  {lab:25s} ({len(fls)} models) common {np.mean(dcs):6.2f} %, area {np.mean(das):6.2f} %: shift '
                  f'{sh.mean():+.3f} +- {sh.std(ddof=1):.3f} points, {100 * rel.mean():+.2f} % of the decline -> scaled '
                  f'to the paper {rel.mean() * dm:+.3f} +- {rel.std(ddof=1) * dm:.3f} points')
