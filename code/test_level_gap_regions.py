#!/usr/bin/env python
"""
test_level_gap_regions.py  (where does the larger level gap of the products come from?)
=========================
The whole-ocean mean Omega_arag lies 5.0 % (1750) and 5.2 % (2024) below the common-area mean in the products, but
4.2 % (1850) and 4.5 % (2024) in the 18 ESMs. The gap is a sum over the regions outside the common area:
    gap = sum_r (A_r / A) (Omega_r / Omega_common - 1)
(A_r area of region r, A of the whole ocean, area weights of the chain), so the contribution of each region shows
where products and models differ. Regions outside the common area: Arctic (north of 60 N), Southern Ocean (south of
55 S, as structural_adjustments.whole_ocean), and the rest (marginal seas and coasts), the rest split into cells a
product covers and cells filled with the regional mean.
Products: as structural_adjustments.whole_ocean (mean of the products' constrained 1750 states and 2024 fits in each
cell), with the previous fill (the mean of the region) and the new one (fill_neighbours, 10 November 2026), and the
decline and shift with each. --no-models: the products only. ESMs: the states of test_polar_shift_models
(1850 and 2024, with the validity check), each model's own regional ratios, then the mean and s.d. over the models.
Run it on the v2 data (with the weights of that run), where the 5.0 and 5.2 % come from:
Usage (code/ of v4):  OA_BASE=<production folder of the whole-cell run> OA_WEIGHTS=cell \
                      OA_RUN=u74 python -u test_level_gap_regions.py
Only for us, no figures.
"""
import pickle
import sys
import warnings

import numpy as np

import oa_config
import calc_omega_arag_obs as co
import calc_pi_xco2 as cpx
import regional_common as rc
import structural_adjustments as sa
import test_polar_shift_models as tp
from test_access_map import on_grid, amean


def regions(lat, lon, cmask, ocean, covered):
    la2 = np.broadcast_to(lat[:, None], ocean.shape)
    out_ = ocean & ~cmask
    return {'Arctic > 60N': out_ & (la2 > 60), 'Southern < 55S': out_ & (la2 < -55),
            'rest, covered': out_ & (la2 >= -55) & (la2 <= 60) & covered,
            'rest, filled': out_ & (la2 >= -55) & (la2 <= 60) & ~covered}


def decompose(f, R, cmask, ocean, w):
    """per region: area share of the ocean, Omega_r / Omega_common - 1 (%), contribution to the gap (%); the gap"""
    ok = ocean & np.isfinite(f)
    wo = w[ok].sum()
    oc = amean(f, cmask & ok, w)
    rows = {}
    for lab, m in R.items():
        m = m & ok
        if not m.any():
            rows[lab] = (0.0, np.nan, 0.0)
            continue
        rel = amean(f, m, w) / oc - 1
        rows[lab] = (100 * w[m].sum() / wo, 100 * rel, 100 * w[m].sum() / wo * rel)
    return rows, 100 * (amean(f, ok, w) / oc - 1)


def main():
    warnings.filterwarnings('ignore', message='Mean of empty slice')
    print(f'data {oa_config.DATA}; weights {oa_config.WEIGHTS}')
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    r = pk['results_common']['UExP']
    lat, lon, cmask = np.asarray(r['lat'], 'f8'), np.asarray(r['lon'], 'f8'), np.asarray(r['mask'], bool)
    ocean = rc.ocean_on(lat, lon).astype(bool)
    w = oa_config.area_weights(lat, lon)

    # products, as structural_adjustments.whole_ocean
    st = {q: sa.product_states(q, lat, lon) for q in sa.PRODUCTS}
    Ps = np.array([st[q][0]['omega_arag'] for q in sa.PRODUCTS]); Ns = np.array([st[q][1]['omega_arag'] for q in sa.PRODUCTS])
    good = np.isfinite(Ps) & np.isfinite(Ns)
    with np.errstate(invalid='ignore'):
        P, N = np.nanmean(np.where(good, Ps, np.nan), axis=0), np.nanmean(np.where(good, Ns, np.nan), axis=0)
    covered = np.isfinite(P) & np.isfinite(N)
    R = regions(lat, lon, cmask, ocean, covered)
    la2 = np.broadcast_to(lat[:, None], ocean.shape)
    region = np.where(la2 > 60, 'Arctic', np.where(la2 < -55, 'Southern', 'other'))
    outside = ocean & ~cmask
    P_r, N_r = sa.regional_fill([P, N], outside & covered, outside, region, w)          # the previous fill
    (P_n, N_n), step = sa.fill_neighbours([P, N], covered & ocean, outside)             # the new fill
    rest = step == -1
    P_n, N_n = sa.regional_fill([P_n, N_n], outside & ~rest & np.isfinite(P_n), rest & outside, region, w)
    print(f'neighbour fill: {int(step.max())} passes, {int(rest.sum())} unreachable cells')
    n_cov = {q: 100 * w[ocean & ~cmask & good[i]].sum() / w[ocean & ~cmask].sum() for i, q in enumerate(sa.PRODUCTS)}
    print('share of the area outside the common area that each product covers: '
          + ', '.join(f'{q} {v:.0f} %' for q, v in n_cov.items()))

    res = {}
    for tag, (p_, n_) in (('regional fill', (P_r, N_r)), ('neighbour fill', (P_n, N_n))):
        res[f'products 1750, {tag}'] = decompose(p_, R, cmask, ocean, w)
        res[f'products 2024, {tag}'] = decompose(n_, R, cmask, ocean, w)
        d_c = 100 * (1 - amean(n_, cmask, w) / amean(p_, cmask, w))
        d_o = 100 * (1 - amean(n_, ocean, w) / amean(p_, ocean, w))
        print(f'products, {tag}: decline common {d_c:.2f} %, ocean {d_o:.2f} %, shift {d_o - d_c:+.3f} points; '
              f'Omega_arag ocean 1750 {amean(p_, ocean, w):.3f}, 2024 {amean(n_, ocean, w):.3f}')
    if '--no-models' in sys.argv:
        res_print(res)
        return

    # ESMs, each with its own ratios
    rows_m = {Y: [] for Y in (1850, 2024)}
    for model, member in cpx.find_runs(None):
        try:
            s_, mla, mlo, _, _ = tp.annual_states(model, member)
            si, po4 = co.load_nutrients(mla, mlo)
            mla2 = np.broadcast_to(mla[:, None], (mla.size, mlo.size))
            om = {Y: tp.omega(s_[Y], si, po4, mla2) for Y in (1850, 2024)}
        except Exception as e:
            print(f'{model:16s} failed: {type(e).__name__}: {e}', flush=True)
            continue
        both = np.isfinite(om[1850]) & np.isfinite(om[2024])
        for Y in (1850, 2024):
            f = np.where(ocean, on_grid(np.where(both, om[Y], np.nan), mla, mlo, lat, lon), np.nan)
            rows_m[Y].append(decompose(f, R, cmask, ocean, w))
    for Y in (1850, 2024):
        L = rows_m[Y]
        res[f'ESMs {Y} (mean of {len(L)})'] = ({k: tuple(np.nanmean([x[0][k][j] for x in L]) for j in range(3)) for k in R},
                                                float(np.mean([x[1] for x in L])))
        res[f'ESMs {Y} s.d.'] = ({k: tuple(np.nanstd([x[0][k][j] for x in L], ddof=1) for j in range(3)) for k in R},
                                  float(np.std([x[1] for x in L], ddof=1)))

    res_print(res)


def res_print(res):
    print('\nper region: area share of the ocean (%) | Omega relative to the common-area mean (%) | contribution to the '
          'gap (%)')
    for lab, (rows, gap) in res.items():
        print(f'  {lab:32s} gap {gap:+6.2f} %: ' + ' | '.join(f'{k} {v[0]:4.1f} {v[1]:+6.1f} {v[2]:+6.2f}' for k, v in rows.items()))


if __name__ == '__main__':
    main()
