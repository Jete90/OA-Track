#!/usr/bin/env python
"""
test_access_cells.py  (which cells outside the common area make ACCESS-ESM1-5 differ?)
====================
test_access_map.py found for ACCESS-ESM1-5, outside the common area between 50 S and 60 N (7.8 % of the ocean), a
level of Omega_arag in 1850 23 % above its common-area mean (the other ESMs -1 %) and a decline of 14.8 % (others
17.4 %). Its maps are capped at 4.5, so extreme values would not show. Here, with the same states (test_polar_shift_models,
nearest neighbour onto the product grid):
  the distribution of Omega_arag 1850 of the model and of the others in that region (percentiles, cells above 6 and 10)
  named marginal seas (boxes): area share, level 1850 of the model and of the others, decline, and their contribution
    to the whole-ocean level gap and to the shift of the decline (whole ocean minus common area) of the model
  the 25 cells with the largest contribution to the model's level excess (area x (Omega - mean of the others)): position,
    Omega 1850 and 2024, decline, and the model's temperature, salinity, TA and DIC of 1850
  the model's shift and level gap without the cells above a threshold, and without the boxes
Only for us, no figures.
Usage (code/):  OA_RUN=u74 python -u test_access_cells.py [--model ACCESS-ESM1-5]
"""
import argparse
import pickle
import warnings

import numpy as np

import oa_config
import calc_omega_arag_obs as co
import calc_pi_xco2 as cpx
import regional_common as rc
import test_polar_shift_models as tp
from test_access_map import on_grid, amean

Y0, Y1 = 1850, 2024
BOXES = {                                   # lon0, lon1 (-180..180), lat0, lat1
    'Red Sea': (32, 44, 12, 30), 'Persian Gulf': (47, 57, 23, 31), 'Mediterranean': (-6, 37, 30, 46),
    'Baltic': (9, 31, 53, 66), 'Hudson Bay': (-95, -75, 51, 66), 'Gulf of St Lawrence': (-70, -56, 44, 52),
    'Okhotsk and Bering shelf': (135, 180, 44, 66), 'Indonesian seas': (95, 140, -11, 8)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='ACCESS-ESM1-5')
    a = ap.parse_args()
    warnings.filterwarnings('ignore', message='Mean of empty slice')
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    r = pk['results_common']['UExP']
    lat, lon, cmask = np.asarray(r['lat'], 'f8'), np.asarray(r['lon'], 'f8'), np.asarray(r['mask'], bool)
    ocean = rc.ocean_on(lat, lon).astype(bool)
    w = oa_config.area_weights(lat, lon)
    la2 = np.broadcast_to(lat[:, None], cmask.shape)
    lo2 = np.broadcast_to(((lon + 180) % 360 - 180)[None, :], cmask.shape)
    rest = ocean & ~cmask & (la2 >= -50) & (la2 <= 60)
    out_ = ocean & ~cmask

    O0, O1, S0 = {}, {}, None
    for model, member in cpx.find_runs(None):
        try:
            st, mla, mlo, _, _ = tp.annual_states(model, member)
            si, po4 = co.load_nutrients(mla, mlo)
            mla2 = np.broadcast_to(mla[:, None], (mla.size, mlo.size))
            om = {y: tp.omega(st[y], si, po4, mla2) for y in (Y0, Y1)}
        except Exception as e:
            print(f'{model:16s} failed: {type(e).__name__}: {e}', flush=True)
            continue
        both = np.isfinite(om[Y0]) & np.isfinite(om[Y1])
        O0[model], O1[model] = (np.where(ocean, on_grid(np.where(both, om[y], np.nan), mla, mlo, lat, lon), np.nan)
                                for y in (Y0, Y1))
        if model == a.model:
            S0 = {v: on_grid(st[Y0][v], mla, mlo, lat, lon) for v in tp.VARS}
    assert a.model in O0, f'{a.model} not among the models: {list(O0)}'
    others = [m for m in O0 if m != a.model]
    o0, o1 = O0[a.model], O1[a.model]
    m0 = np.nanmean([O0[m] for m in others], axis=0)
    m1 = np.nanmean([O1[m] for m in others], axis=0)
    print(f'{a.model} against the mean of {len(others)} others; region: outside the common area, 50 S to 60 N '
          f'({100 * w[rest].sum() / w[ocean].sum():.1f} % of the ocean)')

    q = [1, 10, 50, 90, 99, 100]
    for lab, f in ((a.model, o0), ('others (each cell: mean)', m0)):
        v = f[rest & np.isfinite(f)]
        print(f'  Omega_arag {Y0} {lab:26s} percentiles {q}: ' + ' '.join(f'{x:6.2f}' for x in np.percentile(v, q))
              + f'; cells > 6: {int((v > 6).sum())}, > 10: {int((v > 10).sum())} of {v.size}')

    def shift_gap(excl):
        """the model's shift of the decline (ocean minus common, points) and level gap (ocean / common - 1, %)"""
        oc = ocean & ~excl & np.isfinite(o0) & np.isfinite(o1)
        cm = cmask & oc
        d_o = 100 * (1 - amean(o1, oc, w) / amean(o0, oc, w))
        d_c = 100 * (1 - amean(o1, cm, w) / amean(o0, cm, w))
        return d_o - d_c, 100 * (amean(o0, oc, w) / amean(o0, cm, w) - 1)
    s_all, g_all = shift_gap(np.zeros_like(ocean))
    print(f'\n  {a.model}: shift {s_all:+.2f} points, level gap {g_all:+.1f} % (all cells)')

    print(f'\n  marginal seas outside the common area: share of the ocean | Omega {Y0} model, others | decline model, '
          f'others | shift and gap of the model without the box')
    allbox = np.zeros_like(ocean)
    for name, (x0, x1, y0, y1) in BOXES.items():
        b = out_ & (lo2 >= x0) & (lo2 <= x1) & (la2 >= y0) & (la2 <= y1)
        allbox |= b
        if not b.any():
            continue
        sh = 100 * w[b].sum() / w[ocean].sum()
        dm = 100 * (1 - amean(o1, b, w) / amean(o0, b, w))
        do = 100 * (1 - amean(m1, b, w) / amean(m0, b, w))
        s_, g_ = shift_gap(b)
        print(f'    {name:26s} {sh:5.2f}% | {amean(o0, b, w):5.2f} {amean(m0, b, w):5.2f} | {dm:5.1f} {do:5.1f} | '
              f'{s_:+.2f} {g_:+.1f} %')
    s_, g_ = shift_gap(allbox)
    print(f'    {"without all boxes":26s} shift {s_:+.2f} points, gap {g_:+.1f} %')
    for thr in (10.0, 6.0, 5.0):
        s_, g_ = shift_gap(out_ & (o0 > thr))
        print(f'    without the cells outside the common area with Omega {Y0} > {thr:g}: shift {s_:+.2f}, gap {g_:+.1f} %')

    ex = np.where(rest & np.isfinite(o0) & np.isfinite(m0), w * (o0 - m0), np.nan)
    tot = np.nansum(np.where(rest, w * (o0 - m0), 0.0))
    k = np.argsort(np.where(np.isfinite(ex), -ex, np.inf), axis=None)[:25]
    print(f'\n  the 25 cells with the largest level excess (share of the total excess of the region; the total is '
          f'{tot / w[rest].sum():+.3f} Omega on average)')
    print(f'    {"lat":>6s} {"lon":>7s} {"share":>6s} {"Om1850":>7s} {"others":>7s} {"Om2024":>7s} {"decl%":>6s} '
          f'{"T":>6s} {"S":>6s} {"TA":>7s} {"DIC":>7s}')
    for kk in k:
        j, i = np.unravel_index(kk, ocean.shape)
        print(f'    {lat[j]:6.1f} {lo2[j, i]:7.1f} {100 * ex[j, i] / tot:5.1f}% {o0[j, i]:7.2f} {m0[j, i]:7.2f} '
              f'{o1[j, i]:7.2f} {100 * (1 - o1[j, i] / o0[j, i]):6.1f} {S0["thetao"][j, i]:6.2f} {S0["so"][j, i]:6.2f} '
              f'{S0["talk"][j, i]:7.1f} {S0["dissic"][j, i]:7.1f}', flush=True)


if __name__ == '__main__':
    main()
