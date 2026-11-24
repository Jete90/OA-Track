#!/usr/bin/env python
"""
feedback_pattern_members.py  (17 November 2026)
===========================
The alkalinity-rise pattern of every single member of GFDL-ESM2M-COBALTv2 (coupled, five members), computed exactly as
the ensemble pattern of feedback_pattern.py (drift-corrected nTA onto the product grid, slope of each cell on the
common-area mean over 1860-2014, smoothed over 11 x 11 cells, common-area mean 1), but with each member's own field and
own common-area series. Shows the internal variability of the pattern, above all in the Arctic, for Supplementary
Fig. 23. Writes a separate file and does not touch feedback_pattern.npz, which the chain uses.
Output: data/<run>/structural/feedback_pattern_esm2m_members.npz (pattern_esm2m_<member>, lat, lon, members)
Usage (code/):  OA_RUN=u74 python -u feedback_pattern_members.py
"""
import datetime
import pickle
import warnings

import numpy as np

import feedback_pattern as fp
import oa_config
import regional_common as rc


def main():
    warnings.filterwarnings('ignore', message='Mean of empty slice')
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    r0 = pk['results_common']['UExP']
    lat, lon = np.asarray(r0['lat'], 'f8'), np.asarray(r0['lon'], 'f8')
    cmask = np.asarray(r0['mask'], bool)
    ocean = rc.ocean_on(lat, lon).astype(bool)
    w = oa_config.area_weights(lat, lon)
    ref = np.load(fp.DEST / 'feedback_pattern.npz')
    assert np.allclose(ref['lat'], lat) and np.allclose(ref['lon'], lon), 'feedback_pattern.npz on another grid'
    *ctl, _ = fp.esm2m_control(lat, lon)
    yc, dta, dso = ctl
    sel = yc <= fp.PATTERN_LAST
    out = dict(lat=lat, lon=lon, members=np.array(fp.ESM2M_MEMBERS), smooth=fp.SMOOTH, created=str(datetime.datetime.now()))
    fmt = lambda b: ', '.join(f'{x_[0]}..{x_[1]}: {v_:.2f}' for x_, v_ in zip(fp.BANDS, b))
    print(f'GFDL-ESM2M-COBALTv2 coupled, {fp.PERIOD_ESM2M[0]}-{fp.PATTERN_LAST}, each member alone (common-area mean 1; '
          'negative share; by latitude; correlation with the ensemble pattern over the ocean and the common area)')
    for mem in fp.ESM2M_MEMBERS:
        folder = fp.ESM2M_DIR / 'coupled' / fp.ESM2M_SCEN / mem / 'OA_variables'
        if not folder.is_dir():
            print(f'  {mem}: no folder {folder}')
            continue
        y, ta, so, la, lo = fp.esm2m_ta_s(folder, *fp.PERIOD_ESM2M)
        assert np.array_equal(y, yc), f'{mem}: years differ from the control'
        a = fp.onto(lat, lon, la, lo, fp.nta(ta - dta, so - dso))[sel]
        x = fp.area_series(a, cmask, w)
        p_, neg_ = fp.finish(fp.slope_on(x, a), ocean, cmask, w, lat, fp.SMOOTH)
        out[f'pattern_esm2m_{mem}'] = p_
        print(f'  {mem}: common-area nTA change 1860-79 -> 1995-2014 {x[y[sel] >= 1995].mean() - x[y[sel] <= 1879].mean():+.2f} '
              f'umol/kg; {100 * neg_:4.1f} % negative; {fmt(fp.bands(p_, ocean, w, lat))}; r = '
              f'{fp.corr(ref["pattern_esm2m"], p_, ocean, w):+.2f}, {fp.corr(ref["pattern_esm2m"], p_, cmask, w):+.2f}', flush=True)
    np.savez(fp.DEST / 'feedback_pattern_esm2m_members.npz', **out)
    print('written', fp.DEST / 'feedback_pattern_esm2m_members.npz')


if __name__ == '__main__':
    main()
