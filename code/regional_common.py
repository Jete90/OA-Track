"""
regional_common.py
==================
Shared data and calculations for the figures and tables of the notebook.

For each product, on the common mask and the product's grid (from data/constraint_results.pkl,
written by run_constraint.py with STORE_FIELDS = True):

    pi_scaled   multi-model mean of the bias-adjusted 1750 fields x global constraint factor
    pi_local    local constraint per cell (cell dSST of the three SST products and cell CUC),
                model mean where the local constraint does not reduce the uncertainty
    pi_full     local constraint everywhere, observed predictors taken as they are, no threshold
                (sig_full may exceed the model spread where the observations are too uncertain)
    pi_bayes    local constraint with the observational uncertainty in the predictors (bayes_constraint):
                the observed dSST/CUC are weighted against the model distribution (k_dT, k_CUC = weights,
                1 = observation taken as is, 0 = model mean), no threshold, sigma <= model spread
    sig_*       1 sigma of the above (models; local: fit + observations)
    om          observed annual Omega_arag (year, lat, lon); now = quadratic fit in EVAL_YEAR per cell
    constrained 1 where the cell-by-cell constraint is done, 2 where it cannot be (e.g. no observed dSST) and the
                constrained fields take the multi-model mean instead
    chg_*       relative change now / pi - 1 in %

All arrays are returned on longitudes -180..180 (sorted), ready for plotting.
"""
import os
import oa_config   # run: data folder, mocsy constants, GFDL density (OA_RUN)
import pickle

import numpy as np
import netCDF4

# which run all figures and tables use: set os.environ['OA_RUN'] before importing (or reloading) this module.
#   'u74' = standard (total boron Uppstroem 1974, data), 'l10' = total boron Lee et al. 2010,
#   'm10' = K1, K2 of Millero 2010 with Uppstroem boron (data_k1k2_m10). Data are read from and figures written to
#   the run's folder.
BASE = str(oa_config.BASE)
RUNS = {k: f"{BASE}/{v['data']}" for k, v in oa_config.RUNS.items()}   # oa_config.py
RUN = os.environ.get('OA_RUN', 'u74')
ROOT = RUNS[RUN]
OBS = {'OceanSODA': f'{ROOT}/obs/omega_arag_Oyr_OceanSODA-ETHZ-v2025.nc',
       'UExP': f'{ROOT}/obs/omega_arag_Oyr_UExP-FNN-U-v2026-1.nc',
       'JMA': f'{ROOT}/obs/omega_arag_Oyr_JMA-MLR-v2025.nc',
       'CMEMS': f'{ROOT}/obs/omega_arag_Oyr_CMEMS-LSCE-FFNN.nc'}
FULL = {'OceanSODA': 'OceanSODA-ETHZ v2025', 'UExP': 'UExP-FNN-U v2026-1',
        'JMA': 'JMA-MLR v2025', 'CMEMS': 'CMEMS-LSCE-FFNN'}
COLORS = {'OceanSODA': 'C0', 'UExP': 'C1', 'JMA': 'C2', 'CMEMS': 'C4'}
EVAL_YEAR = 2024
LAST_YEAR = 2024
CUC_SIG = 0.0           # global observed CUC: counted with the mapping term (cuc_mapping_uncertainty.py), not here


def cuc_sigma(ref, which='results_common'):
    """the mapping uncertainty of the observed CUC of a product (cuc_mapping_uncertainty.py): (global value, field on
    the grid of constraint_results.pkl); which: 'results_common' or 'results' (own area)"""
    import os
    p = f'{ROOT}/product_uncertainty/cuc_mapping_{ref}.npz'
    if not os.path.exists(p):
        raise FileNotFoundError(f'{p} missing: run cuc_mapping_uncertainty.py (OA_RUN={RUN}) first')
    a = 'common' if which == 'results_common' else 'own'
    z = np.load(p)
    f = np.asarray(z[f'{a}_field_sigma_cuc'], 'f8')
    w = oa_config.area_weights(z[f'{a}_field_lat'], z[f'{a}_field_lon']) * np.isfinite(f)
    fill = float(np.nansum(np.nan_to_num(f) * w) / w.sum())            # cells without a value: the area mean
    return float(z[f'{a}_sigma_cuc_C2']), np.where(np.isfinite(f), f, fill)
LAB = 'dT + CUC'

# file names of the display items in the notebook (its export cell gives them the names of the manuscript)
FIGDIR = f'{ROOT}/figures'
FIGNAMES = {
    'best': 'Fig1_best_estimate',
    'constraint': 'Fig2_constraint',
    'crossing': 'Fig3_crossing',
    'coverage': 'Fig4_coverage',
    'findlay': 'FigS13_findlay',
    'states_maps': 'FigS14_states_1750_2024',
    'table': 'TableS1_summary',
    'maps_products': 'FigS1_products_2024',
    'global_products': 'FigS2_global_per_product',
    'constraint_product': 'FigS3_constraint',
    'dsst_bias': 'FigS4_dSST_bias',
    'correction_terms': 'FigS5_correction_terms',
    'regional_local': 'FigS6_local_constraint',
    'crossing_co2': 'FigS8_crossing_co2',
    'regional_decline': 'FigS9_regional_decline_per_product',
    'coverage_map': 'FigS10_coverage_map',
    'steps_table': 'TableS2_steps',
}


def figpath(key, suffix='', ext='png'):
    import os
    os.makedirs(FIGDIR, exist_ok=True)
    return f'{FIGDIR}/{FIGNAMES[key]}' + (f'_{suffix}' if suffix else '') + f'.{ext}'


def savefig(fig, key, suffix=''):
    """save as png (200 dpi) and pdf under data/figures with the display-item name (e.g. Fig1_global.png)"""
    for ext in ('png', 'pdf'):
        fig.savefig(figpath(key, suffix, ext), dpi=200)
    print('written:', figpath(key, suffix, '(png, pdf)'))


def fit_now(r, year=EVAL_YEAR):
    """quadratic fit to the annual global means r['ogm'] (common area), evaluated in year (same as the table and,
    per cell, as d['now'])"""
    yo, gm = np.asarray(r['year'], 'f8'), np.asarray(r['ogm'], 'f8')
    ok = np.isfinite(gm) & (yo <= LAST_YEAR)
    return np.polyval(np.polyfit(yo[ok] - 2000.0, gm[ok], 2), year - 2000.0)


def anchor_projection(p, r, x_now, keys=('fixed', 'scaled')):
    """shift a global projection (dict with 'x' and curves in keys, from fig_crossing_projection.pkl) so that it passes
    through the fitted EVAL_YEAR value at x_now (the observed CO2 of EVAL_YEAR) and starts there"""
    now = fit_now(r)
    x = np.asarray(p['x'])
    keep = x > x_now
    out = dict(p, x=np.r_[x_now, x[keep]], x_state=x_now, now=now)
    for k in keys:
        c = np.asarray(p[k])
        c = c + (now - np.interp(x_now, x, c))
        out[k] = np.r_[now, c[keep]]
    return out


def to_180(lon, a):
    l = np.where(lon > 180, lon - 360, lon)
    o = np.argsort(l)
    return l[o], a[..., o]


def local_constraint(Y, DT, CU, dTo, dTs, CUo, ok, return_beta=False, CUs=CUC_SIG):
    """per-cell regression of Y (model, cell) on DT, CU; constrained value and total sigma
    (and the coefficients [intercept, dT, CUC] per cell if return_beta); CUs: 1 sigma of the observed local CUC
    (per cell: the local mapping uncertainty, cuc_sigma)"""
    n = Y.shape[0]
    X = np.stack([np.ones_like(DT), DT, CU], -1).transpose(1, 0, 2)
    yy = Y.T
    inv = np.linalg.inv(np.einsum('cmi,cmj->cij', X, X))
    beta = np.einsum('cij,cmj,cm->ci', inv, X, yy)
    e = yy - np.einsum('cmi,ci->cm', X, beta)
    s2 = (e ** 2).sum(1) / (n - 3)
    x0 = np.stack([np.ones_like(dTo), dTo, CUo], -1)
    yc = np.einsum('ci,ci->c', beta, x0)
    st = np.sqrt(s2 * (1 + np.einsum('ci,cij,cj->c', x0, inv, x0))
                 + (beta[:, 1] * dTs) ** 2 + (beta[:, 2] * CUs) ** 2)
    return (yc, st, beta) if return_beta else (yc, st)


def bayes_constraint(Y, DT, CU, dTo, dTs, CUo, CUs=CUC_SIG):
    """per-cell constraint with observational uncertainty in the predictors (errors in variables):
    the true predictors x = (dT, CUC) are estimated from the model distribution (mean xm, covariance Sm)
    and the observations (xo, covariance So = diag(dTs^2, CUs^2), CUs the local mapping uncertainty of CUC):
        x_best = xm + K (xo - xm),  K = Sm (Sm + So)^-1,  P = (I - K) Sm  (uncertainty of x_best)
    and put into the same regression: y = a + b x_best, var = s2 (1 + leverage) + b' P b.
    K -> I for precise observations (= local_constraint without the obs term), K -> 0 for useless ones
    (= model mean, sigma -> model spread). Returns y, sigma, K[dT,dT], K[CUC,CUC] per cell."""
    n = Y.shape[0]
    X = np.stack([np.ones_like(DT), DT, CU], -1).transpose(1, 0, 2)          # cell, model, 3
    yy = Y.T
    inv = np.linalg.inv(np.einsum('cmi,cmj->cij', X, X))
    beta = np.einsum('cij,cmj,cm->ci', inv, X, yy)
    e = yy - np.einsum('cmi,ci->cm', X, beta)
    s2 = (e ** 2).sum(1) / (n - 3)
    xs = X[:, :, 1:]                                                         # cell, model, 2
    xm = xs.mean(1)
    a_ = xs - xm[:, None, :]
    Sm = np.einsum('cmi,cmj->cij', a_, a_) / (n - 1)
    So = np.zeros_like(Sm)
    So[:, 0, 0] = dTs ** 2
    So[:, 1, 1] = np.broadcast_to(np.asarray(CUs, 'f8'), So[:, 1, 1].shape) ** 2
    K = np.einsum('cij,cjk->cik', Sm, np.linalg.inv(Sm + So))
    xo = np.stack([dTo, CUo], -1)
    xb = xm + np.einsum('cij,cj->ci', K, xo - xm)
    P = Sm - np.einsum('cij,cjk->cik', K, Sm)
    x0 = np.concatenate([np.ones((xb.shape[0], 1)), xb], -1)
    yc = np.einsum('ci,ci->c', beta, x0)
    b = beta[:, 1:]
    st = np.sqrt(s2 * (1 + np.einsum('ci,cij,cj->c', x0, inv, x0)) + np.einsum('ci,cij,cj->c', b, P, b))
    return yc, st, K[:, 0, 0], K[:, 1, 1]


def load(results_file=f'{ROOT}/constraint_results.pkl', which='results_common', refs=None):
    """which: 'results_common' (common mask) or 'results' (each product on its own mask)"""
    with open(results_file, 'rb') as f:
        RC = pickle.load(f)[which]
    if refs is not None:
        RC = {k: v for k, v in RC.items() if k in refs}
    out = {}
    for ref, r in RC.items():
        F = r['fields']
        lat, lon0 = np.asarray(r['lat'], 'f8'), np.asarray(r['lon'], 'f8')
        mask = np.asarray(r['mask']).astype(bool)
        Y, DT, CU = (F[k].astype('f8') for k in ('y1750', 'dT', 'CUC'))
        n = Y.shape[0]
        dobs = np.array([F['dsst_obs'][k] for k in F['dsst_obs']], 'f8')
        dTo, dTs = dobs.mean(0), dobs.std(0, ddof=1)
        CUo = F['CUC_obs'].astype('f8')
        CUs_f = cuc_sigma(ref, which)[1]                  # local mapping uncertainty of the observed CUC
        with netCDF4.Dataset(OBS[ref]) as f:
            yo = np.asarray(f['year'][:])
            om = np.ma.filled(f['omega_arag'][:].astype('f8'), np.nan)
        keep = yo <= LAST_YEAR
        yo, om = yo[keep], om[keep]
        # base: cells with all model 1750 values and observations (multi-model mean possible);
        # ok: base and all predictors (models and observed dSST, CUC), i.e. where the cell-by-cell constraint can be done.
        # In base but not ok (mostly missing observed dSST), the constrained fields take the multi-model mean (flag 2).
        base = mask & np.all(np.isfinite(Y), 0) & np.all(np.isfinite(om), 0)
        ok = (base & np.all(np.isfinite(DT), 0) & np.all(np.isfinite(CU), 0) & np.isfinite(dTo) & np.isfinite(CUo))
        idx = np.where(ok.ravel())[0]
        idb = np.where(base.ravel())[0]
        sel = lambda a: a.reshape(n, -1)[:, idx] if a.ndim == 3 else a.ravel()[idx]
        y = sel(Y)
        yb_all = Y.reshape(n, -1)[:, idb]
        ymean_b, ysd_b = yb_all.mean(0), yb_all.std(0, ddof=1)       # multi-model mean on base
        ymean, ysd = y.mean(0), y.std(0, ddof=1)
        yc_, sc_, rng_ = r['constrained'][LAB]
        scale = yc_ / r['unconstrained'][0]
        sig_rel = sc_ / yc_                        # relative 1 sigma of the constrained global value
        rng_rel = (min(rng_) / yc_, max(rng_) / yc_)  # 284 and 275 ppm relative to 1750
        yc, st = local_constraint(y, sel(DT), sel(CU), sel(dTo), sel(dTs), sel(CUo), ok, CUs=sel(CUs_f))
        use = st < ysd
        yb, sb, k_dT, k_CU = bayes_constraint(y, sel(DT), sel(CU), sel(dTo), sel(dTs), sel(CUo), CUs=sel(CUs_f))

        def to_map(v, ix=idx):
            m_ = np.full(mask.size, np.nan)
            m_[ix] = v
            return m_.reshape(mask.shape)

        mmm, mmsd = to_map(ymean_b, idb), to_map(ysd_b, idb)
        fill = base & ~ok
        cmap_ = lambda v, alt: np.where(fill, alt, to_map(v))           # constrained where ok, else alt
        now = np.polyval(np.polyfit(yo - 2000.0, om.reshape(yo.size, -1)[:, idb], 2), EVAL_YEAR - 2000.0)
        maps = dict(pi_mean=mmm, pi_scaled=mmm * scale, sig_model=mmsd, now=to_map(now, idb),
                    pi_local=cmap_(np.where(use, yc, ymean), mmm), sig_local=cmap_(np.where(use, st, ysd), mmsd),
                    use_local=cmap_(use.astype(float), 0.0),
                    pi_bayes=cmap_(yb, mmm), sig_bayes=cmap_(sb, mmsd), k_dT=cmap_(k_dT, 0.0), k_CUC=cmap_(k_CU, 0.0),
                    pi_full=cmap_(yc, mmm), sig_full=cmap_(st, mmsd),
                    constrained=np.where(ok, 1.0, np.where(fill, 2.0, np.nan)))   # 1 constrained, 2 multi-model mean
        d = maps
        now = d['now']
        d['chg_scaled'] = 100 * (now / d['pi_scaled'] - 1)
        d['chg_local'] = 100 * (now / d['pi_local'] - 1)
        d['chg_bayes'] = 100 * (now / d['pi_bayes'] - 1)
        d['chg_full'] = 100 * (now / d['pi_full'] - 1)
        maps = d
        lon, _ = to_180(lon0, mask)
        maps = {k: to_180(lon0, v)[1] for k, v in maps.items()}
        _, om180 = to_180(lon0, om)
        _, ok180 = to_180(lon0, base)                    # analysis area (constrained or multi-model mean)
        out[ref] = dict(lat=lat, lon=lon, year=yo, om=om180, ok=ok180, scale=scale, sig_rel=sig_rel, rng_rel=rng_rel,
                        area=oa_config.area_weights(lat, lon),
                        anydata=np.isfinite(om180).any(0), **maps)
    # land for the maps: cells where no product has data in any year (as in Figure 1)
    land = ~np.any([d_['anydata'] for d_ in out.values()], axis=0)
    for d_ in out.values():
        d_['land'] = land
    return out


def alkalinity_change(ref, lat, lon):
    """the change of surface Omega_arag by the alkalinity adjustment of the best estimate (make_product_alkalinity.py:
    carbonate_system_<ref>_alkalinity.nc) on the grid of load() (lat, lon in -180..180), as Fig. 4 uses it.
    Returns (change of the constrained 1750 state at the AR6 CO2, a function year -> change of that year's
    annual value (from EVAL_YEAR on that of the EVAL_YEAR trend value), change of the EVAL_YEAR trend value)"""
    with netCDF4.Dataset(f'{ROOT}/product/carbonate_system_{ref}_alkalinity.nc') as f:
        flat, flon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        fy = np.asarray(f['year'][:]).astype(int)
        names = [str(s) for s in np.asarray(f['state'][:]).ravel()]
        g = lambda v, i=Ellipsis: np.ma.filled(f[v][i].astype('f8'), np.nan)
        a50 = g('omega_arag_state_alk', names.index('1750_AR6'))
        a_all = g('omega_arag_alk')                                  # (year, lat, lon)
        a24 = g(f'omega_arag_{EVAL_YEAR}_trend_alk')
    assert np.allclose(flat, lat), f'{ref}: alkalinity file on another latitude grid'
    l180, a50 = to_180(flon, a50)
    assert np.allclose(l180, lon), f'{ref}: alkalinity file on another longitude grid'
    a_all, a24 = to_180(flon, a_all)[1], to_180(flon, a24)[1]

    def year(y):
        return a24 if y >= EVAL_YEAR else a_all[int(np.where(fy == y)[0][0])]
    return a50, year, a24


def area_beyond(dd, key, thr=-20.0):
    """share (%) of the valid area with chg < thr"""
    a, c = dd['area'], dd[key]
    v = np.isfinite(c)
    return 100 * a[v & (c < thr)].sum() / a[v].sum()


def area_beyond_series(dd, pi_key, thr=-20.0, factor=1.0):
    """share (%) of the valid area beyond thr in every year (annual values, not the fit);
    factor scales the preindustrial field (uncertainty and definition range)"""
    a, pi = dd['area'], dd[pi_key] * factor
    v = np.isfinite(pi) & dd['ok']
    return np.array([100 * a[v & (100 * (o / pi - 1) < thr)].sum() / a[v].sum() for o in dd['om']])


# ---------------------------------------------------------------- maps: Robinson projection without cartopy
# All maps of the paper: Robinson, centred on MAP_CENTER (200 E: Pacific in the centre, edge through South Africa at
# 20 E), land and outline as in Fig. 2. Fields are given on their 1-D lon (any convention) and lat grids; the functions
# reorder the columns, project the cell edges and draw the outline on the first use of an axes.
MAP_CENTER = 200.0
_RLAT = np.arange(0, 91, 5.0)
_RX = np.array([1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600, 0.9427, 0.9216, 0.8962, 0.8679, 0.8350,
                0.7986, 0.7597, 0.7186, 0.6732, 0.6213, 0.5722, 0.5322])
_RY = np.array([0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720, 0.4340, 0.4958, 0.5571, 0.6176, 0.6769,
                0.7346, 0.7903, 0.8435, 0.8936, 0.9394, 0.9761, 1.0000])


def _rel_lon(lo):
    return (np.asarray(lo, 'f8') - MAP_CENTER + 180.0) % 360.0 - 180.0


def _robinson(lo_rel, la):
    lo_rel, la = np.asarray(lo_rel, 'f8'), np.asarray(la, 'f8')
    a = np.clip(np.abs(la), 0, 90)
    return 0.8487 * np.interp(a, _RLAT, _RX) * np.deg2rad(lo_rel), 1.3523 * np.interp(a, _RLAT, _RY) * np.sign(la)


def map_xy(lo, la):
    """map coordinates of points (degrees east, degrees north), e.g. for ax.text(*map_xy(lon, lat), ...)"""
    return _robinson(_rel_lon(lo), la)


def _edges(c, lo_, hi_):
    c = np.asarray(c, 'f8')
    mid = 0.5 * (c[1:] + c[:-1])
    return np.clip(np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]]), lo_, hi_)


def _grid(lon, lat):
    rel = _rel_lon(np.asarray(lon, 'f8').ravel())
    o = np.argsort(rel, kind='stable')
    lon_m = rel[o]
    Xe, Ye = _robinson(*np.meshgrid(_edges(lon_m, -180, 180), _edges(lat, -90, 90)))
    Xc, Yc = _robinson(*np.meshgrid(lon_m, np.asarray(lat, 'f8').ravel()))
    return o, Xe, Ye, Xc, Yc


def map_frame(ax, lw=0.6):
    """equal aspect, no axes, Robinson outline (drawn once per axes)"""
    if getattr(ax, '_oa_frame', False):
        return
    ax.set_aspect('equal')
    ax.axis('off')
    b = np.linspace(-90, 90, 181)
    out = np.concatenate([np.column_stack(_robinson(np.full(b.size, -180.0), b)),
                          np.column_stack(_robinson(np.full(b.size, 180.0), b[::-1]))])
    out = np.vstack([out, out[:1]])
    ax.plot(*out.T, color='k', lw=lw, zorder=6, clip_on=False)
    ax.margins(0.01)
    ax._oa_frame = True


def map_mesh(ax, lon, lat, field, shading=None, transform=None, **kw):
    """pcolormesh of field(lat, lon) on the Robinson map (shading/transform accepted and ignored)"""
    map_frame(ax)
    o, Xe, Ye, _, _ = _grid(lon, lat)
    return ax.pcolormesh(Xe, Ye, np.asarray(field)[:, o], shading='flat', **kw)


def map_contour(ax, lon, lat, field, transform=None, **kw):
    map_frame(ax)
    o, _, _, Xc, Yc = _grid(lon, lat)
    return ax.contour(Xc, Yc, np.asarray(field, 'f8')[:, o], **kw)


def map_contourf(ax, lon, lat, field, transform=None, **kw):
    map_frame(ax)
    o, _, _, Xc, Yc = _grid(lon, lat)
    return ax.contourf(Xc, Yc, np.asarray(field, 'f8')[:, o], **kw)


# ---------------------------------------------------------------- the ocean area: one definition for all percentages
# The denominator of every "% of the ocean" (common area, own areas, coverage maps, tables): the cells in which at least
# one of the four products has a value in at least one month of any year (calc_omega_arag_anymonth.py; the 12-month
# annual files where an any-month file is missing), on the products' 1 deg grid, without the inland seas (Black Sea
# with Azov and Marmara, Caspian Sea), which count as land.
ANYMONTH = {'OceanSODA': 'OceanSODA-ETHZ-v2025', 'UExP': 'UExP-FNN-U-v2026-1', 'JMA': 'JMA-MLR-v2025',
            'CMEMS': 'CMEMS-LSCE-FFNN'}
INLAND_BOXES = [(27.0, 42.0, 40.5, 47.5), (46.5, 55.0, 36.5, 47.5)]       # lon0, lon1, lat0, lat1 (-180..180)
_OCEAN = None


def ocean_mask():
    """(lat ascending, lon -180..180, ocean) on the products' grid, see above"""
    global _OCEAN
    if _OCEAN is None:
        lat0 = lon0 = ocean = None
        for q in ANYMONTH:
            p = f'{ROOT}/obs/omega_arag_anymonth_Oyr_{ANYMONTH[q]}.nc'
            if not os.path.exists(p):
                p = OBS[q]
            with netCDF4.Dataset(p) as f:
                lat = np.asarray(f['lat'][:], 'f8')
                lon, a = to_180(np.asarray(f['lon'][:], 'f8'),
                                np.isfinite(np.ma.filled(f['omega_arag'][:].astype('f8'), np.nan)).any(0))
            if lat[0] > lat[-1]:
                lat, a = lat[::-1], a[::-1]
            if ocean is None:
                lat0, lon0, ocean = lat, lon, a
            else:
                assert np.allclose(lat, lat0) and np.allclose(lon, lon0), f'{q}: grid differs'
                ocean = ocean | a
        inland = np.zeros(ocean.shape, bool)
        for lo0, lo1, la0, la1 in INLAND_BOXES:
            inland |= (lat0[:, None] >= la0) & (lat0[:, None] <= la1) & (lon0[None, :] >= lo0) & (lon0[None, :] <= lo1)
        _OCEAN = (lat0, lon0, ocean & ~inland)
    return _OCEAN


def ocean_on(lat, lon):
    """the ocean mask of ocean_mask() on another 1 deg grid of the same cells (any latitude order, any longitude
    convention)"""
    lat0, lon0, oc = ocean_mask()
    jy = np.abs(np.asarray(lat, 'f8')[:, None] - lat0[None, :]).argmin(1)
    l = np.where(np.asarray(lon, 'f8') > 180, np.asarray(lon, 'f8') - 360, np.asarray(lon, 'f8'))
    ix = np.abs(((l[:, None] - lon0[None, :]) + 180) % 360 - 180).argmin(1)
    return oc[np.ix_(jy, ix)]


def pct_ocean(mask, lat, lon):
    """area of mask (on the grid lat, lon) in % of the ocean (ocean_mask)"""
    w = oa_config.area_weights(lat, lon)
    oc = ocean_on(lat, lon)
    return 100 * (w * (np.asarray(mask, bool) & oc)).sum() / (w * oc).sum()
