# Notebook cell: regression constraint on preindustrial Omega_arag for each reference.
# Self-contained: reads only files (mechanism states, drift, xco2 outputs, obs Omega).
# Predictors per model: dT (historical window minus 1850-1900 SST, drift-corrected), CUC = <DIC/R>/pCO2_atm
# of the raw historical window state, dALK. Target: the preindustrial Omega_arag of every model at the CO2 of
# each state (calc_pi_xco2.py): time mean of the monthly Omega_arag of the drift-corrected, bias-adjusted
# piControl with DIC lowered cell by cell to that CO2 (1750: 278.3 ppm). Every state is constrained with its
# own model values; there are no global factors.
import sys, glob, os
import oa_config   # run: data folder, mocsy constants, GFDL density (OA_RUN)
import numpy as np
import netCDF4

CODE = str(oa_config.CODE)
if CODE not in sys.path:
    sys.path.append(CODE)
import calc_omega_arag_obs as co          # nutrients (and mocsy path)
import mocsy
MOCSY_OPTB, MOCSY_OPTK1K2 = oa_config.OPTB, oa_config.OPTK1K2   # total boron, K1/K2 of the run (oa_config.py)

ROOT = str(oa_config.DATA)
# win: reference window, the same as the bias adjustment and the mechanism states (oa_config.WINDOWS).
# The observed SST change from 1850-1900 to win is computed below on each constraint mask from
# three SST products (ERSSTv6, COBE-SST3, DCENT-I); dt_obs is their mean and DT_SIG their spread.
# pco2: mean atmospheric pCO2 of the window (oa_config.WINDOW_PCO2); it only scales CUC and does not change the result.
REFS = {
    'OceanSODA': dict(mech=f'{ROOT}/mechanism', adj=f'{ROOT}/omega_arag_biasadj',
                      obs=f'{ROOT}/obs/omega_arag_Oyr_OceanSODA-ETHZ-v2025.nc', win=oa_config.WINDOWS['OceanSODA'], pco2=oa_config.WINDOW_PCO2['OceanSODA']),
    'UExP': dict(mech=f'{ROOT}/mechanism_UExP', adj=f'{ROOT}/omega_arag_biasadj_UExP',
                 obs=f'{ROOT}/obs/omega_arag_Oyr_UExP-FNN-U-v2026-1.nc', win=oa_config.WINDOWS['UExP'], pco2=oa_config.WINDOW_PCO2['UExP']),
    'JMA': dict(mech=f'{ROOT}/mechanism_JMA', adj=f'{ROOT}/omega_arag_biasadj_JMA',
                obs=f'{ROOT}/obs/omega_arag_Oyr_JMA-MLR-v2025.nc', win=oa_config.WINDOWS['JMA'], pco2=oa_config.WINDOW_PCO2['JMA']),
    'CMEMS': dict(mech=f'{ROOT}/mechanism_CMEMS', adj=f'{ROOT}/omega_arag_biasadj_CMEMS',
                  obs=f'{ROOT}/obs/omega_arag_Oyr_CMEMS-LSCE-FFNN.nc', win=oa_config.WINDOWS['CMEMS'], pco2=oa_config.WINDOW_PCO2['CMEMS']),
}
# Jiang et al. (2023), decade 2010 (Jan 2005 to Dec 2014), only on request (OA_REFS=Jiang, run_constraint_jiang.py);
REFS_EXTRA = {
    'Jiang': dict(mech=f'{ROOT}/mechanism_Jiang', adj=f'{ROOT}/omega_arag_biasadj_Jiang',
                  obs=f'{ROOT}/obs/omega_arag_Oyr_Jiang2023-2010.nc', win=oa_config.WINDOWS['Jiang'], pco2=oa_config.WINDOW_PCO2['Jiang']),
}
REFS_ALL = {**REFS, **REFS_EXTRA}
if os.environ.get('OA_REFS'):
    REFS = {k: REFS_ALL[k] for k in os.environ['OA_REFS'].split(',')}
SST_BASE = (1850, 1900)
ERSST = str(oa_config.OBS2_DIR / 'ERSST/ersst_v6_data/ersst.v6.185001_202512_r.nc')
COBE3 = str(oa_config.OBS2_DIR / 'SST_comparison/COBEv3/cobe-sst3.monthly.{}.nc')
DCENTI = str(oa_config.OBS2_DIR / 'DCENT-I/DCENT_I_1.0.0.0_{}.nc')
VARS = ('thetao', 'so', 'dissic', 'talk')
CUC_SIG = 0.0                       # observed CUC: no set value; its uncertainty is the product's mapping error,
                                    # counted with the mapping term (cuc_mapping_uncertainty.py; notebook, preparation cell)
ALK_SIG_FRAC = 0.5
# preindustrial CO2 is a definition, not an uncertainty: the reference is 1750 (278.3 ppm) and
# the 275-284 ppm range is reported separately as a range, not folded into sigma
ALK_MODELS = ('GFDL-ESM4', 'GFDL-CM4', 'UKESM1-0-LL')
STORE_FIELDS = True                 # keep per-cell fields of models and observations in the results
MODEL_SST = f'{ROOT}/model_sst/sst_{{}}_{{}}.nc'     # from calc_model_sst.py: historical SST period means
DRIFT = True                        # subtract the piControl drift aligned at the branch point (calc_drift.py)
DRIFT_FILE = f'{ROOT}/drift/drift_{{}}_{{}}.nc'


if '_DRIFT_MEANS' not in globals():
    _DRIFT_MEANS = {}


def drift_means(model, member):
    """lat, lon of the drift file and, for 1850-1900 and every window (oa_config.PERIODS), the mean of the drift
    spline over the parallel piControl years minus the piControl mean, {period: {var: (lat, lon)}}, computed with
    drift_tools.Drift.window_mean, the same function as the data product (calc_product_states.py), so that a window
    needs no stored period in the drift file (Drift.window_mean checks itself against a stored one where it exists)"""
    key = (model, member)
    if key not in _DRIFT_MEANS:
        import drift_tools as dt
        d = dt.Drift(model, member)
        _DRIFT_MEANS[key] = (d.lat, d.lon, {p: {v: d.window_mean(v, p) for v in VARS} for p in oa_config.PERIODS})
        del d
    return _DRIFT_MEANS[key]
EVAL_YEAR = 2024


def read2d(path, names):
    with netCDF4.Dataset(path) as f:
        return {n: np.ma.filled(f[n][:].astype('f8'), np.nan) for n in names}


def fit(X, y, x0):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = res @ res / (X.shape[0] - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    r2 = 1 - res @ res / np.sum((y - y.mean()) ** 2)
    return beta, cov, s2, r2, x0 @ beta, np.sqrt(s2 + x0 @ cov @ x0)


# ---------------------------------------------------------------------------------------------
# SST products: period means for 1850-1900 and every window (read once, kept between reruns)
def _years(t):
    return np.array([d.year for d in netCDF4.num2date(t[:], t.units, getattr(t, 'calendar', 'standard'))])


def _nn(src, dst, periodic=False):
    if periodic:
        return np.abs(((dst[:, None] - src[None, :]) + 180) % 360 - 180).argmin(1)
    return np.abs(dst[:, None] - src[None, :]).argmin(1)


_periods = sorted({SST_BASE} | {c['win'] for c in REFS_ALL.values()})
if 'SSTP' not in globals() or set(SSTP['periods']) != set(_periods):
    SSTP = dict(periods=_periods)
    with netCDF4.Dataset(ERSST) as f:
        a = np.ma.filled(f['sst'][:].astype('f8'), np.nan)
        a = a[:, 0] if a.ndim == 4 else a
        yy = _years(f['time'])
        SSTP['ERSSTv6'] = (np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8'), {})
    a[np.abs(a) > 100] = np.nan
    for pp in _periods:
        SSTP['ERSSTv6'][2][pp] = a[(yy >= pp[0]) & (yy <= pp[1])].mean(axis=0)
    SSTP['COBE-SST3'] = (np.arange(-89.5, 90), np.arange(0.5, 360), {})
    for pp in _periods:
        acc = 0
        for y in range(pp[0], pp[1] + 1):
            with netCDF4.Dataset(COBE3.format(y)) as f:
                b = np.ma.filled(f['sst'][:].astype('f8'), np.nan).mean(axis=0)
            b[np.abs(b) > 100] = np.nan
            acc = acc + np.nanmean(b.reshape(180, 4, 360, 4), axis=(1, 3))
        SSTP['COBE-SST3'][2][pp] = acc / (pp[1] - pp[0] + 1)
        print(f'  COBE-SST3 {pp[0]}-{pp[1]} read', flush=True)
    with netCDF4.Dataset(DCENTI.format('mean_spread')) as f:
        a = np.ma.filled(f['ts_mean'][:].astype('f8'), np.nan)
        yy = _years(f['time'])
        SSTP['DCENT-I'] = (np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8'), {})
    for pp in _periods:
        SSTP['DCENT-I'][2][pp] = a[(yy >= pp[0]) & (yy <= pp[1])].mean(axis=0)
    with netCDF4.Dataset(DCENTI.format('diagnostics')) as f:
        SSTP['pure'] = np.nanmax(np.asarray(f['weight'][:], 'f8'), axis=0) < 0.01   # pure SST, no land or ice


def _bilinear(slat, slon, a, lat, lon):
    """Bilinear interpolation of a (slat, slon) field to the target centres; NaN if any of the four
    neighbours is NaN. For grids offset by half a cell (ERSST at whole degrees, products at x.5) this is
    the mean of the neighbouring cells, without the half-cell shift of a nearest-neighbour match."""
    o = np.argsort(slat)
    slat, a = slat[o], a[o]
    j1 = np.clip(np.searchsorted(slat, lat), 1, slat.size - 1)
    j0 = j1 - 1
    wy = np.clip((lat - slat[j0]) / (slat[j1] - slat[j0]), 0, 1)
    sl = np.mod(slon, 360)
    o = np.argsort(sl)
    sl, a = sl[o], a[:, o]
    tl = np.mod(lon, 360)
    i1 = np.searchsorted(sl, tl) % sl.size
    i0 = (i1 - 1) % sl.size
    dx = np.mod(sl[i1] - sl[i0], 360)
    dx = np.where(dx == 0, 360.0, dx)
    wx = np.mod(tl - sl[i0], 360) / dx
    wy, wx = wy[:, None], wx[None, :]
    return ((1 - wy) * (1 - wx) * a[np.ix_(j0, i0)] + (1 - wy) * wx * a[np.ix_(j0, i1)]
            + wy * (1 - wx) * a[np.ix_(j1, i0)] + wy * wx * a[np.ix_(j1, i1)])


def sst_change(win, mask, lat, lon, model_d):
    """dSST (win minus 1850-1900) for ERSSTv6, COBE-SST3, DCENT-I and the models, all on ONE mask: the
    cells of `mask` where every SST product and every model has a valid change (the 'SST mask'; it is used
    for dT only and changes no other mask). ERSSTv6 and COBE-SST3 are interpolated bilinearly to the
    product grid. DCENT-I (combined surface temperature) is used on purely marine 5 deg boxes within the
    SST mask and transferred to the whole SST mask through its offset from the mean of the other two.
    model_d: {model: dSST field on the product grid}."""
    area = oa_config.area_weights(lat, lon)
    gm = lambda d, m: np.sum(np.where(m, d, 0) * area) / (area * m).sum()
    dl, dn = SSTP['DCENT-I'][0], SSTP['DCENT-I'][1]
    d = {}
    for name in ('ERSSTv6', 'COBE-SST3'):
        plat, plon, pm = SSTP[name]
        d[name] = _bilinear(plat, plon, pm[win] - pm[SST_BASE], lat, lon)
    plat, plon, pm = SSTP['DCENT-I']
    d['DCENT-I'] = (pm[win] - pm[SST_BASE])[np.ix_(_nn(plat, lat), _nn(plon, lon, True))]
    sm = mask.copy()
    for a in list(d.values()) + list(model_d.values()):
        sm &= np.isfinite(a)
    pure = sm & SSTP['pure'][np.ix_(_nn(dl, lat), _nn(dn, lon, True))]
    full = {n: gm(d[n], sm) for n in ('ERSSTv6', 'COBE-SST3')}
    onpure = {n: gm(d[n], pure) for n in d}
    two_full, two_pure = np.mean(list(full.values())), np.mean([onpure['ERSSTv6'], onpure['COBE-SST3']])
    full['DCENT-I'] = two_full + onpure['DCENT-I'] - two_pure
    frac = (area * pure).sum() / (area * sm).sum()
    dT_mod = {m: gm(a, sm) for m, a in model_d.items()}
    return full, onpure, frac, sm, dT_mod, d


def run_constraint(common=None, quiet=False):
    """Constraint for every reference. common = (lat, lon, mask) restricts all masks to one common
    ocean area (mapped by nearest neighbour onto each product grid)."""
    results = {}
    for ref, cfg in REFS.items():
        if not os.path.exists(f"{cfg['mech']}/state_{ref}.nc"):
            print(f'[skip] {ref}: no mechanism files yet')
            continue
        PCO2_ATM = cfg['pco2']
        with netCDF4.Dataset(cfg['obs']) as f:
            lat, lon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
            omask = np.asarray(f['mask'][:]).astype(bool)
            yo = np.asarray(f['year'][:])
            ofield = np.ma.filled(f['omega_arag'][:].astype('f8'), np.nan)
        lon2, lat2 = np.meshgrid(lon, lat)
        si, po4 = co.load_nutrients(lat, lon)
        obs = read2d(f"{cfg['mech']}/state_{ref}.nc", VARS)

        files = sorted(p for p in glob.glob(f"{cfg['mech']}/state_*.nc") if not p.endswith(f'state_{ref}.nc'))
        names, st = [], {}
        for p in files:
            model, member = os.path.basename(p)[6:-3].rsplit('_', 1)
            names.append(model)
            st[model] = read2d(p, [f'{v}_{e}' for v in VARS for e in ('hist', 'pi')] + ['revelle_raw_hist'])
            st[model]['member'] = member

        # preindustrial states of every model (calc_pi_xco2.py): per cell, the time mean of the monthly
        # Omega_arag of the drift-corrected, bias-adjusted piControl with DIC at the CO2 of the state
        PI_FIELDS, xco2 = {}, {}
        for m in names:
            fx = glob.glob(f"{cfg['adj']}/piControl/omega_arag_biasadj_xco2_Oyr_{m}_piControl_{st[m]['member']}_*.nc")
            if len(fx) != 1:
                raise FileNotFoundError(f'{ref}: {len(fx)} xco2 files for {m} {st[m]["member"]}')
            with netCDF4.Dataset(fx[0]) as f:
                for vn in f.variables:
                    if vn.startswith('omega_arag_') and vn.endswith('_mean'):
                        tag = vn[len('omega_arag_'):-len('_mean')]
                        PI_FIELDS.setdefault(tag, {})[m] = np.ma.filled(f[vn][:].astype('f8'), np.nan)
                        xco2[tag] = float(f[vn].xco2)
        tags = [k for k, d in PI_FIELDS.items() if len(d) == len(names)]
        for k in ('1750', 'pi_low', 'pi_high', 'pi284'):
            if k not in tags:
                raise ValueError(f'{ref}: preindustrial state {k} missing for some model')

        # common mask: product mask, valid obs, nutrients, every model state and every preindustrial field
        m2 = omask & np.isfinite(si) & np.isfinite(po4)
        for v in VARS:
            m2 &= np.isfinite(obs[v])
        for m in names:
            for k, a in st[m].items():
                if k != 'member':
                    m2 &= np.isfinite(a)
        m_states = m2.copy()
        for k in tags:
            for m in names:
                m2 &= np.isfinite(PI_FIELDS[k][m])
        if common is not None:
            clat, clon, cmask = common
            m2 &= cmask[np.ix_(_nn(clat, lat), _nn(clon, lon, True))]
        w = oa_config.area_weights(lat, lon) * m2
        gm2 = lambda a: np.nansum(a * w) / w[np.isfinite(a)].sum()

        # piControl drift aligned with historical (calc_drift.py), on the product grid:
        # drift[m][v] = trend over the piControl years parallel to the window - piControl mean,
        # driftT[m]  = the same for T, window minus 1850-1900
        w0, w1 = cfg['win']
        drift, driftT = {}, {}
        for m in names:
            if DRIFT:
                dlat, dlon, dm = drift_means(m, st[m]['member'])
                jj, ii = _nn(dlat, lat), _nn(dlon, lon, True)
                rd = lambda p, v: dm[p][v][np.ix_(jj, ii)]
                drift[m] = {v: rd((w0, w1), v) for v in VARS}
                driftT[m] = rd((w0, w1), 'thetao') - rd(SST_BASE, 'thetao')
            else:
                drift[m] = {v: 0.0 for v in VARS}
                driftT[m] = 0.0
        # model change in each variable: historical window minus (drift-corrected) piControl
        chg = {m: {v: st[m][f'{v}_hist'] - st[m][f'{v}_pi'] - drift[m][v] for v in VARS} for m in names}

        # model warming exactly as observed: historical window minus historical 1850-1900, product grid,
        # minus the piControl drift over the same years
        model_d = {}
        for m in names:
            with netCDF4.Dataset(MODEL_SST.format(m, st[m]['member'])) as f:
                slat, slon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
                dd = (np.ma.filled(f[f"sst_{w0}_{w1}"][:].astype('f8'), np.nan)
                      - np.ma.filled(f[f'sst_{SST_BASE[0]}_{SST_BASE[1]}'][:].astype('f8'), np.nan))
            model_d[m] = dd[np.ix_(_nn(slat, lat), _nn(slon, lon, True))] - driftT[m]
        dsst, dsst_pure, frac_pure, sstmask, dT_mod, dsst_fields = sst_change(cfg['win'], m2, lat, lon, model_d)
        DT_OBS = float(np.mean(list(dsst.values())))
        DT_SIG = float(np.std(list(dsst.values()), ddof=1))

        def omega_gm(state):
            t, s, dic, alk = (state[v][m2] for v in VARS)
            r = mocsy.mvars(temp=co.cm.floor_temperature(t, s), sal=s, alk=alk * 1e-6, dic=dic * 1e-6, sil=si[m2] * 1e-6,
                            phos=po4[m2] * 1e-6, patm=np.ones(t.size), depth=np.zeros(t.size),
                            lat=lat2[m2], optcon='mol/kg', optt='Tpot', optp='m', optb=MOCSY_OPTB, optk1k2=MOCSY_OPTK1K2)
            om, R = np.full(m2.shape, np.nan), np.full(m2.shape, np.nan)
            good = co.cm.valid_carbonate(r[0], r[6])
            om[m2], R[m2] = np.where(good, np.asarray(r[6], float), np.nan), np.where(good, np.asarray(r[8], float), np.nan)
            return gm2(om), R, om

        dT = np.array([dT_mod[m] for m in names])           # on the SST mask, relative to 1850-1900
        dA = np.array([gm2(chg[m]['talk']) for m in names])
        CUC = np.array([gm2(st[m]['dissic_hist'] / st[m]['revelle_raw_hist']) / PCO2_ATM for m in names])

        # preindustrial Omega_arag of every model at every state: global mean of the per-cell fields on the mask
        y = {k: np.array([gm2(PI_FIELDS[k][m]) for m in names]) for k in tags}
        y1750 = y['1750']
        # the adjusted piControl itself (pi284) and the mean ratio 1750/pi284 (printed)
        f1750 = y1750 / y['pi284']

        om_obs, R_obs, _ = omega_gm(obs)
        CUC_obs = gm2(obs['dissic'] / R_obs) / PCO2_ATM
        ALK_ref = np.mean([dA[names.index(m)] for m in ALK_MODELS if m in names])

        print(f'\n===== {ref}: {len(names)} models, mask {100 * m2.sum() / omask.sum():.1f} % of product mask '
              f'({100 * (1 - m2.sum() / m_states.sum()):.2f} % of the state mask lost to preindustrial fields) =====')
        print(f'observed dSST {cfg["win"][0]}-{cfg["win"][1]} minus {SST_BASE[0]}-{SST_BASE[1]}: '
              + ', '.join(f'{k} {v:.3f}' for k, v in dsst.items())
              + f' K -> {DT_OBS:.3f} +- {DT_SIG:.3f} K  (DCENT-I from {100 * frac_pure:.0f} % pure-SST area; '
              + 'there: ' + ', '.join(f'{k} {v:.3f}' for k, v in dsst_pure.items()) + ')')
        _w = oa_config.area_weights(lat, lon)
        print(f'SST mask (all SST products and models valid): {100 * (_w * sstmask).sum() / (_w * m2).sum():.1f} % of the '
              f'constraint mask; model dT {dT.min():.3f}-{dT.max():.3f} K (historical window minus 1850-1900)')
        if DRIFT:
            _dd = {v: np.array([gm2(drift[m][v]) for m in names]) for v in VARS}
            print('piControl drift removed (window vs piControl mean, model range): '
                  + ', '.join(f'{v} {_dd[v].min():+.3f} to {_dd[v].max():+.3f}' for v in VARS)
                  + f'; T window minus 1850-1900 {min(gm2(driftT[m]) for m in names):+.3f} to '
                    f'{max(gm2(driftT[m]) for m in names):+.3f} K')
        print(f'observed: dT {DT_OBS:.3f} K, CUC {CUC_obs:.4f} (models {CUC.min():.4f}-{CUC.max():.4f}), '
              f'dALK {ALK_ref:.2f} umol/kg')
        print(f'preindustrial states (monthly, drift-corrected): pi284 {y["pi284"].mean():.3f}, 1750 {y1750.mean():.3f} '
              f'(ratio {f1750.mean():.4f}, models {f1750.min():.4f}-{f1750.max():.4f}), spread 1750 {y1750.std(ddof=1):.3f}')
        out, states, states_by_lab = {}, {}, {}
        for lab, cols, x0v, sig in (
                ('dT + CUC', [dT, CUC], [DT_OBS, CUC_obs], [DT_SIG, CUC_SIG]),
                ('dT + CUC + dALK', [dT, CUC, dA], [DT_OBS, CUC_obs, ALK_ref],
                 [DT_SIG, CUC_SIG, ALK_SIG_FRAC * abs(ALK_ref)])):
            X = np.column_stack([np.ones(len(names))] + cols)
            x0 = np.array([1.0] + x0v)
            beta, cov, s2, r2, yc, sfit = fit(X, y1750, x0)
            sobs = np.sqrt(np.sum((beta[1:] * np.array(sig)) ** 2))
            stot = np.hypot(sfit, sobs)
            # every state constrained with its own model values and the same predictors (no scaling anywhere)
            sk = {}
            for k in tags:
                bk, ck, _, r2k, yck, sfk = fit(X, y[k], x0)
                sok = np.sqrt(np.sum((bk[1:] * np.array(sig)) ** 2))
                sk[k] = dict(xco2=xco2[k], y_raw=float(y[k].mean()), spread=float(y[k].std(ddof=1)),
                             constrained=float(yck), constrained_sigma=float(np.hypot(sfk, sok)),
                             fit_sigma=float(sfk), obs_sigma=float(sok), r2=float(r2k),
                             ratio=float(np.mean(y[k] / y1750)))       # ratio to the 1750 state (the notebook scales with it)
            rng = (sk['pi_high']['constrained'], sk['pi_low']['constrained'])   # 284 and 275 ppm, constrained
            print(f'  {lab:16s} R2 {r2:.2f}  coefs ' + ' '.join(f'{b:+.4f}' for b in beta)
                  + f'\n  {"":16s} 1750 {yc:.3f} +- {stot:.3f} (fit {sfit:.3f}, obs {sobs:.3f}), '
                  f'boundary {0.8 * yc:.3f} +- {0.8 * stot:.3f}'
                  + f'\n  {"":16s} 284-275 ppm: preindustrial {rng[0]:.3f}-{rng[1]:.3f}, '
                  f'boundary {0.8 * rng[0]:.3f}-{0.8 * rng[1]:.3f}')
            out[lab] = (yc, stot, rng)
            states_by_lab[lab] = sk
            if lab == 'dT + CUC':
                states = sk
                print('  states (xco2 ppm: unconstrained, constrained +- sigma; for the record: constrained 1750 x ratio):')
                for k in tags:
                    print(f'    {k:9s} {sk[k]["xco2"]:6.1f}: {sk[k]["y_raw"]:.3f}, {sk[k]["constrained"]:.3f} +- '
                          f'{sk[k]["constrained_sigma"]:.3f}, {yc * sk[k]["ratio"]:.3f}')
        wm = oa_config.area_weights(lat, lon) * m2          # same mask as the constraint
        ogm = np.array([np.nansum(o * wm) / wm[np.isfinite(o)].sum() for o in ofield])
        ok = np.isfinite(ogm)
        fit_ok = ok & (yo <= EVAL_YEAR)              # the same years for every product: up to EVAL_YEAR
        if fit_ok.sum() >= 3:
            q = np.polyfit(yo[fit_ok] - 2000, ogm[fit_ok], 2)
            now = np.polyval(q, EVAL_YEAR - 2000)
            # for comparison (test of the fit uncertainty): the fit through all years of the product
            now_all = np.polyval(np.polyfit(yo[ok] - 2000, ogm[ok], 2), EVAL_YEAR - 2000)
        else:                                        # single state (Jiang et al.): no 2024 value
            now = now_all = np.nan
        _s10 = ok & (yo >= 2005) & (yo <= 2014)
        obs10 = float(np.mean(ogm[_s10])) if _s10.any() else np.nan     # decade 2010 of Jiang et al.
        # unconstrained: model spread only
        yu, su = y1750.mean(), y1750.std(ddof=1)
        rngu = (y['pi_high'].mean(), y['pi_low'].mean())
        print(f'  unconstrained 1750 {yu:.3f} +- {su:.3f}, boundary {0.8 * yu:.3f} +- {0.8 * su:.3f}; '
              f'284-275 ppm {rngu[0]:.3f}-{rngu[1]:.3f}')
        print(f'  obs {EVAL_YEAR} quadratic fit {now:.4f} (years up to {EVAL_YEAR}, constraint mask); fit through all '
              f'years (to {int(yo[ok].max()) if ok.any() else "-"}) {now_all:.4f}, difference {now_all - now:+.4f}')
        for lab, (yc, s, rng) in out.items():
            print(f'  {lab:16s} change {100 * (now / yc - 1):+.1f} %, distance to boundary {now - 0.8 * yc:+.3f} '
                  f'(284 ppm {now - 0.8 * rng[0]:+.3f}, 275 ppm {now - 0.8 * rng[1]:+.3f})')
        results[ref] = dict(names=names, dT=dT, CUC=CUC, dA=dA, y1750=y1750, y_states={k: y[k] for k in tags},
                            CUC_obs=CUC_obs,
                            constrained=out, unconstrained=(yu, su, rngu), now=now, now_all_years=now_all, obs10=obs10, year=yo, ogm=ogm,
                            mask=m2, lat=lat,
                            states=states, states_by_lab=states_by_lab,
                            dicR=obs['dissic'] / R_obs, dsst=dsst, dT_obs=(DT_OBS, DT_SIG), lon=lon, sstmask=sstmask)
        if STORE_FIELDS:
            # per-cell fields for regional / cell-by-cell analyses (float32, product grid)
            f32 = lambda a: np.asarray(a, 'f4')
            results[ref]['fields'] = dict(
                y1750=f32([PI_FIELDS['1750'][m] for m in names]),      # preindustrial at 1750, per cell (monthly mean)
                dT=f32([model_d[m] for m in names]),                              # model dSST, drift-corrected
                CUC=f32([st[m]['dissic_hist'] / st[m]['revelle_raw_hist'] / PCO2_ATM for m in names]),
                CUC_obs=f32(obs['dissic'] / R_obs / PCO2_ATM),
                dsst_obs={k_: f32(v_) for k_, v_ in dsst_fields.items()})

    return results


results = run_constraint()
