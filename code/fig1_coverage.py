# Notebook cell: Figure 1, why previous estimates differ: coverage and the definition of the preindustrial state.
#   a  annual global mean surface Omega_arag of each product on its common area with the ESMs (dashed; cells where the
#      product has complete data and all 18 ESMs have ocean) and on the common area of all products (solid);
#      left column: constrained 1750 value of each product on both areas (open: with the ESMs; filled: all products);
#      right column: the boundary (80 %) on both areas; star: Planetary Health Check (2026)
#   b  decline relative to 1750 over time, observed and preindustrial on the same area: common area with the ESMs
#      (dashed), common area of all products (solid); star: Planetary Health Check (different areas)
#   c  from the Planetary Health Check to this study for UExP-FNN-U, one change per bar, as decline from 1750 (%):
#      published value -> our average of the same data -> mocsy chemistry -> common area with the ESMs -> trend fit for
#      2024 -> common area of all products -> preindustrial from the ESMs (272 ppm) -> CO2 of 1750 from AR6 (278.3 ppm)
#      -> emergent constraint -> mean of the four products (± 1σ); table below: what each bar assumes, changes in bold
# Observed values: quadratic fit to the annual area-weighted means, evaluated in 2024 (as in Table S1).
# Needs data/constraint_results.pkl (with 'states'), the product files (rc.OBS) and regional_common.py.
import os
import sys
import pickle
import importlib
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # the code folder
import oa_config
import regional_common as rc
importlib.reload(rc)

# which run: 'l10' = total boron Lee et al. (2010), the first run (data_boron_lee2010); 'u74' = Uppstroem (1974), the
# standard from now on (data). Data, product files and figures are all taken from / written to that run's folder.
# the run (data, product files, figures) is chosen with os.environ['OA_RUN'] at the top of the notebook (regional_common)
BORON = 'l10' if rc.RUN == 'l10' else 'u74'               # total boron of this run ('m10' uses Uppstroem as well)
ROOT = rc.ROOT
OBSF = dict(rc.OBS)
print(f'run {rc.RUN} (total boron {BORON}): {ROOT}')
LAB = rc.LAB
YEAR = 2024
OMEGA = r'$\Omega_{\mathrm{arag}}$'
# Planetary Health Check (2026), ocean acidification section: observed 2.85 ± 0.01 (2σ) in 2024 from UExP-FNN-U on its
# full area; preindustrial 3.57 = Jiang et al. (2023, Table S5) for 1750; boundary 2.86 (80 % of 3.57, as published)
PHC_OBS, PHC_PI, PHC_B = 2.85, 3.57, 2.86
DETAILED = False                      # panel c: False = the three consistency steps, True = every single change
STEP_REF = 'UExP'                     # product of the Planetary Health Check
BCOL = 'k'                            # boundary

with open(f'{ROOT}/constraint_results.pkl', 'rb') as f:
    PK = pickle.load(f)
RCOM, ROWN = PK['results_common'], PK['results']
refs = [q for q in ('OceanSODA', 'UExP', 'JMA', 'CMEMS') if q in RCOM]


def read_obs(q):
    r = RCOM[q]
    with netCDF4.Dataset(OBSF[q]) as f:
        return (np.asarray(f['year'][:]), np.asarray(r['lat'], 'f8'), np.asarray(r['lon'], 'f8'),
                np.ma.filled(f['omega_arag'][:].astype('f8'), np.nan))


def gm_series(yo, om, lat, lon, mask):
    """annual area-weighted means over mask"""
    wm = oa_config.area_weights(lat, lon) * mask
    return np.array([np.nansum(np.where(np.isfinite(o), o, 0) * wm) / wm[np.isfinite(o)].sum() for o in om])


def fit_now(yo, gm):
    """quadratic fit to the annual means, evaluated in YEAR (as table_summary.py)"""
    ok = np.isfinite(gm) & (yo <= rc.LAST_YEAR)
    return np.polyval(np.polyfit(yo[ok] - 2000.0, gm[ok], 2), YEAR - 2000.0)


# ---------------------------------------------------------------- coverage and observed values per product
cov, V, TS = {}, {}, {}
for q in refs:
    yo, lat, lon, om = read_obs(q)
    full = np.isfinite(om[yo <= rc.LAST_YEAR]).all(0)          # complete: every year (and so every month) up to 2024
    own = np.asarray(ROWN[q]['mask']).astype(bool)
    com = np.asarray(RCOM[q]['mask']).astype(bool)
    g = {k: gm_series(yo, om, lat, lon, m_) for k, m_ in (('full', full), ('own', own), ('com', com))}
    keep = yo <= rc.LAST_YEAR
    TS[q] = dict(year=yo[keep], own=g['own'][keep], com=g['com'][keep])
    V[q] = {k: fit_now(yo, v) for k, v in g.items()}
    # one map grid for all products: longitudes -180..180 (products differ in convention), latitudes ascending
    lon180, full180 = rc.to_180(lon, full)
    com180 = rc.to_180(lon, com)[1]
    if lat[0] > lat[-1]:
        lat, full180, com180 = lat[::-1], full180[::-1], com180[::-1]
    cov[q] = dict(lat=lat, lon=lon180, full=full180, com=com180)
lat, lon = cov[refs[0]]['lat'], cov[refs[0]]['lon']
for q in refs[1:]:
    assert np.allclose(cov[q]['lat'], lat) and np.allclose(cov[q]['lon'], lon), 'products on different grids'
n_prod = sum(cov[q]['full'].astype(int) for q in refs)
common = cov[refs[0]]['com']
w = oa_config.area_weights(lat, lon)
ocean = n_prod > 0
pct = lambda m: 100 * w[m].sum() / w[ocean].sum()
AREA_C = pct(common & ocean)

# preindustrial (constrained, 1750) and decline, own area and common area
dec = lambda o, p: 100 * (1 - o / p)
PI = {}
for q in refs:
    yo_, so_ = ROWN[q]['constrained'][LAB][:2]
    yc_, sc_ = RCOM[q]['constrained'][LAB][:2]
    PI[q] = dict(own=yo_, sown=so_, com=yc_, scom=sc_,
                 dec_own=dec(V[q]['own'], yo_), sdec_own=100 * V[q]['own'] / yo_ ** 2 * so_,
                 dec_com=dec(V[q]['com'], yc_), sdec_com=100 * V[q]['com'] / yc_ ** 2 * sc_)
    print(f"{q:10s} 2024: full area {V[q]['full']:.3f}, own {V[q]['own']:.3f}, common ({AREA_C:.0f} %) {V[q]['com']:.3f}; "
          f"1750: own {yo_:.3f}, common {yc_:.3f}; decline: own {PI[q]['dec_own']:.1f} %, common {PI[q]['dec_com']:.1f} %")
yc_m = np.mean([PI[q]['com'] for q in refs])
B = 0.8 * yc_m
B_sig = 0.8 * np.sqrt(np.mean([PI[q]['scom'] ** 2 for q in refs]))

# ---------------------------------------------------------------- steps from the Planetary Health Check to this study
S = RCOM[STEP_REF]['states']
now_c = V[STEP_REF]['com']
now_m = np.mean([V[q]['com'] for q in refs])
dec_q = np.array([PI[q]['dec_com'] for q in refs])
dec_m = dec_q.mean()
# ---------------------------------------------------------------- uncertainty of the decline (1 sigma, percentage points)
# all components, combined in quadrature; per-product components are averaged as RMS (largely common to all products)
import os
_ddec = lambda q, s_om: 100 * V[q]['com'] / PI[q]['com'] ** 2 * s_om      # d(decline) for an error s_om in 1750


# ---------------------------------------------------------------- estimate of the 2024 state
# main estimator: quadratic fit in time (as in constraint_refs.py). Alternatives: quadratic fit against atmospheric CO2
# (NOAA global annual means) and Enting (1987) smoothing splines in time and in CO2 with long cutoffs. All are linear in
# the annual means y (estimate = w . y); the uncertainty of the main estimator is sqrt(w' C w) with C the covariance of
# the residuals around it, modelled as AR(1) (variance s2, lag-1 autocorrelation r), since interannual variability such
# as ENSO is not independent from year to year. The estimator term is half the range of the alternatives.
import urllib.request
ENT_CUTOFFS = (20.0,)                # years; in CO2 converted with the mean growth rate of the fit period
AERA_WINLEN = 31                     # AERA (Terhaar et al., 2022; Leyvraz, AERA 1.0.0): running mean, linearly extrapolated
_t = urllib.request.urlopen('https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_gl.txt', timeout=60).read()
XCO2 = {int(r_[0]): float(r_[1]) for r_ in (l_.split() for l_ in _t.decode().splitlines())
        if r_ and not r_[0].startswith('#')}


def _series(q):
    r = RCOM[q]
    yo_, gm_ = np.asarray(r['year'], 'f8'), np.asarray(r['ogm'], 'f8')
    ok = np.isfinite(gm_) & (yo_ <= rc.LAST_YEAR)
    assert np.all(np.diff(yo_[ok]) == 1), f'{q}: annual series with gaps'
    return yo_[ok], gm_[ok]


def _w_poly(x, x0, deg):
    """weights of a polynomial fit of degree deg in x, evaluated at x0, and the hat matrix (all fitted values)"""
    X = np.column_stack([x ** k for k in range(deg, -1, -1)])
    H = np.linalg.solve(X.T @ X, X.T)
    v0 = np.array([x0 ** k for k in range(deg, -1, -1)])
    return v0 @ H, X @ H


def _smoother(x, cutoff):
    """Enting (1987) smoothing spline on (possibly uneven) x: penalised least squares with second divided differences,
    lambda chosen for 50 % response at the cutoff (in units of x) for the mean spacing; returns the smoother matrix"""
    n_ = x.size
    h = np.diff(x)
    D = np.zeros((n_ - 2, n_))
    for i in range(n_ - 2):
        D[i, i], D[i, i + 2] = 2 / (h[i] * (h[i] + h[i + 1])), 2 / (h[i + 1] * (h[i] + h[i + 1]))
        D[i, i + 1] = -D[i, i] - D[i, i + 2]
    hm = h.mean()
    lam = hm ** 4 / (2.0 - 2.0 * np.cos(2.0 * np.pi * hm / cutoff)) ** 2
    return np.linalg.inv(np.eye(n_) + lam * D.T @ D)


def _trend(q):
    yrs, y_ = _series(q)
    t = yrs - 2000.0
    xc = np.array([XCO2[int(v)] for v in yrs])
    gr = (xc[-1] - xc[0]) / (yrs[-1] - yrs[0])                  # ppm per year over the fit period
    i24 = list(yrs).index(YEAR)
    e = dict(yrs=yrs, y=y_, xco2=xc)
    wq, Hq = _w_poly(t, YEAR - 2000.0, 2)
    e['time quadratic'], e['fit time quadratic'] = float(wq @ y_), Hq @ y_
    wc, Hc = _w_poly(xc - 400.0, XCO2[YEAR] - 400.0, 2)
    e['CO2 quadratic'], e['fit CO2 quadratic'] = float(wc @ y_), Hc @ y_
    # AERA: 31-year running mean; the last 15 years are the running mean 16 years before the end plus the slope of a
    # linear fit over the last 31 years (aera/core.py, extrapolated_runmean); for comparison only
    if y_.size >= AERA_WINLEN:
        h_ = AERA_WINLEN // 2
        rm = np.convolve(y_, np.ones(AERA_WINLEN) / AERA_WINLEN, mode='same')
        e['AERA (comparison)'] = float(rm[-h_ - 1] + np.polyfit(np.arange(AERA_WINLEN), y_[-AERA_WINLEN:], 1)[0] * h_)
    for co in ENT_CUTOFFS:
        St, Sc = _smoother(yrs, co), _smoother(xc, co * gr)
        e[f'fit time Enting {co:.0f} yr'], e[f'fit CO2 Enting {co:.0f} yr'] = St @ y_, Sc @ y_
        e[f'time Enting {co:.0f} yr'] = float((St @ y_)[i24])
        e[f'CO2 Enting {co * gr:.0f} ppm'] = float((Sc @ y_)[i24])
    res = y_ - e['fit time quadratic']
    s2 = res @ res / (yrs.size - 3)
    r1 = float(np.clip(np.corrcoef(res[:-1], res[1:])[0, 1], 0.0, 0.99))
    C = s2 * r1 ** np.abs(np.subtract.outer(np.arange(yrs.size), np.arange(yrs.size)))
    alt = [v for k, v in e.items() if isinstance(v, float) and k[:4] in ('time', 'CO2 ')]
    e['se'], e['r1'], e['gr'], e['range'] = float(np.sqrt(wq @ C @ wq)), r1, gr, max(alt) - min(alt)
    return e


EST = {q: _trend(q) for q in refs}
print(f'  2024 state by estimator (quadratic fits and Enting splines with {", ".join(f"{c:.0f}" for c in ENT_CUTOFFS)} yr '
      'cutoff; AERA shown for comparison only, not in the uncertainty):')
for q in refs:
    e = EST[q]
    print(f'    {q:10s} {int(e["yrs"][0])}-{int(e["yrs"][-1])} (CO2 +{e["gr"]:.2f} ppm/yr): '
          + ', '.join(f'{k} {v:.3f}' for k, v in e.items() if isinstance(v, float) and k[:4] in ('time', 'CO2 ', 'AERA'))
          + f'; quadratic +- {e["se"]:.3f} (AR(1), r = {e["r1"]:.2f}), range of alternatives {e["range"]:.3f}')
    print(f'    {"":10s} annual mean {YEAR} {e["y"][-1]:.3f}, i.e. {e["y"][-1] - e["time quadratic"]:+.3f} from the quadratic fit '
          f'({100 * (e["y"][-1] - e["time quadratic"]) / e["time quadratic"]:+.2f} %, '
          f'{-100 * (e["y"][-1] - e["time quadratic"]) / PI[q]["com"]:+.2f} points of decline)')


def _co2_sig(q):
    """1750 Omega change for +-1 sigma of the 1750 CO2 (AR6, 278.3 +- 1.8 ppm), from the states 1750_lo/_hi"""
    st = RCOM[q]['states']
    if '1750_lo' not in st or '1750_hi' not in st:
        return np.nan
    return PI[q]['com'] * abs(st['1750_lo']['ratio'] - st['1750_hi']['ratio']) / 2


def _run_means(path):
    """mean decline, 2024 state and 1750 state over the products in a run (fit in YEAR, constrained 1750, common mask)"""
    with open(path, 'rb') as f:
        rr = pickle.load(f)['results_common']
    now_, pi_ = np.array([rc.fit_now(rr[q]) for q in refs]), np.array([rr[q]['constrained'][LAB][0] for q in refs])
    return np.mean(100 * (1 - now_ / pi_)), now_.mean(), pi_.mean()


rms = lambda a: float(np.sqrt(np.nanmean(np.square(a))))
K_ESM, K_CO2, K_PROD_D = '1750 state estimated from the ESMs (after the emergent constraint)', 'atmospheric CO$_2$ in 1750 (AR6, 278.3 ± 1.8 ppm)', 'differences between the four observation-based products'
K_FIT, K_METH, K_B, K_K12 = 'forced 2024 state: statistical uncertainty of the fit (natural variability)', 'forced 2024 state: choice of fitting method', 'total boron (Uppström 1974 or Lee et al. 2010)', 'K$_1$, K$_2$ (Lueker et al. 2000 or Millero 2010)'
SIG = {K_ESM: rms([PI[q]['sdec_com'] for q in refs]),
       K_CO2: rms([_ddec(q, _co2_sig(q)) for q in refs]),
       K_PROD_D: float(dec_q.std(ddof=1)),
       K_FIT: rms([100 * EST[q]['se'] / PI[q]['com'] for q in refs]),
       K_METH: rms([100 * EST[q]['range'] / 2 / PI[q]['com'] for q in refs])}
# the same components in absolute Omega_arag (1 sigma): (1750 state, 2024 state); nan = does not affect that state
_v24 = np.array([V[q]['com'] for q in refs])
_v50 = np.array([PI[q]['com'] for q in refs])
ABS = {K_ESM: (rms([PI[q]['scom'] for q in refs]), np.nan),
       K_CO2: (rms([_co2_sig(q) for q in refs]), np.nan),
       K_PROD_D: (float(_v50.std(ddof=1)), float(_v24.std(ddof=1))),
       K_FIT: (np.nan, rms([EST[q]['se'] for q in refs])),
       K_METH: (np.nan, rms([EST[q]['range'] / 2 for q in refs]))}
# equilibrium constants: difference of the decline between two formulations taken as 2 sigma (as Orr et al., 2018, for
# total boron); from the sensitivity runs, if available
_here = _run_means(f'{ROOT}/constraint_results.pkl')
_other = rc.RUNS['u74' if BORON == 'l10' else 'l10'] + '/constraint_results.pkl'
_m10 = rc.RUNS['m10'] + '/constraint_results.pkl'
for lab_, path_, needs in ((K_B, _other, rc.RUN == 'u74'), (K_K12, _m10, rc.RUN == 'u74')):
    try:
        if not needs:
            raise FileNotFoundError("only evaluated for the standard run (OA_RUN = 'u74')")
        _o = _run_means(path_)
        SIG[lab_] = abs(_o[0] - _here[0]) / 2
        ABS[lab_] = (abs(_o[2] - _here[2]) / 2, abs(_o[1] - _here[1]) / 2)
    except Exception as e_:
        SIG[lab_], ABS[lab_] = np.nan, (np.nan, np.nan)
        print(f'  {lab_}: not included ({e_})')
sdec_m = float(np.sqrt(np.nansum(np.square(list(SIG.values())))))
print('uncertainty of the decline (1 sigma, percentage points):')
for k_, v_ in SIG.items():
    print(f'  {k_.replace("$", ""):55s} {v_:.2f}')
print(f'  {"total (quadrature)":55s} {sdec_m:.2f}')
X_J, X_1750 = S['jiang']['xco2'], S['1750']['xco2']
# observed side for STEP_REF, from the product as published to our value on the common area
yo_s, lat_s, lon_s, om_s = read_obs(STEP_REF)
with netCDF4.Dataset(OBSF[STEP_REF]) as f:
    om_p = np.ma.filled(f['omega_arag_product'][:].astype('f8'), np.nan)    # the product's own Omega
i24 = list(yo_s).index(YEAR)
mean24 = lambda fld, m_: gm_series(yo_s[i24:i24 + 1], fld[i24:i24 + 1], lat_s, lon_s, m_)[0]
own_s = np.asarray(ROWN[STEP_REF]['mask']).astype(bool)
OBS_STEPS = dict(prod=mean24(om_p, np.isfinite(om_p[i24])), mocsy=mean24(om_s, np.isfinite(om_s[i24])),
                 own=mean24(om_s, own_s), own_fit=V[STEP_REF]['own'], com_fit=now_c,
                 com24=mean24(om_s, np.asarray(RCOM[STEP_REF]['mask']).astype(bool)))
print(f'{STEP_REF} 2024: product Omega {OBS_STEPS["prod"]:.3f}, mocsy {OBS_STEPS["mocsy"]:.3f}, own area {OBS_STEPS["own"]:.3f}, '
      f'fit {OBS_STEPS["own_fit"]:.3f}, common area (fit) {OBS_STEPS["com_fit"]:.3f}')
# per step: label, present value, its area, chemistry, estimate, preindustrial value, its source, its CO2, product
# Jiang et al. (2023) 1750 field (NCEI 0259391, median of 14 ESMs, CO2System.jl with the same constants as mocsy here)
# evaluated on our common area: separates the area effect from the observed baseline the models are anchored to
JFILE = f'{ROOT}/jiang2023/Aragonite_median_historical.nc'
with netCDF4.Dataset(JFILE) as f:
    _jy = np.asarray(f['time'][:]).astype(int)
    _jlat = np.asarray(f['latitude'][:], 'f8')[:, 0]
    _jlon = (np.asarray(f['longitude'][:], 'f8')[0, :] + 180.0) % 360.0 - 180.0
    _J = np.ma.filled(f['aragonite'][:].astype('f8'), np.nan)
_J[~np.isfinite(_J) | (_J <= 0) | (_J > 10)] = np.nan
_o = np.argsort(_jlon)
_jlon, _J = _jlon[_o], _J[:, :, _o]
_wj = oa_config.area_weights(_jlat, _jlon)


def jiang_mean(field, lat_, lon_, mask_):
    lon180, m_ = rc.to_180(np.asarray(lon_, 'f8'), np.asarray(mask_).astype(bool))
    m_ = m_[np.ix_(np.abs(np.asarray(lat_)[None, :] - _jlat[:, None]).argmin(1),
                   np.abs(lon180[None, :] - _jlon[:, None]).argmin(1))] & np.isfinite(field)
    return np.sum(field[m_] * _wj[m_]) / _wj[m_].sum()


_rc = RCOM[STEP_REF]
PI_J_OWN = np.sum(np.where(np.isfinite(_J[0]), _J[0], 0) * _wj) / _wj[np.isfinite(_J[0])].sum()
PI_J_COM = jiang_mean(_J[list(_jy).index(1750)], _rc['lat'], _rc['lon'], _rc['mask'])
J10_COM = jiang_mean(_J[list(_jy).index(2010)], _rc['lat'], _rc['lon'], _rc['mask'])
_o10 = np.mean([np.nanmean(np.asarray(RCOM[q]['ogm'])[(np.asarray(RCOM[q]['year']) >= 2005)
                                                       & (np.asarray(RCOM[q]['year']) <= 2014)]) for q in refs])
_pi272 = np.mean([RCOM[q]['states']['jiang']['y_raw'] for q in refs])
print(f'Jiang et al. 1750: own coverage {PI_J_OWN:.3f}, common area {PI_J_COM:.3f} (ours at the same CO2 {_pi272:.3f}); '
      f'2010s on the common area: Jiang {J10_COM:.3f}, products {_o10:.3f}; decline 1750-2010s: Jiang '
      f'{dec(J10_COM, PI_J_COM):.1f} %, ours {dec(_o10, _pi272):.1f} %')
# 1750 from our 18 ESMs, but relative to the observations of Jiang et al.: their 2010s state on the common coverage
# times our modelled ratio 1750 / 2010s for STEP_REF (approximation; separates the models from the observations)
_y_ = np.asarray(RCOM[STEP_REF]['year'])
U10_COM = np.nanmean(np.asarray(RCOM[STEP_REF]['ogm'])[(_y_ >= 2005) & (_y_ <= 2014)])
# total boron: this study uses Uppstroem (1974) (MOCSY_OPTB = 'u74', as UExP-FNN-U); Jiang et al. (2023) use
# Lee et al. (2010) (check in their methods; set J_BORON = 'u74' if not). Their fields provide only Omega_arag, so they are
# converted with the ratio u74 / l10 of the same quantity in our two runs (same states, only the boron differs):
#   1750 values: ratio of our 1750 state at their CO2 (states['jiang']) on the common coverage
#   2010s value: ratio of the observed 2005-2014 mean of STEP_REF on the common coverage
J_BORON = 'l10'
OTHER_PKL = rc.RUNS['l10' if BORON == 'u74' else 'u74'] + '/constraint_results.pkl'
if J_BORON != BORON:
    with open(OTHER_PKL, 'rb') as f:
        _L = pickle.load(f)['results_common'][STEP_REF]
    _yl = np.asarray(_L['year'])
    F_J = S['jiang']['y_raw'] / _L['states']['jiang']['y_raw']
    F_J10 = U10_COM / np.nanmean(np.asarray(_L['ogm'])[(_yl >= 2005) & (_yl <= 2014)])
else:
    F_J = F_J10 = 1.0
PHC_PI_U, PI_J_COM_U = PHC_PI * F_J, PI_J_COM * F_J
# preindustrial states of STEP_REF after the constraint: at the CO2 of Jiang et al. (constrained 1750 value times the
# modelled ratio of the two CO2 states) and at 278.3 ppm (1750)
PI_JC = PI[STEP_REF]['com'] * S['jiang']['ratio']
PI_1750C = PI[STEP_REF]['com']
PI_E18_J = J10_COM * F_J10 * PI_JC / U10_COM
print(f'boron conversion of Jiang et al. ({J_BORON} -> {BORON}): 1750 x {F_J:.4f}, 2010s x {F_J10:.4f}; 1750 own coverage '
      f'{PHC_PI_U:.3f}, common {PI_J_COM_U:.3f}, 18 ESMs relative to Jiang et al. {PI_E18_J:.3f}')
o = OBS_STEPS
J = ('Jiang 2023', f'{X_J:.0f} (1750)')             # Jiang et al. (2023)
STEPS = [('Planetary\nHealth Check\n(2026)', PHC_OBS, 'full', 'product', '2024', PHC_PI, *J, 'UExP'),
         ('same data,\nour average', o['prod'], 'full', 'product', '2024', PHC_PI, *J, 'UExP'),
         ('mocsy\nchemistry', o['mocsy'], 'full', 'mocsy', '2024', PHC_PI, *J, 'UExP'),
         ('common area\nwith ESMs', o['own'], 'with ESMs', 'mocsy', '2024', PHC_PI, *J, 'UExP'),
         ('trend fit\nfor 2024', o['own_fit'], 'with ESMs', 'mocsy', 'fit', PHC_PI, *J, 'UExP'),
         ('common area\nof all products', now_c, 'all products', 'mocsy', 'fit', PHC_PI, *J, 'UExP'),
         ('Jiang 1750 on\nthe common area', now_c, 'all products', 'mocsy', 'fit', PI_J_COM, 'Jiang, common area', J[1],
          'UExP'),
         ('same observed\nbaseline (ESMs)', now_c, 'all products', 'mocsy', 'fit', S['jiang']['y_raw'], 'ESMs', J[1],
          'UExP'),
         ('CO$_2$ of 1750\nfrom AR6', now_c, 'all products', 'mocsy', 'fit', S['1750']['y_raw'], 'ESMs', f'{X_1750:.1f} (1750)',
          'UExP'),
         ('emergent\nconstraint', now_c, 'all products', 'mocsy', 'fit', PI[STEP_REF]['com'], 'ESMs, constr.',
          f'{X_1750:.1f} (1750)', 'UExP'),
         ('this study\n(four\nproducts)', now_m, 'all products', 'mocsy', 'fit', yc_m, 'ESMs, constr.', f'{X_1750:.1f} (1750)',
          'mean of 4')]
STEPS_DET = STEPS                                    # every single change (Table S2)
vals_det = [dec(s_[1], s_[5]) for s_ in STEPS_DET]
vals_det[-1] = dec_m
if not DETAILED:                                     # the three consistency steps only (detailed chain: DETAILED = True)
    STEPS = [STEPS[0],
             ('same equilibrium\nconstants for 1750\nand present day', o['mocsy'], 'full', 'mocsy', '2024', PHC_PI_U, *J, 'UExP'),
             ('same ocean\ncoverage for 1750\nand present day', o['com24'], 'all products', 'mocsy', '2024', PI_J_COM_U,
              'Jiang, common area', J[1], 'UExP'),
             ('2024 from the\ntrend fit instead\nof a single year', now_c, 'all products', 'mocsy', 'fit', PI_J_COM_U,
              'Jiang, common area', J[1], 'UExP'),
             ('18 constrained\nESMs instead of the\n14 ESMs of Jiang et al.', now_c, 'all products', 'mocsy', 'fit', PI_E18_J,
              'ESMs (mocsy)', J[1], 'UExP'),
             ('1750 calculated\nrelative to the product\nused for present day', now_c, 'all products', 'mocsy', 'fit', PI_JC,
              'ESMs (mocsy)', J[1], 'UExP'),
             (f'CO$_2$ in 1750 of\n{X_1750:.1f} ppm (AR6)\ninstead of {X_J:.0f} ppm\n(Jiang et al.)', now_c, 'all products', 'mocsy', 'fit', PI_1750C, 'ESMs (mocsy)',
              f'{X_1750:.1f} (1750)', 'UExP'),
             ('this study\n(mean of the\nfour products)', now_m, 'all products', 'mocsy', 'fit', yc_m,
              'ESMs, constr.', f'{X_1750:.1f} (1750)', 'mean of 4')]
vals = [dec(s_[1], s_[5]) for s_ in STEPS]
vals[-1] = dec_m
# consistency of each bar: same equilibrium constants, same ocean area, 1750 consistent with the present-day estimate,
# CO2 of 1750 (1 = consistent)
CHECKS = [(0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 0, 0), (1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 1, 1), (1, 1, 1, 1)] if not DETAILED else None
print('steps: ' + '; '.join(f'{s_[0].replace(chr(10), " ")} {v:.1f} %' for s_, v in zip(STEPS, vals))
      + f' ± {sdec_m:.1f} %')

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(12, 11))
outer = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 0.85], hspace=0.3, left=0.1, right=0.98, top=0.96, bottom=0.25)
top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[1.45, 1], wspace=0.2)

# a: annual global means on the own area (dashed) and the common area (solid), with the 1750 value and the boundary of
#    each product on both areas (columns; open = own area, filled = common area)
sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=top[0], width_ratios=[0.2, 1, 0.2], wspace=0.08)
ax0 = fig.add_subplot(sub[0])
ax = fig.add_subplot(sub[1], sharey=ax0)
ax2 = fig.add_subplot(sub[2], sharey=ax0)
xq = {q: -0.45 + 0.3 * k for k, q in enumerate(refs)}


def column(a_, factor):
    for q in refs:
        c = rc.COLORS[q]
        a_.errorbar(xq[q] - 0.06, factor * PI[q]['own'], yerr=factor * PI[q]['sown'], fmt='o', mfc='white', mec=c,
                    mew=1.2, ms=4.5, color=c, capsize=2, lw=0.9)
        a_.errorbar(xq[q] + 0.06, factor * PI[q]['com'], yerr=factor * PI[q]['scom'], fmt='o', color=c, ms=4.5,
                    capsize=2, lw=0.9)
    a_.set_xlim(-0.65, 0.85)
    a_.set_xticks([0])
    a_.spines[['top', 'right']].set_visible(False)


column(ax0, 1.0)
ax0.plot(0.7, PHC_PI, marker='*', color='k', ms=10, ls='none', zorder=4)       # Planetary Health Check
ax0.set_xticklabels(['1750'])
ax0.set_ylabel(f'Global mean surface {OMEGA}', fontsize=10)
column(ax2, 0.8)
ax2.plot(0.7, PHC_B, marker='*', color='k', ms=10, ls='none', zorder=4)
ax2.set_xticklabels(['boundary'])
ax2.tick_params(labelleft=False)
for q in refs:
    ax.plot(TS[q]['year'], TS[q]['own'], color=rc.COLORS[q], lw=1.0, ls=(0, (3, 2)))
    ax.plot(TS[q]['year'], TS[q]['com'], color=rc.COLORS[q], lw=1.3)
ax.plot(YEAR, PHC_OBS, marker='*', color='k', ms=10, ls='none', zorder=4)
ax.set_xlim(1985, YEAR + 1)
ax.set_xticks(range(1990, YEAR + 1, 10))          # no ticks at the edges, next to the columns
ax.tick_params(labelleft=False)
ax.spines[['top', 'right']].set_visible(False)
hh = [Line2D([], [], color='0.3', lw=1.0, ls=(0, (3, 2))), Line2D([], [], color='0.3', lw=1.3)] + \
     [Line2D([], [], color=rc.COLORS[q], lw=1.3) for q in refs] + \
     [Line2D([], [], marker='*', color='k', ms=9, ls='none')]
ax.legend(hh, ['common area with the ESMs (open)', f'common area of all products, {AREA_C:.0f}% (filled)'] + [rc.FULL[q] for q in refs]
          + ['Planetary Health Check (2026)'],
          fontsize=7.5, frameon=False, loc='upper right')
lo_ = min(min(TS[q]['own'].min(), TS[q]['com'].min(), 0.8 * (PI[q]['own'] - PI[q]['sown'])) for q in refs)
lo_ = min(lo_, PHC_OBS) - 0.03
hi_ = max(PI[q][k] + PI[q]['s' + k] for q in refs for k in ('own', 'com')) + 0.03
ax0.set_ylim(lo_, hi_)
ax0.text(-1.1, 1.02, 'a', transform=ax0.transAxes, fontsize=16, fontweight='bold', va='bottom')

# b: decline relative to 1750 over time, each product on its own area (dashed) and on the common area (solid)
ax = fig.add_subplot(top[1])
for q in refs:
    ax.plot(TS[q]['year'], dec(TS[q]['own'], PI[q]['own']), color=rc.COLORS[q], lw=1.0, ls=(0, (3, 2)))
    ax.plot(TS[q]['year'], dec(TS[q]['com'], PI[q]['com']), color=rc.COLORS[q], lw=1.3)
ax.axhline(20, color=BCOL, lw=1.0, ls=(0, (4, 3)))
ax.text(1986, 19.85, 'boundary (20% decline)', ha='left', va='top', fontsize=8.5, color=BCOL)
ax.plot(YEAR, vals[0], marker='*', color='k', ms=10, ls='none', zorder=4)
ax.annotate('Planetary Health\nCheck (2026)', xy=(YEAR, vals[0] - 0.12), xytext=(YEAR - 2, vals[0] - 1.1), ha='right',
            va='center', fontsize=8, arrowprops=dict(arrowstyle='-|>,head_length=0.25,head_width=0.12', color='0.4',
                                                     lw=0.6, shrinkA=2, shrinkB=0))      # label below the line, arrow ends just under the star
ax.set_xlim(1985, YEAR + 1)
ax.set_ylabel('Decline from 1750 (%)', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.text(-0.16, 1.02, 'b', transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom')

# c: steps from the Planetary Health Check to this study; below: which of the three comparisons is consistent
ax = fig.add_subplot(outer[1])
n = len(STEPS)
C_START, C_END, C_DOWN, C_UP = '0.6', '0.15', '#3d6f8e', '#b9cbd6'
BW = 0.5
for i, v in enumerate(vals):
    if i == 0:
        ax.bar(i, v, width=BW, color=C_START, zorder=2)
    elif i == n - 1:
        ax.bar(i, v, width=BW, color=C_END, zorder=2)
    else:
        lo_, hi_ = sorted((vals[i - 1], v))
        ax.bar(i, max(hi_ - lo_, 0.02), bottom=lo_, width=BW, color=C_DOWN if v < vals[i - 1] else C_UP, zorder=2)
    if i > 0:
        ax.plot([i - 1 + BW / 2, i - BW / 2], [vals[i - 1]] * 2, color='0.6', lw=0.6, zorder=1)
    lab = f'{v:.1f}%'
    down = 0 < i < n - 1 and v < vals[i - 1]           # label at the new value: below a decrease, above otherwise
    y_lab = v - 0.12 if down else v + 0.12
    ax.text(i, y_lab, lab, ha='center', va='top' if down else 'bottom', fontsize=9,
            fontweight='bold' if i in (0, n - 1) else 'normal')
ax.axhline(20, color=BCOL, lw=0.9, ls=(0, (4, 3)), zorder=1)
ax.text(n - 0.45, 20.08, 'boundary: 20% decline', ha='right', va='bottom', fontsize=8.5, color=BCOL)
ax.set_xticks(range(n))
ax.set_xticklabels([s_[0] for s_ in STEPS], fontsize=8, linespacing=1.1)
ax.tick_params(axis='x', length=0)
ax.set_xlim(-0.6, n - 0.4)
ax.set_ylim(np.floor(min(vals) - 1.0), np.ceil((max(vals) + 0.4) * 2) / 2)     # top: just above the highest label
ax.set_ylabel(f'Decline of global mean\nsurface {OMEGA} from 1750 (%)', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.text(-0.075, 1.02, 'c', transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom')
# consistency matrix: filled = consistent, open = not; the full assumptions are in Table S2
if CHECKS is not None:
    tr = ax.get_xaxis_transform()
    names = ('same constants', 'same coverage', 'same product', 'CO$_2$ of 1750 (AR6)')
    for r_, name in enumerate(names):
        y_ = -0.3 - 0.1 * r_
        ax.text(-0.6, y_, name, transform=tr, ha='right', va='center', fontsize=9, color='0.25', clip_on=False)
        ax.plot([-0.45, n - 0.55], [y_ - 0.05] * 2, transform=tr, color='0.88', lw=0.6, clip_on=False)
        for i, ck in enumerate(CHECKS):
            ax.plot(i, y_, transform=tr, marker='o', ms=8, mew=1.2, mec=C_DOWN, mfc=C_DOWN if ck[r_] else 'white',
                    clip_on=False, ls='none')
    ax.plot([-0.45, n - 0.55], [-0.25] * 2, transform=tr, color='0.88', lw=0.6, clip_on=False)
    ax.text(n - 0.4, -0.3 - 0.1 * (len(names) - 1) - 0.12, 'filled: the same for today and 1750,  open: different',
            transform=tr, ha='right', va='top', fontsize=8, color='0.4', clip_on=False)
rc.savefig(fig, 'coverage')
plt.show()

# ---------------------------------------------------------------- Table S2: every single change, both sides of the ratio
own180 = rc.to_180(np.asarray(ROWN[STEP_REF]['lon'], 'f8'), np.asarray(ROWN[STEP_REF]['mask']).astype(bool))[1]
A_FULL = pct(cov[STEP_REF]['full'])
P_ = 'UExP-FNN-U'
# constants (K1, K2; total boron B): UExP-FNN-U v2026-1 (Zenodo 10.5281/zenodo.22110725): PyCO2SYS, Lueker, Uppstroem;
# this study: mocsy 2.4.0 with optb='u74'; Jiang et al. (2023): CO2System.jl, Lueker, J_BORON
BNAME = {'l10': 'Lee', 'u74': 'Uppström'}
K_PROD = 'Lueker; B: Uppström'
K_OURS = 'Lueker; B: ' + BNAME[BORON]
K_J = 'Lueker; B: ' + BNAME[J_BORON]
# one row per bar of Fig. 1c: step, then present day (Omega, product, K1 K2, coverage, year) and 1750 (Omega, source,
# K1 K2, coverage, relative to, CO2), and the decline; entries that change from the previous row are set in bold
T_ROWS = [
    ('Planetary Health Check (2026)', PHC_OBS, P_, K_PROD, P_, '2024',
     PHC_PI, 'Jiang et al. (2023)', K_J, 'Jiang et al.', 'Jiang et al.', f'{X_J:.0f}'),
    ('same equilibrium constants for 1750 and present day', o['mocsy'], P_, K_OURS, P_, '2024',
     PHC_PI_U, 'Jiang et al. (2023)', K_OURS, 'Jiang et al.', 'Jiang et al.', f'{X_J:.0f}'),
    ('same ocean coverage for 1750 and present day', o['com24'], P_, K_OURS, 'common', '2024',
     PI_J_COM_U, 'Jiang et al. (2023)', K_OURS, 'common', 'Jiang et al.', f'{X_J:.0f}'),
    ('2024 from the trend fit instead of a single year', now_c, P_, K_OURS, 'common', '2024 (fit)',
     PI_J_COM_U, 'Jiang et al. (2023)', K_OURS, 'common', 'Jiang et al.', f'{X_J:.0f}'),
    ('18 constrained ESMs instead of the 14 ESMs of Jiang et al.', now_c, P_, K_OURS, 'common', '2024 (fit)',
     PI_E18_J, '18 ESMs, constrained', K_OURS, 'common', 'Jiang et al.', f'{X_J:.0f}'),
    ('1750 calculated relative to the product used for present day', now_c, P_, K_OURS, 'common', '2024 (fit)',
     PI_JC, '18 ESMs, constrained', K_OURS, 'common', P_, f'{X_J:.0f}'),
    (f'CO$_2$ in 1750 of {X_1750:.1f} ppm (AR6) instead of {X_J:.0f} ppm', now_c, P_, K_OURS, 'common', '2024 (fit)',
     PI_1750C, '18 ESMs, constrained', K_OURS, 'common', P_, f'{X_1750:.1f}'),
    ('this study (mean of the four products)', now_m, 'four products', K_OURS, 'common', '2024 (fit)',
     yc_m, '18 ESMs, constrained', K_OURS, 'common', 'each product', f'{X_1750:.1f}')]
T_DEC = [dec(r_[1], r_[6]) for r_ in T_ROWS]
T_DEC[-1] = dec_m
assert np.allclose(T_DEC, vals), 'Table S2 and Fig. 1c differ'
GROUPS = [('', 2), ('Present day', 5), ('Preindustrial (1750)', 6), ('', 1)]
HEAD = ['', 'Step (bars of Fig. 1c)', 'Ω$_{arag}$', 'product', 'K$_1$, K$_2$', 'coverage', 'year',
        'Ω$_{arag}$', 'source', 'K$_1$, K$_2$', 'coverage', 'relative to', 'CO$_2$ (ppm)', 'Decline (%)']
CELLS = [[str(k + 1), r_[0], f'{r_[1]:.2f}', r_[2], r_[3], r_[4], r_[5], f'{r_[6]:.2f}', r_[7], r_[8], r_[9], r_[10],
          r_[11], f'{d_:.1f}']
         for k, (r_, d_) in enumerate(zip(T_ROWS, T_DEC))]
NOTES = [
    f'Coverage: {P_} = all cells of {P_} with data in all months ({A_FULL:.1f}% of the ocean); Jiang et al. = the coverage '
    'of their 1° product (without the Mediterranean, Red Sea, Hudson Bay and part of the coasts); common = cells shared '
    f'by the four products and all 18 ESMs ({AREA_C:.1f}%).',
    'Constants: K$_1$, K$_2$ and total boron (B). All use K$_1$, K$_2$ of Lueker et al. (2000). UExP-FNN-U computes '
    'Ω$_{arag}$ with PyCO2SYS and total boron of Uppström (1974) (doi:10.5281/zenodo.22110725). This study recomputes '
    'Ω$_{arag}$ with mocsy 2.4.0 from T, S, DIC and TA (GLODAPv2 Si and PO$_4$), with total boron of '
    + {'l10': 'Lee et al. (2010)', 'u74': 'Uppström (1974)'}[BORON] +
    ', K$_{SO4}$ Dickson (1990), K$_F$ Perez and Fraga (1987) and the aragonite solubility of Mucci (1983). '
    + (f'Jiang et al. (2023) use total boron of {BNAME[J_BORON]}; from step 2 on, their Ω$_{{arag}}$ is converted to '
       f'{BNAME[BORON]} with the ratio of the same states computed with both formulations (×{F_J:.3f} for 1750).'
       if J_BORON != BORON else 'Jiang et al. (2023) use the same constants.'),
    'Relative to: the observations the preindustrial models are bias-adjusted to, so that 1750 = present-day observations '
    'minus the modelled change (Jiang et al.: their gridded SOCAT, GLODAPv2 and CODAP-NA climatology). Step 4: 1750 of '
    f'Jiang et al. replaced by their 2010s state on the common coverage times the ratio 1750 / 2005-2014 of the 18 '
    f'bias-adjusted ESMs for {P_}, i.e. our models relative to their observations. Year: 2024 = '
    'annual mean of 2024; 2024 (fit) = quadratic fit to the annual means, evaluated in 2024. CO$_2$: 272 = effective CO$_2$ of the 1750 value of Jiang et al. (their 1850 state minus the '
    'ice-core change 1752-1852); 278.3 = AR6 for 1750.',
    'Decline = 1 − present day / 1750. Bold: what changes from the previous row. Last row: mean of the four products; '
    'its uncertainty is given in Table S3.']

import csv
with open(rc.figpath('steps_table', ext='csv'), 'w', newline='') as f_:
    wr = csv.writer(f_)
    wr.writerow(['#', 'Step', 'Present Omega', 'Present product', 'Present K1 K2', 'Present coverage', 'Present year',
                 '1750 Omega', '1750 source', '1750 K1 K2', '1750 coverage', '1750 relative to', '1750 CO2 (ppm)',
                 'Decline (%)'])
    wr.writerows([[c_.replace('$', '') for c_ in r_] for r_ in CELLS])
    for n_ in NOTES:
        wr.writerow([n_.replace('$', '')])

# rendered table (booktabs style: rules, no grid, changed entries in bold)
import textwrap
CW = [0.3, 4.2, 0.55, 1.05, 1.5, 0.75, 0.8, 0.55, 1.55, 1.5, 0.85, 1.0, 0.85, 1.05]      # column widths (inches)
RH, X0 = 0.27, 0.15
WIDTH = X0 * 2 + sum(CW)
note_lines = [l_ for n_ in NOTES for l_ in textwrap.wrap(n_, 230)]
HEIGHT = 0.35 + 2 * RH + RH * len(CELLS) + 0.25 + 0.19 * len(note_lines) + 0.15
fig = plt.figure(figsize=(WIDTH, HEIGHT))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WIDTH)
ax.set_ylim(HEIGHT, 0)
ax.axis('off')
xe = X0 + np.concatenate([[0], np.cumsum(CW)])
ALIGN = ['center', 'left'] + ['center'] * 12
xt = lambda j: xe[j] + 0.06 if ALIGN[j] == 'left' else (xe[j] + xe[j + 1]) / 2
y = 0.25
ax.plot([X0, WIDTH - X0], [y, y], color='k', lw=1.2)
j0 = 0
for g_, nspan in GROUPS:                             # group headers with a short rule underneath
    if g_:
        ax.text((xe[j0] + xe[j0 + nspan]) / 2, y + RH / 2, g_, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.plot([xe[j0] + 0.08, xe[j0 + nspan] - 0.08], [y + RH - 0.03] * 2, color='0.3', lw=0.7)
    j0 += nspan
y += RH
for j, h_ in enumerate(HEAD):
    ax.text(xt(j), y + RH / 2, h_, ha=ALIGN[j], va='center', fontsize=9.5, fontweight='bold')
y += RH
ax.plot([X0, WIDTH - X0], [y, y], color='k', lw=0.7)
for k, row in enumerate(CELLS):
    last = k == len(CELLS) - 1
    if last:
        ax.plot([X0, WIDTH - X0], [y, y], color='0.5', lw=0.5)
    for j, c_ in enumerate(row):
        changed = k > 0 and 2 <= j <= 12 and c_ != CELLS[k - 1][j]
        if changed:
            ax.add_patch(plt.Rectangle((xe[j] + 0.02, y + 0.03), CW[j] - 0.04, RH - 0.06, color='#e3edf3', lw=0))
        ax.text(xt(j), y + RH / 2, c_, ha=ALIGN[j], va='center', fontsize=9,
                fontweight='bold' if (changed or last or j == 13) else 'normal',
                color='k' if (changed or j < 2 or j == 13) else '0.4')
    y += RH
ax.plot([X0, WIDTH - X0], [y, y], color='k', lw=1.2)
y += 0.22
for l_ in note_lines:
    ax.text(X0, y, l_, ha='left', va='center', fontsize=8, color='0.25')
    y += 0.19
rc.savefig(fig, 'steps_table')
plt.show()

# ---------------------------------------------------------------- Table S3: uncertainty of the decline (1 sigma)
rc.FIGNAMES.setdefault('uncertainty_table', 'TableS3_uncertainty')
HOW = {K_ESM: 'regression across the 18 bias-adjusted ESMs and the observed predictors (ΔSST, CUC)',
       K_CO2: 'preindustrial states at 278.3 ± 1.8 ppm (AR6 very likely range ±2.9 ppm)',
       K_PROD_D: 'standard deviation across the four products; offsets largely cancel in the decline',
       K_FIT: 'standard error of the 2024 value of the quadratic fit, residuals as AR(1) process',
       K_METH: 'half the range of quadratic fits and Enting splines (20 yr), in time and against CO$_2$',
       K_B: 'half the difference between runs with both formulations (difference = 2σ, as Orr et al., 2018)',
       K_K12: 'half the difference between runs with both formulations (same assumption as for boron)'}
f2 = lambda v: f'{v:.2f}' if np.isfinite(v) else '–'
f3 = lambda v: f'{v:.3f}' if np.isfinite(v) else '–'
T3 = [(k_, f3(ABS[k_][0]), f3(ABS[k_][1]),
       f2(v_) if np.isfinite(v_) else 'after the sensitivity run', HOW.get(k_, '')) for k_, v_ in SIG.items()]
_s50 = float(np.sqrt(np.nansum([a_[0] ** 2 for a_ in ABS.values()])))
_s24 = float(np.sqrt(np.nansum([a_[1] ** 2 for a_ in ABS.values()])))
T3_TOT = [('total (in quadrature)', f3(_s50), f3(_s24), f2(sdec_m), '')]
HEAD3 = ['Component', '1σ of Ω$_{arag}$ 1750', '1σ of Ω$_{arag}$ 2024', '1σ of decline (%)', 'Estimated from']
with open(rc.figpath('uncertainty_table', ext='csv'), 'w', newline='') as f_:
    wr = csv.writer(f_)
    wr.writerow(['Component', '1 sigma Omega_arag 1750', '1 sigma Omega_arag 2024', '1 sigma decline (percentage points)',
                 'Estimated from'])
    wr.writerows([[c_.replace('$', '') for c_ in r_] for r_ in T3 + T3_TOT])
T3 = T3 + T3_TOT
CW3, RH3 = [5.2, 1.6, 1.6, 1.6, 6.3], 0.3
W3, H3 = X0 * 2 + sum(CW3), 0.35 + RH3 * (len(T3) + 1) + 0.8
fig = plt.figure(figsize=(W3, H3))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W3)
ax.set_ylim(H3, 0)
ax.axis('off')
xe3 = X0 + np.concatenate([[0], np.cumsum(CW3)])
y = 0.2
ax.plot([X0, W3 - X0], [y, y], color='k', lw=1.2)
for j, h_ in enumerate(HEAD3):
    ax.text(xe3[j] + 0.06, y + RH3 / 2, h_, ha='left', va='center', fontsize=9.5, fontweight='bold')
y += RH3
ax.plot([X0, W3 - X0], [y, y], color='k', lw=0.7)
for k, row in enumerate(T3):
    tot = k >= len(T3) - len(T3_TOT)
    if k == len(T3) - len(T3_TOT):
        ax.plot([X0, W3 - X0], [y, y], color='0.5', lw=0.5)
    for j, c_ in enumerate(row):
        ax.text(xe3[j] + 0.06, y + RH3 / 2, c_, ha='left', va='center', fontsize=9 if j < 4 else 8.5,
                fontweight='bold' if tot else 'normal', color='k' if j < 4 else '0.3')
    y += RH3
ax.plot([X0, W3 - X0], [y, y], color='k', lw=1.2)
NOTE3 = ('All values 1σ; components combined in quadrature; per-product values as root mean square over the four '
         'products. Systematic errors of K$_2$, K$_{SO4}$, K$_F$ and the aragonite solubility product affect 2024 and 1750 '
         'alike and cancel in the decline.')
for l_ in textwrap.wrap(NOTE3, 200):
    ax.text(X0, y + 0.25, l_, fontsize=8, color='0.25', va='center')
    y += 0.2
rc.savefig(fig, 'uncertainty_table')
plt.show()
# ---------------------------------------------------------------- Fig. S11: estimators of the 2024 state, one figure per product
# a: in time, b: against atmospheric CO2; each with the quadratic fit and the Enting spline (20 yr, in CO2 the equivalent
# cutoff); dots: annual means on the common coverage; insets: the last 15 years
rc.FIGNAMES.setdefault('trend_fits', 'FigS11_trend_fits')
co = ENT_CUTOFFS[0]
for q in refs:
    e, c = EST[q], rc.COLORS[q]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for k_, (ax_, xk, tag, ttl) in enumerate(((axs[0], 'yrs', 'time', 'in time'),
                                              (axs[1], 'xco2', 'CO2', 'against atmospheric CO$_2$'))):
        lines = ((f'fit {tag} quadratic', 'quadratic fit', 'k', '-'),
                 (f'fit {tag} Enting {co:.0f} yr', f'Enting spline ({co:.0f} yr)' if tag == 'time'
                  else f'Enting spline ({co * e["gr"]:.0f} ppm, {co:.0f} yr)', '0.45', (0, (4, 2))))
        xs = e[xk]
        ax_.plot(xs, e['y'], 'o', ms=3.5, color=c, alpha=0.7, zorder=1, label='annual mean')
        for kk, lab_, col, ls_ in lines:
            ax_.plot(xs, e[kk], color=col, lw=1.3, ls=ls_, label=lab_)
        ax_.set_title(f'Fits {ttl}', fontsize=10)
        ax_.text(-0.12 if k_ == 0 else -0.06, 1.03, 'ab'[k_], transform=ax_.transAxes, fontsize=14, fontweight='bold')
        ax_.spines[['top', 'right']].set_visible(False)
        ax_.legend(fontsize=8, frameon=False, loc='upper right')
        ins = ax_.inset_axes([0.08, 0.08, 0.4, 0.4])
        sel = e['yrs'] >= YEAR - 14
        ins.plot(xs[sel], e['y'][sel], 'o', ms=3, color=c, alpha=0.7)
        for kk, lab_, col, ls_ in lines:
            ins.plot(xs[sel], e[kk][sel], color=col, lw=1.2, ls=ls_)
        ins.tick_params(labelsize=7)
        ins.set_title('last 15 years', fontsize=7)
    axs[0].set_ylabel(f'Global mean surface {OMEGA}', fontsize=10)
    axs[0].set_xlabel('Year')
    axs[1].set_xlabel('Atmospheric CO$_2$ (ppm, NOAA global annual mean)')
    fig.suptitle(f'{rc.FULL[q]}, 2024: ' + ', '.join(f'{k} {v:.3f}' for k, v in e.items() if isinstance(v, float)
                                                    and k[:4] in ('time', 'CO2 ')) + f', annual mean {e["y"][-1]:.3f}',
                 fontsize=9)
    fig.tight_layout()
    rc.savefig(fig, 'trend_fits', suffix=q)
    plt.show()
