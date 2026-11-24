#!/usr/bin/env python
"""
Uncertainty of the 2024 global-mean surface Omega_arag of an observation-based product from its
mapping uncertainties, with spatially correlated errors (Ford et al. 2024, GBC). Each uncertainty
component is a random field with an exponential spatial correlation (decorrelation length L),
scaled by the local 1 sigma of the product. Dissociation constants are held fixed.
As Ford et al., the length is perturbed within its uncertainty in every draw: for every draw and component one
factor f of oa_config.L_FACTORS (log-uniform between 1/2 and 2, equally probable) is drawn and the field has the
length f L; the 12 monthly fields of a draw (C1) share it. The result therefore includes the uncertainty of L.

C1: fields independent between months (as Ford et al. 2024)
C2: same field in all twelve months (conservative: persistent regional biases)
A missing sigma in a valid cell-month is filled with the monthly area mean of that component (as level_uncertainty.py).

Products
  UExP       fCO2 and TA: net (1800 km), para (2400 km), val (2400 km), 2 sigma in the file;
             SST, SSS: linear Omega components (2400 km). fCO2/TA propagated through mocsy.
  OceanSODA  sfco2_uncert, talk_uncert (RMSE-based totals, 1 sigma; 2400 km assumed),
             propagated through mocsy.
  JMA        no uncertainty fields: OceanSODA-ETHZ sfco2_uncert and talk_uncert used as proxy
             (both are regression-based estimates), propagated through mocsy.
  CMEMS      its own pCO2 and TA uncertainties (spco2_uncertainty, talk_uncertainty, 1 sigma; 2400 km assumed) on the
             1 degree grid of the chain (make_cmems_unc.py; T OSTIA and S CNR as in the product), propagated through
             mocsy; SST, SSS: the UExP-FNN-U Omega components as proxy. Not its omega_ar_uncertainty: that total
             includes the equilibrium constants, which are assessed separately (product_constants_uncertainty.py).

Usage:  python -u product_uncertainty_mc_C.py PRODUCT [NDRAW] [LFAC]
        PRODUCT in UExP, OceanSODA, JMA, CMEMS; LFAC scales all central decorrelation lengths (default 1)
Output: data/<run>/product_uncertainty/mcC_<PRODUCT>_2024_L<LFAC>_r<L_RANGE>_n<NDRAW>.npz
"""
import sys
import numpy as np
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator

import oa_config
import calc_omega_arag_obs as co
import mocsy
mvars = mocsy.mvars

PRODUCT = sys.argv[1]
NDRAW = int(sys.argv[2]) if len(sys.argv) > 2 else 100
LFAC = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
YEAR = 2024
SEED = 2
COARSE = 4.0
R_EARTH = 6371.
OPT = dict(optcon='mol/kg', optt='Tpot', optp='m',
           optb=oa_config.OPTB, optk1k2=oa_config.OPTK1K2)
OBS = str(oa_config.OBS_DIR) + '/'
FILES = dict(
    UExP=OBS + 'UExP-FNN-U/Fordetal_UExP-FNN-U_surface-carbonate-system_v2026-1.nc',
    OceanSODA=OBS + 'OceanSODA_ETHZ-v2025/0220059/6.6/data/0-data/OceanSODA_ETHZ-v2025.OCADS.01-1982-2024.nc',
    JMA=OBS + 'JMA-MLR_v2025/JMA_co2map_1990-2024_v2025.nc',
    JMA_TS=OBS + 'JMA-MLR_v2025/JMA_co2map_TS_1990-2024_v2025.nc',
    CMEMS=OBS + 'CMEMS-LSCE-FFNN/CMEMS-LSCE-FFNN_1deg_1985-2025.nc',                 # make_cmems_ts.py
    CMEMS_UNC=OBS + 'CMEMS-LSCE-FFNN/CMEMS-LSCE-FFNN_1deg_uncertainty_1985-2025.nc',   # make_cmems_unc.py
)


# ------------------------------------------------------------------ reading
def months_of_year(ds):
    tv = ds['time']
    units = tv.units
    if units.lower().startswith('months since'):                    # JMA
        y0 = int(units.split('since')[1].strip()[:4])
        y = y0 + np.asarray(tv[:], 'i8') // 12
        return np.where(y == YEAR)[0]
    if units.lower().startswith('days since 1970-01-15'):          # UExP
        d = np.datetime64('1970-01-15') + tv[:].astype('i8').astype('timedelta64[D]')
        y = d.astype('datetime64[Y]').astype(int) + 1970
        return np.where(y == YEAR)[0]
    dt = nc.num2date(tv[:], units, getattr(tv, 'calendar', 'standard'))
    return np.array([i for i, x in enumerate(dt) if x.year == YEAR])


def rd(ds, name, it):
    """12 months of a variable as (12, lat, lon) float64, NaN for missing"""
    v = ds[name]
    dims = list(v.dimensions)
    sl = tuple(slice(int(it[0]), int(it[-1]) + 1) if d == 'time' else slice(None) for d in dims)
    a = np.ma.filled(np.ma.asarray(v[sl]).astype('f8'), np.nan)
    lat_d = [d for d in dims if d.startswith('lat')][0]
    lon_d = [d for d in dims if d.startswith('lon')][0]
    a = np.transpose(a, [dims.index('time'), dims.index(lat_d), dims.index(lon_d)])
    a[np.abs(a) > 1e10] = np.nan
    return a


def coord(ds, *names):
    for k in names:
        if k in ds.variables:
            return np.asarray(ds[k][:], 'f8')
    raise KeyError(names)


ds = nc.Dataset(FILES[PRODUCT])
it = months_of_year(ds)
assert it.size == 12, f'{PRODUCT}: {it.size} months in {YEAR}'
print(f'{PRODUCT}: {ds.filepath()}, months {it[0]}-{it[-1]}')
lat = coord(ds, 'lat', 'latitude')
lon = coord(ds, 'lon', 'longitude') % 360.

UNC_FILE = None                       # uncertainties from another file (JMA: none of its own)
if PRODUCT == 'UExP':
    V = dict(t='subskin_temp', s='subskin_salinity', ta='ta', dic='dic')
    NONLIN = [('fco2', 'fco2_net_unc', 1800.), ('fco2', 'fco2_para_unc', 2400.), ('fco2', 'fco2_val_unc', 2400.),
              ('ta', 'ta_net_unc', 1800.), ('ta', 'ta_para_unc', 2400.), ('ta', 'ta_val_unc', 2400.)]
    LIN = [('saturation_aragonite_sst_unc', 2400.), ('saturation_aragonite_sss_unc', 2400.)]
    SIGFAC = 0.5                                                     # file gives 2 sigma
elif PRODUCT in ('OceanSODA', 'JMA'):
    V = dict(t='temperature', s='salinity', ta='talk', dic='dic')
    NONLIN = [('fco2', 'sfco2_uncert', 2400.), ('ta', 'talk_uncert', 2400.)]
    LIN = []
    SIGFAC = 1.0                                                     # RMSE-based, 1 sigma
    if PRODUCT == 'JMA':
        V = dict(ta='ta', dic='dic')
        UNC_FILE = FILES['OceanSODA']                                # OceanSODA fields as proxy
else:                                                                # CMEMS: its own fields, 1 degree grid of the chain
    V = dict(t='sst', s='sss', ta='ta', dic='dic')
    NONLIN = [('fco2', 'fco2_unc', 2400.), ('ta', 'ta_unc', 2400.)]
    LIN = []
    SIGFAC = 1.0                                                     # standard deviation
    UNC_FILE = FILES['CMEMS_UNC']

raw = {k: rd(ds, n, it) for k, n in V.items()}
if UNC_FILE is None:
    unc = {n: rd(ds, n, it) for _, n, _ in NONLIN}
    unc.update({n: rd(ds, n, it) for n, _ in LIN})
ds.close()

if PRODUCT == 'CMEMS':
    with nc.Dataset(UNC_FILE) as du:
        iu = months_of_year(du)
        if not (np.allclose(coord(du, 'lat', 'latitude'), lat) and np.allclose(coord(du, 'lon', 'longitude') % 360., lon)):
            raise SystemExit(f'{UNC_FILE}: grid differs from {FILES["CMEMS"]}')
        unc = {n: rd(du, n, iu) for _, n, _ in NONLIN}
    print(f'  uncertainties: CMEMS-LSCE-FFNN pCO2 and TA ({UNC_FILE})')

if PRODUCT == 'JMA':
    with nc.Dataset(FILES['JMA_TS']) as dt:
        its = months_of_year(dt)
        raw['t'], raw['s'] = rd(dt, 'sst', its), rd(dt, 'sss', its)
    with nc.Dataset(UNC_FILE) as du:
        iu = months_of_year(du)
        ulat = coord(du, 'lat', 'latitude')
        ulon = coord(du, 'lon', 'longitude') % 360.
        jy = np.abs(lat[:, None] - ulat[None, :]).argmin(1)
        ix = np.abs(((lon[:, None] - ulon[None, :]) + 180) % 360 - 180).argmin(1)
        unc = {n: rd(du, n, iu)[:, jy][:, :, ix] for _, n, _ in NONLIN}
    print(f'  uncertainties: OceanSODA-ETHZ fields as proxy ({UNC_FILE})')

# SST and SSS: only UExP-FNN-U gives their effect on Omega; used as proxy for OceanSODA, JMA and CMEMS
if PRODUCT in ('OceanSODA', 'JMA', 'CMEMS'):
    LIN = [('saturation_aragonite_sst_unc', 2400.), ('saturation_aragonite_sss_unc', 2400.)]
    with nc.Dataset(FILES['UExP']) as du:
        iu = months_of_year(du)
        ulat = coord(du, 'lat', 'latitude')
        ulon = coord(du, 'lon', 'longitude') % 360.
        jy = np.abs(lat[:, None] - ulat[None, :]).argmin(1)
        ix = np.abs(((lon[:, None] - ulon[None, :]) + 180) % 360 - 180).argmin(1)
        for n_, _ in LIN:
            unc[n_] = rd(du, n_, iu)[:, jy][:, :, ix] * 0.5 / SIGFAC    # UExP: 2 sigma
    print('  SST, SSS: UExP-FNN-U Omega components as proxy')

ok = np.ones(next(iter(raw.values())).shape, bool)
for a in raw.values():
    ok &= np.isfinite(a)
IM, IY, IX = np.nonzero(ok)
LAT, LON = lat[IY], lon[IX]
W = oa_config.area_weights(lat, lon)[IY, IX]            # the weights of the run (OA_WEIGHTS)
n = IM.size
print(f'{n} valid cell-months, grid {lat.size} x {lon.size}')
f = {k: a[ok] for k, a in raw.items()}
# a missing sigma in a valid cell-month (sea-ice edge, coasts, cells where the proxy field has no value) is filled with
# the area-weighted mean sigma of the same component and month, as in level_uncertainty.py; setting it to zero would
# treat these cells as error-free (review A7)
_w2 = oa_config.area_weights(lat, lon)
for k_, a_ in unc.items():
    filled = 0
    for m_ in range(a_.shape[0]):
        miss = ok[m_] & ~np.isfinite(a_[m_])
        good = ok[m_] & np.isfinite(a_[m_])
        if miss.any() and good.any():
            a_[m_][miss] = np.sum(a_[m_][good] * _w2[good]) / _w2[good].sum()
            filled += int(miss.sum())
    print(f'  {k_}: {100 * filled / max(int(ok.sum()), 1):.2f} % of the valid cell-months without sigma, filled with the '
          f'monthly area mean', flush=True)
SIG = {k: np.nan_to_num(a[ok]) * SIGFAC for k, a in unc.items()}

REG = {
    'global': np.ones(n, bool),
    'tropics (15S-15N)': np.abs(LAT) < 15,
    'subtropics (15-45)': (np.abs(LAT) >= 15) & (np.abs(LAT) < 45),
    'Southern Ocean (<45S)': LAT < -45,
    'subpolar N Pacific (45-65N)': (LAT >= 45) & (LAT < 65) & (LON > 140) & (LON < 240),
    'subpolar N Atlantic (45-65N)': (LAT >= 45) & (LAT < 65) & ((LON > 290) | (LON < 20)),
    'Arctic (>65N)': LAT >= 65,
}
SEL = {(r, k): np.where(m & (IM == k))[0] for r, m in REG.items() for k in range(12)}


def regmeans(om):
    res = {}
    for r in REG:
        mm = []
        for k in range(12):
            s = SEL[(r, k)]
            if s.size == 0:
                continue
            v = np.isfinite(om[s])
            mm.append(np.sum(om[s][v] * W[s][v]) / np.sum(W[s][v]))
        res[r] = np.mean(mm) if mm else np.nan
    return res


# ------------------------------------------------------------------ chemistry
if NONLIN:
    sil2d, po42d = co.load_nutrients(lat, lon)                       # (lat, lon) umol/kg
    SIL = np.clip(np.nan_to_num(sil2d)[IY, IX], 0, None) * 1e-6      # GLODAP has small negative values
    PHOS = np.clip(np.nan_to_num(po42d)[IY, IX], 0, None) * 1e-6
    T = f['t'] - 273.15 if np.nanmean(f['t']) > 100 else f['t']
    S = f['s']
    TA0, DIC0 = f['ta'] * 1e-6, f['dic'] * 1e-6
    for _, k, _ in NONLIN:
        if k.startswith('ta') or k.startswith('talk'):
            SIG[k] = SIG[k] * 1e-6
    ZERO, PATM = np.zeros(n), np.ones(n)

    def carb(ta, dic):
        out = mvars(temp=T, sal=S, alk=ta, dic=dic, sil=SIL, phos=PHOS,
                    patm=PATM, depth=ZERO, lat=LAT, **OPT)
        om = np.asarray(out[6], float)
        return np.asarray(out[2], float), np.where((om > 0) & (om < 20), om, np.nan)

    def solve(ta, ftarget, niter=4):
        dic = DIC0.copy()
        with np.errstate(divide='ignore', invalid='ignore'):
            for _ in range(niter):
                f0, _ = carb(ta, dic)
                f1, _ = carb(ta, dic + 1e-6)
                der = (f1 - f0) / 1e-6
                step = np.where(np.isfinite(der) & (der > 0), (ftarget - f0) / der, 0.)
                dic = dic + np.clip(np.nan_to_num(step), -5e-4, 5e-4)
        fc, om = carb(ta, dic)
        return np.where(np.abs(fc - ftarget) < 0.5, om, np.nan)

    F0, OM0 = carb(TA0, DIC0)
    good0 = np.isfinite(F0) & np.isfinite(OM0)
    OM0 = np.where(good0, OM0, np.nan)
    print(f'  cells where mocsy fails in the base state: {np.sum(~good0)} of {n}')
else:
    OM0 = f['om']
base = regmeans(OM0)
print(f'base global mean {base["global"]:.4f}')

# ------------------------------------------------------------------ correlated random fields
clat = np.arange(-90 + COARSE / 2, 90, COARSE)
clon = np.arange(COARSE / 2, 360, COARSE)
GLA, GLO = np.meshgrid(np.deg2rad(clat), np.deg2rad(clon), indexing='ij')
xyz = np.stack([np.cos(GLA) * np.cos(GLO), np.cos(GLA) * np.sin(GLO), np.sin(GLA)], -1).reshape(-1, 3)
dist = R_EARTH * np.arccos(np.clip(xyz @ xyz.T, -1, 1))
LS = sorted({l * LFAC * f for _, _, l in NONLIN for f in oa_config.L_FACTORS}
            | {l * LFAC * f for _, l in LIN for f in oa_config.L_FACTORS})
# one Cholesky factor per length (2 central lengths x len(L_FACTORS)), kept in single precision (about 65 MB each)
CHOL = {l: np.linalg.cholesky(np.exp(-dist / l) + 1e-8 * np.eye(dist.shape[0])).astype('f4') for l in LS}
print(f'  {len(LS)} decorrelation lengths: {", ".join(f"{l:.0f}" for l in LS)} km')
plon = np.concatenate([clon[-1:] - 360, clon, clon[:1] + 360])
plat = np.concatenate([[-90.], clat, [90.]])
QA, QO = np.meshgrid(lat, lon, indexing='ij')
QPTS = np.stack([QA.ravel(), QO.ravel()], -1)                       # (lat, lon) target


def field(rng, l):
    z = (CHOL[l] @ rng.standard_normal(dist.shape[0])).reshape(clat.size, clon.size)
    z = np.concatenate([z[:, -1:], z, z[:, :1]], 1)
    z = np.concatenate([np.full((1, z.shape[1]), z[0].mean()), z, np.full((1, z.shape[1]), z[-1].mean())], 0)
    return RegularGridInterpolator((plat, plon), z)(QPTS).reshape(lat.size, lon.size)


rng0 = np.random.default_rng(99)
NORM = {}
for l in LS:
    s2 = np.zeros((lat.size, lon.size))
    for _ in range(100):
        s2 += field(rng0, l)**2
    NORM[l] = 1 / np.sqrt(s2 / 100)
    print(f'  L={l:.0f} km: mean interpolation std {np.mean(np.sqrt(s2 / 100)):.3f} (normalised to 1)')


def sample(rng, l, per_month):
    if per_month:
        return np.stack([field(rng, l) * NORM[l] for _ in range(12)])[IM, IY, IX]
    return (field(rng, l) * NORM[l])[IY, IX]


def lfac(rng):
    """the factor of the decorrelation length of one component in one draw (oa_config.L_FACTORS, equally probable)"""
    return oa_config.L_FACTORS[rng.integers(len(oa_config.L_FACTORS))]


# ------------------------------------------------------------------ Monte Carlo
res = {}
for v, per_month in [('C1', True), ('C2', False)]:
    rng = np.random.default_rng(SEED)
    out = {r: [] for r in REG}
    for i in range(NDRAW):
        if NONLIN:
            dF = sum(sample(rng, l * LFAC * lfac(rng), per_month) * SIG[k] for t, k, l in NONLIN if t == 'fco2')
            dTA = sum(sample(rng, l * LFAC * lfac(rng), per_month) * SIG[k] for t, k, l in NONLIN if t == 'ta')
            om = solve(TA0 + dTA, F0 + dF)
        else:
            om = OM0.copy()
        for k, l in LIN:
            om = om + sample(rng, l * LFAC * lfac(rng), per_month) * SIG[k]
        r = regmeans(om)
        for q in REG:
            out[q].append(r[q] - base[q])
        if (i + 1) % 10 == 0:
            print(f'  {v}: {i + 1}/{NDRAW}  running global 1 sigma {np.std(out["global"]):.4f}', flush=True)
    res[v] = {q: np.array(x) for q, x in out.items()}

print(f'\n{PRODUCT}, L x {LFAC} (each length log-uniform within a factor {oa_config.L_RANGE:g}), {NDRAW} draws: '
      f'change of the {YEAR} annual mean Omega_arag')
for v in res:
    print(f'  {v} ({"months independent" if v == "C1" else "same pattern all months"})')
    print(f'    {"region":30s} {"base":>6s} {"1 sigma":>8s} {"p2.5":>8s} {"p97.5":>8s}')
    for q in REG:
        x = res[v][q]
        p = np.percentile(x, [2.5, 97.5])
        print(f'    {q:30s} {base[q]:6.3f} {x.std():8.4f} {p[0]:+8.4f} {p[1]:+8.4f}')

OUTD = oa_config.DATA / 'product_uncertainty'
OUTD.mkdir(parents=True, exist_ok=True)
OUTF = OUTD / f'mcC_{PRODUCT}_{YEAR}_L{LFAC:g}_r{oa_config.L_RANGE:g}_n{NDRAW}.npz'
np.savez(OUTF, **{f'{v}_{q}': res[v][q] for v in res for q in REG},
         base=np.array([base[q] for q in REG]), regions=np.array(list(REG)))
print(f'\nsaved {OUTF}')
