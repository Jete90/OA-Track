#!/usr/bin/env python
"""
level_uncertainty.py
====================
The level uncertainty of the data product that comes from the reference itself, per cell and for the area means,
stored separately from the constraint uncertainty (calc_product_obs.py for the product years,
calc_product_states.py for the model years, make_product.py carries both):

mapping    the mapping uncertainty of the observation-based product (Ford et al. 2024, GBC). Every uncertainty
           component c of the product is an error field e_c with the local 1 sigma of the product and an
           exponential spatial correlation exp(-d / L_c); components are independent of each other.
             UExP-FNN-U   fCO2 and TA: net (L 1800 km), para (2400 km), val (2400 km), the file gives 2 sigma;
                          SST and SSS: the Omega_arag components of the file (2400 km, 2 sigma)
             OceanSODA    sfco2_uncert and talk_uncert (1 sigma, RMSE based; 2400 km assumed);
                          SST and SSS: the UExP-FNN-U components as proxy
             CMEMS-LSCE-FFNN  its own pCO2 and TA uncertainties (1 sigma, 2400 km assumed; make_cmems_unc.py,
                          fCO2 = 0.9965 pCO2), SST and SSS as for OceanSODA (its total Omega_arag uncertainty is
                          not used: it includes the equilibrium constants, assessed separately)
             JMA-MLR, Jiang et al. (2023): the OceanSODA-ETHZ fCO2 and TA fields as proxy (no fields of their own),
                          SST and SSS as for OceanSODA
           The errors are given in fCO2, TA, T and S (fCO2 and TA mapped, DIC derived, as in these products). They
           are turned into errors of the state (DIC, TA, T, S) at fixed fCO2 with the derivatives of mocsy at the
           product state (finite differences), and every indicator follows from the derivatives at the state it
           is evaluated at:  e_q = sum_x dq/dx * dX_x.  The SST and SSS components are Omega_arag errors; they are
           converted to T and S errors with dOmega_arag/dT and dOmega_arag/dS at fixed fCO2 and TA.
           Annual mean of the monthly errors, two limits:
             C1  errors independent between months: sigma = sqrt(sum_m sum_c e_cm^2) / 12
             C2  the same error in all twelve months: sigma = sqrt(sum_c (sum_m e_cm)^2) / 12  (conservative)
           Area means: quadratic form of the box sums (equal-area boxes of about 440 km: 4 degree latitude bands,
           4 degree / cos(lat) wide; the error fully correlated inside a box) with the correlation between the box
           centres (great circle). The lengths L_c are uncertain (oa_config.L_RANGE: log-uniform between L_c / 2 and
           2 L_c, as Ford et al. perturb their lengths in every draw), so the correlation is the expected one,
           the mean of exp(-d / (f L_c)) over the factors f of oa_config.L_FACTORS; for a zero-mean error this gives
           the variance averaged over the distribution of the length. The per-cell sigmas do not depend on L.
           Product years: the errors of that year. Model years: the state is bias adjusted to the product over
           its window W (calc_bias_adjusted.py), so the product's error of the window enters every model year as
           a fixed offset of the state: the window mean of the state errors dX_c (per month), propagated with the
           derivatives at the constrained state of the year (fully correlated between years).
           Cells where the product has a value but a component has no sigma get the area-weighted mean sigma of
           that component and month (reported); years outside a source are taken from its nearest year.
constants  the uncertainty of the dissociation constants and of total boron (Orr et al. 2018: pK0 0.002,
           pK1 0.0075, pK2 0.015, pKb 0.01, pKw 0.01, pKspa 0.02, pKspc 0.02, total boron 2 %), mocsy.merrors on
           the annual-mean state; fully correlated in space and time, so the area mean is the area mean of the
           cell sigma. Not provided by merrors: the Revelle factor (NaN); DIC, TA, T and S have none.
"""
from __future__ import annotations

import numpy as np
import netCDF4

import oa_config
import calc_bias_adjusted as cb
import calc_omega_arag_obs as co
import calc_omega_arag_products as cp
import calc_pi_xco2 as px
from make_jma_ts import year_month

VARS = cb.VARS                                              # thetao, so, dissic, talk
QUANTITIES = list(px.INDICATOR_NAMES) + ['thetao', 'so', 'talk']
DX = ('dissic', 'talk', 'thetao', 'so')                     # the state variables the errors are expressed in
STEP = {'dissic': 1.0, 'talk': 1.0, 'thetao': 0.01, 'so': 0.01}
BOX = 4.0
R_EARTH = 6371.0
EPK = np.array([0.002, 0.0075, 0.015, 0.01, 0.01, 0.02, 0.02])     # pK0, pK1, pK2, pKb, pKw, pKspa, pKspc
EBT = 0.02
SOURCES = {'UExP': dict(file=cp.PRODUCTS['UExP-FNN-U-v2026-1']['file'], order=cp.PRODUCTS['UExP-FNN-U-v2026-1']['order']),
           'OceanSODA': dict(file=cb.REFERENCES['OceanSODA']['file'], order=cb.REFERENCES['OceanSODA']['order']),
           'CMEMS': dict(file=cp.OBS / 'CMEMS-LSCE-FFNN' / 'CMEMS-LSCE-FFNN_1deg_uncertainty_1985-2025.nc',
                         order='time,lat,lon')}                              # make_cmems_unc.py
# component: (name, kind, source, variable, L in km, factor to 1 sigma)
_TS = [('sst', 'omega_t', 'UExP', 'saturation_aragonite_sst_unc', 2400.0, 0.5),
       ('sss', 'omega_s', 'UExP', 'saturation_aragonite_sss_unc', 2400.0, 0.5)]
_OS = [('fco2', 'fco2', 'OceanSODA', 'sfco2_uncert', 2400.0, 1.0),
       ('ta', 'ta', 'OceanSODA', 'talk_uncert', 2400.0, 1.0)]
COMPONENTS = {
    'UExP': [(f'{k}_{p}', k, 'UExP', f'{k}_{p}_unc', L, 0.5) for k in ('fco2', 'ta')
             for p, L in (('net', 1800.0), ('para', 2400.0), ('val', 2400.0))] + _TS,
    'OceanSODA': _OS + _TS, 'JMA': _OS + _TS, 'Jiang': _OS + _TS,
    'CMEMS': [('fco2', 'fco2', 'CMEMS', 'fco2_unc', 2400.0, 1.0), ('ta', 'ta', 'CMEMS', 'ta_unc', 2400.0, 1.0)] + _TS,
}
CONST_INDEX = {'omega_arag': 6, 'omega_calc': 7, 'pco2': 1, 'fco2': 2, 'co2': 3, 'hco3': 4, 'co3': 5}
CONST_SCALE = {'co2': 1e6, 'hco3': 1e6, 'co3': 1e6}


# ================================================================================================ chemistry
def chemistry(fields, si, po4, lat2):
    """{var: (k, ny, nx)} -> {q: (k, ny, nx)} for QUANTITIES, the state itself (no CO2 target)"""
    _, res = next(px.target_states(fields, si, po4, lat2, {'s': px.XCO2_MODEL}, [0.0]))
    for v in ('thetao', 'so', 'talk'):
        res[v] = np.where(np.isfinite(res['omega_arag']), fields[v], np.nan)
    return res


def jacobian(fields, si, po4, lat2):
    """state {var: (ny, nx)} -> base {q: (ny, nx)}, J {q: {x: dq/dx (ny, nx)}} (forward differences)"""
    stack = {v: np.stack([fields[v]] + [fields[v] + (STEP[x] if x == v else 0.0) for x in DX]) for v in VARS}
    r = chemistry(stack, si, po4, lat2)
    base = {q: r[q][0] for q in QUANTITIES}
    J = {q: {x: (r[q][i + 1] - r[q][0]) / STEP[x] for i, x in enumerate(DX)} for q in QUANTITIES}
    return base, J


def obs_to_state(J, kind, sigma):
    """an error `sigma` of the product in fCO2 (uatm), TA (umol/kg, fixed fCO2) or Omega_arag through T or S
    (fixed fCO2 and TA) -> the error of the state {x: (ny, nx)} (DIC, TA, T, S)"""
    fD = J['fco2']['dissic']
    with np.errstate(divide='ignore', invalid='ignore'):
        if kind == 'fco2':
            return {'dissic': sigma / fD}
        if kind == 'ta':
            return {'talk': sigma, 'dissic': -sigma * J['fco2']['talk'] / fD}
        x = {'omega_t': 'thetao', 'omega_s': 'so'}[kind]
        dodx = J['omega_arag'][x] - J['omega_arag']['dissic'] * J['fco2'][x] / fD     # dOmega/dx at fixed fCO2, TA
        sx = sigma / np.abs(dodx)
        sx = np.where(np.isfinite(sx), sx, 0.0)
        return {x: sx, 'dissic': -sx * J['fco2'][x] / fD}
    raise ValueError(kind)


def propagate(J, dX):
    """{q: sum_x dq/dx dX_x}"""
    out = {}
    for q in QUANTITIES:
        e = 0.0
        for x, d in dX.items():
            e = e + J[q][x] * d
        out[q] = e
    return out


def constants_sigma(fields, si, po4, lat2):
    """1 sigma from the dissociation constants and total boron (mocsy.merrors) of the state {var: (ny, nx)}"""
    import mocsy
    t, s, dic, alk = (fields[v] for v in VARS)
    ok = np.isfinite(t) & np.isfinite(s) & np.isfinite(dic) & np.isfinite(alk) & np.isfinite(si) & np.isfinite(po4) & (dic > 0) & (alk > 0)
    out = {q: np.full(t.shape, np.nan) for q in QUANTITIES}
    for q in ('dic', 'talk', 'thetao', 'so'):
        out[q] = np.where(ok, 0.0, np.nan)
    if not ok.any():
        return out
    n = int(ok.sum())
    z = np.zeros(n)
    args = (co.cm.floor_temperature(t[ok], s[ok]), s[ok], alk[ok] * 1e-6, dic[ok] * 1e-6, np.clip(si[ok], 0, None) * 1e-6,
            np.clip(po4[ok], 0, None) * 1e-6, np.ones(n), z, lat2[ok])
    r = mocsy.merrors(*args, z, z, z, z, z, z, 'mol/kg', 'Tpot', 'm', optb=oa_config.OPTB, optk1k2=oa_config.OPTK1K2,
                      epk=EPK, ebt=EBT)
    st = px.mvars_all(t[ok], s[ok], dic[ok] * 1e-6, alk[ok] * 1e-6, args[4], args[5], lat2[ok])
    h = 10.0 ** (-np.asarray(st[0], float))
    eh = np.asarray(r[0], float)
    vals = {q: np.asarray(r[i], float) * CONST_SCALE.get(q, 1.0) for q, i in CONST_INDEX.items()}
    vals['hplus'] = eh * 1e9
    vals['hplus_free'] = eh * 1e9 * px.total_to_free(t[ok], s[ok])
    vals['ph'] = eh / (h * np.log(10.0))
    for q, v in vals.items():
        out[q][ok] = np.where(np.isfinite(v) & (np.abs(v) < 1e10), v, np.nan)
    return out


# ================================================================================================ area means
class BoxCov:
    """expected exp(-d/(f L)) correlation (mean over oa_config.L_FACTORS) between equal-area boxes (4 degree latitude bands, 4 degree / cos(lat) wide, about
    440 km), the boxes that contain cells of the grid; box sums of a weighted field with a sparse operator, the
    variance of the sum of the boxes var = E' K E for many E at once"""
    _cache = {}

    def __init__(self, lat, lon, L, active=None):
        lat, lon = np.asarray(lat, 'f8'), np.asarray(lon, 'f8')
        nband = int(180 / BOX)
        band_lat = -90 + BOX * (np.arange(nband) + 0.5)
        nlon = np.maximum(1, np.round(360 / BOX * np.cos(np.deg2rad(band_lat)))).astype(int)
        off = np.concatenate([[0], np.cumsum(nlon)])
        jb = np.clip(((lat + 90.0) // BOX).astype(int), 0, nband - 1)
        LAT, LON = np.meshgrid(jb, lon % 360.0, indexing='ij')
        ib = np.minimum((LON / 360.0 * nlon[LAT]).astype(int), nlon[LAT] - 1)
        box_all = (off[LAT] + ib).ravel()
        act = np.unique(box_all) if active is None else np.unique(box_all[np.asarray(active).ravel()])
        index = -np.ones(off[-1], int)
        index[act] = np.arange(act.size)
        self.box = index[box_all]                                  # -1: cell in no active box
        self.nb = act.size
        key = (L, oa_config.L_FACTORS, act.tobytes())
        if key not in BoxCov._cache:
            band = np.searchsorted(off, act, side='right') - 1
            clat = np.deg2rad(band_lat[band])
            clon = np.deg2rad((act - off[band] + 0.5) * 360.0 / nlon[band])
            xyz = np.stack([np.cos(clat) * np.cos(clon), np.cos(clat) * np.sin(clon), np.sin(clat)], -1)
            d = R_EARTH * np.arccos(np.clip(xyz @ xyz.T, -1, 1))
            K = np.mean([np.exp(-d / (f * L)) for f in oa_config.L_FACTORS], axis=0)   # expected correlation
            np.fill_diagonal(K, 1.0)
            BoxCov._cache[key] = K
        self.K = BoxCov._cache[key]

    def operator(self, weight):
        """sparse (nb, ncell) operator: box sums of weight * field"""
        from scipy.sparse import csr_matrix
        w = np.asarray(weight, 'f8').ravel()
        sel = (self.box >= 0) & (w != 0)
        cols = np.nonzero(sel)[0]
        return csr_matrix((w[sel], (self.box[sel], cols)), shape=(self.nb, w.size))

    def sums(self, field, weight):
        """box sums of weight * field (NaN as 0) -> (nb,)"""
        a = np.where(np.isfinite(field), field, 0.0)
        return self.operator(weight) @ a.ravel()

    def var(self, E):
        """E (nb,) or (nb, k) -> sum over the columns of E' K E"""
        E = np.asarray(E, 'f8')
        return float(np.sum(E * (self.K @ E)))

    def var_cols(self, E):
        """E (nb, k) -> (k,) E_j' K E_j"""
        return np.sum(E * (self.K @ E), axis=0)


class LevelYear:
    """the level uncertainty of one year: monthly mapping errors (per component) and the constants, annual result"""

    def __init__(self, lev: 'Level'):
        self.L = lev
        nq = len(QUANTITIES)
        self.ss = None                                                           # C1: sum of e^2 (ncell, nq)
        self.sm = {c: None for c in lev.names}                                   # C2: sum over months of e
        self.E1 = {c: [] for c in lev.names}                                     # C1: box sums per month (nb, nq) per mask
        self.valid = None
        self.n = 0

    def add(self, errors):
        """errors {component: {q: (ny, nx)}} of one month"""
        lev = self.L
        for c, eq in errors.items():
            X = np.stack([np.asarray(eq[q], 'f8').ravel() for q in QUANTITIES], axis=1)       # (ncell, nq)
            fin = np.isfinite(X)
            self.valid = fin if self.valid is None else (self.valid & fin)
            X = np.where(fin, X, 0.0)
            self.ss = X * X if self.ss is None else self.ss + X * X
            self.sm[c] = X.copy() if self.sm[c] is None else self.sm[c] + X
            ops = lev.ops[lev.length[c]]
            self.E1[c].append({k: ops[k] @ X for k in lev.masks})
        self.n += 1

    def result(self, const=None):
        """{q: dict(map1, map2, const, series={mask: (c1, c2, const)})}"""
        lev = self.L
        shape = lev.w.shape
        nm = float(max(self.n, 1))
        nq = len(QUANTITIES)
        if self.n == 0:
            ok = np.zeros((shape[0] * shape[1], nq), bool)
            m1 = m2 = np.full(ok.shape, np.nan)
        else:
            ok = self.valid
            m1 = np.where(ok, np.sqrt(self.ss) / nm, np.nan)
            m2 = np.where(ok, np.sqrt(sum(self.sm[c] ** 2 for c in lev.names)) / nm, np.nan)
        # area variances: C1 the monthly box sums, C2 the box sums of the monthly sums, per component (its L)
        v1 = {k: np.zeros(nq) for k in lev.masks}
        v2 = {k: np.zeros(nq) for k in lev.masks}
        for c in lev.names:
            bc = lev.boxcov[lev.length[c]]
            for k in lev.masks:
                if not self.E1[c]:
                    continue
                A = np.concatenate([e[k] for e in self.E1[c]], axis=1)                    # (nb, 12 nq)
                v1[k] += bc.var_cols(A).reshape(-1, nq).sum(axis=0)
                v2[k] += bc.var_cols(sum(e[k] for e in self.E1[c]))
        out = {}
        for j, q in enumerate(QUANTITIES):
            okq = ok[:, j].reshape(shape)
            mc = const[q] if const is not None else np.full(shape, np.nan)
            ser = {}
            for k, m in lev.masks.items():
                W = float((lev.w * (m & okq)).sum())
                okc = m & np.isfinite(mc)
                sc_ = float(np.sum(np.where(okc, mc, 0.0) * lev.w) / (lev.w * okc).sum()) if okc.any() else np.nan
                if W <= 0:
                    ser[k] = (np.nan, np.nan, sc_)
                else:
                    ser[k] = (np.sqrt(v1[k][j]) / (nm * W), np.sqrt(v2[k][j]) / (nm * W), sc_)
            out[q] = dict(map1=m1[:, j].reshape(shape), map2=m2[:, j].reshape(shape), const=mc, series=ser)
        return out


class Level:
    """the level uncertainty of one reference on a grid (lat, lon) with area masks {name: (ny, nx) bool}"""

    def __init__(self, ref, lat, lon, si, po4, masks):
        self.ref = ref
        self.lat, self.lon = np.asarray(lat, 'f8'), np.asarray(lon, 'f8')
        self.lat2 = np.broadcast_to(self.lat[:, None], (self.lat.size, self.lon.size))
        self.si, self.po4 = si, po4
        self.masks = {k: np.asarray(m, bool) for k, m in masks.items()}
        self.w = oa_config.area_weights(self.lat, self.lon)
        self.comps = COMPONENTS[ref]
        self.names = [c[0] for c in self.comps]
        self.kind = {c[0]: c[1] for c in self.comps}
        self.length = {c[0]: c[4] for c in self.comps}
        act = np.any([m for m in self.masks.values()], axis=0) if self.masks else None
        self.boxcov = {L: BoxCov(self.lat, self.lon, L, act) for L in set(self.length.values())}
        self.ops = {L: {k: b.operator(self.w * m) for k, m in self.masks.items()} for L, b in self.boxcov.items()}
        self.notes = {}
        self._src = {}

    # -------------------------------------------------------------------------------- the product's sigma fields
    def available(self):
        """drop components whose source file or variable is missing (reported); returns the notes"""
        keep = []
        for c in self.comps:
            name, kind, src, var, L, fac = c
            p = SOURCES[src]['file']
            try:
                with netCDF4.Dataset(p) as f:
                    if var not in f.variables:
                        raise KeyError(var)
                keep.append(c)
            except (OSError, FileNotFoundError, KeyError) as e:
                self.notes[name] = f'missing ({type(e).__name__}: {e}); component not included'
        self.comps = keep
        self.names = [c[0] for c in keep]
        return self.notes

    def _source(self, src):
        if src not in self._src:
            cfg = SOURCES[src]
            rd = cp.Reader(cfg['file'], cfg['order'])
            ym = year_month(rd.f['time'])
            slat = np.asarray(rd.f['lat'][:], 'f8')
            slon = np.asarray(rd.f['lon'][:], 'f8')
            idx = {}
            for i, (y, m) in enumerate(ym):
                idx.setdefault(int(y), {})[int(m)] = i
            years = sorted(y for y, d in idx.items() if len(d) == 12)
            self._src[src] = dict(rd=rd, idx=idx, years=np.array(years), jy=co.nearest_index(slat, self.lat),
                                  ix=co.nearest_index(slon, self.lon, periodic=True))
        return self._src[src]

    def sigmas(self, year, valid):
        """{component: (12, ny, nx)} 1 sigma of the product in `year` (valid: (12, ny, nx) cells with a product
        value; there a missing sigma is filled with the area mean of the month)"""
        out = {}
        for name, kind, src, var, L, fac in self.comps:
            S = self._source(src)
            yy = int(S['years'][np.argmin(np.abs(S['years'] - year))])
            if yy != year:
                self.notes[f'{name}_years'] = self.notes.get(f'{name}_years', set()) | {f'{year}->{yy}'}
            i0 = S['idx'][yy][1]
            a = S['rd'](var, i0, i0 + 12)[:, S['jy']][:, :, S['ix']] * fac
            a = np.abs(a)
            filled = 0
            for mo in range(12):
                v = valid[mo]
                miss = v & ~np.isfinite(a[mo])
                good = v & np.isfinite(a[mo])
                if miss.any():
                    mean = float(np.sum(np.where(good, a[mo], 0) * self.w) / max(np.sum(self.w * good), 1e-30)) if good.any() else 0.0
                    a[mo][miss] = mean
                    filled += int(miss.sum())
                a[mo][~v] = np.nan
            nv = int(valid.sum())
            self.notes[f'{name}_filled'] = self.notes.get(f'{name}_filled', 0.0) + filled
            self.notes[f'{name}_valid'] = self.notes.get(f'{name}_valid', 0.0) + nv
            out[name] = a
        return out

    def close(self):
        for s in self._src.values():
            s['rd'].close()

    def report(self):
        lines = []
        for name in [c[0] for c in COMPONENTS[self.ref]]:
            if name in self.notes and isinstance(self.notes[name], str):
                lines.append(f'{name}: {self.notes[name]}')
                continue
            fv = self.notes.get(f'{name}_valid', 0.0)
            ff = self.notes.get(f'{name}_filled', 0.0)
            yrs = sorted(self.notes.get(f'{name}_years', set()))
            lines.append(f'{name} (L {self.length.get(name, np.nan):.0f} km, {self.kind.get(name, "")}): '
                         f'{100 * ff / fv if fv else 0:.2f} % of the cell-months filled with the monthly area mean'
                         + (f'; years from the nearest year: {", ".join(yrs[:3])}{" ..." if len(yrs) > 3 else ""}' if yrs else ''))
        return lines

    # -------------------------------------------------------------------------------- one year
    def product_year(self, fields12, year):
        """a product year: fields12 {var: (12, ny, nx)} the product state. Returns (result, dX {c: {x: (12, ny, nx)}})"""
        valid = np.isfinite(fields12['dissic']) & np.isfinite(fields12['talk'])
        sig = self.sigmas(year, valid)
        acc = LevelYear(self)
        dX = {c: {x: np.zeros((12,) + self.w.shape) for x in DX} for c in self.names}
        for mo in range(12):
            st = {v: fields12[v][mo] for v in VARS}
            n_ok = int(np.sum(np.all([np.isfinite(st[v]) for v in VARS], axis=0) & np.isfinite(self.si) & np.isfinite(self.po4)))
            if n_ok == 0:
                print(f'  level uncertainty: {self.ref} {year} month {mo + 1}: no cell with T, S, DIC, TA and nutrients '
                      f'(finite cells: ' + ', '.join(f'{v} {int(np.isfinite(st[v]).sum())}' for v in VARS) + ')', flush=True)
            _, J = jacobian(st, self.si, self.po4, self.lat2)
            err = {}
            for c in self.names:
                d = obs_to_state(J, self.kind[c], sig[c][mo])
                for x, a in d.items():
                    dX[c][x][mo] = a
                err[c] = propagate(J, d)
            acc.add(err)
        const = constants_sigma({v: np.nanmean(fields12[v], axis=0) for v in VARS}, self.si, self.po4, self.lat2)
        return acc.result(const), dX

    def model_year(self, fields12, dX):
        """a model year: fields12 {var: (12, ny, nx)} the constrained state, dX {c: {x: (12, ny, nx)}} the window
        mean of the product's state errors"""
        acc = LevelYear(self)
        for mo in range(12):
            _, J = jacobian({v: fields12[v][mo] for v in VARS}, self.si, self.po4, self.lat2)
            acc.add({c: propagate(J, {x: dX[c][x][mo] for x in DX}) for c in self.names})
        const = constants_sigma({v: np.nanmean(fields12[v], axis=0) for v in VARS}, self.si, self.po4, self.lat2)
        return acc.result(const)


# ================================================================================================ files
def write_dx(g, lev: Level, dX, window):
    """the window mean of the state errors per component (for the model years, calc_product_states.py)"""
    g.createDimension('component', len(lev.names))
    g.createDimension('month', 12)
    x = g.createVariable('component', str, ('component',)); x[:] = np.array(lev.names, dtype=object)
    x = g.createVariable('component_length', 'f8', ('component',)); x.units = 'km'; x[:] = [lev.length[c] for c in lev.names]
    x.long_name = 'decorrelation length of the exponential spatial correlation'
    x = g.createVariable('component_kind', str, ('component',)); x[:] = np.array([lev.kind[c] for c in lev.names], dtype=object)
    for v in DX:
        x = g.createVariable(f'level_dx_{v}', 'f4', ('component', 'month', 'lat', 'lon'), zlib=True, complevel=4, fill_value=cb.FILL)
        x.units = cb.UNITS[v]
        x.long_name = (f'mapping error of the product as an error of the state ({v}), 1 sigma, signed by the derivative; '
                       f'window mean {window[0]}-{window[1]} (the bias-adjustment window of the models)')
        if lev.names:
            x[:] = np.ma.masked_invalid(np.array([dX[c][v] for c in lev.names], 'f4'))
    g.level_window = f'{window[0]}-{window[1]}'


def read_dx(path, lat, lon):
    """(names, lengths, kinds, dX) from an obs file, on the grid (lat, lon); None if the file has none"""
    with netCDF4.Dataset(path) as f:
        if 'level_dx_dissic' not in f.variables or 'component' not in f.dimensions or len(f.dimensions['component']) == 0:
            return None
        names = [str(s) for s in f['component'][:]]
        jy = co.nearest_index(np.asarray(f['lat'][:], 'f8'), np.asarray(lat, 'f8'))
        ix = co.nearest_index(np.asarray(f['lon'][:], 'f8'), np.asarray(lon, 'f8'), periodic=True)
        dX = {c: {} for c in names}
        for v in DX:
            a = np.ma.filled(f[f'level_dx_{v}'][:].astype('f8'), np.nan)[:, :, jy][:, :, :, ix]
            for i, c in enumerate(names):
                dX[c][v] = a[i]
        return names, dX


def window_of(ref, years_obs):
    w0, w1 = cb.REFERENCES[ref].get('window', cb.OVERLAP)
    return max(w0, int(years_obs[0])), min(w1, int(years_obs[-1]), 2014)
