"""
drift_tools.py
==============
Reading side of the piControl drift (calc_drift.py writes the files): the Drift class turns the stored
Enting-spline anomalies s(k) - mean(piControl) into the drift correction of the chain for any reference window,

    X'(t) = X(t) - [s(k(t)) - mean of s over W']        W' = piControl years parallel to the window W,

for piControl months (k = the file year) and historical months (k = branch_year + (year - first historical
year)). Used by calc_bias_adjusted.py and calc_pi_xco2.py. No dependency on those modules (avoids import cycles).
"""
from __future__ import annotations

import netCDF4
import numpy as np

import oa_config
import calc_omega_arag as cm

OUT = oa_config.DATA / 'drift'
VARS = cm.VARS
PERIODS = oa_config.PERIODS        # 1850-1900 and the window of every reference (oa_config.WINDOWS)
CUTOFF = 50.0                       # years, 50 % response of the Enting smoothing spline


def drift_path(model, member):
    return OUT / f'drift_{model}_{member}.nc'


def smoother(n, cutoff=CUTOFF):
    """Enting (1987) smoothing spline as a matrix: penalised least squares with a second-difference penalty,
    lambda chosen so that the response is 50 % at the cutoff period. One matrix serves all cells."""
    lam = 1.0 / (2.0 - 2.0 * np.cos(2.0 * np.pi / cutoff)) ** 2
    D = np.diff(np.eye(n), 2, axis=0)
    return np.linalg.inv(np.eye(n) + lam * D.T @ D)


def parallel_years(by, hist0, y0, y1):
    """piControl years parallel to the historical years y0..y1."""
    return by + (y0 - hist0), by + (y1 - hist0)


def window_slice(years, q0, q1):
    """index range [k0, k1) of the piControl years q0..q1 in `years`; a range partly or fully outside the file is
    clipped to the nearest available years (at least one year), as calc_drift.py warns"""
    n = years.size
    k0 = int(np.clip(np.searchsorted(years, q0), 0, n - 1))
    k1 = int(np.clip(np.searchsorted(years, q1, side='right'), k0 + 1, n))
    return k0, k1


class Drift:
    """The drift correction of one model for any reference window, from drift_<model>_<member>.nc.

    term(var, exp, years, window) returns  s(k(t)) - mean of s over W'  for the given years of the given
    experiment (one field per entry of `years`, which may repeat for monthly steps), on the model grid or,
    with jy and ix, sampled onto another grid by nearest neighbour as in calc_bias_adjusted.ModelRun.read.
    Subtracting it from the model fields removes the drift relative to the window years (see module doc)."""

    def __init__(self, model, member):
        path = drift_path(model, member)
        if not path.exists():
            raise FileNotFoundError(f'{path}: run calc_drift.py first')
        with netCDF4.Dataset(path) as f:
            self.years = np.asarray(f['year'][:], 'i8')
            self.by = int(f.branch_year)
            self.hist0 = int(f.hist_first_year)
            self.lat = np.asarray(f['lat'][:], 'f8')
            self.lon = np.asarray(f['lon'][:], 'f8')
            self.anom = {v: np.ma.filled(f[f'{v}_smooth_anom'][:].astype('f4'), np.nan) for v in VARS}  # ~65 MB/var
            self.stored = {k: np.ma.filled(f[k][:].astype('f8'), np.nan) for k in f.variables
                           if '_drift_' in k}
        self.tag = f'{model} {member}'
        self._warned = set()
        self._checked = set()

    def index(self, exp, years):
        """index into the piControl years for each year of `exp` (historical years via the branch point)"""
        years = np.asarray(years, 'i8')
        k = years if exp == 'piControl' else self.by + (years - self.hist0)
        idx = np.searchsorted(self.years, k)
        out = (idx < self.years.size) & (self.years[np.minimum(idx, self.years.size - 1)] == k)
        if not np.all(out):
            if exp not in self._warned:                    # once per experiment and model
                self._warned.add(exp)
                bad = k[~out]
                print(f'WARNING {self.tag}: {exp} years map to piControl years outside the file '
                      f'{self.years[0]}-{self.years[-1]} (first {bad.min()}): the nearest available year is used '
                      f'for them', flush=True)
            idx = np.clip(np.searchsorted(self.years, k), 0, self.years.size - 1)
        return idx

    def window_mean(self, var, window):
        """mean of s over the piControl years parallel to the historical window (y0, y1), minus piControl mean"""
        q0, q1 = parallel_years(self.by, self.hist0, *window)
        k0, k1 = window_slice(self.years, q0, q1)
        m = self.anom[var][k0:k1].astype('f8').mean(axis=0)
        key = f'{var}_drift_{window[0]}_{window[1]}'
        if key in self.stored and key not in self._checked:
            self._checked.add(key)
            d = np.abs(m - self.stored[key])
            if np.isfinite(d).any() and not (np.nanmax(d) < 1e-3):
                print(f'WARNING {self.tag}: window mean of {var} differs from the stored {key} by '
                      f'{np.nanmax(d):.2e}', flush=True)
        return m

    def term(self, var, exp, years, window, jy=None, ix=None):
        """s(k(t)) - mean of s over W', shape (len(years), ny, nx); NaN where the spline is undefined"""
        d = self.anom[var][self.index(exp, years)].astype('f8') - self.window_mean(var, window)
        if jy is not None:
            d = d[:, jy][:, :, ix]
        return d
