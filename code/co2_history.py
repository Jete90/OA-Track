"""
co2_history.py
==============
Atmospheric CO2 (global annual mean, ppm) and its uncertainty for every year of the data product, 1750-2100.

Sources (files in inputs/co2/, downloaded from input4MIPs on ESGF (netCDF) or from the University of Melbourne GHG
site, greenhousegases.science.unimelb.edu.au (csv); either format, yearly or monthly, monthly values are averaged
to annual means; the hemispheric-mean files "gr1-GMNHSH", of which the global mean (sector 0 / data_mean_global)
is used):
  1750-2014   CMIP6 historical concentrations, Meinshausen et al. (2017, GMD 10, 2057-2116), the forcing of the
              historical runs: mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_CMIP_UoM-CMIP-1-2-0_gr1-GMNHSH_000001-201412.nc
              (or _0000-2014, or .csv)
  2015-2100   CMIP6 scenario concentrations, Meinshausen et al. (2020, GMD 13, 3571-3605), the forcing of the
              ScenarioMIP runs: ..._ScenarioMIP_UoM-IMAGE-ssp126-1-2-1_gr1-GMNHSH_201501-250012.nc, likewise
              UoM-MESSAGE-GLOBIOM-ssp245, UoM-AIM-ssp370, UoM-REMIND-MAGPIE-ssp585 (the version string and the
              yearly/monthly choice do not matter: files are found by pattern *carbon*dioxide*...*GMNHSH*)
  1959-2024   NOAA global annual means (inputs/co2_annmean_gl.txt), used by the paper for the observed years
Uncertainty (1 sigma, ppm):
  1750-1850   AR6 very likely ranges (Gulev et al. 2021): 2.9/1.645 in 1750 to 2.1/1.645 in 1850, interpolated
  1851-1957   1.2, the Law Dome ice core measurements (Etheridge et al. 1996; Rubino et al. 2019)
  1958-       0.1, direct measurements (Mauna Loa and the global network); the same for the scenario years, as
              decided for the product (the scenario CO2 is a prescribed pathway, not a measurement)
"""
from __future__ import annotations

import csv
import glob
import re
from pathlib import Path

import netCDF4
import numpy as np

import oa_config

CO2_DIR = oa_config.INPUTS / 'co2'
NOAA = oa_config.INPUTS / 'co2_annmean_gl.txt'
SCENARIOS = ('ssp126', 'ssp245', 'ssp370', 'ssp585')
SIG_1750, SIG_1850 = 2.9 / 1.645, 2.1 / 1.645      # AR6, 1 sigma
SIG_ICE, SIG_DIRECT = 1.2, 0.1                       # Law Dome; direct measurements and scenarios


def _annual(years, values):
    """{year: mean of the values of the year}"""
    acc, n = {}, {}
    for y, v in zip(years, values):
        y = int(y)
        acc[y] = acc.get(y, 0.0) + float(v)
        n[y] = n.get(y, 0) + 1
    return {y: acc[y] / n[y] for y in acc}


def _years_from_name(path, nt):
    """years of the nt time steps from the time range in an input4MIPs file name (last token YYYYMM-YYYYMM for
    monthly, YYYY-YYYY for yearly files), counted backwards from the end of the range (the start token of the
    historical files, 000001, is not a calendar date); None if the name has no range"""
    m = re.search(r'_(\d{4,6})-(\d{4,6})\.[a-z]+$', str(path))
    if not m:
        return None
    a, b = m.group(1), m.group(2)
    if len(a) == 6 and len(b) == 6:                   # monthly: absolute month number of the last step
        last = int(b[:4]) * 12 + int(b[4:]) - 1
        return [(last - (nt - 1 - i)) // 12 for i in range(nt)]
    if len(a) == 4 and len(b) == 4:                   # yearly
        return [int(b) - (nt - 1 - i) for i in range(nt)]
    return None


def _is_ppm(units):
    u = str(units).strip().lower()
    if u in ('ppm', 'ppmv', 'umol mol-1', 'micromol mol-1'):
        return True
    try:
        return abs(float(u) - 1e-6) < 1e-12                # '1e-6', '1.e-6'
    except ValueError:
        return False


PPM_RANGE = (150.0, 3000.0)          # any atmospheric CO2 of 1750-2500 in ppm


def _to_ppm(path, x, units):
    """values in ppm: the units attribute is used when the values in that unit are atmospheric CO2; when they are not,
    but the values themselves are in the ppm range, the attribute is wrong (some ScenarioMIP files of UoM carry
    '1.e-12' with ppm values): the values are used as ppm and this is printed; anything else is an error"""
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if _is_ppm(units):
        f = 1.0
    else:
        try:
            f = float(str(units).strip()) / 1e-6
        except ValueError:
            f = None
    if f is not None and PPM_RANGE[0] <= lo * f and hi * f <= PPM_RANGE[1]:
        return x * f
    if PPM_RANGE[0] <= lo and hi <= PPM_RANGE[1]:
        print(f'co2_history: {Path(path).name}: units attribute {units!r} does not match the values ({lo:.1f} to '
              f'{hi:.1f}), which are ppm; used as ppm')
        return x
    raise ValueError(f'{path}: values {lo:.4g} to {hi:.4g} with units {units!r} are not atmospheric CO2')


def _years_from_axis(f):
    """years of the time axis of a netCDF file through cftime; None if the units or the calendar cannot be
    decoded (reference year 0 in a Gregorian calendar)"""
    t = f['time']
    for cal in (getattr(t, 'calendar', 'standard'), 'proleptic_gregorian', 'noleap'):
        try:
            d = netCDF4.num2date(t[:], t.units, cal, only_use_cftime_datetimes=True)
            return [int(x.year) for x in d]
        except (ValueError, TypeError):
            continue
    return None


def read_input4mips(path):
    """{year: annual global mean} from an input4MIPs GHG concentration file: csv (columns year, data_mean_global;
    yearly or monthly rows) or netCDF (variable *carbon_dioxide*(time, sector), sector 0 = global mean; the years
    from the time axis, checked against the time range in the file name, or from the name alone if the axis
    cannot be decoded)"""
    path = Path(path)
    if path.suffix == '.csv':
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows or 'year' not in rows[0] or 'data_mean_global' not in rows[0]:
            raise ValueError(f'{path}: expected columns year and data_mean_global, got {list(rows[0]) if rows else []}')
        return _annual([float(r['year']) for r in rows], [r['data_mean_global'] for r in rows])
    with netCDF4.Dataset(path) as f:
        names = [v for v in f.variables if 'carbon_dioxide' in v]
        if len(names) != 1:
            raise ValueError(f'{path}: expected one *carbon_dioxide* variable, found {names}')
        v = f[names[0]]
        if v.ndim != 2 or v.dimensions[0] != 'time':
            raise ValueError(f'{path}: expected dimensions (time, sector), got {v.dimensions}')
        if 'sector' in f.variables:
            ids = getattr(f['sector'], 'ids', '')
            if ids and not ids.lstrip().lower().startswith('0: global'):
                raise ValueError(f'{path}: sector 0 is not the global mean: {ids}')
        x = np.ma.filled(v[:, 0].astype('f8'), np.nan)
        if 'units' in v.ncattrs():
            x = _to_ppm(path, x, v.units)
        y_axis = _years_from_axis(f)
    y_name = _years_from_name(path, x.size)
    if y_axis is not None and y_name is not None and y_axis[-1] != y_name[-1]:
        raise ValueError(f'{path}: the time axis ends in {y_axis[-1]}, the file name in {y_name[-1]}')
    years = y_axis if y_axis is not None else y_name
    if years is None:
        raise ValueError(f'{path}: cannot decode the time axis and the file name has no time range')
    if not np.all(np.isfinite(x)):
        raise ValueError(f'{path}: missing values in the global mean')
    return _annual(years, x)


def _find(pattern):
    """the file matching the pattern (csv or netCDF; csv first, then sorted order: every form of the file gives
    the same annual means)"""
    hits = sorted(glob.glob(str(CO2_DIR / (pattern + '.csv'))) + glob.glob(str(CO2_DIR / (pattern + '.nc'))))
    return Path(hits[0]) if hits else None


def historical():
    """{year: ppm} for the CMIP6 historical concentrations, or None if the file is missing"""
    p = _find('*carbon*dioxide*GHGConcentrations_CMIP_UoM*GMNHSH*')
    return read_input4mips(p) if p else None


def scenario(ssp):
    p = _find(f'*carbon*dioxide*ScenarioMIP*{ssp}*GMNHSH*')
    return read_input4mips(p) if p else None


def noaa():
    """{year: ppm} from the NOAA global annual mean file (year, mean, unc)"""
    out = {}
    if NOAA.exists():
        for l in open(NOAA):
            if l.strip() and not l.startswith('#'):
                a, b = l.split()[:2]
                out[int(a)] = float(b)
    return out


def sigma(year):
    """1 sigma of the atmospheric CO2 of a year (ppm), see module doc"""
    year = np.asarray(year, 'f8')
    s = np.where(year <= 1850, SIG_1750 + (SIG_1850 - SIG_1750) * (year - 1750) / 100.0,
                 np.where(year <= 1957, SIG_ICE, SIG_DIRECT))
    return s if s.ndim else float(s)


class CO2History:
    """CO2 of every year 1750-2100 for the product: CMIP6 historical to 2014, then the scenario; NOAA for the
    observed years where the chain uses it (reported separately)"""

    def __init__(self):
        self.hist = historical()
        self.ssp = {s: scenario(s) for s in SCENARIOS}
        self.noaa = noaa()
        self.missing = ([] if self.hist else ['historical']) + [s for s in SCENARIOS if not self.ssp[s]]
        # every scenario must continue the historical concentrations (2014 -> 2015 within a few ppm): this also
        # confirms the units of files whose units attribute was not used
        for sc, d in self.ssp.items():
            if d and self.hist and 2014 in self.hist and 2015 in d and abs(d[2015] - self.hist[2014]) > 5.0:
                raise ValueError(f'{sc}: CO2 of 2015 ({d[2015]:.2f}) does not continue the historical 2014 ({self.hist[2014]:.2f})')

    def require(self, what='historical'):
        if what == 'historical' and not self.hist:
            raise FileNotFoundError(f'CMIP6 historical CO2 file not found in {CO2_DIR} (see co2_history.py)')
        if what in SCENARIOS and not self.ssp[what]:
            raise FileNotFoundError(f'CMIP6 {what} CO2 file not found in {CO2_DIR} (see co2_history.py)')

    def value(self, year, ssp=None):
        if year <= 2014:
            self.require('historical')
            src, lab = self.hist, 'historical'
        else:
            self.require(ssp)
            src, lab = self.ssp[ssp], ssp
        if year not in src:
            raise KeyError(f'{lab} CO2 file has no year {year} (years {min(src)}-{max(src)})')
        return src[year]

    def series(self, years, ssp=None):
        return np.array([self.value(int(y), ssp) for y in years])
