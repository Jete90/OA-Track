#!/usr/bin/env python
"""
make_woa_nutrients.py
=====================
Annual surface silicate and phosphate of the World Ocean Atlas 2023 (all decades, 1 degree) as the second nutrient
climatology of nutrient_sensitivity.py (run_checks.sh, NUT_ALT): the mean of the 12 monthly objectively analysed
climatologies (i_an, p_an; months 01 to 12, not the annual or seasonal files) at the surface (first depth level),
in umol/kg, on the WOA grid (lat -89.5..89.5, lon -179.5..179.5; --lon360: 0.5..359.5). nutrient_sensitivity.py
matches the grids by coordinate, so the longitude convention does not change its results.

A cell takes the mean over the months with a value; --strict requires all 12 months.
If the output files exist already, the fields are compared with them first (maximum difference, cells only in
one of them) and the files are rewritten only with --force.

Input:  <OBS_DIR>/WOA2023/{silicate,phosphate}/woa23_all_{i,p}MM_01.nc  (MM = 01 .. 12)
Output: data/<run>/obs/woa23_si_surface_annual.nc (silicate), woa23_po4_surface_annual.nc (phosphate)

Usage (code folder):  OA_RUN=u74 python make_woa_nutrients.py [--woa DIR] [--strict] [--lon360] [--force]
"""
import argparse
import glob
import os
import warnings

import netCDF4
import numpy as np

import oa_config

WOA = str(oa_config.OBS_DIR / 'WOA2023')
FIELDS = {'silicate': dict(sub='silicate', tag='i', var='i_an', out='woa23_si_surface_annual.nc'),
          'phosphate': dict(sub='phosphate', tag='p', var='p_an', out='woa23_po4_surface_annual.nc')}


def monthly_files(woa, f):
    files = [os.path.join(woa, f['sub'], f'woa23_all_{f["tag"]}{m:02d}_01.nc') for m in range(1, 13)]
    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        have = sorted(glob.glob(os.path.join(woa, f['sub'], '*.nc')))
        raise FileNotFoundError(f'missing {missing[:3]} ...; files in {os.path.join(woa, f["sub"])}: '
                                f'{[os.path.basename(h) for h in have[:20]]}')
    return files


def surface(path, var):
    with netCDF4.Dataset(path) as d:
        a = np.ma.filled(d[var][0, 0].astype('f8'), np.nan)          # time 0, first depth level (0 m)
        lat, lon = np.asarray(d['lat'][:], 'f8'), np.asarray(d['lon'][:], 'f8')
    return a, lat, lon


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--woa', default=WOA)
    ap.add_argument('--strict', action='store_true', help='require all 12 months in a cell')
    ap.add_argument('--force', action='store_true', help='rewrite existing output files')
    ap.add_argument('--lon360', action='store_true', help='write longitudes 0.5 .. 359.5 E (default: as WOA, -179.5 .. 179.5)')
    a = ap.parse_args()
    out_dir = os.path.join(str(oa_config.DATA), 'obs')
    os.makedirs(out_dir, exist_ok=True)
    for name, f in FIELDS.items():
        months = [surface(p, f['var']) for p in monthly_files(a.woa, f)]
        lat, lon = months[0][1], months[0][2]
        stack = np.array([m[0] for m in months])                     # (12, lat, lon)
        n = np.isfinite(stack).sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)          # land: no month
            field = np.nanmean(stack, axis=0)
        if a.strict:
            field[n < 12] = np.nan
        print(f'{name}: {np.isfinite(field).sum()} cells ({np.sum((n > 0) & (n < 12))} with fewer than 12 months, '
              f'{"left out" if a.strict else "mean of the months with a value"}); '
              f'range {np.nanmin(field):.3f} to {np.nanmax(field):.3f} umol/kg')
        if a.lon360:                                                 # 0.5 .. 359.5 E instead of -179.5 .. 179.5
            o = np.argsort(lon % 360.0)
            lon, field = lon[o] % 360.0, field[:, o]
        path = os.path.join(out_dir, f['out'])
        if os.path.exists(path):
            with netCDF4.Dataset(path) as d:
                old = np.ma.filled(d[name][:].astype('f8'), np.nan)
                olat, olon = np.asarray(d['lat'][:], 'f8'), np.asarray(d['lon'][:], 'f8')
            print(f'  existing file: lat {olat[0]:.1f} .. {olat[-1]:.1f}, lon {olon[0]:.1f} .. {olon[-1]:.1f}; new: lat '
                  f'{lat[0]:.1f} .. {lat[-1]:.1f}, lon {lon[0]:.1f} .. {lon[-1]:.1f}')
            # compare at the same coordinates (any longitude convention, either latitude order)
            jy = np.abs(olat[:, None] - lat[None, :]).argmin(1)
            ix = np.abs(((olon[:, None] - lon[None, :]) + 180.0) % 360.0 - 180.0).argmin(1)
            same_grid = np.allclose(lat[jy], olat) and np.allclose(((lon[ix] - olon) + 180.0) % 360.0 - 180.0, 0.0)
            if not same_grid:
                print('  WARNING: the grids do not match cell by cell; the comparison uses the nearest cells')
            new_on_old = field[np.ix_(jy, ix)]
            print(f'  same coordinate order as the existing file: {np.array_equal(jy, np.arange(lat.size)) and np.array_equal(ix, np.arange(lon.size))}'
                  + ('' if a.lon360 else ' (--lon360 writes longitudes 0.5 .. 359.5)'))
            field_cmp = new_on_old
            both = np.isfinite(old) & np.isfinite(field_cmp)
            print(f'  against the existing {f["out"]} at the same coordinates: max |difference| '
                  f'{np.nanmax(np.abs(field_cmp - old)[both]):.2e} on {both.sum()} cells; only in the existing file '
                  f'{np.sum(np.isfinite(old) & ~np.isfinite(field_cmp))}, only in the new one '
                  f'{np.sum(~np.isfinite(old) & np.isfinite(field_cmp))}')
            if not a.force:
                print(f'  kept {path} (--force to rewrite)')
                continue
        with netCDF4.Dataset(path, 'w') as d:
            d.createDimension('lat', lat.size)
            d.createDimension('lon', lon.size)
            d.createVariable('lat', 'f8', ('lat',))[:] = lat
            d.createVariable('lon', 'f8', ('lon',))[:] = lon
            v = d.createVariable(name, 'f8', ('lat', 'lon'), fill_value=np.nan)
            v.units = 'umol kg-1'
            v[:] = field
            d.source = (f'WOA23 {name}, all decades, 1 degree, surface, mean of the 12 monthly climatologies'
                        + (' (cells with all 12 months)' if a.strict else ''))
            d.history = f'make_woa_nutrients.py from {a.woa}/{f["sub"]} ({f["var"]}, months 01-12)'
        print(f'  written {path}')


if __name__ == '__main__':
    main()
