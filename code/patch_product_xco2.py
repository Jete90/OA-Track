#!/usr/bin/env python
"""
patch_product_xco2.py
=====================
Completes the atmospheric CO2 of the data product files in place, so that every value of the product comes with the
CO2 of its year:
  xco2 (year)            the product years after the CMIP6 history, 2015 to 2024 (2025 for UExP-FNN-U and
                         CMEMS-LSCE-FFNN), with the NOAA global annual means (inputs/co2_annmean_gl.txt,
                         co2_history.noaa; the record the paper uses for the observed years). make_product.py left
                         these years empty, since the product years are not forced by the CMIP6 history.
  xco2_<ssp> (year_ssp)  the CO2 of each scenario series 1750-2100: the CMIP6 history to 2014 and the ScenarioMIP
                         concentrations afterwards (co2_history.CO2History; the forcing of the models), one new
                         variable per scenario in the file.
The variable descriptions and the history attribute say what was done; nothing else in the files changes. Files
already complete are left alone, so the script can be run again.

Usage (code/):  OA_RUN=u74 python -u patch_product_xco2.py            (all carbonate_system_*.nc without _alkalinity)
                OA_RUN=u74 python -u patch_product_xco2.py --dry-run   (report only)
"""
import argparse
import datetime
import sys

import netCDF4
import numpy as np

import oa_config
import co2_history

OUT = oa_config.DATA / 'product'
NOTE = 'NOAA global annual means from 2015 (Lan et al.; the product years)'


def patch_years(f, years, noaa, dry):
    """xco2 of the product years from the NOAA record; returns (note, changed)"""
    x = f['xco2']
    v = np.ma.filled(x[:].astype('f8'), np.nan)
    empty = np.isnan(v) & (years >= 2015)
    missing = [int(y) for y in years[empty] if int(y) not in noaa]
    todo = empty & np.isin(years, list(noaa))
    tail = f'; no NOAA value for {missing}, left empty' if missing else ''
    if not todo.any():
        return 'xco2 complete' + tail, False
    span = f'{years[todo].min()}-{years[todo].max()}'
    if dry:
        return f'would fill xco2 for {int(todo.sum())} years ({span})' + tail, False
    v[todo] = [noaa[int(y)] for y in years[todo]]
    x[:] = v
    x.long_name = str(getattr(x, 'long_name', 'atmospheric CO2 of the year')) + f'; {NOTE}'
    return f'xco2 filled for {int(todo.sum())} years ({span}) with the NOAA record' + tail, True


def patch_scenarios(f, co2, dry):
    """xco2_<ssp> of the scenario series present in the file; returns (notes, changed)"""
    if 'year_ssp' not in f.dimensions:
        return [], False
    ys = np.asarray(f['year_ssp'][:]).astype(int)
    notes, changed = [], False
    for s in co2_history.SCENARIOS:
        if f'omega_arag_{s}' not in f.variables:
            continue
        if f'xco2_{s}' in f.variables:
            notes.append(f'xco2_{s} present')
            continue
        try:
            vals = co2.series(ys, s)
        except (FileNotFoundError, KeyError) as e:
            notes.append(f'xco2_{s} not written ({e})')
            continue
        if dry:
            notes.append(f'would write xco2_{s} ({ys[0]}-{ys[-1]}, {vals[-1]:.1f} ppm in {ys[-1]})')
            continue
        x = f.createVariable(f'xco2_{s}', 'f8', ('year_ssp',), fill_value=np.nan)
        x.units = 'ppm'
        x.long_name = (f'atmospheric CO2 of the year of the {s} series: CMIP6 history (Meinshausen et al. 2017) to 2014, '
                       f'ScenarioMIP {s} concentrations (Meinshausen et al. 2020) from 2015; the forcing of the models')
        x[:] = vals
        notes.append(f'xco2_{s} written ({ys[0]}-{ys[-1]}, {vals[-1]:.1f} ppm in {ys[-1]})')
        changed = True
    return notes, changed


def patch(path, noaa, co2, dry):
    with netCDF4.Dataset(path, 'r' if dry else 'a') as f:
        years = np.asarray(f['year'][:]).astype(int)
        note, c1 = patch_years(f, years, noaa, dry)
        notes, c2 = patch_scenarios(f, co2, dry)
        if c1 or c2:
            done = ('xco2 of the product years from the NOAA record' if c1 else '') + (', ' if c1 and c2 else '') \
                   + ('xco2_<ssp> of the scenario series from the CMIP6 and ScenarioMIP record' if c2 else '')
            f.history = f'{datetime.datetime.now():%Y-%m-%d %H:%M} patch_product_xco2.py: {done}\n' + str(getattr(f, 'history', ''))
    return '; '.join([note] + notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    noaa = co2_history.noaa()
    if not noaa:
        raise SystemExit(f'{co2_history.NOAA} missing or empty')
    co2 = co2_history.CO2History()
    if co2.missing:
        print(f'CO2 files missing in {co2_history.CO2_DIR}: {co2.missing} (their series are not written)')
    files = sorted(p for p in OUT.glob('carbonate_system_*.nc') if not p.name.endswith('_alkalinity.nc'))
    print(f'{len(files)} files; NOAA record {min(noaa)}-{max(noaa)}' + (' (dry run)' if a.dry_run else ''))
    for p in files:
        print(f'  {p.name}: {patch(p, noaa, co2, a.dry_run)}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
