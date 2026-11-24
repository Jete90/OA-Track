#!/usr/bin/env python
"""
oatrack_inventory.py  (24 November 2026: the variable list of OA-Track for its README)
====================
Reads only the headers of the OA-Track files and writes OA-Track_variables.md: for each kind of file (a reference,
OA-Track_common, OA-Track_all) the dimensions, the global attributes and every variable, with the 15 quantities
folded into one line each (<q> stands for omega_arag, ph, dic, ...; <quantity> in the description for its name),
so that the list stays readable (about 3600 variables per file). Then the size and md5 of every file, if --md5.

Usage (code/ of v4, env_jens):
  python -u oatrack_inventory.py --dir ../data/u74/oatrack_seanoe --out ../data/u74/oatrack_seanoe/OA-Track_variables.md [--md5]
"""
import argparse
import hashlib
from collections import OrderedDict
from pathlib import Path

import netCDF4

QUANTITIES = ['omega_arag', 'omega_calc', 'ph', 'pco2', 'fco2', 'co2', 'hco3', 'co3', 'revelle', 'hplus', 'hplus_free',
              'dic', 'thetao', 'so', 'talk']
UNITS = {'omega_arag': '1', 'omega_calc': '1', 'ph': 'total scale', 'pco2': 'uatm', 'fco2': 'uatm', 'co2': 'umol kg-1',
         'hco3': 'umol kg-1', 'co3': 'umol kg-1', 'revelle': '1', 'hplus': 'nmol kg-1', 'hplus_free': 'nmol kg-1',
         'dic': 'umol kg-1', 'thetao': 'degC', 'so': 'psu', 'talk': 'umol kg-1'}
LONG = {'omega_arag': 'aragonite saturation state', 'omega_calc': 'calcite saturation state', 'ph': 'pH (total scale)',
        'hplus': 'hydrogen ion concentration (total scale)', 'hplus_free': 'hydrogen ion concentration (free scale)',
        'pco2': 'partial pressure of CO2', 'fco2': 'fugacity of CO2', 'co2': 'dissolved CO2', 'hco3': 'bicarbonate ion',
        'co3': 'carbonate ion', 'revelle': 'Revelle factor', 'dic': 'dissolved inorganic carbon', 'thetao': 'temperature',
        'so': 'salinity', 'talk': 'total alkalinity'}
BY_LENGTH = sorted(QUANTITIES, key=len, reverse=True)     # hplus_free before hplus, omega_arag before ...


def fold(name, long_name):
    """(pattern, description) with the quantity replaced, or the name itself if it belongs to no quantity"""
    for q in BY_LENGTH:
        if name == q or name.startswith(q + '_'):
            rest = name[len(q):]
            d = long_name.replace(LONG[q], '<quantity>') if long_name else ''
            return '<q>' + rest, d, q
    return name, long_name, None


def kind(path):
    n = path.name
    if n.startswith('OA-Track_common'):
        return 'OA-Track_common[_noalk].nc (mean of the four fCO2-products, common area)'
    if n.startswith('OA-Track_all'):
        return 'OA-Track_all[_noalk].nc (mean of the fCO2-products covering each cell)'
    return 'OA-Track_<reference>[_noalk].nc (one fCO2-product or Jiang et al.)'


def describe(path):
    lines = []
    with netCDF4.Dataset(path) as f:
        lines += [f'Example file: `{path.name}`', '', '**Dimensions**', '']
        for d in f.dimensions.values():
            v = f.variables.get(d.name)
            rng = ''
            if v is not None and v.ndim == 1 and len(d) > 0:
                a = v[:]
                rng = f': {a[0]} to {a[-1]}' if len(d) > 1 else f': {a[0]}'
            lines.append(f'- `{d.name}` ({len(d)}){rng}')
        lines += ['', '**Global attributes**', '']
        for k in f.ncattrs():
            val = str(f.getncattr(k)).replace('\n', ' ')
            lines.append(f'- `{k}`: {val[:300]}{" ..." if len(val) > 300 else ""}')
        pats = OrderedDict()
        for name, v in f.variables.items():
            if name in f.dimensions:
                continue
            pat, desc, q = fold(name, getattr(v, 'long_name', ''))
            key = (pat, v.dimensions)
            if key not in pats:
                unit = '(of the quantity)' if q and getattr(v, 'units', '') == UNITS[q] else getattr(v, 'units', '')
                pats[key] = dict(desc=desc, unit=unit, qs=set())
            if q:
                pats[key]['qs'].add(q)
        lines += ['', f'**Variables** ({len(f.variables)}; `<q>` is one of the quantities listed in the README, '
                      '`<quantity>` its name)', '', '| Variable | Dimensions | Units | Description |', '|---|---|---|---|']
        for (pat, dims), d in pats.items():
            only = ''
            if pat.startswith('<q>') and len(d['qs']) < len(QUANTITIES):
                only = f' (only {", ".join(sorted(d["qs"]))})'
            lines.append(f'| `{pat}` | {", ".join(dims) or "scalar"} | {d["unit"]} | {d["desc"].replace("|", "/")}{only} |')
    return lines, len(pats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='the folder with the OA-Track files (only read)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--md5', action='store_true', help='also the md5 of every file (reads every file once)')
    a = ap.parse_args()
    files = sorted(Path(a.dir).glob('OA-Track_*.nc'))
    assert files, f'no OA-Track files in {a.dir}'
    out = ['# OA-Track: variables of the files', '',
           'Generated from the file headers by oatrack_inventory.py. The files without the alkalinity adjustment '
           '(_noalk) have the same variables except those with "alkalinity" in the name.', '']
    done = set()
    for p in files:
        k = kind(p)
        if k in done or p.name.endswith('_noalk.nc'):
            continue
        done.add(k)
        lines, n = describe(p)
        out += [f'## {k}', ''] + lines + ['']
        print(f'{p.name}: {n} variable patterns', flush=True)
    out += ['## Files', '', '| File | Size (GB) |' + (' md5 |' if a.md5 else ''), '|---|---|' + ('---|' if a.md5 else '')]
    for p in files:
        row = f'| {p.name} | {p.stat().st_size / 1e9:.1f} |'
        if a.md5:
            h = hashlib.md5()
            with open(p, 'rb') as fh:
                for block in iter(lambda: fh.read(1 << 24), b''):
                    h.update(block)
            row += f' {h.hexdigest()} |'
            print(f'{p.name} {h.hexdigest()}', flush=True)
        out.append(row)
    Path(a.out).write_text('\n'.join(out) + '\n')
    print(f'written {a.out}')


if __name__ == '__main__':
    main()
