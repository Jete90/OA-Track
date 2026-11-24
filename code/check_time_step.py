#!/usr/bin/env python
"""
check_time_step.py
==================
Checks, for every model input the chain reads (historical, piControl, ssp126/245/370/585), that the time step the
code uses (ModelRun.step of calc_bias_adjusted.py: 12 = monthly, 1 = annual) agrees with the files themselves:

  - the number of time steps per calendar year on the time axis (all years, not only the first two), the test
  - the global attribute 'frequency' of the four input files (printed only: cdo yearmean keeps 'mon')
  - the time axis length of the four variables
  - the step stored with the derived files: bias file (number of slots, attribute 'slots') and smoothed-state
    file of calc_smooth_states.py (attribute 'step'), for the reference given by --reference

Prints one line per run and a summary; 'MISMATCH' marks any disagreement.

Usage (code/):  OA_RUN=u74 python -u check_time_step.py [--reference OceanSODA]
"""
import argparse
from collections import Counter

import netCDF4
import numpy as np

import calc_bias_adjusted as cb
import calc_smooth_states as cs
import calc_pi_xco2 as cpx

EXPS = ('historical', 'piControl', 'ssp126', 'ssp245', 'ssp370', 'ssp585')


def file_info(path):
    with netCDF4.Dataset(path) as f:
        freq = getattr(f, 'frequency', '?')
        t = f['time']
        years, _ = cb.cm.year_month(t[:], t.units, getattr(t, 'calendar', 'standard'))
    per_year = Counter(int(y) for y in years)
    return freq, len(years), sorted(set(per_year.values())), int(min(per_year)), int(max(per_year))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', default='OceanSODA', choices=list(cb.REFERENCES))
    a = ap.parse_args()
    cb.set_reference(a.reference)
    runs = cpx.find_runs(None)
    bad, steps = [], Counter()
    for model, member in runs:
        for exp in EXPS:
            paths = {v: cb.cm.input_path(v, model, exp, member) for v in cb.VARS}
            if all(p is None for p in paths.values()):
                continue
            tag = f'{model:16s} {exp:10s}'
            if any(p is None for p in paths.values()):
                print(f'{tag} MISMATCH: missing {[v for v, p in paths.items() if p is None]}')
                bad.append(tag)
                continue
            info = {v: file_info(p) for v, p in paths.items()}
            run = cb.ModelRun(model, exp, member)
            step, nt, y0, y1 = run.step, run.nt, int(run.years[0]), int(run.years[-1])
            run.close()
            msg = []
            for v, (freq, n, per, fy, ly) in info.items():
                # the 'frequency' attribute is only printed: cdo yearmean keeps 'mon' from the source files,
                # so the time axis (steps per calendar year) is the test
                if per != [step]:
                    msg.append(f'{v} steps per year {per}')
            lens = {v: i[1] for v, i in info.items()}
            if len(set(lens.values())) > 1:
                msg.append(f'lengths {lens}')
            steps[(exp, step)] += 1
            freqs = '/'.join(sorted({i[0] for i in info.values()}))
            print(f'{tag} step {step:2d} ({freqs}), {nt} steps, years {y0}-{y1}'
                  + (f'  MISMATCH: {"; ".join(msg)}' if msg else ''))
            if msg:
                bad.append(tag)
        # derived files
        try:
            with netCDF4.Dataset(cb.bias_path(model, member)) as f:
                nslot = f['bias_thetao'].shape[0]
                slots = getattr(f, 'slots', '?')
            print(f'{model:16s} bias file ({a.reference}): {nslot} slot(s), "{slots}"')
            if nslot != 1:
                bad.append(f'{model} bias')
        except (FileNotFoundError, OSError):
            print(f'{model:16s} bias file ({a.reference}): not found')
        try:
            with netCDF4.Dataset(cs.out_path(model, member)) as f:
                st = int(f.step)
            print(f'{model:16s} smoothed states: step {st}')
            if st != 1:
                bad.append(f'{model} smoothed')
        except (FileNotFoundError, OSError, AttributeError):
            print(f'{model:16s} smoothed states: not found')
    print('\nsummary (experiment, step): ' + ', '.join(f'{e} {s}: {n}' for (e, s), n in sorted(steps.items())))
    print('all consistent' if not bad else f'MISMATCH or not annual in {len(bad)} case(s): {bad}')


if __name__ == '__main__':
    main()
