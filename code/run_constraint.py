#!/usr/bin/env python
"""
run_constraint.py
=================
Runs the emergent constraint for all products (constraint_refs.py) and again on the common mask
(common_mask_cell.py), prints both tables, and saves everything the figure notebooks need:

    data/constraint_results.pkl   dict with
        results         per product, each on its own constraint mask
        results_common  per product, on the common mask
        cm, lat_c, lon_c  the common mask and its grid (OceanSODA grid)

Run once from the terminal (and again whenever inputs or the constraint change):
    cd <production folder>/code
    $PY run_constraint.py | tee run_constraint.log
"""
import pickle
import oa_config   # run: data folder, mocsy constants, GFDL density (OA_RUN)
from pathlib import Path

_HERE = Path(__file__).resolve().parent      # not CODE: constraint_refs.py redefines CODE as a string
_OUT = oa_config.DATA / 'constraint_results.pkl'

for _cell in ('constraint_refs.py', 'common_mask_cell.py'):
    print(f'\n##### {_cell}', flush=True)
    exec(compile((_HERE / _cell).read_text(), str(_HERE / _cell), 'exec'))

with open(_OUT, 'wb') as f:
    pickle.dump(dict(results=results, results_common=results_common, cm=cm, lat_c=lat_c, lon_c=lon_c), f)
print(f'\nsaved to {_OUT}')
