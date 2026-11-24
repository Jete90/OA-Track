#!/usr/bin/env python
"""
check_jiang_std.py  (18 November 2026)
==================
Where does the +-0.06 of the present-day state of Findlay et al. (2025) come from? Prints the sheets and the rows of
Jiang et al. (2023) Regions_median_0m.xlsx and Regions_std_0m.xlsx that concern Omega_arag (aragonite), so that the
spread across their models can be read for every decade (1750, 1850, 2010, 2020, ...). Reads only.
Usage (code/):  OA_RUN=u74 python -u check_jiang_std.py
"""
import pandas as pd

import regional_common as rc

pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.max_rows', 200)
for name in ('Regions_median_0m.xlsx', 'Regions_std_0m.xlsx'):
    path = f'{rc.ROOT}/jiang2023/{name}'
    xl = pd.ExcelFile(path)
    print(f'\n===== {name}: sheets {xl.sheet_names}')
    for sh in xl.sheet_names:
        df = xl.parse(sh, header=None)
        rows = df[df.apply(lambda r: r.astype(str).str.contains('ragon|Omega|global|Global|Year|year|Decade|decade',
                                                                regex=True).any(), axis=1)]
        print(f'--- sheet {sh}: {df.shape[0]} rows x {df.shape[1]} columns; first rows:')
        print(df.head(4).to_string())
        print('--- rows mentioning aragonite / global / year:')
        print(rows.head(40).to_string())
