#!/usr/bin/env python
"""
make_indicator_table.py  (Supplementary Table 2 and Supplementary Data 1: the 15 quantities of OA-Track, 1750-2024)
=======================
The best estimate of the paper for every year from 1750 to 2024, global mean on the common area, with the alkalinity
adjustment: the mean of the four fCO2-products,
    1750-1989   the constrained ESMs (global constraint, the method of the paper; OA-Track <q>_gm_common_global), at
                the CMIP6 CO2 of the year (277.15 ppm in 1750)
    1990-2024   the fCO2-products themselves (<q>_gm_common_obs), the years all four cover (as Fig. 1)
and two extra rows, the values of the paper:
    first row   the 1750 state at the CO2 of AR6, 278.3 ppm (<q>_gm_common_global_state, state 1750)
    last row    the 2024 trend value (quadratic fit of the product years, <q>_gm_common_2024_trend)
1 sigma of an absolute value, all components in quadrature (Supplementary Information, OA-Track): per fCO2-product
the constraint (model years and the 1750 state; none in product years), the mapping (C2, the conservative one, as the
uncertainty budget), the equilibrium constants and the alkalinity adjustment (its full size, |alk - noalk|); then for
the mean of the four sqrt(mean of the squares + spread^2), as the combined OA-Track files. The 2024 trend value takes
the mapping and constants components of the product year 2024.
Columns: the 15 quantities, in the order of Jiang et al. (2023, Tables S5 to S8) with pCO2, CO2* and HCO3- added; CO2
of the row (CMIP6, NOAA, AR6).
Outputs (rc.FIGDIR, and with the manuscript names in rc.FIGDIR/manuscript):
    TableS5_indicators_decades.png/.pdf -> SuppTable2   1750, 1760, ..., 2020, 2024 and the two extra rows
    Supplementary_Data_1.xlsx                          every year, value and 1 sigma, with a sheet describing it
Needs: OA-Track_<ref>.nc and OA-Track_<ref>_noalk.nc of the four fCO2-products (make_oatrack.py), openpyxl.
Usage (code/):  OA_RUN=u74 OMP_NUM_THREADS=2 python -u make_indicator_table.py
"""
import datetime

import netCDF4
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import oa_config
import regional_common as rc
import make_product as mp

OATRACK = oa_config.DATA / 'oatrack'
PRODUCTS = mp.PRODUCTS
EVAL_YEAR = mp.EVAL_YEAR
SWITCH = mp.SWITCH_COMMON                            # 1990: first year all four fCO2-products cover
ORDER = ['thetao', 'so', 'dic', 'talk', 'pco2', 'fco2', 'co2', 'hco3', 'co3', 'hplus', 'hplus_free', 'ph',
         'omega_arag', 'omega_calc', 'revelle']
HEAD = {'thetao': ('T', '°C'), 'so': ('S', ''), 'dic': ('DIC', 'µmol kg⁻¹'), 'talk': ('TA', 'µmol kg⁻¹'),
        'pco2': ('pCO₂', 'µatm'), 'fco2': ('fCO₂', 'µatm'), 'co2': ('CO₂*', 'µmol kg⁻¹'), 'hco3': ('HCO₃⁻', 'µmol kg⁻¹'),
        'co3': ('CO₃²⁻', 'µmol kg⁻¹'), 'hplus': ('[H⁺] total', 'nmol kg⁻¹'), 'hplus_free': ('[H⁺] free', 'nmol kg⁻¹'),
        'ph': ('pH', 'total scale'), 'omega_arag': ('Ωarag', ''), 'omega_calc': ('Ωcalc', ''), 'revelle': ('Revelle', '')}
LEVELS = ('sigma_mapping_c2', 'sigma_constants')
DECADES = list(range(1750, 2021, 10)) + [EVAL_YEAR]


def rd(f, name):
    return np.ma.filled(f[name][...].astype('f8'), np.nan)


def per_product(ref):
    """value and squared 1 sigma of every row for one fCO2-product: rows 'AR6 1750', 1750..2024, 'trend 2024'"""
    with netCDF4.Dataset(OATRACK / f'OA-Track_{ref}.nc') as f, netCDF4.Dataset(OATRACK / f'OA-Track_{ref}_noalk.nc') as g:
        yh = np.asarray(f['year'][:]).astype(int)
        yo = np.asarray(f['year_obs'][:]).astype(int)
        states = [str(s) for s in f['state'][:]]
        i17 = states.index('1750')
        ko = np.searchsorted(yo, EVAL_YEAR)
        assert yo[ko] == EVAL_YEAR, f'{ref}: no product year {EVAL_YEAR}'
        out, co2 = {}, {}
        for q in ORDER:
            p = f'{q}_gm_common'
            mh, mo = yh < SWITCH, (yo >= SWITCH) & (yo <= EVAL_YEAR)
            # value: model years, product years, then the two states of the paper
            v = np.r_[rd(f, f'{p}_global_state')[i17], rd(f, f'{p}_global')[mh], rd(f, f'{p}_obs')[mo],
                      float(rd(f, f'{p}_{EVAL_YEAR}_trend'))]
            v0 = np.r_[rd(g, f'{p}_global_state')[i17], rd(g, f'{p}_global')[mh], rd(g, f'{p}_obs')[mo],
                       float(rd(g, f'{p}_{EVAL_YEAR}_trend'))]
            con = np.r_[rd(f, f'{p}_global_state_sigma')[i17], rd(f, f'{p}_global_sigma')[mh], np.zeros(mo.sum()), 0.0]
            lev = sum(np.r_[rd(f, f'{p}_{lv}_state')[i17], rd(f, f'{p}_{lv}')[mh], rd(f, f'{p}_obs_{lv}')[mo],
                            rd(f, f'{p}_obs_{lv}')[ko]] ** 2 for lv in LEVELS)
            out[q] = (v, con ** 2 + lev + (v - v0) ** 2)
        co2 = np.r_[float(rd(f, 'state_xco2')[i17]), rd(f, 'xco2')[yh < SWITCH], rd(f, 'xco2_obs')[mo], np.nan]
        years = np.r_[yh[mh], yo[mo]]
    assert np.array_equal(years, np.arange(1750, EVAL_YEAR + 1)), f'{ref}: years {years[0]}..{years[-1]}, not continuous'
    return out, co2


def combine():
    P = {r: per_product(r) for r in PRODUCTS}
    rows = ['1750 (AR6, 278.3 ppm)'] + [str(y) for y in range(1750, EVAL_YEAR + 1)] + [f'{EVAL_YEAR} (trend value)']
    V, S = {}, {}
    for q in ORDER:
        A = np.array([P[r][0][q][0] for r in PRODUCTS])
        S2 = np.array([P[r][0][q][1] for r in PRODUCTS])
        assert np.all(np.isfinite(A)), f'{q}: missing values in {[r for r in PRODUCTS if not np.all(np.isfinite(P[r][0][q][0]))]}'
        V[q] = A.mean(0)
        S[q] = np.sqrt(S2.mean(0) + A.std(0, ddof=1) ** 2)
    co2 = P[PRODUCTS[0]][1]
    src = ['state'] + ['constrained ESMs'] * (SWITCH - 1750) + ['fCO2-products'] * (EVAL_YEAR - SWITCH + 1) + ['trend fit']
    return rows, V, S, co2, src


def decimals(s):
    """decimal places that give a 1 sigma of s two significant digits"""
    s = np.asarray(s, 'f8')
    s = s[np.isfinite(s) & (s > 0)]
    return max(0, 1 - int(np.floor(np.log10(np.median(s))))) if s.size else 2


def fmt(v, s, d=None):
    """value and 1 sigma with d decimal places (default: two significant digits of the sigma)"""
    d = decimals([s]) if d is None else d
    return f'{v:.{d}f}', (f'{s:.{d}f}' if np.isfinite(s) else '')


def table_png(rows, V, S, co2):
    keep = [0] + [1 + y - 1750 for y in DECADES] + [len(rows) - 1]
    blocks = (ORDER[:8], ORDER[8:])
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 9.4))
    for ax, cols in zip(axes, blocks):
        ax.set_axis_off()
        head = ['Year', 'CO₂\n(ppm)'] + [HEAD[q][0] + f'\n{HEAD[q][1]}' for q in cols]
        dec = {q: decimals(S[q][keep]) for q in cols}     # one precision per column
        cell = []
        for i in keep:
            r_ = [rows[i].replace(' (AR6, 278.3 ppm)', ' AR6').replace(' (trend value)', ' trend'),
                  '' if not np.isfinite(co2[i]) else f'{co2[i]:.2f}'.rstrip('0').rstrip('.')]
            for q in cols:
                a, b = fmt(V[q][i], S[q][i], dec[q])
                r_.append(f'{a} ± {b}' if b else a)
            cell.append(r_)
        t = ax.table(cellText=cell, colLabels=head, loc='upper center', cellLoc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(5.6)
        t.scale(1.0, 0.95)
        for c_ in range(len(head)):
            t[0, c_].set_height(t[0, c_].get_height() * 1.9)
        for (r_, c_), c in t.get_celld().items():
            c.set_linewidth(0.3)
            c.set_edgecolor('#bbbbbb')
            if r_ == 0:
                c.set_text_props(fontweight='bold')
                c.set_facecolor('#eef1f5')
            elif r_ in (1, len(keep)):
                c.set_facecolor('#f6efe4')        # the two values of the paper
    fig.tight_layout(h_pad=0.6)
    return fig


def excel(rows, V, S, co2, src, path):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = 'global mean, common area'
    head = ['year', 'source', 'CO2 (ppm)']
    for q in ORDER:
        u = f' ({HEAD[q][1]})' if HEAD[q][1] else ''
        head += [f'{HEAD[q][0]}{u}', f'{HEAD[q][0]} 1 sigma']
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True)
    hl = PatternFill('solid', fgColor='F6EFE4')
    for i, r_ in enumerate(rows):
        line = [int(r_) if r_.isdigit() else r_, src[i], None if not np.isfinite(co2[i]) else round(float(co2[i]), 2)]
        for q in ORDER:
            line += [float(V[q][i]), float(S[q][i])]
        ws.append(line)
        if not r_.isdigit():
            for c in ws[ws.max_row]:
                c.fill = hl
    for col in ws.iter_cols(min_col=4, min_row=2):
        for c in col:
            c.number_format = '0.0000'
    ws.freeze_panes = 'D2'
    info = wb.create_sheet('read me')
    for line in (__doc__.strip().split('\n') + ['', f'created {datetime.datetime.now():%Y-%m-%d %H:%M}, run {oa_config.RUN}, '
                                                    f'area weights: {oa_config.WEIGHTS_TEXT}']):
        info.append([line])
    wb.save(path)


def main():
    rows, V, S, co2, src = combine()
    out = rc.FIGDIR
    man = f'{out}/manuscript'
    import os
    os.makedirs(man, exist_ok=True)
    fig = table_png(rows, V, S, co2)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}/TableS5_indicators_decades.{ext}', dpi=300)
        fig.savefig(f'{man}/SuppTable2.{ext}', dpi=300)
    excel(rows, V, S, co2, src, f'{man}/Supplementary_Data_1.xlsx')
    k17, k24 = 0, len(rows) - 1
    a, b = fmt(V['omega_arag'][k17], S['omega_arag'][k17])
    c, d = fmt(V['omega_arag'][k24], S['omega_arag'][k24])
    print(f'Omega_arag 1750 (AR6) {a} +- {b}, {EVAL_YEAR} trend value {c} +- {d}; decline '
          f'{100 * (1 - V["omega_arag"][k24] / V["omega_arag"][k17]):.2f} % (paper, common area, with alkalinity)')
    print(f'written {man}/SuppTable2.png/.pdf and {man}/Supplementary_Data_1.xlsx ({len(rows)} rows)')


if __name__ == '__main__':
    main()
