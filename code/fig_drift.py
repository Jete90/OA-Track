#!/usr/bin/env python
"""
fig_drift.py  (candidate Supplementary Figure: the drift of the 18 ESMs and its removal)
============
For each ESM, the global mean surface Omega_arag on the common area of the paper, from the raw annual-mean surface
states of piControl and historical, without and with the drift correction of the chain:

    X'(t) = X(t) - [s(k(t)) - mean of s over W']     (drift_tools.Drift.term; s the Enting spline with a 50-year
                                                       cutoff through the piControl annual means, stored per cell
                                                       for T, S, DIC and alkalinity by calc_drift.py)

with W the bias window of OceanSODA-ETHZ (1983-2014; the window only sets the constant of the correction, not its
shape). Omega_arag of the annual-mean state with mocsy and the constants of the run (calc_pi_xco2.mvars_all, DIC and
alkalinity per kg as calc_omega_arag.py: in-situ density, GFDL-CM4/-ESM4 the reference density), GLODAP Si and PO4.
The piControl years are drawn at the historical years they run parallel to (branch year), so that the drift removed
from a historical year can be read off directly. This is a check of the drift step, not of the bias adjustment: the
states are the raw model states.

Printed per model: the drift of Omega_arag over the piControl years parallel to 1850-2014 (raw minus corrected, 2014
minus 1850), its largest absolute value over the drawn piControl years (--pi-range, default parallel to 1700-2100), and the standard deviation of the corrected
piControl (the variability left). Output:
    data/<run>/drift/drift_omega_common.npz                  the series (for a notebook cell)
    data/<run>/figures/FigS31_drift.png/.pdf                 the figure (18 panels)
In the notebook: fd.load() if fd.DEST exists, else fd.compute() (30 to 60 minutes, once), then fd.plot(...).
Usage (code/):  OA_RUN=u74 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
                python -u fig_drift.py 2>&1 | tee ../logs/u74/fig_drift.log
"""
import argparse
import glob
import pickle
from pathlib import Path

import netCDF4
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import oa_config
import calc_omega_arag as cm
import calc_omega_arag_obs as co
import calc_pi_xco2 as px
import drift_tools as dt
import regional_common as rc

WINDOW = oa_config.WINDOWS['OceanSODA']
CHUNK_YEARS = 20


def onto(lat, lon, la, lo, a):
    jy = np.abs(lat[:, None] - la[None, :]).argmin(1)
    ix = np.abs(((lon[:, None] - lo[None, :]) + 180) % 360 - 180).argmin(1)
    return a[..., jy, :][..., ix]


class Raw:
    """annual-mean surface T (degC), S, DIC and alkalinity (umol/kg) of one run, as calc_omega_arag.py reads them"""

    def __init__(self, model, exp, member):
        self.model, self.tag = model, f'{model} {exp} {member}'
        paths = {v: cm.input_path(v, model, exp, member) for v in cm.VARS}
        miss = [v for v, p in paths.items() if p is None]
        if miss:
            raise FileNotFoundError(f'{self.tag}: no input for {miss}')
        self.src = {v: netCDF4.Dataset(paths[v]) for v in cm.VARS}
        self.var = {v: self.src[v].variables[v] for v in cm.VARS}
        self.lat = np.asarray(self.src['dissic']['lat'][:], 'f8')
        self.lon = np.asarray(self.src['dissic']['lon'][:], 'f8')
        self.lon2, self.lat2 = np.meshgrid(self.lon, self.lat)
        tv = self.src['dissic']['time']
        nt = min(self.var[v].shape[0] for v in cm.VARS)
        yy, mm = cm.year_month(tv[:nt], tv.units, getattr(tv, 'calendar', 'standard'))
        self.step = 12 if nt > 1 and (yy[1], mm[1]) != (yy[0] + 1, mm[0]) else 1
        nt -= nt % self.step
        self.years = np.asarray(yy[:nt:self.step], 'i8')
        self.nt = nt
        t0, s0 = cm.read(self.var['thetao'], 0, self.step), cm.read(self.var['so'], 0, self.step)
        kelvin = getattr(self.var['thetao'], 'units', '').strip() in ('K', 'degK', 'kelvin') or np.nanmedian(t0) > 100
        self.toff = -273.15 if kelvin else 0.0
        self.sscale = 1000.0 if np.nanmedian(s0) < 1 else 1.0
        self.cunits = {v: getattr(self.var[v], 'units', 'mol m-3') for v in ('dissic', 'talk')}

    def years_block(self, k0, k1):
        """annual means of years k0..k1-1 (indices into self.years); DIC and alkalinity in umol/kg"""
        i0, i1 = k0 * self.step, k1 * self.step
        ann = lambda a: a.reshape(-1, self.step, *a.shape[1:]).mean(axis=1)
        t = cm.read(self.var['thetao'], i0, i1) + self.toff
        s = cm.read(self.var['so'], i0, i1) * self.sscale
        out = {'thetao': ann(t), 'so': ann(s)}
        rho = cm.model_density(self.model, t, s, self.lon2, self.lat2)
        for v in ('dissic', 'talk'):
            per_m3 = cm.to_mol_m3(cm.read(self.var[v], i0, i1), self.cunits[v], cm.surface_density(t, s, self.lon2, self.lat2))
            out[v] = ann(per_m3 / rho * 1e6)                       # umol/kg, as the drift files and the chain
        return out

    def close(self):
        for f in self.src.values():
            f.close()


def omega(st, si, po4, lat2):
    t, s, d, a = (st[v] for v in cm.VARS)
    ok = np.isfinite(t) & np.isfinite(s) & np.isfinite(d) & np.isfinite(a) & (s > 0) & (d > 0) & (a > 0) \
        & np.isfinite(si) & np.isfinite(po4)
    om = np.full(t.shape, np.nan)
    if ok.any():
        b = lambda x: np.broadcast_to(x, t.shape)[ok]
        r = px.mvars_all(t[ok], s[ok], d[ok] * 1e-6, a[ok] * 1e-6, b(si * 1e-6), b(po4 * 1e-6), b(lat2))
        o_ = np.asarray(r[6], 'f8')
        om[ok] = np.where(px.valid_omega(o_, np.asarray(r[0], 'f8')), o_, np.nan)
    return om


def series(raw, drift, exp, mask, w, lat_c, lon_c, si, po4, ka, kb):
    """global mean Omega_arag on the common area, raw and drift-corrected, for the years ka..kb-1 of the run"""
    n = kb - ka
    g0, g1 = np.full(n, np.nan), np.full(n, np.nan)
    for j0 in range(0, n, CHUNK_YEARS):
        j1 = min(j0 + CHUNK_YEARS, n)
        k0, k1 = ka + j0, ka + j1
        st = raw.years_block(k0, k1)
        yrs = raw.years[k0:k1]
        cor = {v: st[v] - drift.term(v, exp, yrs, WINDOW) for v in cm.VARS}
        for g, x in ((g0, st), (g1, cor)):
            om = onto(lat_c, lon_c, raw.lat, raw.lon, omega(x, si, po4, raw.lat2))
            ok = mask[None] & np.isfinite(om)
            g[j0:j1] = np.sum(np.where(ok, om * w, 0), axis=(1, 2)) / np.sum(np.where(ok, w, 0), axis=(1, 2))
    return g0, g1


DEST = dt.OUT / 'drift_omega_common.npz'
EXPS = ('piControl', 'historical')
KEYS = ('years', 'raw', 'corrected')


def compute(models=None, pi_range=(1700, 2100)):
    """the series of every ESM (dict model -> dict exp -> (years, raw, corrected)), stored in DEST"""
    pk = pickle.load(open(oa_config.DATA / 'constraint_results.pkl', 'rb'))
    r0 = pk['results_common']['OceanSODA']
    lat_c, lon_c, mask = np.asarray(r0['lat'], 'f8'), np.asarray(r0['lon'], 'f8'), np.asarray(r0['mask'], bool)
    w = oa_config.area_weights(lat_c, lon_c)
    models = sorted({m for m, _ in px.find_runs(models)})
    print(f'{len(models)} ESMs; window of the correction {WINDOW}; common area of the paper; weights {oa_config.WEIGHTS}')
    res = {}
    for model in models:
        dfiles = sorted(glob.glob(str(dt.OUT / f'drift_{model}_*.nc')))
        assert len(dfiles) == 1, f'{model}: drift files {dfiles}'
        member = Path(dfiles[0]).stem[len(f'drift_{model}_'):]
        drift = dt.Drift(model, member)
        runs = {e: [m for mo, ex, m in cm.find_runs([e], [model])] for e in EXPS}
        out, mems = {}, {}
        for exp in EXPS:
            mem = member if member in runs[exp] else runs[exp][0]
            raw = Raw(model, exp, mem)
            assert raw.lat.size == drift.lat.size and raw.lon.size == drift.lon.size, f'{model}: grids differ'
            si, po4 = co.load_nutrients(raw.lat, raw.lon)
            par = raw.years if exp == 'historical' else drift.hist0 + (raw.years - drift.by)   # piControl: parallel year
            sel = np.where((par >= pi_range[0]) & (par <= pi_range[1]))[0] if exp == 'piControl' else np.arange(par.size)
            ka, kb = int(sel.min()), int(sel.max()) + 1
            g0, g1 = series(raw, drift, exp, mask, w, lat_c, lon_c, si, po4, ka, kb)
            out[exp], mems[exp] = (par[ka:kb], g0, g1), mem
            raw.close()
        res[model] = out
        summary(model, out, mems)
    np.savez(DEST, models=np.array(models), window=np.array(WINDOW),
             **{f'{m}|{e}|{k}': res[m][e][i] for m in models for e in EXPS for i, k in enumerate(KEYS)})
    print(f'written {DEST}')
    return res


def load():
    z = np.load(DEST)
    res = {str(m): {e: tuple(z[f'{m}|{e}|{k}'] for k in KEYS) for e in EXPS} for m in z['models']}
    for m, out in res.items():
        summary(m, out)
    return res


def drift_hist(out):
    """drift of Omega_arag removed over 1850-2014 (parallel piControl years): raw minus corrected, 2014 minus 1850"""
    yp, p0, p1 = out['piControl']
    par = (yp >= 1850) & (yp <= 2014)
    dd = p0 - p1
    return float(np.interp(2014, yp[par], dd[par]) - np.interp(1850, yp[par], dd[par])) if par.sum() > 1 else np.nan


def summary(model, out, mems=None):
    yp, p0, p1 = out['piControl']
    dd = p0 - p1
    print(f'  {model:16s} drift of Omega_arag 1850-2014 (parallel piControl) {drift_hist(out):+.4f}; largest |drift - mean| '
          f'over the drawn piControl ({yp.size} yr) {np.nanmax(np.abs(dd - np.nanmean(dd))):.4f}; sd of the corrected '
          f'piControl {np.nanstd(p1, ddof=1):.4f}' + (f'; members {mems["piControl"]} / {mems["historical"]}' if mems else ''),
          flush=True)


def plot(res):
    """one panel per ESM: piControl (grey) and historical (red), raw (light) and drift removed (dark)"""
    models = sorted(res)
    n, nc = len(models), 6
    fig, axes = plt.subplots(int(np.ceil(n / nc)), nc, figsize=(13.5, 2.1 * np.ceil(n / nc)), sharex=True, squeeze=False)
    for ax, m in zip(axes.flat, models):
        for e, c0, c1 in (('piControl', '#b8bcc4', '#4a4f57'), ('historical', '#f2b8a2', '#c0392b')):
            yrs, g0, g1 = res[m][e]
            ax.plot(yrs, g0, color=c0, lw=0.6)
            ax.plot(yrs, g1, color=c1, lw=0.7)
        ax.set_title(f'{m}  ({drift_hist(res[m]):+.3f})', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axvspan(*WINDOW, color='0.92', zorder=0)
    for ax in axes.flat[n:]:
        ax.set_axis_off()
    for ax in axes[:, 0]:
        ax.set_ylabel('Ω$_{arag}$', fontsize=8)
    fig.legend([plt.Line2D([], [], color=c) for c in ('#b8bcc4', '#4a4f57', '#f2b8a2', '#c0392b')],
               ['piControl, raw', 'piControl, drift removed', 'historical, raw', 'historical, drift removed'],
               loc='lower center', ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+')
    ap.add_argument('--pi-range', nargs=2, type=int, default=(1700, 2100),
                    help='piControl years drawn, as the historical years they run parallel to (default 1700 2100: the '
                         'years of the preindustrial states, the historical runs and the scenarios)')
    a = ap.parse_args()
    fig = plot(compute(a.models, tuple(a.pi_range)))
    for ext in ('png', 'pdf'):
        p = Path(rc.FIGDIR) / f'FigS31_drift.{ext}'
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200)
    print(f'written {Path(rc.FIGDIR) / "FigS31_drift.png"}')


if __name__ == '__main__':
    main()
