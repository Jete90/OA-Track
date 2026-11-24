#!/usr/bin/env python
"""
test_diseq_models.py
====================
The change of the air-sea disequilibrium (surface ocean pCO2 minus atmospheric pCO2, moist air at 1 atm) in each
of the 18 ESMs, from their raw output, on the common area of the paper (cos(latitude)):

  pCO2 from the model's surface DIC, TA, T, S (annual) with mocsy (constants of the chain, GLODAP Si and PO4),
  DIC and TA converted from mol m-3 with the surface density (EOS-80 at 0 dbar; GFDL: 1035 kg m-3, as the chain),
  atmospheric CO2 of the CMIP6 history and SSP2-4.5 (co2_history), the control at the 1850 value;
  drift: the same for the parallel piControl years, linear fit, subtracted.

Printed per model: the disequilibrium in 1850-1869, the change to 1995-2014 and to 2015-2024 (SSP2-4.5), and the
sensitivity of the decline to it (0.17 points per uatm, test_disequilibrium_alk.py) as the spread of the decline
that the disequilibrium alone creates across the models.

Usage (code/):  OA_RUN=u74 python -u test_diseq_models.py
"""
import numpy as np

import calc_omega_arag as cm
import calc_omega_arag_obs as co
import calc_pi_xco2 as cpx
import co2_history
import drift_tools as dt
import mocsy
from test_alk_diseq_models import common_mask_on, series, ph2o
from test_alk_models_trend import surface

SENS = 0.17                                   # points of decline per uatm
W0, W1, W2 = (1850, 1869), (1995, 2014), (2015, 2024)
NAMES = {'t': ('thetao', 'tos'), 's': ('so', 'sos'), 'dic': ('dissic',), 'ta': ('talk',)}


def rho0(t, s):
    """EOS-80 density at 0 dbar (kg m-3)"""
    a = 999.842594 + 6.793952e-2 * t - 9.095290e-3 * t ** 2 + 1.001685e-4 * t ** 3 - 1.120083e-6 * t ** 4 + 6.536332e-9 * t ** 5
    b = 8.24493e-1 - 4.0899e-3 * t + 7.6438e-5 * t ** 2 - 8.2467e-7 * t ** 3 + 5.3875e-9 * t ** 4
    c = -5.72466e-3 + 1.0227e-4 * t - 1.6546e-6 * t ** 2
    return a + b * s + c * s ** 1.5 + 4.8314e-4 * s ** 2


def fields(model, member, exp):
    out = {}
    for k, names in NAMES.items():
        p = next(((n, cm.input_path(n, model, exp, member)) for n in names if cm.input_path(n, model, exp, member)), None)
        if p is None:
            return None
        y, a, lat, lon = surface(p[1], p[0])
        out[k] = (y, a)
    ys = [v[0] for v in out.values()]
    yc = sorted(set(ys[0]).intersection(*map(set, ys[1:])))
    sel = {k: np.isin(v[0], yc) for k, v in out.items()}
    return np.array(yc), {k: v[1][sel[k]] for k, v in out.items()}, lat, lon


def diseq(model, years, F, lat, lon, xco2):
    """area mean of ocean minus air pCO2 for each year"""
    m = common_mask_on(lat, lon)
    si, po4 = co.load_nutrients(lat, lon)
    la = np.broadcast_to(lat[:, None], m.shape)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones(m.shape)
    out = np.full(years.size, np.nan)
    for i in range(years.size):
        t, s = F['t'][i], F['s'][i]
        if np.nanmedian(t) > 100:
            t = t - 273.15
        ok = m & np.isfinite(t) & np.isfinite(s) & np.isfinite(F['dic'][i]) & np.isfinite(F['ta'][i])
        rho = cm.REF_DENSITY.get(model) or rho0(t[ok], s[ok])
        n = int(ok.sum())
        r = mocsy.mvars(temp=t[ok], sal=s[ok], alk=F['ta'][i][ok] / rho, dic=F['dic'][i][ok] / rho,
                        sil=si[ok] * 1e-6, phos=po4[ok] * 1e-6, patm=np.ones(n), depth=np.zeros(n), lat=la[ok],
                        optcon='mol/kg', optt='Tpot', optp='m', optb=co.MOCSY_OPTB, optk1k2=co.MOCSY_OPTK1K2)
        d = np.asarray(r[1], 'f8') - xco2[i] * (1 - ph2o(t[ok], s[ok]))
        good = np.isfinite(d)
        out[i] = np.sum(d[good] * w[ok][good]) / np.sum(w[ok][good])
    return out


def main():
    yh, xh = series(co2_history.historical())
    ys, xs = series(co2_history.scenario('ssp245'))
    atm = dict(zip(yh, xh)); atm.update({y: x for y, x in zip(ys, xs) if y not in atm})
    x1850 = atm[1850]
    print(f'{"model":16s} {"1850-69":>8s} {"->95-14":>8s} {"->15-24":>8s} {"drift":>7s}  (uatm, drift-corrected changes)')
    rows = {}
    for model, member in cpx.find_runs(None):
        try:
            H = fields(model, member, 'historical'); S = fields(model, member, 'ssp245'); C = fields(model, member, 'piControl')
            if H is None or S is None or C is None:
                print(f'{model:16s} files missing (historical {H is not None}, ssp245 {S is not None}, piControl {C is not None})', flush=True)
                continue
            years = np.concatenate([H[0], S[0][S[0] > H[0][-1]]])
            F = {k: np.concatenate([H[1][k], S[1][k][S[0] > H[0][-1]]]) for k in NAMES}
            dis = diseq(model, years, F, H[2], H[3], np.array([atm[y] for y in years]))
            d = dt.Drift(model, member)
            par = d.by + (years - 1850)
            ok = (par >= C[0].min()) & (par <= C[0].max())
            ic = np.array([np.argmin(np.abs(C[0] - p_)) for p_ in par[ok]])
            Fc = {k: C[1][k][ic] for k in NAMES}
            disc = diseq(model, par[ok], Fc, C[2], C[3], np.full(ok.sum(), x1850))
            b = np.polyfit(par[ok], disc, 1)
            dr = np.polyval(b, par) - np.polyval(b, par[0])
            dc = dis - dr
            a = lambda w_: dc[(years >= w_[0]) & (years <= w_[1])].mean()
            rows[model] = (a(W0), a(W1) - a(W0), a(W2) - a(W0))
            print(f'{model:16s} {a(W0):+8.1f} {a(W1) - a(W0):+8.1f} {a(W2) - a(W0):+8.1f} '
                  f'{dr[(years >= W2[0]) & (years <= W2[1])].mean():+7.1f}', flush=True)
        except Exception as e:
            print(f'{model:16s} failed: {type(e).__name__}: {e}', flush=True)
    if rows:
        v = np.array(list(rows.values()))
        for j, lab in ((1, '1850-69 -> 1995-2014'), (2, '1850-69 -> 2015-2024')):
            print(f'  change {lab}: mean {v[:, j].mean():+.1f}, sd {v[:, j].std(ddof=1):.1f}, range '
                  f'{v[:, j].min():+.1f} to {v[:, j].max():+.1f} uatm ({len(v)} models); spread of the decline from the '
                  f'disequilibrium alone {SENS} x sd = {SENS * v[:, j].std(ddof=1):.2f} points')
        print(f'  level 1850-1869: mean {v[:, 0].mean():+.1f}, sd {v[:, 0].std(ddof=1):.1f} uatm (cancels through the anchoring)')


if __name__ == '__main__':
    main()
