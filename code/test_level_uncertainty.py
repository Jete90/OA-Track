#!/usr/bin/env python
"""unit tests of level_uncertainty.py: the state conversion at fixed fCO2, the box quadratic form in its two limits,
the C1/C2 limits of the annual mean. Usage: python test_level_uncertainty.py"""
import numpy as np
import oa_config  # noqa: F401
import level_uncertainty as lu

ny, nx = 6, 12
lat = np.linspace(-75, 75, ny)
lon = np.arange(15, 360, 30.0)
lat2 = np.broadcast_to(lat[:, None], (ny, nx))
si, po4 = np.full((ny, nx), 5.0), np.full((ny, nx), 0.5)
rng = np.random.default_rng(1)
st = {'thetao': 15 + 10 * rng.random((ny, nx)), 'so': 34 + rng.random((ny, nx)),
      'dissic': 2000 + 50 * rng.random((ny, nx)), 'talk': 2300 + 20 * rng.random((ny, nx))}
_, J = lu.jacobian(st, si, po4, lat2)
ok = True


def check(name, a, b, tol):
    global ok
    d = float(np.nanmax(np.abs(np.asarray(a) - np.asarray(b))))
    ok &= d < tol
    print(f'  {"ok  " if d < tol else "FAIL"} {name}: max |difference| {d:.2e} (tol {tol:.0e})')


print('state conversion (linear, fixed fCO2):')
e = lu.propagate(J, lu.obs_to_state(J, 'fco2', np.full((ny, nx), 10.0)))
check('fco2 error -> fco2', e['fco2'], 10.0, 1e-6)
e = lu.propagate(J, lu.obs_to_state(J, 'ta', np.full((ny, nx), 5.0)))
check('ta error -> fco2 unchanged', e['fco2'], 0.0, 1e-6)
check('ta error -> talk', e['talk'], 5.0, 1e-9)
for k in ('omega_t', 'omega_s'):
    e = lu.propagate(J, lu.obs_to_state(J, k, np.full((ny, nx), 0.05)))
    check(f'{k} error -> |omega_arag| 0.05', np.abs(e['omega_arag']), 0.05, 1e-6)
    check(f'{k} error -> fco2 unchanged', e['fco2'], 0.0, 1e-6)
# against a nonlinear recomputation at fixed fCO2: DIC solved for the fCO2 of the base after TA + 5
d = lu.obs_to_state(J, 'ta', np.full((ny, nx), 5.0))
p = {v: st[v] + d.get(v, 0.0) for v in lu.VARS}
r0, r1 = lu.chemistry({v: st[v][None] for v in lu.VARS}, si, po4, lat2), lu.chemistry({v: p[v][None] for v in lu.VARS}, si, po4, lat2)
check('ta error: nonlinear fco2 change (relative)', (r1['fco2'] - r0['fco2']) / r0['fco2'], 0.0, 2e-3)

print('box quadratic form:')
w = np.cos(np.deg2rad(lat))[:, None] * np.ones(nx)
f = rng.normal(size=(ny, nx))
bc = lu.BoxCov(lat, lon, 1e9)
E = bc.sums(f, w)
check('L -> infinity: var = (sum)^2', bc.var(E), np.sum(f * w) ** 2, 1e-4 * np.sum(f * w) ** 2)
bc = lu.BoxCov(lat, lon, 1.0)
E = bc.sums(f, w)
check('L -> 0: var = sum of box sums^2', bc.var(E), np.sum(E ** 2), 1e-9 * np.sum(E ** 2))

print('annual limits C1, C2 (one component, the same error in every month):')
lev = lu.Level.__new__(lu.Level)
lev.masks = {'own': np.ones((ny, nx), bool)}
lev.w = w
lev.names, lev.length = ['a'], {'a': 2400.0}
lev.boxcov = {2400.0: lu.BoxCov(lat, lon, 2400.0)}
lev.ops = {2400.0: {'own': lev.boxcov[2400.0].operator(w)}}
acc = lu.LevelYear(lev)
err = {q: np.full((ny, nx), 0.3) for q in lu.QUANTITIES}
for _ in range(12):
    acc.add({'a': err})
r = acc.result()['omega_arag']
check('C2 = e', r['map2'], 0.3, 1e-12)
check('C1 = e / sqrt(12)', r['map1'], 0.3 / np.sqrt(12), 1e-12)
check('C2 area mean = e (same error everywhere, correlated)', r['series']['own'][1], 0.3 * np.sqrt(lev.boxcov[2400.0].var(
    lev.boxcov[2400.0].sums(np.ones((ny, nx)), w))) / w.sum(), 1e-9)

print('constants (mocsy.merrors):')
c = lu.constants_sigma(st, si, po4, lat2)
print('  omega_arag sigma', float(np.nanmean(c['omega_arag'])), ' ph sigma', float(np.nanmean(c['ph'])),
      ' revelle', float(np.nanmean(c['revelle'])), ' dic', float(np.nanmean(c['dic'])))
ok &= bool(np.all(np.isfinite(c['omega_arag'])) and np.all(c['dic'] == 0))
print('ALL OK' if ok else 'SOME FAILED')
