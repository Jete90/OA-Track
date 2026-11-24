#!/usr/bin/env python
"""
compare_uexp_versions.py
========================
The 2024 global mean surface Omega_arag of UExP-FNN-U in the pre-release used by the Planetary Health Check 2026
(v2026-pre1, Zenodo 19605928) and in the released v2026-1, computed the same way for both, so that the difference is
the product version alone:

  the product's own saturation_aragonite, annual mean of the 12 months of 2024 per cell (all 12 months required),
  area-weighted with cos(latitude) over
    a) all cells with a value in each version (the Planetary Health Check uses the product's full area)
    b) the cells with a value in both versions (same area)

The Planetary Health Check weights with an ocean-fraction mask (ESA CCI water bodies) and ellipsoidal cell areas;
the difference between its 2.85 and a) of v2026-pre1 is that averaging (and the rounding of 2.85).

Usage:  python compare_uexp_versions.py PRE_RELEASE_FILE [RELEASED_FILE [OCEAN_FRACTION_FILE]]
        (OCEAN_FRACTION_FILE from ocean_fraction_1deg.py: adds the weighting of the Planetary Health Check)
"""
import sys

import netCDF4
import numpy as np

import oa_config
REL = str(oa_config.OBS_DIR / 'UExP-FNN-U/Fordetal_UExP-FNN-U_surface-carbonate-system_v2026-1.nc')
VAR = 'saturation_aragonite'
YEAR = 2024


def year_index(t):
    units = t.units
    if units.lower().startswith('days since 1970-01-15'):
        d = np.datetime64('1970-01-15') + np.asarray(t[:], 'i8').astype('timedelta64[D]')
        y = d.astype('datetime64[Y]').astype(int) + 1970
    else:
        y = np.array([x.year for x in netCDF4.num2date(t[:], units, getattr(t, 'calendar', 'standard'))])
    return np.where(y == YEAR)[0]


def annual_2024(path):
    with netCDF4.Dataset(path) as f:
        v = f[VAR]
        it = year_index(f['time'])
        assert it.size == 12, f'{path}: {it.size} months in {YEAR}'
        dims = list(v.dimensions)
        sl = tuple(slice(int(it[0]), int(it[-1]) + 1) if d == 'time' else slice(None) for d in dims)
        a = np.ma.filled(np.ma.asarray(v[sl]).astype('f8'), np.nan)
        lat_d = [d for d in dims if d.startswith('lat')][0]
        lon_d = [d for d in dims if d.startswith('lon')][0]
        a = np.transpose(a, [dims.index('time'), dims.index(lat_d), dims.index(lon_d)])
        lat = np.asarray(f[next(k for k in ('lat', 'latitude') if k in f.variables)][:], 'f8')
        lon = np.asarray(f[next(k for k in ('lon', 'longitude') if k in f.variables)][:], 'f8')
        hist = getattr(f, 'history', '')[:120]
        ver = {k: getattr(f, k) for k in f.ncattrs() if 'version' in k.lower()}
    a[(a <= 0) | (a > 20)] = np.nan
    return a.mean(axis=0), lat, lon, ver, hist        # NaN unless all 12 months exist


def gm(field, lat, mask, w=None):
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones(field.shape) if w is None else w
    return float(np.sum(field[mask] * w[mask]) / w[mask].sum())


def ellipsoid_area(lat, nlon):
    """area of the 1 deg cells on the WGS84 ellipsoid (authalic latitude formula), (lat, lon)"""
    a_, f_ = 6378137.0, 1 / 298.257223563
    e = np.sqrt(f_ * (2 - f_))
    b2 = (a_ * (1 - f_)) ** 2

    def q(phi):
        s_ = np.sin(np.deg2rad(phi))
        return s_ / (1 - e ** 2 * s_ ** 2) + np.log((1 + e * s_) / (1 - e * s_)) / (2 * e)
    band = np.pi * b2 * (q(np.minimum(lat + 0.5, 90)) - q(np.maximum(lat - 0.5, -90)))   # area of the latitude band
    return (band / nlon)[:, None] * np.ones(nlon)


def phc_weights(frac_file, lat, lon):
    """the weights of the Planetary Health Check: ellipsoidal cell area x ocean fraction (ESA CCI water bodies)"""
    with netCDF4.Dataset(frac_file) as f:
        flat, flon = np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8')
        fr = np.asarray(f['ocean_fraction'][:], 'f8')
    jy = np.abs(lat[:, None] - flat[None, :]).argmin(1)
    l = np.where(lon > 180, lon - 360, lon)
    ix = np.abs(((l[:, None] - flon[None, :]) + 180) % 360 - 180).argmin(1)
    return ellipsoid_area(lat, lon.size) * fr[np.ix_(jy, ix)]


def main():
    pre = sys.argv[1]
    rel = sys.argv[2] if len(sys.argv) > 2 else REL
    P, latp, lonp, verp, _ = annual_2024(pre)
    R, latr, lonr, verr, _ = annual_2024(rel)
    assert np.allclose(latp, latr) and np.allclose(lonp % 360, lonr % 360), 'the two versions are on different grids'
    mp, mr = np.isfinite(P), np.isfinite(R)
    both = mp & mr
    print(f'pre-release: {pre} {verp}')
    print(f'released:    {rel} {verr}')
    print(f'cells with a 2024 value: pre-release {mp.sum()}, released {mr.sum()}, both {both.sum()}')
    a_p, a_r = gm(P, latp, mp), gm(R, latr, mr)
    b_p, b_r = gm(P, latp, both), gm(R, latr, both)
    dec = lambda o: 100 * (1 - o / 3.57)
    print(f'a) full area of each version: pre-release {a_p:.4f} ({dec(a_p):.2f} %), released {a_r:.4f} ({dec(a_r):.2f} %), '
          f'version effect {dec(a_r) - dec(a_p):+.2f} points')
    print(f'b) same cells:                pre-release {b_p:.4f} ({dec(b_p):.2f} %), released {b_r:.4f} ({dec(b_r):.2f} %), '
          f'version effect {dec(b_r) - dec(b_p):+.2f} points')
    print(f'Planetary Health Check 2.85 ({dec(2.85):.2f} %): minus pre-release a) {2.85 - a_p:+.4f} '
          f'(their averaging and the rounding of 2.85; decline {dec(2.85) - dec(a_p):+.2f} points)')
    if len(sys.argv) > 3:                                   # the ocean-fraction file of ocean_fraction_1deg.py
        W = phc_weights(sys.argv[3], latp, lonp)
        WA = ellipsoid_area(latp, lonp.size)
        c_p, c_r = gm(P, latp, mp, W), gm(R, latr, mr, W)
        e_p = gm(P, latp, mp, WA)
        print(f'c) ellipsoidal cell area only:  pre-release {e_p:.4f} ({dec(e_p):.2f} %)')
        print(f'd) as the Planetary Health Check (ellipsoidal area x ocean fraction): pre-release {c_p:.4f} '
              f'({dec(c_p):.2f} %), released {c_r:.4f} ({dec(c_r):.2f} %); published 2.85: difference {2.85 - c_p:+.4f}')
        print(f'   step from our averaging to theirs (pre-release): {dec(c_p) - dec(a_p):+.2f} points')
    d = R - P
    print(f'released minus pre-release per cell: mean {np.nanmean(d[both]):+.4f}, 5-95 % {np.nanpercentile(d[both], 5):+.4f} '
          f'to {np.nanpercentile(d[both], 95):+.4f}')


if __name__ == '__main__':
    main()
