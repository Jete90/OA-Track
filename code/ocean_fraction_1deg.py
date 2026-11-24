#!/usr/bin/env python
"""
ocean_fraction_1deg.py
======================
The ocean fraction of every 1 x 1 degree cell from the ESA CCI Land Cover water bodies ocean map (150 m,
ESACCI-LC-L4-WB-Ocean-Map-150m-P13Y-2000-v4.0.tif; 0 = ocean, 1 = land and inland water), as used for the area weights
of the Planetary Health Check 2026 (ocean proportion of each pixel). The GeoTIFF is read in strips of one degree of
latitude (rasterio, or GDAL's Python bindings), so memory stays small.

Output: ocean_fraction_1deg.nc next to the GeoTIFF, variable ocean_fraction(lat, lon), lat -89.5..89.5,
lon -179.5..179.5.

Usage:  python ocean_fraction_1deg.py PATH/ESACCI-LC-L4-WB-Ocean-Map-150m-P13Y-2000-v4.0.tif
"""
import sys
from pathlib import Path

import netCDF4
import numpy as np


def open_raster(path):
    try:
        import rasterio
        ds = rasterio.open(path)

        def read(r0, nrows):
            from rasterio.windows import Window
            return ds.read(1, window=Window(0, r0, ds.width, nrows))
        return ds.width, ds.height, ds.transform[0], ds.transform[2], ds.transform[5], read
    except ImportError:
        from osgeo import gdal
        ds = gdal.Open(str(path))
        gt = ds.GetGeoTransform()
        band = ds.GetRasterBand(1)
        return ds.RasterXSize, ds.RasterYSize, gt[1], gt[0], gt[3], lambda r0, n: band.ReadAsArray(0, r0, ds.RasterXSize, n)


def main():
    path = Path(sys.argv[1])
    w, h, dx, x0, y0, read = open_raster(path)
    ppd_x, ppd_y = round(1 / dx), round(h / 180)
    print(f'{path.name}: {w} x {h} pixels, {ppd_x} x {ppd_y} per degree, west {x0}, north {y0}')
    assert abs(x0 + 180) < 1e-6 and abs(y0 - 90) < 1e-6 and w == 360 * ppd_x and h == 180 * ppd_y, 'unexpected grid'
    frac = np.full((180, 360), np.nan)
    vals = set()
    for j in range(180):                                   # north to south
        a = read(j * ppd_y, ppd_y)
        if j % 30 == 0:
            vals |= set(np.unique(a[:, ::97]).tolist())
        ocean = (a == 0).reshape(ppd_y, 360, ppd_x)
        frac[179 - j] = ocean.sum(axis=(0, 2)) / (ppd_y * ppd_x)     # south to north (sum: no large float copy)
        if j % 20 == 0:
            print(f'  row {j}/180', flush=True)
    print(f'pixel values seen: {sorted(vals)} (0 = ocean expected)')
    out = path.parent / 'ocean_fraction_1deg.nc'
    with netCDF4.Dataset(out, 'w') as g:
        g.createDimension('lat', 180)
        g.createDimension('lon', 360)
        g.createVariable('lat', 'f8', ('lat',))[:] = np.arange(-89.5, 90)
        g.createVariable('lon', 'f8', ('lon',))[:] = np.arange(-179.5, 180)
        x = g.createVariable('ocean_fraction', 'f4', ('lat', 'lon'))
        x.long_name = f'fraction of the 1 deg cell that is ocean in {path.name} (value 0)'
        x[:] = frac
    print(f'written {out}; ocean area fraction of the globe (cos lat) '
          f'{np.sum(frac * np.cos(np.deg2rad(np.arange(-89.5, 90)))[:, None]) / (360 * np.cos(np.deg2rad(np.arange(-89.5, 90))).sum()):.3f}')


if __name__ == '__main__':
    main()
