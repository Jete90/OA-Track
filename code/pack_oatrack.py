#!/usr/bin/env python
"""
pack_oatrack.py
===============
Packs OA-Track for its SEANOE record (https://doi.org/10.17882/121840) from the output of make_oatrack_product.py:

    OA-Track_README.pdf                         the README
    OA-Track_v<version>_global_means.zip        the folder OA-Track_v<version>_global_means/ with the global means of
                                                every quantity (.nc and .csv), the README (.md and .pdf), and the list
                                                of all variables (OA-Track_variables.pdf and .xlsx)
    OA-Track_<q>_maps_obs_based_products.nc     one file per quantity (hard links to the output)
    OA-Track_<q>_maps_esms.nc                   one file per quantity, once they exist
    MANIFEST.md5                                the md5 of every file, those in the zip included

The README and the list of variables are expected next to the output (--src). Every file of oatrack_names.files()
must be there, except the maps of the ESMs, which are added when they exist. The zip is checked against its sources.

    OA_RUN=u74 python pack_oatrack.py [--src <dir>] [--out <dir>] [--version <version>]
"""
import argparse
import hashlib
import os
import shutil
import zipfile
from pathlib import Path

import oa_config
import oatrack_names as N

DOCS = ('OA-Track_README.md', 'OA-Track_README.pdf', 'OA-Track_variables.pdf', 'OA-Track_variables.xlsx')


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(Path(oa_config.DATA) / 'oatrack_product'))
    ap.add_argument('--out', default=None, help='default: <src>/../oatrack_seanoe_v<version>')
    ap.add_argument('--version', default='1.0')
    a = ap.parse_args()
    src = Path(a.src)
    out = Path(a.out) if a.out else src.parent / f'oatrack_seanoe_v{a.version}'
    names = [F.name for F in N.files()]
    glob_ = [n for n in names if '_global_mean.' in n] + [n.replace('.nc', '.csv') for n in names
                                                           if n.endswith('_global_mean.nc')]
    maps_obs = [n for n in names if n.endswith('_maps_obs_based_products.nc')]
    maps_esm = [n for n in names if n.endswith('_maps_esms.nc')]
    missing = [n for n in glob_ + maps_obs + list(DOCS) if not (src / n).is_file()]
    assert not missing, f'missing in {src}: {missing}'
    esm_have = [n for n in maps_esm if (src / n).is_file()]
    assert len(esm_have) in (0, len(maps_esm)), f'only {len(esm_have)} of {len(maps_esm)} maps of the ESMs'
    assert not out.exists(), f'{out} exists: remove it first'
    out.mkdir(parents=True)

    folder = f'OA-Track_v{a.version}_global_means'
    zname = out / f'{folder}.zip'
    members = sorted(glob_) + list(DOCS)
    with zipfile.ZipFile(zname, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for n in members:
            z.write(src / n, f'{folder}/{n}')
    sums = {f'{folder}/{n}': md5(src / n) for n in members}
    with zipfile.ZipFile(zname) as z:                                     # check the zip against its sources
        assert z.testzip() is None
        for n in members:
            h = hashlib.md5(z.read(f'{folder}/{n}')).hexdigest()
            assert h == sums[f'{folder}/{n}'], n
    shutil.copy2(src / 'OA-Track_README.pdf', out / 'OA-Track_README.pdf')
    for n in maps_obs + esm_have:
        link_or_copy(src / n, out / n)
    top = ['OA-Track_README.pdf', zname.name] + maps_obs + esm_have
    for n in top:
        sums[n] = md5(out / n)
    assert sums['OA-Track_README.pdf'] == sums[f'{folder}/OA-Track_README.pdf']
    with open(out / 'MANIFEST.md5', 'w') as f:
        for n in top + [f'{folder}/{m}' for m in members]:
            f.write(f'{sums[n]}  {n}\n')
    size = lambda p: p.stat().st_size / 1e6
    print(f'{out}: {len(top) + 1} files, {sum(size(out / n) for n in top):.0f} MB')
    print(f'  {zname.name}: {len(members)} files, {size(zname):.0f} MB (unpacked '
          f'{sum(size(src / n) for n in members):.0f} MB)')
    print(f'  maps: {len(maps_obs)} of the obs-based products, {len(esm_have)} of the ESMs'
          + ('' if esm_have else ' (added once they exist)'))
    print(f'  MANIFEST.md5: {len(top) + len(members)} lines, md5 {md5(out / "MANIFEST.md5")}')


if __name__ == '__main__':
    main()
