#!/bin/bash
# download_mgdsst.sh
# ------------------
# Downloads JMA MGDSST (Merged satellite and in-situ data Global Daily SST, 0.25 deg,
# reanalysis since 1982; Kurihara et al. 2006, Sakurai et al. 2005/2007), the SST used for
# the JMA Ocean CO2 Map (Iida et al. 2021, Table 1), from the NEAR-GOOS RRTDB archive
#   https://www.data.jma.go.jp/goos/data/pub/JMA-product/mgd_sst_glb_D/
# Free and unrestricted access (IOC data exchange policy). Daily files, ~12,800 for 1990-2024.
#
# Usage:  nohup bash download_mgdsst.sh > download_mgdsst.log 2>&1 &
# Re-running continues where it stopped (wget -nc).

set -u
OUT=${OA_OBS:-/oceandata01/observational_data}/MGDSST/daily
BASE=https://www.data.jma.go.jp/goos/data/pub/JMA-product/mgd_sst_glb_D
Y0=1990
Y1=2024

mkdir -p $OUT
cd $OUT
wget -q -O Readme_mgd_sst_glb_D $BASE/Readme_mgd_sst_glb_D

for y in $(seq $Y0 $Y1); do
    mkdir -p $y
    # the archive is organised in year folders; accept only data files, no index pages
    wget -q -r -l 2 -np -nd -nc -e robots=off -P $y \
         -R "index.html*,*.html,*.htm,*.php" $BASE/$y/ \
        || echo "check year $y"
    echo "$y: $(ls $y | wc -l) files"
done
