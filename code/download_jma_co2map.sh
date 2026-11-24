#!/bin/bash
# download_jma_co2map.sh
# ----------------------
# Downloads the global JMA Ocean CO2 Map (JMA-MLR, v2025, 1990-2024, monthly, 1x1 deg;
# Iida et al., 2021, J. Oceanogr.) from
#   https://www.data.jma.go.jp/kaiyou/english/co2_flux/co2_flux_data_en.html
# unzips the yearly files and merges them into one file with cdo.
#
# Variables: pCO2s, flux, dic, ndic, ta, nta, pH (no SST/SSS in the global files).
#
# Usage:  bash download_jma_co2map.sh            (skips files already downloaded)

set -u
OUT=${OA_OBS:-/oceandata01/observational_data}/JMA-MLR_v2025
URL=https://www.data.jma.go.jp/kaiyou/data/english/co2_flux/grid
VERSION=v2025
Y0=1990
Y1=2024

mkdir -p $OUT/zip $OUT/yearly
cd $OUT

for y in $(seq $Y0 $Y1); do
    z=JMA_co2map_${y}_${VERSION}.ZIP
    if [ ! -s zip/$z ]; then
        wget -q --tries=3 --timeout=60 -O zip/$z $URL/$z || { echo "FAILED download $y"; rm -f zip/$z; continue; }
    fi
    unzip -o -q -d yearly zip/$z || echo "FAILED unzip $y"
    echo "ok $y"
done

# merge all years into one file (time is 'months since <year>-01-15' per file; cdo handles this)
ls yearly/JMA_co2map_*_${VERSION}.nc > /dev/null 2>&1 &&
cdo -s -O mergetime yearly/JMA_co2map_*_${VERSION}.nc JMA_co2map_${Y0}-${Y1}_${VERSION}.nc &&
echo "merged: $OUT/JMA_co2map_${Y0}-${Y1}_${VERSION}.nc"

cdo -s ntime JMA_co2map_${Y0}-${Y1}_${VERSION}.nc 2>/dev/null | xargs -I{} echo "time steps: {} (expected $(( (Y1 - Y0 + 1) * 12 )))"
