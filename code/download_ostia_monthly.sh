#!/bin/bash
# download_ostia_monthly.sh
# -------------------------
# OSTIA SST as used by CMEMS-LSCE-FFNN (QUID 015_008, Table 2):
#   SST_GLO_SST_L4_REP_OBSERVATIONS_010_011  (METOFFICE-GLO-SST-L4-REP-OBS-SST)     1985-01 .. 2026-03
#   (CMEMS-LSCE used REP to 2022-05 and NRT after; NRT-V2 starts only on 2024-01-17, so the reanalysis, which
#    reaches 2026-03, is used throughout)
#   SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001  (METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2)  2022-06 .. end
# Daily 0.05 deg is ~100 MB per day, so each month is downloaded, averaged to a monthly mean,
# block-averaged 20x20 to 1 deg (cell centres -89.5..89.5, -179.5..179.5) and the daily file deleted.
#
# Usage (in the cmems env, which has copernicusmarine and cdo on the PATH):
#   bash download_ostia_monthly.sh 1985 1994 &     # year range, run several ranges in parallel
#   bash download_ostia_monthly.sh 1995 2004 &
#   ...
# Output: $OA_OBS/OSTIA/monthly_1deg/ostia_sst_YYYYMM_1deg.nc  (analysed_sst, K)

set -u
Y0=${1:-1985}
Y1=${2:-2025}
OUT=${OA_OBS:-/oceandata01/observational_data}/OSTIA/monthly_1deg
TMP=${OA_OBS:-/oceandata01/observational_data}/OSTIA/tmp
mkdir -p $OUT $TMP
command -v cdo > /dev/null || { echo "cdo not found in this env"; exit 1; }

for y in $(seq $Y0 $Y1); do
  for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    f=$OUT/ostia_sst_${y}${m}_1deg.nc
    [ -s $f ] && continue
    if [ $y$m -le 202603 ]; then ds=METOFFICE-GLO-SST-L4-REP-OBS-SST; else ds=METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2; fi
    last=$(date -d "$y-$m-01 +1 month -1 day" +%Y-%m-%d)
    raw=$TMP/ostia_${y}${m}_daily.nc
    copernicusmarine subset -i $ds -v analysed_sst -t $y-$m-01T00:00:00 -T ${last}T23:59:59 \
        -o $TMP -f $(basename $raw) --overwrite > $TMP/log_${y}${m}.txt 2>&1
    if [ ! -s $raw ]; then echo "FAILED download $y-$m ($ds), see $TMP/log_${y}${m}.txt"; continue; fi
    nd=$(cdo -s ntime $raw)
    if cdo -s -L -f nc4 -z zip_4 gridboxmean,20,20 -monmean -selname,analysed_sst $raw $f.tmp; then
      mv $f.tmp $f && rm -f $raw $TMP/log_${y}${m}.txt && echo "ok $y-$m ($nd days, $ds)"
    else
      echo "FAILED cdo $y-$m"; rm -f $f.tmp
    fi
  done
done
