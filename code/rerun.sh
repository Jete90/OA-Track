#!/bin/bash
# Full run of the OA_boundary chain for one run of oa_config.py:   ./rerun.sh u74   (likewise l10,
# m10, m10l10). Everything is computed into data/<run>. Inputs that depend neither on the model DIC/ALK nor on
# the chemistry (linked CMIP6 files raw/, GLODAP nutrients and product links obs/, Jiang et al. 2023, Findlay
# et al. 2025, NOAA CO2) are linked from inputs/. Stops at the first error. Logs in logs/<run>.
#
# Order: the drift is computed before the bias adjustment, because the adjusted state of both
# experiments is drift-corrected; the preindustrial states come from the monthly adjusted piControl.
set -e
RUN=$1
[ -n "$RUN" ] || { echo "usage: ./rerun.sh <run>  (runs in oa_config.py)"; exit 1; }
cd "$(dirname "$0")"
export OA_RUN=$RUN
BASE=$(python -c "import oa_config; print(oa_config.BASE)")
DATA=$(python -c "import oa_config; print(oa_config.DATA)")
IN=$BASE/inputs
L=$BASE/logs/$RUN
mkdir -p $L $DATA/obs
REFS="OceanSODA UExP JMA CMEMS Jiang"
JOBS=${JOBS:-4}
log() { echo "$(date '+%F %T') $*" | tee -a $L/rerun.log; }
# the data product (rerun_product.sh) needs the CMIP6 CO2 files in inputs/co2/: warn now, do not stop
python -c "import co2_history as c; h = c.CO2History(); assert not h.missing, h.missing" 2>/dev/null \
    || log "WARNING: CMIP6 CO2 files missing in $IN/co2 (needed by rerun_product.sh, see co2_history.py; link_ssp_inputs.sh links them)"

log "0 run $RUN -> $DATA; $(python -c 'import oa_config; print(oa_config.VERSION); print(oa_config.CONSTANTS, "| REF_DENSITY", oa_config.REF_DENSITY)')"
for x in raw jiang2023 findlay2025 co2_annmean_gl.txt; do
    [ -e $DATA/$x ] || ln -s $IN/$x $DATA/$x
done
for f in $IN/obs/*; do
    [ -e $DATA/obs/$(basename $f) ] || ln -s $f $DATA/obs/$(basename $f)
done
for v in Temperature Salinity DIC TA; do
    [ -e $DATA/jiang2023/${v}_median_historical.nc ] || { log "missing $DATA/jiang2023/${v}_median_historical.nc"; exit 1; }
done

log "1 raw model Omega (historical, piControl)"
python calc_omega_arag.py --force --jobs $JOBS > $L/omega_models.log 2>&1

log "2 observation-based products and the reference state of Jiang et al. (decade 2010)"
python calc_omega_arag_obs.py > $L/omega_obs.log 2>&1
python calc_omega_arag_products.py > $L/omega_products.log 2>&1
python calc_omega_arag_anymonth.py > $L/omega_anymonth.log 2>&1
python make_jiang_ref.py > $L/jiang_ref.log 2>&1

log "3 piControl drift (Enting spline per cell and year) and model SST"
python calc_drift.py --jobs $JOBS > $L/drift.log 2>&1
python calc_model_sst.py > $L/model_sst.log 2>&1

log "4 bias-adjusted, drift-corrected Omega (historical and piControl), all references"
for r in $REFS; do
    python calc_bias_adjusted.py --reference $r --force --jobs $JOBS > $L/bias_$r.log 2>&1
    log "   bias $r done"
done

log "5 preindustrial CO2 states from the monthly adjusted piControl (1750, 1850, Jiang et al., ...)"
for r in $REFS; do
    python calc_pi_xco2.py --reference $r --jobs $JOBS > $L/pi_xco2_$r.log 2>&1
    log "   pi_xco2 $r done"
done

log "6 mechanism states (CUC, raw Revelle and Omega)"
for r in $REFS; do
    python calc_mechanism.py --reference $r --jobs $JOBS > $L/mechanism_$r.log 2>&1
    log "   mechanism $r done"
done

log "7 emergent constraint, own and common masks -> $DATA/constraint_results.pkl"
python run_constraint.py > $L/run_constraint.log 2>&1

log "8 Jiang et al. (2023) as fifth reference -> $DATA/constraint_results_jiang.pkl"
python run_constraint_jiang.py > $L/run_constraint_jiang.log 2>&1

log "all done"
