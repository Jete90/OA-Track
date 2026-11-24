#!/bin/bash
# The data product for one run, after rerun.sh:   ./rerun_product.sh u74   (likewise l10, m10, m10l10)
# Steps:
#   P0  calc_smooth_states.py   once for all runs (data/smoothed): the drift-removed, smoothed states (annual for all models) of
#                               every model (skipped when the files exist; FORCE_SMOOTH=1 recomputes)
#   P1  calc_product_obs.py     all indicators of the products and of Jiang's decade, with their level uncertainty
#                               (mapping, constants; level_uncertainty.py) and the state errors for the model years
#   P1b cuc_mapping_uncertainty.py  mapping uncertainty of the observed CUC of every reference (needs P1), used by P2
#                               and by the notebook (regional_common.cuc_sigma)
#   P2  calc_product_states.py  per reference: the constrained states (pi, historical, scenarios), the long step
#   P3  make_product.py         the product files
# Logs in logs/<run>. JOBS (default 4): models in parallel in P0, parts in parallel in P2; REFS to restrict;
# SKIP_P1=1: keep the obs files of an earlier run of this script (P1 is deterministic).
# PARALLEL_REFS=1: the references of P2 at the same time (up to JOBS processes each, about 5 GB per process).
# Needs inputs/co2/ (CMIP6 CO2 files, see co2_history.py) and the scenario links (link_ssp_inputs.sh).
set -e
RUN=$1
[ -n "$RUN" ] || { echo "usage: ./rerun_product.sh <run>"; exit 1; }
cd "$(dirname "$0")"
export OA_RUN=$RUN
BASE=$(python -c "import oa_config; print(oa_config.BASE)")
DATA=$(python -c "import oa_config; print(oa_config.DATA)")
L=$BASE/logs/$RUN
mkdir -p $L
JOBS=${JOBS:-4}
REFS=${REFS:-"OceanSODA UExP JMA CMEMS Jiang"}
log() { echo "$(date '+%F %T') $*" | tee -a $L/rerun_product.log; }
python -c "import co2_history as c; h = c.CO2History(); assert not h.missing, 'CO2 files missing: ' + str(h.missing)" \
    || { log "CMIP6 CO2 files missing in $BASE/inputs/co2 (see co2_history.py, link_ssp_inputs.sh)"; exit 1; }
[ -f $DATA/constraint_results.pkl ] || { log "no $DATA/constraint_results.pkl: run rerun.sh first"; exit 1; }

# P0 is shared by all runs: one run computes it under a lock, runs started at the same time wait for it; the
# marker COMPLETE is written only when every model is done, so a partial folder is never taken as finished
SM=$BASE/data/smoothed
mkdir -p $SM
(
    flock 9
    if [ -n "$FORCE_SMOOTH" ] || [ ! -f $SM/COMPLETE ]; then
        rm -f $SM/COMPLETE
        log "P0 smoothed model states (once for all runs) -> $SM"
        python -u calc_smooth_states.py --jobs $JOBS > $L/smooth_states.log 2>&1 \
            || { log "P0 failed, see $L/smooth_states.log"; exit 1; }
        date '+%F %T' > $SM/COMPLETE
    else
        log "P0 smoothed model states complete in $SM since $(cat $SM/COMPLETE) (FORCE_SMOOTH=1 to recompute)"
    fi
) 9> $SM/.lock || exit 1

if [ -n "$SKIP_P1" ]; then
    log "P1 skipped (SKIP_P1): the existing $DATA/product/obs_*.nc are used"
    ls $DATA/product/obs_*.nc > /dev/null 2>&1 || { log "no obs files: run without SKIP_P1"; exit 1; }
else
    log "P1 carbonate system of the products and of Jiang's decade"
    python -u calc_product_obs.py > $L/product_obs.log 2>&1 || { log "P1 failed, see $L/product_obs.log"; exit 1; }
fi

log "P1b mapping uncertainty of the observed CUC -> $DATA/product_uncertainty/cuc_mapping_<ref>.npz"
python -u cuc_mapping_uncertainty.py > $L/cuc_mapping.log 2>&1 || { log "P1b failed, see $L/cuc_mapping.log"; exit 1; }

log "P2 constrained states per reference (pi, historical, scenarios)$([ -n "$PARALLEL_REFS" ] && echo ', references in parallel')"
p2() {
    local r=$1 rc=0
    python -u calc_product_states.py --reference $r --jobs $JOBS > $L/states_$r.log 2>&1 || rc=$?
    log "   states $r: $(grep -c "^  \[done\]" $L/states_$r.log) parts done, $(grep -cE "^  \[(FAILED|KILLED)" $L/states_$r.log) failed or killed$(grep -q "nothing to do" $L/states_$r.log && echo " (reference not in this chain)")$([ $rc -ne 0 ] && echo ", EXIT $rc")"
    [ $rc -eq 0 ] || touch $L/states_$r.FAILED
}
rm -f $L/states_*.FAILED
for r in $REFS; do
    if [ -n "$PARALLEL_REFS" ]; then p2 $r & else p2 $r; fi
done
wait
if ls $L/states_*.FAILED > /dev/null 2>&1; then
    log "P2 not finished for: $(ls $L/states_*.FAILED | sed 's#.*/states_##; s#.FAILED##' | tr '\n' ' ')- P3 not run"
    exit 1
fi

log "P3 product files -> $DATA/product/carbonate_system_*.nc"
python -u make_product.py > $L/make_product.log 2>&1 || { log "P3 failed, see $L/make_product.log"; exit 1; }
log "done: see $L/make_product.log (tables per reference, checks against the paper, switch diagnostics)"
