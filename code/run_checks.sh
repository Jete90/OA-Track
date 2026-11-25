#!/bin/bash
# The uncertainty of the fCO2-products and the checks of the emergent constraint for one run, after its rerun.sh
# (u74 for the paper); the other checks behind statements of the paper run on their own (README.md):
#   ./run_checks.sh u74
# Runs in parallel (one process each, they only read the chain's files; the product runs are not touched):
#   state        test_state_constraint.py      the state constraint against the direct regression (per product)
#   loo          test_constraint_loo.py        perfect-model leave-one-out of the 1750 value (global and cells)
#   loo_changes  test_constraint_loo.py --mechanism --changes   the same for the changes of T, S, DIC, ALK, DIC-ALK
#   loo_mech     test_constraint_loo.py --mechanism   the per-model values of the mechanism
#   constants    product_constants_uncertainty.py     -> data/<run>/product_uncertainty/constants_UExP_2024.npz
#   mapping      product_uncertainty_mc_C.py   the four products, 100 draws each -> data/<run>/product_uncertainty/
#   nutrients    nutrient_sensitivity.py       Si and PO4 (zero, x0.8, x1.2; a second climatology with NUT_ALT="--alt-si FILE VAR
#                                              --alt-po4 FILE VAR", e.g. World Ocean Atlas 2023)
# Output: logs/<run>/checks/<name>.log and logs/<run>/checks/summary.txt (the end of every log).
RUN=${1:-u74}
cd "$(dirname "$0")"
export OA_RUN=$RUN
BASE=$(python -c "import oa_config; print(oa_config.BASE)")
DATA=$(python -c "import oa_config; print(oa_config.DATA)")
C=$BASE/logs/$RUN/checks
mkdir -p $C $DATA/product_uncertainty
[ -f $DATA/constraint_results.pkl ] || { echo "no $DATA/constraint_results.pkl: run rerun.sh $RUN first"; exit 1; }
log() { echo "$(date '+%F %T') $*" | tee -a $C/run_checks.log; }
log "checks of run $RUN -> $C"

run() {   # name, command...
    local name=$1; shift
    ( "$@" > $C/$name.log 2>&1 && log "  done   $name" || log "  FAILED $name (see $C/$name.log)" ) &
}
run state       python -u test_state_constraint.py --data $DATA
run loo         python -u test_constraint_loo.py --data $DATA
run loo_changes python -u test_constraint_loo.py --data $DATA --mechanism --changes --no-cells
run loo_mech    python -u test_constraint_loo.py --data $DATA --mechanism --no-cells
run constants   python -u product_constants_uncertainty.py
run nutrients   python -u nutrient_sensitivity.py $NUT_ALT

# mapping uncertainty of the four products, always computed here (100 draws each)
for p in UExP OceanSODA JMA CMEMS; do
    run mapping_$p python -u product_uncertainty_mc_C.py $p 100
done

wait
{
    echo "checks of run $RUN, $(date '+%F %T')"
    cat $C/run_checks.log
    for f in state loo loo_changes loo_mech constants nutrients mapping_UExP mapping_OceanSODA mapping_JMA mapping_CMEMS; do
        [ -f $C/$f.log ] || continue
        echo; echo "=================================================================== $f (last 40 lines)"
        tail -n 40 $C/$f.log
    done
} > $C/summary.txt
log "all done: $C/summary.txt"
