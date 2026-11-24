# Files of the working code folder not in this archive

Development and operations of the study on its server, not needed to reproduce the paper:

- `check_a7.py`: follow-up of an internal review item (Red Sea cells of one ESM), no statement of the paper
- `check_a7b.py`: follow-up of an internal review item (Red Sea cells of one ESM), no statement of the paper
- `check_jiang_coverage.py`: check before the first run with Jiang et al. (2023), on folders of an early version
- `check_monthly_cmip6.py`: inventory of monthly CMIP6 output on our disks (set-up)
- `check_picontrol_monthly.py`: inventory of monthly piControl files on our disks (set-up)
- `est_dic_from_fco2_pyco2sys.py`: first estimate with PyCO2SYS, replaced by test_dic_from_fco2.py with mocsy
- `find_monthly_cmip6.py`: searches our disks for monthly CMIP6 output (set-up of the study)
- `make_for_marine_masks.py`: masks for a colleague, not part of the paper
- `new_production.sh`: creates a production folder from an earlier one on our server
- `rerun_window.sh`: one-off rerun after the change of the UExP-FNN-U reference period (12 November 2026)
- `run_alkfix_20261117.sh`: one-off rerun after a fix of the alkalinity series (17 November 2026)
- `test_barrett_lonfix.py`: test of a bug fix in barrett_sampling.py (10 November 2026), fixed since
- `test_oatrack_patternsigma.py`: test of a code change in make_oatrack.py before a run (16 November 2026)
- `tidy_code.sh`: moved superseded files out of the code folder (3 November 2026), done

Dated copies of earlier states of a script, kept in the working folder when a script was changed (the
archive holds the current state of each script, with which the results of the paper were computed):

- `fig_box_schematic_before_9steps_20261118.py`
- `fig_box_schematic_before_panela_20261116.py`
- `fig_box_schematic_flat_20261116.py`
- `make_for_marine_masks_before_cellweights_20261117.py`
- `make_for_marine_masks_before_datetag_20261116.py`
- `make_for_marine_masks_before_subfolder_20261117.py`
- `make_for_marine_masks_before_weights_20261117.py`
- `make_indicator_table_before_patternsigma_20261116.py`
- `make_oatrack_before_patternsigma_20261116.py`
- `make_oatrack_patternsigma_global_20261116.py`
- `make_product_alkalinity_before_patternsigma_20261116.py`
- `projection_before_quadfit_20261116.py`
- `regional_common_before_alkfigs_20261120.py`
- `run_structural_before_patternsigma_20261116.sh`
- `run_structural_before_singlepatterns_20261116.sh`
- `structural_adjustments_before_alkdirect_20261116.py`
- `structural_adjustments_before_esm2m1850_20261117.py`
- `structural_adjustments_before_patternsigma_20261116.py`
- `structural_adjustments_before_unconstrained_20261120.py`
- `structural_adjustments_patternsigma_global_20261116.py`
- `test_polar_shift_models_before_filled_20261120.py`

Comments in the code and the notebook still name earlier scripts (for example fig_crossing.py, fig3_crossing.py,
table_summary.py, calc_cuc.py of an earlier study), whose code now lives in the notebook or in the scripts of
this archive; they record where a piece of code came from and are not needed.
