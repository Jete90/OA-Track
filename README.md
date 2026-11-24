# Code of "Precise tracking of ocean acidification as it nears its planetary boundary"

Jens Terhaar, Tianfei Xue, and Helene Hollitzer, University of Bern

Code of: Terhaar, J., Xue, T. & Hollitzer, H. Precise tracking of ocean acidification as it nears its planetary
boundary (submitted). Archived on Zenodo: https://doi.org/10.5281/zenodo.24493310. Repository:
https://github.com/Jete90/OA-Track.

This archive holds the code of the analysis of the paper and of its data product OA-Track: the surface ocean
aragonite saturation state (Omega_arag) and the carbonate system from 1750 to 2024 and to 2100, from four
observation-based fCO2-products and 18 CMIP6 Earth system models, with one consistent carbonate chemistry
(mocsy 2.4.0) and an emergent constraint on the preindustrial state. OA-Track itself is archived at SEANOE
(https://doi.org/10.17882/121840), the output of GFDL-ESM2M-COBALTv2 used here at Zenodo
(https://doi.org/10.5281/zenodo.24493378).

## Layout and running

The scripts sit in one folder, `code/`, inside a production folder that also holds `inputs/`, `data/<run>/` and
`logs/`; `oa_config.py` derives every path from the location of `code/`. The runs (`OA_RUN`): `u74` (the paper:
K1 and K2 of Lueker et al. 2000, total boron of Uppstroem 1974), `l10`, `m10` and `m10l10` (sensitivities to the
constants). Area weights: `OA_WEIGHTS=ocean` (the paper: cell area times ocean fraction) or `cell`.

1. Set the locations of the raw inputs in `oa_config.py` or through the environment variables `OA_OBS`,
   `OA_OBS2`, `OA_MODELS`, `OA_ESM2M` and `OA_MOCSY` (see the block "raw inputs" in `oa_config.py`), and build
   mocsy 2.4.0 (https://github.com/jamesorr/mocsy) with its Python interface.
2. Download and prepare the inputs (`download_*.sh`, `make_cmems_ts.py`, `make_jma_ts.py`,
   `make_woa_nutrients.py`, `ocean_fraction_1deg.py`, `prepare_inputs.sh`, `link_ssp_inputs.sh`).
3. `./run_production.sh u74` runs the main analysis, the data product, the adjustments to the global mean and
   the summaries in order (each step can be skipped with `SKIP`; `JOBS` sets the processes in parallel).
   `python make_oatrack.py` then writes OA-Track and `python quantize_oatrack.py` the archived copy (maps with
   five significant digits, every 1 sigma and spread with three), `python make_indicator_table.py`
   Supplementary Table 2 and Supplementary Data 1, and `./run_checks.sh u74` the checks.
4. The figures and tables: `OA_boundary_figures.ipynb` from top to bottom (open it from `code/`); its last cell
   copies every display item under the name the manuscript uses. Box 1: `fig_box_schematic.py`.

Requirements: Python 3 with numpy, scipy, netCDF4 (with netCDF-C 4.9 or later for the storage of OA-Track),
matplotlib, pandas, openpyxl, gsw and mocsy 2.4.0; rasterio and GDAL for `ocean_fraction_1deg.py`; adjustText
(optional, labels in one figure); cdo for the remapping of the CMIP6 output. `environment.yml` lists the conda environment of the runs of the paper (without build strings).

## Display items and the notebook cells that draw them

| Manuscript | Notebook file name | Cell |
|---|---|---|
| Fig1 | Fig1_best_estimate | 8 |
| Fig2 | Fig2_constraint | 15 |
| Fig3 | Fig4_coverage | 25 |
| Fig4 | Fig3_crossing | 17 |
| EDFig1 | FigS11_trend_fits | 59 |
| EDFig2 | FigS15_jiang_chain | 71 |
| EDFig3 | FigS12_state_constraint | 61 |
| EDFig4 | FigS28_adjustments | 35 |
| EDFig5 | FigS13_findlay | 65 |
| EDFig6 | FigS8_crossing_co2 | 53 |
| EDFig7 | FigS14_states_1750_2024 | 69 |
| EDTable1 | TableS1_summary | 27 |
| EDTable2 | TableS3_uncertainty | 31 |
| EDTable3 | TableS2_steps | 67 |
| SuppFig1 | FigS1_products_2024 | 37 |
| SuppFig2 | FigS10_coverage_map | 57 |
| SuppFig3 | FigS2_global_per_product | 8 |
| SuppFig4 | FigS29_trends | 13 |
| SuppFig5 | FigS3_constraint | 41 |
| SuppFig6 | FigS4_dSST_bias | 43 |
| SuppFig7 | FigS5_correction_terms | 45 |
| SuppFig8 | FigS6_local_constraint | 47 |
| SuppFig9 | FigS9_regional_decline_per_product | 55 |
| SuppFig10 | FigS16_indicator_omega_arag | 73 |
| SuppFig11 | FigS17_indicator_omega_calc | 73 |
| SuppFig12 | FigS18_indicator_ph | 73 |
| SuppFig13 | FigS19_indicator_hplus | 73 |
| SuppFig14 | FigS20_indicator_hplus_free | 73 |
| SuppFig15 | FigS21_indicator_pco2 | 73 |
| SuppFig16 | FigS22_indicator_fco2 | 73 |
| SuppFig17 | FigS23_indicator_co2 | 73 |
| SuppFig18 | FigS24_indicator_hco3 | 73 |
| SuppFig19 | FigS25_indicator_co3 | 73 |
| SuppFig20 | FigS26_indicator_revelle | 73 |
| SuppFig21 | FigS27_indicator_dic | 73 |
| SuppFig22 | FigS30_leave_one_out | 80 |
| SuppFig23 | FigS32_feedback_patterns | 84 |
| SuppTable1 | TableS4_product_indicators | 75 |
| Box 1 figure | Box1 | `fig_box_schematic.py` |
| Supplementary Table 2, Supplementary Data 1 | SuppTable2, Supplementary_Data_1.xlsx | `make_indicator_table.py` |

## Scripts

### Running the chain

| File | What it does |
|---|---|
| `link_ssp_inputs.sh` | Links the CMIP6 CO2 concentration files into inputs/co2/ and the CMIP6 scenario files (thetao, so, dissic, talk; ssp126, ssp245, ssp370, ssp585) into inputs/raw/<ssp>/ of ... |
| `prepare_inputs.sh` | 1. Links the CMIP6 files used in this study into data/raw/<exp>/ for every historical and piControl run with thetao, so, dissic and talk. These files |
| `rerun.sh` | Full run of the OA_boundary chain (version 2) for one run of oa_config.py: ./rerun.sh u74 (likewise l10, m10, m10l10). Everything is computed into data/<run>. Inputs that ... |
| `rerun_product.sh` | The data product for one run, after rerun.sh: ./rerun_product.sh u74 (likewise l10, m10, m10l10) Steps: |
| `run_checks.sh` | All checks behind the numbers of the manuscript for one run, after its rerun.sh (u74 for the paper): ./run_checks.sh u74 (OLD=<old production folder> to compare with anot ... |
| `run_production.sh` | the whole chain of one run, from the raw model output to the data product with its add-on, unattended, stopping at the first failing step (nothing runs on stale input). S ... |
| `run_structural.sh` | the structural adjustments of the decline, in order (start from code/ in env_jens): OA_RUN=u74 nohup bash run_structural.sh > ../logs/u74/run_structural.log 2>&1 & |

### Downloading and preparing the inputs

| File | What it does |
|---|---|
| `download_jma_co2map.sh` | Downloads the global JMA Ocean CO2 Map (JMA-MLR, v2025, 1990-2024, monthly, 1x1 deg; Iida et al., 2021, J. Oceanogr.) from |
| `download_mgdsst.sh` | Downloads JMA MGDSST (Merged satellite and in-situ data Global Daily SST, 0.25 deg, reanalysis since 1982; Kurihara et al. 2006, Sakurai et al. 2005/2007), the SST used f ... |
| `download_obs_products.sh` | Downloads the two remaining observation-based carbonate-system products used in the Planetary Health Check: |
| `download_ostia_monthly.sh` | OSTIA SST as used by CMEMS-LSCE-FFNN (QUID 015_008, Table 2): SST_GLO_SST_L4_REP_OBSERVATIONS_010_011 (METOFFICE-GLO-SST-L4-REP-OBS-SST) 1985-01 .. 2026-03 |
| `make_cmems_ts.py` | CMEMS-LSCE-FFNN on a 1x1 degree monthly grid, with the temperature and salinity the product itself uses (QUID 015_008, Table 2): |
| `make_cmems_unc.py` | The uncertainty fields of CMEMS-LSCE-FFNN (1 sigma) on the 1 x 1 degree monthly grid of CMEMS-LSCE-FFNN_1deg_1985-2025.nc (make_cmems_ts.py), with the same block means of ... |
| `make_jiang_ref.py` | Observation-based surface state of Jiang et al. (2023, NCEI 0259391, median of 14 ESMs bias-adjusted to SOCAT, GLODAPv2 and CODAP-NA) for their decade 2010 (mean of Janua ... |
| `make_jma_ts.py` | Temperature and salinity for the JMA Ocean CO2 Map (JMA-MLR, v2025), as used by JMA (Iida et al. 2021, Table 1): |
| `make_woa_nutrients.py` | Annual surface silicate and phosphate of the World Ocean Atlas 2023 (all decades, 1 degree) as the second nutrient climatology of nutrient_sensitivity.py (run_checks.sh,  ... |
| `ocean_fraction_1deg.py` | The ocean fraction of every 1 x 1 degree cell from the ESA CCI Land Cover water bodies ocean map (150 m, ESACCI-LC-L4-WB-Ocean-Map-150m-P13Y-2000-v4.0.tif; 0 = ocean, 1 = ... |

### Configuration and shared modules

| File | What it does |
|---|---|
| `co2_history.py` | Atmospheric CO2 (global annual mean, ppm) and its uncertainty for every year of the data product, 1750-2100. Sources (files in inputs/co2/, downloaded from input4MIPs on  ... |
| `constraint_refs.py` | Notebook cell: regression constraint on preindustrial Omega_arag for each reference. Self-contained: reads only files (mechanism states, drift, xco2 outputs, obs Omega). |
| `drift_tools.py` | Reading side of the piControl drift (calc_drift.py writes the files): the Drift class turns the stored Enting-spline anomalies s(k) - mean(piControl) into the drift corre ... |
| `fill_tools.py` | 20 November 2026 Filling of ocean cells without a value from the cells around them, used for the extension of the decline to the whole |
| `jiang_chem.py` | The surface states of Jiang et al. (2023, NCEI 0259391, median of 14 ESMs) recomputed with mocsy, as every other carbonate-system calculation of this study. |
| `level_uncertainty.py` | The level uncertainty of the data product that comes from the reference itself, per cell and for the area means, stored separately from the constraint uncertainty (calc_p ... |
| `oa_config.py` | The runs of the OA_boundary chain. The run is chosen with the environment variable OA_RUN (set before python starts, e.g. by rerun.sh, or in the first notebook cell); it  ... |
| `pi_states.py` | The preindustrial CO2 states (atmospheric xCO2, ppm) used in the paper, shared by calc_pi_xco2.py (which computes Omega_arag for each of them with mocsy) and table_summar ... |
| `regional_common.py` | Shared data and calculations for the regional figures (fig_regional.py, fig_regional_local.py). For each product, on the common mask and the product's grid (from data/con ... |
| `state_constraint.py` | The emergent constraint applied to the model STATE (the method of the data product, decided 2 November 2026): For a set of n models with the predictors dSST (warming of t ... |

### Main analysis (rerun.sh)

| File | What it does |
|---|---|
| `calc_bias_adjusted.py` | Bias-adjusted, drift-corrected surface Omega_arag for one CMIP6 model, using an observation-based product as reference (OceanSODA-ETHZ v2025 by default). |
| `calc_drift.py` | piControl drift of surface T, S, DIC and ALK for every model, aligned with the historical run through its branch point. Per cell, the annual means of the whole available  ... |
| `calc_mechanism.py` | Mean surface states for testing why the bias-adjusted preindustrial Omega_arag depends on the raw model bias. |
| `calc_model_sst.py` | Top-level ocean temperature (thetao) of every model's historical run, averaged over the periods the constraint needs, so that the model warming is defined exactly as the  ... |
| `calc_omega_arag.py` | Surface aragonite saturation state (Omega_arag) for the CMIP6 historical and piControl runs linked by prepare_inputs.sh, computed with mocsy 2.0 |
| `calc_omega_arag_anymonth.py` | Annual surface Omega_arag of the four observation-based products from ALL available months: a cell enters a year as soon as it has one valid month (valid: T, S, DIC, TA p ... |
| `calc_omega_arag_obs.py` | Surface aragonite saturation state from OceanSODA-ETHZ v2025 (1982-2024), recomputed with mocsy 2.4.0 from the product's talk, dic, temperature and |
| `calc_omega_arag_products.py` | Surface Omega_arag from further observation-based products, computed exactly as for OceanSODA in calc_omega_arag_obs.py (mocsy 2.4.0, optcon='mol/kg', optt='Tpot', optp=' ... |
| `calc_pi_xco2.py` | Preindustrial Omega_arag of every model for the preindustrial CO2 states used in the paper (1750 and 1850 of AR6 with their +-1 sigma, their midpoint, the effective CO2 o ... |
| `common_mask_cell.py` | Notebook cell (after the constraint cell): all products on one common ocean area. The common mask is the intersection of the constraint masks of all products that are don ... |
| `run_constraint.py` | Runs the emergent constraint for all products (constraint_refs.py) and again on the common mask (common_mask_cell.py), prints both tables, and saves everything the figure ... |
| `run_constraint_jiang.py` | The full chain of this study with the observation-based state of Jiang et al. (2023, decade 2010 = January 2005 to December 2014) as a fifth reference: bias adjustment to ... |

### Data product OA-Track (rerun_product.sh, make_oatrack.py)

| File | What it does |
|---|---|
| `calc_product_members.py` | Members of the data product: for one reference product and every model, the annual means of all carbonate system indicators of the adjusted (drift-corrected, bias-adjuste ... |
| `calc_product_obs.py` | The observation-based part of the data product: all carbonate system indicators of every reference product (OceanSODA-ETHZ, UExP-FNN-U, JMA-MLR, CMEMS-LSCE-FFNN; monthly  ... |
| `calc_product_states.py` | data product, pass 2: per run and reference The constrained and the unconstrained carbonate system of every year, from the models' states, with the |
| `calc_smooth_states.py` | data product, pass 1: once for all references and all configurations The forced monthly state of every model for the data product: T, S, DIC and ALK of the historical run ... |
| `cuc_mapping_uncertainty.py` | The mapping uncertainty of the observed CUC (the predictor of the emergent constraint) and its correlation with the mapping error of the observed Omega_arag, per product, ... |
| `diagnose_product.py` | data product: where are the irregular cells, and why The summaries of the product files (summarize_product.py, 6 November 2026) show a few irregular cells. This script |
| `make_indicator_table.py` | Supplementary Table 2 and Supplementary Data 1: the 15 quantities of OA-Track, 1750-2024 The best estimate of the paper for every year from 1750 to 2024, global mean on t ... |
| `make_oatrack.py` | data product OA-Track: the files of the release Writes OA-Track, the data product of the paper (Supplementary Methods, 'the data product OA-Track'), from the files |
| `make_product.py` | data product, final step: the product files from the states and the observation-based products Per reference (OceanSODA-ETHZ, UExP-FNN-U, JMA-MLR, CMEMS-LSCE-FFNN; Jiang  ... |
| `make_product_alkalinity.py` | data product: the alkalinity adjustment of the best estimate The adjustment of the carbonate system for the surface alkalinity rise that the observation-based products an ... |
| `patch_product_xco2.py` | Completes the atmospheric CO2 of the data product files in place, so that every value of the product comes with the CO2 of its year: |
| `summarize_product.py` | A compact text summary of every file of the data product, for checks without the files themselves (they are too large to pass on). Per file: the global attributes, the di ... |

### Adjustments to the global mean and projection (run_structural.sh, notebook)

| File | What it does |
|---|---|
| `barrett_sampling.py` | The sampling of the observed surface TA trend of Barrett et al. (2025, GBC), on the 1 degree grid of the products, written to data/<run>/structural/barrett_sampling.npz.  ... |
| `feedback_pattern.py` | The spatial pattern of the calcification feedback in surface alkalinity, for the alkalinity adjustment of structural_adjustments.py, written to data/<run>/structural/feed ... |
| `feedback_pattern_members.py` | 17 November 2026 The alkalinity-rise pattern of every single member of GFDL-ESM2M-COBALTv2 (coupled, five members), computed exactly as |
| `projection.py` | The projections of the surface carbonate system that the figures use, computed once per run and cached in the run's data folder (delete a cache file to recompute it). Eve ... |
| `structural_adjustments.py` | The two adjustments of the decline 1750-2024 beyond the analysis on the common area (Extended Data Table 3, Extended Data Fig. 5, Extended Data Table 1), for each of the  ... |
| `test_polar_shift_models.py` | The whole ocean against the common area of the paper in each of the 18 ESMs: the shift of the decline and the gap of the level of Omega_arag. structural_adjustments.py ta ... |

### Archived records: OA-Track at SEANOE, the ESM2M output at Zenodo

| File | What it does |
|---|---|
| `oatrack_inventory.py` | 24 November 2026: the variable list of OA-Track for its README Reads only the headers of the OA-Track files and writes OA-Track_variables.md: for each kind of file (a ref ... |
| `pack_esm2m.py` | 24 November 2026: the GFDL-ESM2M-COBALTv2 output of the paper for its own Zenodo record Copies the part of the GFDL-ESM2M-COBALTv2 simulations that the analysis reads (fe ... |
| `quantize_oatrack.py` | 24 November 2026 The release copy of OA-Track for SEANOE (one record of about 175 GB): every file of data/<run>/oatrack/ copied into a |
| `test_oatrack_compression.py` | 20 November 2026: how small can OA-Track get for SEANOE, 150 GB per record? OA-Track of v4 is 279 GB (14 files). This test copies a sample of map variables of one OA-Trac ... |

### Figures

| File | What it does |
|---|---|
| `OA_boundary_figures.ipynb` | the figures and tables of the paper, from top to bottom; the last cell exports them under the manuscript names |
| `fig1_coverage.py` | Notebook cell: Figure 1, why previous estimates differ: coverage and the definition of the preindustrial state. a annual global mean surface Omega_arag of each product on ... |
| `fig_box_schematic.py` | draft of the schematic for Box 1: the method and what can go wrong a the method: observed state of an fCO2-product minus the change simulated by each ESM gives one preind ... |
| `fig_drift.py` | candidate Supplementary Figure: the drift of the 18 ESMs and its removal For each ESM, the global mean surface Omega_arag on the common area of the paper, from the raw an ... |

### Checks behind statements of the paper (run_checks.sh and single runs)

| File | What it does |
|---|---|
| `calc_pi1750_nutrients.py` | The 1750 Omega_arag of every model, exactly as calc_pi_xco2.py computes it (every month of the drift-corrected, bias-adjusted piControl; DIC lowered until surface pCO2 ha ... |
| `check_alk_trend.py` | exploratory, not part of the chain Is the models' recent change of surface alkalinity related to their constrained quantity, the bias-adjusted |
| `check_annual_state_omega.py` | The models enter the chain as annual means (step 1: one value per year of T, S, DIC, TA), so their Omega_arag (1750 and historical) is the saturation state of the annual- ... |
| `check_inputs.py` | The input checks of the code review (code_review_2026-11-05.md, section C), in one run. Nothing is changed; every item ends with OK, NOTE (to read) or PROBLEM. |
| `check_j_models.py` | Does the offset J = (mean of the monthly Omega_arag) - (Omega_arag of the annual-mean state) change between an early, near-preindustrial period and the present in the mod ... |
| `check_jiang_std.py` | 18 November 2026 Where does the +-0.06 of the present-day state of Findlay et al. (2025) come from? Prints the sheets and the rows of |
| `check_ocean_fraction.py` | the area weights of version 4: what the ocean-fraction map does to the grid Reads inputs/ESACCI_water_bodies/ocean_fraction_1deg.nc (oa_config.OCEAN_FRACTION_FILE) and pr ... |
| `check_time_step.py` | Checks, for every model input the chain reads (historical, piControl, ssp126/245/370/585), that the time step the code uses (ModelRun.step of calc_bias_adjusted.py: 12 =  ... |
| `compare_pi_methods.py` | Old against new preindustrial states, per reference and model, from the constraint results of two production folders: the earlier method (Omega_arag of the annual-mean dr ... |
| `compare_runs.py` | Sensitivity of the results to the carbonate constants: the runs of oa_config.py (u74 main run; l10 total boron Lee et al. 2010; m10 K1, K2 Millero 2010; m10l10 both). Per ... |
| `compare_uexp_versions.py` | The 2024 global mean surface Omega_arag of UExP-FNN-U in the pre-release used by the Planetary Health Check 2026 (v2026-pre1, Zenodo 19605928) and in the released v2026-1 ... |
| `compare_v3_v4.py` | the effect of the area weights on the main analysis: whole cells (v3) against cell area x ocean fraction (v4) Reads constraint_results.pkl of two production folders and p ... |
| `nutrient_sensitivity.py` | How much the choice of the nutrients (silicate and phosphate) changes the observed state, the constrained 1750 state and the decline, per product on its own and on the co ... |
| `phc_2020.py` | 18 November 2026 The decline of the Planetary Health Check (2026) for other years than 2024, in its own framework: global mean of |
| `phc_area.py` | 18 November 2026 Why does averaging 1750 and 2024 over different areas make the decline of the Planetary Health Check (2026) larger? |
| `phc_baseline_regions.py` | 18 November 2026 Do the observation-based estimates of the Planetary Health Check (2026) differ more outside our common area? In |
| `product_constants_uncertainty.py` | Uncertainty of the global-mean surface Omega_arag from the equilibrium constants (Orr et al. 2018), for the UExP-FNN-U v2026-1 2024 state, and its effect on the relative  ... |
| `product_uncertainty_mc_C.py` | Uncertainty of the 2024 global-mean surface Omega_arag of an observation-based product from its mapping uncertainties, with spatially correlated errors (Ford et al. 2024, ... |
| `test_access_cells.py` | which cells outside the common area make ACCESS-ESM1-5 differ? test_access_map.py found for ACCESS-ESM1-5, outside the common area between 50 S and 60 N (7.8 % of the oce ... |
| `test_access_map.py` | why does ACCESS-ESM1-5 give a smaller decline over the whole ocean than on the common area? In Extended Data Fig. 5c,d ACCESS-ESM1-5 is the only model whose decline 1850  ... |
| `test_alk_before_constraint.py` | does the order of emergent constraint and alkalinity adjustment matter? The chain applies the alkalinity adjustment after the emergent constraint: the constrained 1750 st ... |
| `test_alk_diseq_models.py` | Two checks of what the 1750 state takes from the models (see test_disequilibrium_alk.py), on the common area of the paper, cos(latitude): |
| `test_alk_models_trend.py` | Is the calcification feedback in surface alkalinity (Barrett et al. 2025: +0.072 +- 0.023 umol/kg per year in global surface TA over about 30 years, measured minus S/T/nu ... |
| `test_alk_residuals.py` | Is the alkalinity change of the models already in the residual spread of the constraint? For each product (common area): the 18 per-model values of constraint_results.pkl ... |
| `test_baseline_sensitivity.py` | does the decline depend on the present-day map the simulated change is applied to? The chain computes the preindustrial state as baseline minus the simulated change. Here ... |
| `test_cell_constraint_terms.py` | 17 November 2026 Which of the two predictors moves the cell-by-cell constrained 1750 Omega_arag, and in which direction? |
| `test_cell_crossing.py` | by how much does the cell-by-cell constraint move the local crossing of the boundary? For each observation-based product, cell by cell: the forced series of the product y ... |
| `test_constraint_idealised.py` | Idealised (perfect-knowledge) test of the emergent constraint for every carbonate system indicator: does the regression on dSST and CUC recover the true 1750 value, and w ... |
| `test_constraint_loo.py` | Perfect-model (leave-one-out) test of the emergent constraint with the real models, per carbonate system indicator and per predictor set (dSST only, CUC only, both): does ... |
| `test_decline_cuc.py` | what does a constraint of the relative decline itself give, compared with the best estimate? Variant B of test_baseline_sensitivity.py showed that the relative decline ha ... |
| `test_decline_ownworld.py` | the relative decline 1850 -> 2024 of every ESM in its own world, without bias adjustment, compared with the chain |
| `test_dic_from_fco2.py` | does it matter that we take the DIC of the fCO2-products instead of their fCO2? The chain takes DIC and alkalinity (with T and S) of every fCO2-product and computes Omega ... |
| `test_diseq_models.py` | The change of the air-sea disequilibrium (surface ocean pCO2 minus atmospheric pCO2, moist air at 1 atm) in each of the 18 ESMs, from their raw output, on the common area ... |
| `test_diseq_residual.py` | 18 November 2026 Does the third control of the simulated carbon uptake, how closely surface waters follow the rising atmospheric CO2, |
| `test_disequilibrium_alk.py` | How much of the 1750 surface state, and with it the decline 1750-2024, rests on the models, and how far idealised extremes of the air-sea disequilibrium and of alkalinity ... |
| `test_drift_cutoff.py` | does the 50-year cutoff of the drift spline matter for the decline of the models? The chain removes the piControl drift with an Enting spline of 50-year cutoff (calc_drif ... |
| `test_esm4_so.py` | why GFDL-ESM4 has no salinity in a few coastal cells of the historical series Reads the raw GFDL-ESM4 files (historical, piControl and the four scenarios, the members of  ... |
| `test_level_gap_regions.py` | where does the larger level gap of the products come from? The whole-ocean mean Omega_arag lies 5.0 % (1750) and 5.2 % (2024) below the common-area mean in the products,  ... |
| `test_level_uncertainty.py` | unit tests of level_uncertainty.py: the state conversion at fixed fCO2, the box quadratic form in its two limits, the C1/C2 limits of the annual mean. Usage: python test_ ... |
| `test_model_masks.py` | do the raw files of every model cover the same cells in every experiment and variable? For every model and member of the chain, and every experiment (historical, piContro ... |
| `test_mri_area.py` | how much of the common area is lost to the incomplete MRI-ESM2-0 salinity? The constraint area is the set of cells every one of the 18 ESMs covers (all four variables, hi ... |
| `test_polar_shift.py` | How much larger is the decline 1750-2024 over the whole ocean than over the common area of the paper, and how well is that shift known? Cell by cell, so that each part of ... |
| `test_polar_smoothing.py` | does the whole-ocean shift of the ESMs depend on the smoothing of their states? structural_adjustments.py takes the whole-ocean shift of the 18 ESMs from test_polar_shift ... |
| `test_product_alk_trend.py` | do the products already contain an alkalinity trend beyond salinity? The alkalinity adjustment assumes that the products miss the rise of surface alkalinity from weaker c ... |
| `test_product_cells.py` | two questions behind the irregular cells of diagnose_product.py, 7 November 2026 1 mocsy at cold temperatures: the state of a Jiang cell with an extreme dSST sigma (81.5  ... |
| `test_r2_mechanism.py` | how much of the model spread do CUC and dSST explain: of the 1750 Omega_arag, and of the change of DIC minus alkalinity? |
| `test_sampling_polar.py` | Two of the structural uncertainties, on the data of the chain. A Sampling effect on the products' trends. The decline uses the product's 2024 state relative to its bias w ... |
| `test_smooth_prefix.py` | how large is the end effect of the 30-year smoothing at 1850? calc_smooth_states.py smooths every model's drift-removed state, X(t) - a(k(t)), with a 30-year Enting splin ... |
| `test_spna_crossing.py` | how much earlier does the regional constraint put the local crossing of the boundary in the subpolar North Atlantic? |
| `test_state_constraint.py` | One product, one state (1750): the constrained STATE against the constrained INDICATOR. direct the paper: Omega_arag of every model's 1750 state, regressed on dSST and CU ... |
| `test_whole_ocean_coverage.py` | 18 November 2026 The three estimates of the whole-ocean adjustment (structural_adjustments.py, step 2) rest on different cells: the |

## Not in this archive

Scripts for the development and the operation of the study on its server (inventories of our disks, memory
guards, one-off reruns, the set-up of production folders) and dated copies of earlier states of scripts are not
included, as they are not needed to reproduce the paper. Comments in the code and the notebook still name some
earlier scripts (for example fig_crossing.py, fig3_crossing.py, and table_summary.py), whose code now lives in
the notebook or in the scripts of this archive.

## Licence

MIT licence (`LICENSE`). OA-Track itself is published at SEANOE under CC BY 4.0.
