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
constants). Area weights: `OA_WEIGHTS=ocean` (the paper: cell area times ocean fraction) or `cell` (whole cells).

1. Set the locations of the raw inputs in `oa_config.py` or through the environment variables `OA_OBS`,
   `OA_OBS2`, `OA_MODELS`, `OA_ESM2M` and `OA_MOCSY` (see the block "raw inputs" in `oa_config.py`), and build
   mocsy 2.4.0 (https://github.com/jamesorr/mocsy) with its Python interface.
2. Download and prepare the inputs (`download_*.sh`, `make_cmems_ts.py`, `make_cmems_unc.py`, `make_jma_ts.py`,
   `make_woa_nutrients.py`, `ocean_fraction_1deg.py`, `prepare_inputs.sh`, `link_ssp_inputs.sh`).
3. `./run_production.sh u74` runs the main analysis, the data product and the adjustments to the global mean in
   order (each step can be skipped with `SKIP`; `JOBS` sets the processes in parallel), and `./run_checks.sh u74`
   the uncertainty of the fCO2-products and the checks of the emergent constraint. `python make_oatrack.py` then
   writes the complete output for each fCO2-product, and `python make_indicator_table.py` Supplementary Table 2
   and Supplementary Data 1. OA-Track as archived at SEANOE: `python make_oatrack_product.py --part global` (the
   global means, .nc and .csv), `--part obs_based_products` (the maps of the obs-based products) and
   `--part esms` (the maps of the ESMs, which are added to this code and to the record before the paper is
   published), with the names and descriptions of `oatrack_names.py`;
   `python list_oatrack_names.py` lists every variable, and `python pack_oatrack.py` packs the record (zip of the
   global means, maps, MANIFEST.md5).
4. The figures and tables: `OA_boundary_figures.ipynb` from top to bottom (open it from `code/`); its last cell
   copies every display item under the name the manuscript uses. Box 1: `fig_box_schematic.py`.
5. The other checks behind statements of the paper (last table below) run on their own, each with
   `OA_RUN=u74 python -u <script>` from `code/` (options in the docstring of each script).

Requirements: Python 3 with numpy, scipy, netCDF4 (with netCDF-C 4.9 or later for the storage of OA-Track),
matplotlib, pandas, openpyxl, gsw and mocsy 2.4.0; rasterio and GDAL for `ocean_fraction_1deg.py`; adjustText
(optional, labels in one figure); cdo for the remapping of the CMIP6 output. `environment.yml` lists the conda environment of the runs of the paper (without build strings).

## Display items and the notebook cells that draw them

| Display item | Notebook cell |
|---|---|
| Fig. 1 | 8 |
| Fig. 2 | 15 |
| Fig. 3 | 25 |
| Fig. 4 | 17 |
| Extended Data Fig. 1 | 57 |
| Extended Data Fig. 2 | 69 |
| Extended Data Fig. 3 | 59 |
| Extended Data Fig. 4 | 35 |
| Extended Data Fig. 5 | 63 |
| Extended Data Fig. 6 | 51 |
| Extended Data Fig. 7 | 67 |
| Extended Data Table 1 | 27 |
| Extended Data Table 2 | 31 |
| Extended Data Table 3 | 65 |
| Supplementary Fig. 1 | 37 |
| Supplementary Fig. 2 | 55 |
| Supplementary Fig. 3 | 39 |
| Supplementary Fig. 4 | 13 |
| Supplementary Fig. 5 | 41 |
| Supplementary Fig. 6 | 43 |
| Supplementary Fig. 7 | 45 |
| Supplementary Fig. 8 | 47 |
| Supplementary Fig. 9 | 53 |
| Supplementary Fig. 10 | 71 |
| Supplementary Fig. 11 | 71 |
| Supplementary Fig. 12 | 71 |
| Supplementary Fig. 13 | 71 |
| Supplementary Fig. 14 | 71 |
| Supplementary Fig. 15 | 71 |
| Supplementary Fig. 16 | 71 |
| Supplementary Fig. 17 | 71 |
| Supplementary Fig. 18 | 71 |
| Supplementary Fig. 19 | 71 |
| Supplementary Fig. 20 | 71 |
| Supplementary Fig. 21 | 71 |
| Supplementary Fig. 22 | 77 |
| Supplementary Fig. 23 | 79 |
| Supplementary Table 1 | 73 |
| Box 1 | `fig_box_schematic.py` |
| Supplementary Table 2, Supplementary Data 1 | `make_indicator_table.py` |

The last cell of the notebook copies every display item into `data/<run>/figures/manuscript/` under these names
(Fig1, EDFig1, SuppFig1, ...).

## Scripts

### Running the chain

| File | What it does |
|---|---|
| `link_ssp_inputs.sh` | Links the CMIP6 CO2 concentration files into inputs/co2/ and the CMIP6 scenario files (thetao, so, dissic, talk; ssp126, ssp245, ssp370, ssp585) into inputs/raw/<ssp>/ of ... |
| `prepare_inputs.sh` | 1. Links the CMIP6 files used in this study into data/raw/<exp>/ for every historical and piControl run with thetao, so, dissic and talk. These files |
| `rerun.sh` | Full run of the OA_boundary chain for one run of oa_config.py: ./rerun.sh u74 (likewise l10, m10, m10l10). Everything is computed into data/<run>. Inputs that depend neit ... |
| `rerun_product.sh` | The data product for one run, after rerun.sh: ./rerun_product.sh u74 (likewise l10, m10, m10l10) Steps: |
| `run_checks.sh` | The uncertainty of the fCO2-products and the checks of the emergent constraint for one run, after its rerun.sh (u74 for the paper); the other checks behind statements of  ... |
| `run_production.sh` | the whole chain of one run, from the raw model output to the data product with its add-on, unattended, stopping at the first failing step (nothing runs on stale input). S ... |
| `run_structural.sh` | the structural adjustments of the decline, in order (start from code/): OA_RUN=u74 nohup bash run_structural.sh > ../logs/u74/run_structural.log 2>&1 & |

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
| `fill_tools.py` | Filling of ocean cells without a value from the cells around them, used for the extension of the decline to the whole ocean (structural_adjustments.py, test_polar_shift_m ... |
| `jiang_chem.py` | The surface states of Jiang et al. (2023, NCEI 0259391, median of 14 ESMs) recomputed with mocsy, as every other carbonate-system calculation of this study. |
| `level_uncertainty.py` | The level uncertainty of the data product that comes from the reference itself, per cell and for the area means, stored separately from the constraint uncertainty (calc_p ... |
| `oa_config.py` | The runs of the OA_boundary chain. The run is chosen with the environment variable OA_RUN (set before python starts, e.g. by rerun.sh, or in the first notebook cell); it  ... |
| `pi_states.py` | The preindustrial CO2 states (atmospheric xCO2, ppm) used in the paper, shared by calc_pi_xco2.py (which computes Omega_arag for each of them with mocsy) and the notebook ... |
| `regional_common.py` | Shared data and calculations for the figures and tables of the notebook. For each product, on the common mask and the product's grid (from data/constraint_results.pkl, |
| `state_constraint.py` | The emergent constraint applied to the model STATE (the method of the data product): For a set of n models with the predictors dSST (warming of the reference window relat ... |

### Main analysis (rerun.sh)

| File | What it does |
|---|---|
| `calc_bias_adjusted.py` | Bias-adjusted, drift-corrected surface Omega_arag for one CMIP6 model, using an observation-based product as reference (OceanSODA-ETHZ v2025 by default). |
| `calc_drift.py` | piControl drift of surface T, S, DIC and ALK for every model, aligned with the historical run through its branch point. Per cell, the annual means of the whole available  ... |
| `calc_mechanism.py` | Mean surface states for testing why the bias-adjusted preindustrial Omega_arag depends on the raw model bias. |
| `calc_model_sst.py` | Top-level ocean temperature (thetao) of every model's historical run, averaged over the periods the constraint needs, so that the model warming is defined exactly as the  ... |
| `calc_omega_arag.py` | Surface aragonite saturation state (Omega_arag) for the CMIP6 historical and piControl runs linked by prepare_inputs.sh, computed with mocsy 2.4 |
| `calc_omega_arag_anymonth.py` | Annual surface Omega_arag of the four observation-based products from ALL available months: a cell enters a year as soon as it has one valid month (valid: T, S, DIC, TA p ... |
| `calc_omega_arag_obs.py` | Surface aragonite saturation state from OceanSODA-ETHZ v2025 (1982-2024), recomputed with mocsy 2.4.0 from the product's talk, dic, temperature and |
| `calc_omega_arag_products.py` | Surface Omega_arag from further observation-based products, computed exactly as for OceanSODA in calc_omega_arag_obs.py (mocsy 2.4.0, optcon='mol/kg', optt='Tpot', optp=' ... |
| `calc_pi_xco2.py` | Preindustrial Omega_arag of every model for the preindustrial CO2 states used in the paper (1750 and 1850 of AR6 with their +-1 sigma, their midpoint, the effective CO2 o ... |
| `common_mask_cell.py` | Notebook cell (after the constraint cell): all products on one common ocean area. The common mask is the intersection of the constraint masks of all products that are don ... |
| `run_constraint.py` | Runs the emergent constraint for all products (constraint_refs.py) and again on the common mask (common_mask_cell.py), prints both tables, and saves everything the figure ... |
| `run_constraint_jiang.py` | The full chain of this study with the observation-based state of Jiang et al. (2023, decade 2010 = January 2005 to December 2014) as a fifth reference: bias adjustment to ... |

### Uncertainty of the fCO2-products (run_checks.sh)

| File | What it does |
|---|---|
| `product_constants_uncertainty.py` | Uncertainty of the global-mean surface Omega_arag from the equilibrium constants (Orr et al. 2018), for the UExP-FNN-U v2026-1 2024 state, and its effect on the relative  ... |
| `product_uncertainty_mc_C.py` | Uncertainty of the 2024 global-mean surface Omega_arag of an observation-based product from its mapping uncertainties, with spatially correlated errors (Ford et al. 2024, ... |

### Data product OA-Track (rerun_product.sh, run_production.sh)

| File | What it does |
|---|---|
| `calc_product_members.py` | Members of the data product: for one reference product and every model, the annual means of all carbonate system indicators of the adjusted (drift-corrected, bias-adjuste ... |
| `calc_product_obs.py` | The observation-based part of the data product: all carbonate system indicators of every reference product (OceanSODA-ETHZ, UExP-FNN-U, JMA-MLR, CMEMS-LSCE-FFNN; monthly  ... |
| `calc_product_states.py` | data product, pass 2: per run and reference The constrained and the unconstrained carbonate system of every year, from the models' states, with the |
| `calc_smooth_states.py` | data product, pass 1: once for all references and all configurations The forced monthly state of every model for the data product: T, S, DIC and ALK of the historical run ... |
| `cuc_mapping_uncertainty.py` | The mapping uncertainty of the observed CUC (the predictor of the emergent constraint) and its correlation with the mapping error of the observed Omega_arag, per product, ... |
| `fit_sigma.py` | The 1 sigma of the forced state estimated from annual values (the 'sigma_fit' of Extended Data Table 2), for many series at once and every year of the record: the same es ... |
| `make_indicator_table.py` | Supplementary Table 2 and Supplementary Data 1: the 15 quantities of OA-Track, 1750-2024 The best estimate of the paper for every year from 1750 to 2024, global mean on t ... |
| `make_oatrack.py` | data product OA-Track: the complete output for each fCO2-product, from which make_oatrack_product.py writes the archived record Writes the complete output of OA-Track, th ... |
| `make_oatrack_product.py` | OA-Track as published Writes OA-Track as defined in oatrack_names.py (every file, variable, dimension, long name and description from there), |
| `make_oatrack_simple.py` | the estimate of the forced state, the temperature flag and the storage precision that make_oatrack_product.py uses; its own compact files are not part of the record Write ... |
| `make_product.py` | data product, final step: the product files from the states and the observation-based products Per reference (OceanSODA-ETHZ, UExP-FNN-U, JMA-MLR, CMEMS-LSCE-FFNN; Jiang  ... |
| `make_product_alkalinity.py` | data product: the alkalinity adjustment of the best estimate The adjustment of the carbonate system for the surface alkalinity rise that the observation-based products an ... |
| `oatrack_names.py` | the files and variables of OA-Track as published The one definition of the files and variables of OA-Track (https://doi.org/10.17882/121840), used by the script that |
| `patch_product_xco2.py` | Completes the atmospheric CO2 of the data product files in place, so that every value of the product comes with the CO2 of its year: |

### Adjustments to the global mean and projection (run_structural.sh, notebook)

| File | What it does |
|---|---|
| `barrett_sampling.py` | The sampling of the observed surface TA trend of Barrett et al. (2025, GBC), on the 1 degree grid of the products, written to data/<run>/structural/barrett_sampling.npz.  ... |
| `feedback_pattern.py` | The spatial pattern of the calcification feedback in surface alkalinity, for the alkalinity adjustment of structural_adjustments.py, written to data/<run>/structural/feed ... |
| `projection.py` | The projections of the surface carbonate system that the figures use, computed once per run and cached in the run's data folder (delete a cache file to recompute it). Eve ... |
| `structural_adjustments.py` | The two adjustments of the decline 1750-2024 beyond the analysis on the common area (Extended Data Fig. 4, Extended Data Tables 1 and 2), for each of the four products, w ... |
| `test_polar_shift_models.py` | The whole ocean against the common area of the paper in each of the 18 ESMs: the shift of the decline and the gap of the level of Omega_arag. structural_adjustments.py ta ... |

### Archived records: OA-Track at SEANOE, the ESM2M output at Zenodo

| File | What it does |
|---|---|
| `list_oatrack_names.py` | Writes the full list of the files and variables of OA-Track from oatrack_names.py: OA-Track_variables.pdf every file and every variable with dimensions and units; long na ... |
| `pack_esm2m.py` | the GFDL-ESM2M-COBALTv2 output of the paper for its own Zenodo record Copies the part of the GFDL-ESM2M-COBALTv2 simulations that the analysis reads (feedback_pattern.py: |
| `pack_oatrack.py` | Packs OA-Track for its SEANOE record (https://doi.org/10.17882/121840) from the output of make_oatrack_product.py: OA-Track_README.pdf the README |

### Figures and tables

| File | What it does |
|---|---|
| `OA_boundary_figures.ipynb` | the figures and tables of the paper, from top to bottom; the last cell exports them under the manuscript names |
| `fig_box_schematic.py` | the schematic of Box 1: the method and what can go wrong a the method: observed state of an fCO2-product minus the change simulated by each ESM gives one preindustrial st ... |

### Checks behind statements of the paper

| File | What it does | Statement of the paper |
|---|---|---|
| `calc_pi1750_nutrients.py` | The 1750 Omega_arag of every model, exactly as calc_pi_xco2.py computes it (every month of the drift-corrected, bias-adjusted piControl; DIC lowered until surface pCO2 ha ... | Methods, 'Carbonate chemistry': the 1750 state with the nutrients of the World Ocean Atlas 2023 |
| `check_annual_state_omega.py` | The models enter the chain as annual means (step 1: one value per year of T, S, DIC, TA), so their Omega_arag (1750 and historical) is the saturation state of the annual- ... | Methods, 'Carbonate chemistry': annual-mean states against the mean of monthly saturation states in the fCO2-products |
| `check_j_models.py` | Does the offset J = (mean of the monthly Omega_arag) - (Omega_arag of the annual-mean state) change between an early, near-preindustrial period and the present in the mod ... | Methods, 'Carbonate chemistry': the same in the ESMs, and its change from 1850-1869 to 1995-2014 |
| `nutrient_sensitivity.py` | How much the choice of the nutrients (silicate and phosphate) changes the observed state, the constrained 1750 state and the decline, per product on its own and on the co ... | Methods, 'Carbonate chemistry': the nutrients (GLODAPv2 scaled by ±20 %, World Ocean Atlas 2023); run_checks.sh |
| `phc_2020.py` | The decline of the Planetary Health Check (2026) for other years than 2024, in its own framework: global mean of saturation_aragonite of UExP-FNN-U (mean of the 12 months ... | Extended Data Table 3: the year 2020 in the framework of the Planetary Health Check |
| `phc_area.py` | Why does averaging 1750 and 2024 over different areas make the decline of the Planetary Health Check (2026) larger? In its framework (UExP-FNN-U v2026-1, 2024 = mean of t ... | Main text, 'Consistency with earlier estimates': the areas of the Planetary Health Check |
| `phc_baseline_regions.py` | Do the observation-based estimates of the Planetary Health Check (2026) differ more outside our common area? In the step from its areas to the common area (Fig. 3c), only ... | Main text, 'Consistency with earlier estimates': the observational baselines inside and outside the common area |
| `test_cell_crossing.py` | by how much does the cell-by-cell constraint move the local crossing of the boundary? For each observation-based product, cell by cell: the forced series of the product y ... | Methods, 'Emergent constraint on the preindustrial state': the cell-by-cell constraint and the local crossing of the boundary |
| `test_constraint_loo.py` | Perfect-model (leave-one-out) test of the emergent constraint with the real models, per carbonate system indicator and per predictor set (dSST only, CUC only, both): does ... | Methods, 'Emergent constraint on the preindustrial state', and Supplementary Methods: the leave-one-out test (Supplementary Fig. 22); run_checks.sh |
| `test_dic_from_fco2.py` | does it matter that we take the DIC of the fCO2-products instead of their fCO2? The chain takes DIC and alkalinity (with T and S) of every fCO2-product and computes Omega ... | Methods, 'Carbonate chemistry': DIC solved from the mapped fCO2 instead of the DIC of the fCO2-products |
| `test_diseq_residual.py` | Does the third control of the simulated carbon uptake, how closely surface waters follow the rising atmospheric CO2, explain the part of the model spread that CUC and dSS ... | Methods, 'Emergent constraint on the preindustrial state': the air-sea disequilibrium |
| `test_drift_cutoff.py` | does the 50-year cutoff of the drift spline matter for the decline of the models? The chain removes the piControl drift with an Enting spline of 50-year cutoff (calc_drif ... | Methods, 'Ocean acidification outside the observed period': the cutoff of the drift spline |
| `test_polar_smoothing.py` | does the whole-ocean shift of the ESMs depend on the smoothing of their states? structural_adjustments.py takes the whole-ocean shift of the 18 ESMs from test_polar_shift ... | Methods, 'Extension to the whole ocean': the smoothing of the states of the ESMs |
| `test_product_alk_trend.py` | do the products already contain an alkalinity trend beyond salinity? The alkalinity adjustment assumes that the products miss the rise of surface alkalinity from weaker c ... | Methods, 'Alkalinity adjustment': no alkalinity rise in the fCO2-products |
| `test_spna_crossing.py` | how much earlier does the regional constraint put the local crossing of the boundary in the subpolar North Atlantic? | Main text: the subpolar North Atlantic and the region south of Greenland |
| `test_state_constraint.py` | One product, one state (1750): the constrained STATE against the constrained INDICATOR. direct the paper: Omega_arag of every model's 1750 state, regressed on dSST and CU ... | Extended Data Fig. 3f: the state constraint against the direct regression; run_checks.sh |

## Licence

MIT licence (`LICENSE`). OA-Track itself is published at SEANOE under CC BY 4.0.
