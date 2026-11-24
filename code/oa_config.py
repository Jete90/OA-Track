"""
oa_config.py
============
The runs of the OA_boundary chain. The run is chosen with the environment variable OA_RUN (set before python
starts, e.g. by rerun.sh, or in the first notebook cell); it sets

  BASE         the production folder: the parent of the folder this file is in (code/), so the whole folder can
               be copied or renamed without editing any path; OA_BASE in the environment overrides it
  DATA         data folder all calc scripts read from and write to (BASE/data/<run>)
  OPTB         mocsy total boron  ('u74' Uppstroem 1974, 'l10' Lee et al. 2010)
  OPTK1K2      mocsy K1, K2       ('l' Lueker et al. 2000, 'm10' Millero 2010)
  REF_DENSITY  models that carry DIC and alkalinity per kg and write the CMIP6 mol m-3 output as
               tracer * constant density: GFDL-CM4 (BLING) and GFDL-ESM4 (COBALT), RHO_0 = 1035 kg m-3
               (generic_BLING.F90, generic_COBALT.F90). All other models carry per-volume tracers (or per-kg
               tracers written with in-situ density) and are converted with the in-situ surface density.
  CONSTANTS    text for the netCDF attributes
  WEIGHTS      the area weights of every mean and area on the 1-degree product grid (OA_WEIGHTS in the environment):
               'ocean' (default): cell area (cos latitude) times the ocean fraction of the cell from the ESA CCI Land
               Cover ocean map (inputs/ESACCI_water_bodies/ocean_fraction_1deg.nc, ocean_fraction_1deg.py), so that coastal cells count
               by their sea surface; 'cell': the cell area alone (the weights of version 3). area_weights(lat, lon)
               gives the weights of a grid. Diagnostics printed on the native model grids (raw Omega, drift, model
               SST) use the cell area in both cases.

u74 is the main run of the paper; l10, m10 and m10l10 are the sensitivity runs to the carbonate constants.
regional_common.py (figures, tables) uses RUNS for the folders and chooses the run itself from OA_RUN.

Version 2 of the chain (November 2026): the adjusted model state is the same for both experiments and every
reference, X(t) - [s(k(t)) - mean of s over the window years] + bias, with s the Enting spline of the
piControl (calc_drift.py); the preindustrial states are computed from the adjusted piControl with DIC
lowered cell by cell to the target CO2 (calc_pi_xco2.py), no global factors.

Time step: the code handles monthly and annual model output (calc_bias_adjusted.ModelRun.step, 12 or 1). All 18
CMIP6 models of this study are ANNUAL MEANS (step 1, check_time_step.py), so wherever docstrings, comments or
netCDF attributes say 'month' or 'monthly', read 'time step': the bias is one annual field, the preindustrial
state is the mean over all control years, and the 12 'monthly climatology' slots of calc_pi_xco2.py and
calc_product_states.py are identical copies of the time mean. The observed Omega_arag (calc_omega_arag_obs.py,
calc_omega_arag_products.py, calc_product_obs.py) is the saturation state of the annual-mean T, S, DIC and TA,
the same definition as for the models (5 November 2026; before, the mean of the monthly saturation states,
0.002 higher, stored as omega_arag_monthly_mean).
"""
import math
import os
from pathlib import Path

VERSION = ('v4 (November 2026): v3 (drift-corrected adjusted annual states, preindustrial states cell by cell, annual-mean-state '
           'Omega_arag; temperatures floored at the freezing point before the chemistry and unsolved carbonate systems NaN; '
           'alkalinity adjustment from the time series of the four ESMs with saturation-dependent calcification; complete '
           'GFDL-ESM4 (gr) and MRI-ESM2-0 salinity files) with area weights cell area x ocean fraction (OA_WEIGHTS); '
           'UExP-FNN-U adjusted over its own years 1980-2014 (12 November 2026)')

BASE = Path(os.environ.get('OA_BASE', Path(__file__).resolve().parents[1]))

RUNS = {
    'u74': dict(data='data/u74', optb='u74', optk1k2='l'),        # main run
    'l10': dict(data='data/l10', optb='l10', optk1k2='l'),        # total boron Lee et al. (2010)
    'm10': dict(data='data/m10', optb='u74', optk1k2='m10'),      # K1, K2 Millero (2010)
    'm10l10': dict(data='data/m10l10', optb='l10', optk1k2='m10'),  # both
}
REF_DENSITY = {'GFDL-CM4': 1035.0, 'GFDL-ESM4': 1035.0}

_K1K2 = {'l': 'Lueker et al. (2000)', 'm10': 'Millero (2010)'}
_BORON = {'u74': 'Uppstroem (1974)', 'l10': 'Lee et al. (2010)'}

RUN = os.environ.get('OA_RUN', 'u74')
if RUN not in RUNS:
    raise ValueError(f"OA_RUN = '{RUN}' unknown; runs: {', '.join(RUNS)}")
DATA = BASE / RUNS[RUN]['data']
INPUTS = BASE / 'inputs'
CODE = BASE / 'code'

# ------------------------------------------------------------------------------------------------ raw inputs
# Where the raw inputs lie, outside the production folder (the defaults are the folders on the server of the study;
# set the environment variables to run elsewhere; README.md lists what each folder must contain):
#   OBS_DIR    OA_OBS     the observation-based products (fCO2-products, SST and nutrient climatologies as downloaded)
#   OBS2_DIR   OA_OBS2    SST reconstructions (ERSSTv6, COBE-SST3, DCENT-I) and the GLODAPv2 mapped climatology
#   MODEL_DIR  OA_MODELS  CMIP6 output on the 1 degree grid (CMIP_DATA/CMIP6_DATA/<experiment>) and the CO2 forcing
#   ESM2M_DIR  OA_ESM2M   output of the GFDL-ESM2M-COBALTv2 simulations and the GLODAPv2.2023 merged master file
#   MOCSY_PATH OA_MOCSY   the folder of the compiled mocsy (mocsy 2.4.0, https://github.com/jamesorr/mocsy)
OBS_DIR = Path(os.environ.get('OA_OBS', '/oceandata01/observational_data'))
OBS2_DIR = Path(os.environ.get('OA_OBS2', '/oceandata02/terhaar/Data/observation_based'))
MODEL_DIR = Path(os.environ.get('OA_MODELS', '/oceandata02/terhaar/Data/model_data'))
ESM2M_DIR = Path(os.environ.get('OA_ESM2M', '/storage/climatestor/ArcticECO/data/Fei'))
MOCSY_PATH = os.environ.get('OA_MOCSY', '/home/terhaar/mocsy_env_jens/mocsy-2.4.0/')
OPTB = RUNS[RUN]['optb']
OPTK1K2 = RUNS[RUN]['optk1k2']
CONSTANTS = (f"constants: K1, K2 of {_K1K2[OPTK1K2]} (optk1k2={OPTK1K2}), total boron of {_BORON[OPTB]} "
             f"(optb={OPTB}), mocsy defaults otherwise")

# ------------------------------------------------------------------------------------------------ reference windows
# The reference period W of the bias adjustment of every observation-based estimate: the years the product shares with
# the historical simulations (to 2014), from its first complete year (OceanSODA-ETHZ 1983, UExP-FNN-U 1980,
# CMEMS-LSCE-FFNN 1985, JMA-MLR 1990), and for Jiang et al. (2023) their decade 2010 (2005-2014). Every script of the
# chain takes the windows from here (bias, mechanism states, constraint, CUC mapping, structural adjustments).
WINDOWS = {'OceanSODA': (1983, 2014), 'UExP': (1980, 2014), 'JMA': (1990, 2014), 'CMEMS': (1985, 2014),
           'Jiang': (2005, 2014)}
# mean atmospheric pCO2 of each window (uatm; the products: NOAA global mean xCO2 of the window times about 0.978 for
# the water vapour; Jiang et al.: their xCO2). It scales the CUC of the models and of the observations alike and
# therefore does not change the constraint.
WINDOW_PCO2 = {'OceanSODA': 359.8, 'UExP': 357.5, 'JMA': 366.0, 'CMEMS': 361.0, 'Jiang': 387.7}
# the periods whose means calc_drift.py (drift) and calc_model_sst.py (model SST) store: 1850-1900 and every window
PERIODS = sorted({(1850, 1900)} | set(WINDOWS.values()))

# ------------------------------------------------------------------------------------------------ mapping errors
# The decorrelation lengths of the mapping errors of the products (Ford et al. 2024, GBC: UExP-FNN-U, about 1800 and
# 2400 km) are uncertain, and assumed for all products. As Ford et al. perturb their lengths within their uncertainty
# in every draw, every length L is taken as log-uniformly distributed between L / L_RANGE and L * L_RANGE,
# represented by N_L equally probable factors (the midpoints of N_L equal intervals in log space).
# level_uncertainty.py: the expected correlation, mean of exp(-d / (f L)) over the factors f (exact for the variance);
# product_uncertainty_mc_C.py: one factor drawn per draw and error component.
L_RANGE = 2.0
N_L = 9
L_FACTORS = tuple(math.exp(math.log(L_RANGE) * (2.0 * (k + 0.5) / N_L - 1.0)) for k in range(N_L))

# ------------------------------------------------------------------------------------------------ area weights
WEIGHTS = os.environ.get('OA_WEIGHTS', 'ocean')
if WEIGHTS not in ('ocean', 'cell'):
    raise ValueError(f"OA_WEIGHTS = '{WEIGHTS}' unknown; 'ocean' (cell area x ocean fraction) or 'cell' (cell area)")
OCEAN_FRACTION_FILE = INPUTS / 'ESACCI_water_bodies' / 'ocean_fraction_1deg.nc'
WEIGHTS_TEXT = {'ocean': 'cell area (cos latitude) times the ocean fraction of the cell (ESA CCI Land Cover ocean map, '
                         'ocean_fraction_1deg.nc)',
                'cell': 'cell area (cos latitude)'}[WEIGHTS]
_FRACTION = {}
if WEIGHTS == 'ocean' and not OCEAN_FRACTION_FILE.exists():
    raise FileNotFoundError(f'{OCEAN_FRACTION_FILE} missing (ocean_fraction_1deg.py); OA_WEIGHTS=cell uses the cell area alone')


def _ocean_fraction(lat, lon):
    """the ocean fraction of the cells of the grid lat (ny), lon (nx): nearest cell of the 1-degree map, longitude periodic
    (a longitude exactly between two map cells takes the lower index; the x.5 grids of the chain never do)"""
    import numpy as np
    if 'grid' not in _FRACTION:
        import netCDF4
        if not OCEAN_FRACTION_FILE.exists():
            raise FileNotFoundError(f'{OCEAN_FRACTION_FILE} missing (ocean_fraction_1deg.py); OA_WEIGHTS=cell uses the cell area alone')
        with netCDF4.Dataset(OCEAN_FRACTION_FILE) as f:
            _FRACTION['grid'] = (np.asarray(f['lat'][:], 'f8'), np.asarray(f['lon'][:], 'f8'),
                                 np.clip(np.ma.filled(f['ocean_fraction'][:].astype('f8'), 0.0), 0.0, 1.0))
    flat, flon, frac = _FRACTION['grid']
    lat, lon = np.asarray(lat, 'f8').reshape(-1), np.asarray(lon, 'f8').reshape(-1)
    jy = np.abs(lat[:, None] - flat[None, :]).argmin(axis=1)
    d = np.abs(((lon[:, None] - flon[None, :]) + 180.0) % 360.0 - 180.0)
    ix = d.argmin(axis=1)
    return frac[np.ix_(jy, ix)]


def area_weights(lat, lon):
    """(ny, nx) area weights of the grid lat (ny), lon (nx) according to WEIGHTS: cos(latitude), times the ocean
    fraction of every cell for 'ocean'. Every mean, area and fraction of the chain on the product grid uses these."""
    import numpy as np
    lat, lon = np.asarray(lat, 'f8'), np.asarray(lon, 'f8')
    assert lat.ndim == 1 and lon.ndim == 1, f'area_weights takes the 1-D latitude and longitude vectors, got {lat.shape}, {lon.shape}'
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones(lon.size)
    if WEIGHTS == 'ocean':
        w = w * _ocean_fraction(lat, lon)
    return w
