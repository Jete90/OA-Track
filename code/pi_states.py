"""
pi_states.py
============
The preindustrial CO2 states (atmospheric xCO2, ppm) used in the paper, shared by calc_pi_xco2.py (which
computes Omega_arag for each of them with mocsy) and table_summary.py.

AR6 (Gulev et al. 2021, Ch. 2): 278.3 +- 2.9 ppm in 1750 and 285.5 +- 2.1 ppm in 1850, very likely ranges
(the chapter's convention) -> 1 sigma = range / 1.645. '_lo'/'_hi' = -/+ 1 sigma, for the uncertainty of each
state. 'mid' = midpoint of 1750 and 1850 (sigma = mean of the two: correlated ice-core errors). 'jiang' =
effective preindustrial CO2 of Jiang et al. (2023): the CMIP6 forcing of their 1850 state (1850-1854 mean
284.59 ppm, Meinshausen et al. 2017) minus their ice-core difference 1852 - 1752 (288.57 - 276.39 ppm).
'x280' = the nominal CO2 of the original boundary (Guinotte and Fabry 2008). 'pi_low'/'pi_high' (275,
284 ppm): the earlier sensitivity range, kept for compatibility.

The data product (calc_product_states.py) does not use per-model states of other years: the years 1750-1849 are
computed at the CO2 of each year (CMIP6 history, Meinshausen et al. 2017) on the constrained monthly climatology
of the adjusted piControl, exactly, once per reference. The per-model named states are the paper's values and
the check of the constrained state against the direct regression. Per-model states of every year
(calc_pi_xco2.py --explicit-years) exist only as an option for tests.
"""
X1750, S1750 = 278.3, 2.9 / 1.645
X1850, S1850 = 285.5, 2.1 / 1.645
XMID, SMID = 0.5 * (X1750 + X1850), 0.5 * (S1750 + S1850)
X1850_CMIP6 = 284.59                       # Meinshausen et al. (2017), 1850-1854 mean
XJIANG = X1850_CMIP6 - (288.57 - 276.39)
TARGETS = {'1750': X1750, '1750_lo': X1750 - S1750, '1750_hi': X1750 + S1750,
           '1850': X1850, '1850_lo': X1850 - S1850, '1850_hi': X1850 + S1850,
           'mid': XMID, 'mid_lo': XMID - SMID, 'mid_hi': XMID + SMID,
           'jiang': XJIANG, 'x280': 280.0, 'pi_low': 275.0, 'pi_high': 284.0}
# 1 sigma (ppm) of the CO2 of each state: AR6 for 1750 and 1850, their mean for the midpoint, the 1750 ice-core
# value for the Jiang et al. state (derived from the ice cores), none for 280 ppm (a set value, not an estimate)
XCO2_SIGMA = {'1750': S1750, '1850': S1850, 'mid': SMID, 'jiang': S1750, 'x280': 0.0}
# the years 1750-1849 at their CO2 (CMIP6 history, Meinshausen et al. 2017, the forcing of the historical runs):
# targets 'y1750' ... 'y1849' for calc_pi_xco2.py --explicit-years (tests only, see above)
YEARS_EXPLICIT = list(range(1750, 1850))


def year_targets():
    """{'y1750': xCO2, ...} from the CMIP6 historical CO2 file (inputs/co2/); raises if the file is missing"""
    import co2_history
    h = co2_history.CO2History()
    h.require('historical')
    return {f'y{y}': float(h.value(y)) for y in YEARS_EXPLICIT}


# the point states of the data product (their CO2 term: the constrained state lowered to xCO2 -/+ sigma)
PRODUCT_STATES = ['1750', '1850', 'mid', 'jiang', 'x280']
