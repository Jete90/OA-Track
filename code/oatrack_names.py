#!/usr/bin/env python
"""
oatrack_names.py  (the files and variables of OA-Track as published)
================
The one definition of the files and variables of OA-Track (https://doi.org/10.17882/121840), used by the script that
writes them and written out by list_oatrack_names.py.

Per quantity <q> three files and a CSV:
    OA-Track_<q>_global_mean.nc (.csv)    global means 1750-2100: the preindustrial states, the fCO2-products and the
                                          ESMs (historical and four SSPs), each fCO2-product and their mean, on three areas
    OA-Track_<q>_maps_obs_based_products.nc  maps of the preindustrial states and of the obs-based products (fCO2-products)
                                          1980-2025, each
                                          fCO2-product and their mean
    OA-Track_<q>_maps_esms.nc             maps of the ESMs 1750-2100 (historical and four SSPs), the mean of the
                                          fCO2-products

Every name is   <q>_<block>[_gm][_change][_<term>]
    blocks      preindustrial (state 1750, 1850), obs_annual, obs_annual_forced, esms_annual_forced (scenario)
    _gm         a global mean, with the dimension area; otherwise a map
    _change     the change from the preindustrial state of the dimension state, relative (%; absolute for pH,
                temperature and salinity)
    terms       sigma (every term included), sigma_upper_limit (changes of the ESM years), alkalinity_adjustment, and in
                the maps sigma_alkalinity (its part of the 1 sigma, with the spread of the patterns)
    dimensions  area (common_area, product_area, whole_ocean), product (mean, OceanSODA, UExP, JMA, CMEMS), scenario
                (historical, ssp126, ssp245, ssp370, ssp585), state (1750, 1850), year
"""
QUANTITIES = ['omega_arag', 'omega_calc', 'ph', 'hplus', 'hplus_free', 'co3', 'hco3', 'co2', 'pco2', 'fco2', 'dic',
              'talk', 'revelle', 'thetao', 'so']
UNITS = {'omega_arag': '1', 'omega_calc': '1', 'ph': '1', 'pco2': 'uatm', 'fco2': 'uatm', 'co2': 'umol kg-1',
         'hco3': 'umol kg-1', 'co3': 'umol kg-1', 'revelle': '1', 'hplus': 'nmol kg-1', 'hplus_free': 'nmol kg-1',
         'dic': 'umol kg-1', 'thetao': 'degC', 'so': 'psu', 'talk': 'umol kg-1'}
LONG = {'omega_arag': 'aragonite saturation state', 'omega_calc': 'calcite saturation state',
        'ph': 'pH (total scale)', 'pco2': 'partial pressure of CO2', 'fco2': 'fugacity of CO2',
        'co2': 'dissolved CO2 (CO2*)', 'hco3': 'bicarbonate ion', 'co3': 'carbonate ion', 'revelle': 'Revelle factor',
        'hplus': 'hydrogen ion concentration (total scale)', 'hplus_free': 'hydrogen ion concentration (free scale)',
        'dic': 'dissolved inorganic carbon', 'thetao': 'temperature', 'so': 'salinity', 'talk': 'total alkalinity'}
ALK = ('omega_arag', 'omega_calc', 'ph', 'hplus', 'hplus_free', 'co3', 'hco3', 'dic', 'talk', 'revelle')
ABSOLUTE_CHANGE = ('ph', 'thetao', 'so')          # change in the units of the quantity; all others relative (%)
NO_CONSTANTS = ('revelle',)                      # mocsy gives no uncertainty of the Revelle factor from the constants
T_DEPENDENT = ('thetao', 'pco2', 'fco2', 'co2', 'ph', 'hplus', 'hplus_free', 'dic', 'hco3', 'co3', 'revelle')
WHOLE_OCEAN = ('omega_arag',)                    # whole-ocean global means in this version
PRODUCTS = ['mean', 'OceanSODA', 'UExP', 'JMA', 'CMEMS']
SCENARIOS = ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
YEARS_GLOBAL = (1750, 2100)
YEARS_OBS = (1980, 2025)

# ------------------------------------------------------------------------------------------------ texts
BLOCK = {
    'preindustrial': (['state'], 'preindustrial state',
                      'constrained preindustrial state at the CO2 of IPCC AR6 (278.3 ppm in 1750, 285.5 ppm in 1850)'),
    'obs_annual': (['year'], 'annual values of the fCO2-products',
                   'annual means of the fCO2-products 1980-2025, with the real year-to-year variability'),
    'obs_annual_forced': (['year'], 'forced state of the fCO2-products',
                          'quadratic fit of the annual values of each fCO2-product up to 2024, evaluated in the years '
                          '1980-2024; 2024 is the state of the paper'),
    'esms_annual_forced': (['scenario', 'year'], 'forced state of the constrained ESMs',
                           'ESMs bias-adjusted to the fCO2-product and constrained (emergent constraint), smoothed (30-year '
                           'Enting spline); historical: all 18 ESMs 1750-2014; ssp126 to ssp585: the ESMs that provide the '
                           'scenario, 1750-2100 (their past differs slightly from historical); 1750-1849 the constrained '
                           'preindustrial ocean at the CMIP6 CO2 of the year'),
}
PRODUCT_TEXT = ('product mean: the mean of the fCO2-products (1 sigma with their spread); OceanSODA, UExP, JMA, CMEMS: '
                'each fCO2-product, with the ESMs bias-adjusted and constrained to it (1 sigma without the spread)')
AREA_TEXT = ('area common_area: cells covered by all four fCO2-products in every year and by all 18 ESMs (84% of the '
             'ocean); product_area: cells covered by the fCO2-product in every year and by all 18 ESMs (empty for the '
             'mean); whole_ocean: the whole ocean as in the paper (empty for the annual values)')


DELTA = ('delta (the change from the reference period of the fCO2-products, today, to the state or year, from the '
         'bias-adjusted and constrained ESMs, including the uncertainty of the atmospheric CO2)')


def sigma_words(q, block, change=False, gm=False):
    """what _sigma of a value or change contains"""
    obs = block.startswith('obs')
    if change:
        w = ['delta of the preindustrial state' if obs else 'delta of both years',
             'spread of the change between the fCO2-products (mean only)']
    else:
        w = ([] if obs else [DELTA]) + ['spread between the fCO2-products (mean only)',
              'mapping error'] + ([] if q in NO_CONSTANTS else ['equilibrium constants'])
    if block == 'obs_annual_forced':
        w.append('fit of the forced state')
    if q in ALK:
        w.append('alkalinity adjustment')
    if gm and q in WHOLE_OCEAN:
        w.append('extension to the whole ocean (whole_ocean only)')
    return w


TEXT = {
    '_sigma': 'every term included ({terms}); use it alone',
    '_sigma_change': 'every term included ({terms}; mapping error and constants cancel); use it alone',
    '_sigma_upper_limit': 'upper limit ({terms}); delta is correlated between the two years, so the true 1 sigma is '
                          'smaller (exact value in version 1.1)',
    '_alkalinity_adjustment_gm': 'included in the value and, as |a|, in {sig} (the rise has been measured only since '
                                 '1991); without the adjustment: value minus a, with 1 sigma sqrt({sig}^2 - a^2)',
    '_alkalinity_adjustment_map': 'included in the value; without the adjustment: value minus this, with 1 sigma '
                                  'sqrt({sig}^2 - {salk}^2)',
    '_sigma_alkalinity': 'the part of {sig} from the alkalinity adjustment, sqrt(a^2 + s^2): a the adjustment, s the '
                         'spread between the patterns of the four ESMs with saturation-dependent calcification (already '
                         'included in {sig}; only to remove the adjustment)',
}


# ------------------------------------------------------------------------------------------------ rows
class File:
    def __init__(self, q, name, title, sizes):
        self.q, self.name, self.title, self.sizes, self.rows = q, name, title, sizes, []

    def add(self, name, dims, units, long_name, desc):
        assert all(r['name'] != name for r in self.rows), f'{self.name}: {name} twice'
        self.rows.append(dict(file=self.name, name=name, dims=', '.join(dims) or 'scalar', dimlist=list(dims),
                              units=units, long_name=long_name, description=desc))

    def size(self):
        """uncompressed size in bytes (maps float32, global means float64); the historical scenario holds 265 of the
        351 years"""
        b = 0
        for r in self.rows:
            n = 1
            for d in r['dimlist']:
                n *= self.sizes[d]
            if 'scenario' in r['dimlist'] and 'year' in r['dimlist']:
                n = n // (self.sizes['scenario'] * self.sizes['year']) * (4 * 351 + 265)
            b += n * (4 if 'lat' in r['dimlist'] else 8)
        return b


def _block(F, q, block, gm, product, alk=True):
    """a value with _sigma and the alkalinity adjustment (and in the maps its part of the 1 sigma)"""
    tdims, short, desc = BLOCK[block]
    dims = (['area'] if gm else []) + (['product'] if product else []) + tdims + ([] if gm else ['lat', 'lon'])
    n = f'{q}_{block}' + ('_gm' if gm else '')
    head = f'{LONG[q]}, {"global mean" if gm else "map"}: {short}'
    F.add(n, dims, UNITS[q], head, desc + ('; ' + PRODUCT_TEXT if product else '; the mean of the fCO2-products')
          + ('; ' + AREA_TEXT if gm else ''))
    F.add(n + '_sigma', dims, UNITS[q], head + ', 1 sigma',
          TEXT['_sigma'].format(terms=', '.join(sigma_words(q, block, gm=gm))))
    if alk and q in ALK:
        _alk(F, n, dims, UNITS[q], head, '_sigma', gm)


def _change(F, q, block, gm, product):
    tdims, short, _ = BLOCK[block]
    dims = (['area'] if gm else []) + (['product'] if product else []) + ['state'] + tdims + ([] if gm else ['lat', 'lon'])
    n = f'{q}_{block}' + ('_gm' if gm else '') + '_change'
    head = f'{LONG[q]}, {"global mean" if gm else "map"}: change of the {short} from the preindustrial state'
    u = UNITS[q] if q in ABSOLUTE_CHANGE else '%'
    kind = 'change' if q in ABSOLUTE_CHANGE else 'relative change (%)'
    F.add(n, dims, u, head, f'{kind} from the preindustrial state of the same fCO2-product and area (dimension state: '
          '1750 or 1850) to the year')
    terms = ', '.join(sigma_words(q, block, change=True, gm=gm))
    if block.startswith('obs'):
        sig = '_sigma'
        F.add(n + sig, dims, u, head + ', 1 sigma', TEXT['_sigma_change'].format(terms=terms))
    else:
        sig = '_sigma_upper_limit'
        F.add(n + sig, dims, u, head + ', upper limit of the 1 sigma', TEXT['_sigma_upper_limit'].format(terms=terms))
    if q in ALK:
        _alk(F, n, dims, u, head, sig, gm)


def _alk(F, n, dims, u, head, sig, gm):
    F.add(n + '_alkalinity_adjustment', dims, u, head + ', alkalinity adjustment',
          TEXT['_alkalinity_adjustment_gm' if gm else '_alkalinity_adjustment_map'].format(sig=n + sig,
                                                                                          salk=n + '_sigma_alkalinity'))
    if not gm:
        F.add(n + '_sigma_alkalinity', dims, u, head + ', 1 sigma of the alkalinity adjustment',
              TEXT['_sigma_alkalinity'].format(sig=n + sig))


def _coords(F, q, names, years):
    areas = 'common_area, product_area' + (', whole_ocean' if q in WHOLE_OCEAN else '')
    text = {'lat': ('degrees_north', 'latitude', 'cell centre, 1 degree'),
            'lon': ('degrees_east', 'longitude', 'cell centre, 1 degree'),
            'year': ('', 'year', f'{years[0]}-{years[1]}'),
            'state': ('', 'preindustrial state', '1750, 1850 (at the CO2 of IPCC AR6, 278.3 and 285.5 ppm)'),
            'scenario': ('', 'scenario', 'historical (18 ESMs, 1750-2014), ssp126, ssp245, ssp370, ssp585 (1750-2100)'),
            'product': ('', 'fCO2-product', 'mean, OceanSODA, UExP, JMA, CMEMS'),
            'area': ('', 'area of the global mean', areas)}
    for d in names:
        F.add(d, [d], *text[d])


def _masks(F, q, product_area, constrained):
    F.add('cell_weight', ['lat', 'lon'], '1', 'area weight of the cell', 'cos(latitude) times the ocean fraction of the '
          'cell (ESA CCI Land Cover ocean map), proportional to the ocean area of the cell: the weights of every area '
          'mean of OA-Track, for regional means as the global means')
    F.add('mask_common_area', ['lat', 'lon'], '', 'common area: cells covered by all four fCO2-products in every year '
          'and by all 18 ESMs', '1 inside, 0 outside')
    if product_area:
        F.add('mask_product_area', ['product', 'lat', 'lon'], '', 'area of each fCO2-product: cells covered by the '
              'product in every year and by all 18 ESMs', 'mean: cells covered by at least one fCO2-product; 1 inside, '
              '0 outside')
    F.add('filled_flag', ['lat', 'lon'], '', 'cells of the mean filled from their neighbours', '1 where no fCO2-product '
          'covers the cell and the mean is filled from the nearest covered cells of the same region, as in the paper '
          '(2% of the ocean); the single fCO2-products are not filled')
    if constrained:
        F.add('constrained_flag', ['lat', 'lon'], '', 'how the cell is constrained', '1 cell-by-cell emergent '
              'constraint; 2 multi-model mean (no local observation-based predictors)')
    if q in T_DEPENDENT:
        levels = ('the preindustrial states and the ESM years' if constrained else
                  'the preindustrial states, the annual values and the forced states' if q == 'thetao' else
                  'the preindustrial states')
        F.add('sigma_flag', ['lat', 'lon'], '', 'cells where the 1 sigma of the levels is left out', '1 where the mapping '
              'error of temperature of one of the fCO2-products exceeds 1 degC in one of its years (cold water; known '
              f'limitation of version 1.0): there the 1 sigma of {levels} is left out, as it rests on this mapping error; '
              'the changes keep their 1 sigma, as the mapping error cancels in them')


def _co2(F, which):
    if 'esms' in which:
        F.add('xco2_esms_annual', ['scenario', 'year'], 'ppm', 'atmospheric CO2 of esms_annual_forced',
              'CMIP6 historical concentrations (Meinshausen et al. 2017) to 2014, ScenarioMIP afterwards')
    if 'obs' in which:
        F.add('xco2_obs_annual', ['year'], 'ppm', 'atmospheric CO2 of the fCO2-products',
              'NOAA global annual mean (Lan et al.)')
    F.add('xco2_preindustrial', ['state'], 'ppm', 'atmospheric CO2 of the preindustrial states',
          'IPCC AR6: 278.3 ppm in 1750 and 285.5 ppm in 1850')


def files():
    out = []
    base = {'lat': 180, 'lon': 360, 'state': 2, 'product': len(PRODUCTS), 'scenario': len(SCENARIOS)}
    for q in QUANTITIES:
        n_area = 2 + (q in WHOLE_OCEAN)
        F = File(q, f'OA-Track_{q}_global_mean.nc', f'{LONG[q]}: global means 1750-2100',
                 dict(base, area=n_area, year=YEARS_GLOBAL[1] - YEARS_GLOBAL[0] + 1))
        _coords(F, q, ['area', 'product', 'scenario', 'state', 'year'], YEARS_GLOBAL)
        _block(F, q, 'preindustrial', True, True)
        _block(F, q, 'obs_annual', True, True)
        _block(F, q, 'obs_annual_forced', True, True)
        _change(F, q, 'obs_annual_forced', True, True)
        _block(F, q, 'esms_annual_forced', True, True)
        _change(F, q, 'esms_annual_forced', True, True)
        _co2(F, ['esms', 'obs'])
        out.append(F)
        F = File(q, f'OA-Track_{q}_maps_obs_based_products.nc', f'{LONG[q]}: maps of the preindustrial states and of the '
                 'fCO2-products 1980-2025, each fCO2-product and their mean', dict(base, year=YEARS_OBS[1] - YEARS_OBS[0] + 1))
        _coords(F, q, ['lat', 'lon', 'product', 'state', 'year'], YEARS_OBS)
        _block(F, q, 'preindustrial', False, True)
        _block(F, q, 'obs_annual', False, True)
        _block(F, q, 'obs_annual_forced', False, True)
        _change(F, q, 'obs_annual_forced', False, True)
        _masks(F, q, True, False)
        _co2(F, ['obs'])
        out.append(F)
        F = File(q, f'OA-Track_{q}_maps_esms.nc', f'{LONG[q]}: maps of the ESMs 1750-2100 (historical and four '
                 'scenarios), mean of the fCO2-products', dict(base, year=YEARS_GLOBAL[1] - YEARS_GLOBAL[0] + 1))
        _coords(F, q, ['lat', 'lon', 'product', 'scenario', 'state', 'year'], YEARS_GLOBAL)
        _block(F, q, 'preindustrial', False, True)
        _block(F, q, 'esms_annual_forced', False, False)
        _change(F, q, 'esms_annual_forced', False, False)
        _masks(F, q, False, True)
        _co2(F, ['esms'])
        out.append(F)
    return out


def csv_columns(F):
    """the columns of the CSV of the global-mean file, in long format: one row per value"""
    return ['variable', 'area', 'product', 'scenario', 'state', 'year', 'value', 'sigma', 'sigma_upper_limit'] + (
        ['alkalinity_adjustment'] if F.q in ALK else [])
