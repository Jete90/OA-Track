# Notebook cell (after the constraint cell): all products on one common ocean area.
# The common mask is the intersection of the constraint masks of all products that are done,
# on the OceanSODA grid; run_constraint maps it onto each product grid.
done = [r for r in ('OceanSODA', 'UExP', 'JMA', 'CMEMS') if r in results]
lat_c, lon_c = results['OceanSODA']['lat'], results['OceanSODA']['lon']
cm = results['OceanSODA']['mask'].copy()
for r in done[1:]:
    cm &= results[r]['mask'][np.ix_(_nn(results[r]['lat'], lat_c), _nn(results[r]['lon'], lon_c, True))]
area = oa_config.area_weights(lat_c, lon_c)          # the weights of the run (run_constraint.py imports oa_config)
print(f'common mask of {", ".join(done)}: '
      f'{100 * (area * cm).sum() / (area * results["OceanSODA"]["mask"]).sum():.1f} % of the OceanSODA constraint mask')

results_common = run_constraint(common=(lat_c, lon_c, cm))

rows = [('dSST (K)', lambda r: f"{r['dT_obs'][0]:.3f} ± {r['dT_obs'][1]:.3f}"),
        ('CUC obs', lambda r: f"{r['CUC_obs']:.4f}"),
        ('1750 unconstrained', lambda r: f"{r['unconstrained'][0]:.3f} ± {r['unconstrained'][1]:.3f}"),
        ('1750 constrained', lambda r: f"{r['constrained']['dT + CUC'][0]:.3f} ± {r['constrained']['dT + CUC'][1]:.3f}"),
        ('boundary', lambda r: f"{0.8 * r['constrained']['dT + CUC'][0]:.3f}"),
        ('boundary 284-275 ppm', lambda r: '{:.3f}-{:.3f}'.format(*(0.8 * np.array(r['constrained']['dT + CUC'][2])))),
        ('fit 2024', lambda r: f"{r['now']:.3f}"),
        ('change vs 1750 (%)', lambda r: f"{100 * (r['now'] / r['constrained']['dT + CUC'][0] - 1):+.1f}"),
        ('distance', lambda r: f"{r['now'] - 0.8 * r['constrained']['dT + CUC'][0]:+.3f}")]
print(f'\n{"common mask":22s}' + ''.join(f'{r:>16s}' for r in results_common))
for lab, fn in rows:
    print(f'{lab:22s}' + ''.join(f'{fn(results_common[r]):>16s}' for r in results_common))
d = np.array([results_common[r]['now'] - 0.8 * results_common[r]['constrained']['dT + CUC'][0] for r in results_common])
print(f'\ndistance across products: mean {d.mean():+.3f}, spread (std) {d.std(ddof=1):.3f}')
