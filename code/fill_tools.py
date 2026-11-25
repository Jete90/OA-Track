"""
fill_tools.py
=============
Filling of ocean cells without a value from the cells around them, used for the extension of the decline to the whole
ocean (structural_adjustments.py, test_polar_shift_models.py), so that the fCO2-products and the ESMs are filled
with the same rule.

fill_neighbours: the weighted mean over the 5 x 5 cells around a cell, the 8 direct neighbours with twice the weight
of the 16 cells of the outer ring, in passes from the edges of a gap inward.
regional_fill: cells no pass can reach (an isolated basin): the area mean of the cells of the same region (Arctic
north of 60 N, Southern Ocean south of 55 S, other).
fill_to: both in one call, for a list of fields on one grid.
"""
import numpy as np

FILL_RINGS = {1: 2.0, 2: 1.0}      # Chebyshev distance of a donor cell -> weight (3 x 3 twice the 5 x 5 ring)


def _shift(a, dj, di):
    """a moved so that out[j, i] = a[j + dj, i + di]; longitudes wrap, latitudes beyond the grid give 0 (False)"""
    r = np.roll(a, -di, axis=1)
    if dj == 0:
        return r
    out = np.zeros_like(r)
    if dj > 0:
        out[:-dj] = r[dj:]
    else:
        out[-dj:] = r[:dj]
    return out


def fill_neighbours(fields, have, need):
    """fills the cells `need` of the 2-D fields from the cells `have` around them: the weighted mean over the 5 x 5
    cells around a cell, the 8 direct neighbours with twice the weight of the 16 cells of the outer ring (FILL_RINGS);
    longitudes wrap, latitudes do not. A cell without a donor within 5 x 5 is filled in a later pass from the cells
    filled before, so that a gap is closed from its edges inward. Returns the filled fields and the pass in which
    each cell was filled (0: donor or not needed, -1: no donor reachable, e.g. an isolated basin)"""
    fields = [np.array(f, 'f8') for f in fields]
    known, todo = have.copy(), need & ~have
    step = np.zeros(have.shape, int)
    k = 0
    while todo.any():
        k += 1
        den = np.zeros(have.shape)
        num = [np.zeros(have.shape) for _ in fields]
        for dj in range(-2, 3):
            for di in range(-2, 3):
                d = max(abs(dj), abs(di))
                if d == 0:
                    continue
                kn = _shift(known, dj, di)
                den += FILL_RINGS[d] * kn
                for n_, f in zip(num, fields):
                    n_ += FILL_RINGS[d] * _shift(np.where(known, f, 0.0), dj, di)
        new = todo & (den > 0)
        if not new.any():
            break
        for n_, f in zip(num, fields):
            f[new] = n_[new] / den[new]
        known |= new
        todo &= ~new
        step[new] = k
    step[todo] = -1
    return fields, step


def regional_fill(fields, have, need, region, w):
    """for the cells fill_neighbours cannot reach (and for comparison): the area mean of the cells
    `have` of the same region (Arctic north of 60 N, Southern south of 55 S, other)"""
    fields = [np.array(f, 'f8') for f in fields]
    for g in ('Arctic', 'Southern', 'other'):
        h, m = have & (region == g), need & ~have & (region == g)
        if h.any() and m.any():
            for f in fields:
                f[m] = np.sum(f[h] * w[h]) / w[h].sum()
    return fields


def regions(lat, shape):
    """the region of every cell for regional_fill: Arctic north of 60 N, Southern south of 55 S, other"""
    la2 = np.broadcast_to(np.asarray(lat, 'f8')[:, None], shape)
    return np.where(la2 > 60, 'Arctic', np.where(la2 < -55, 'Southern', 'other'))


def fill_to(fields, have, need, region, w):
    """the fields filled in `need` from `have`: fill_neighbours, then regional_fill for the cells it cannot reach.
    Returns the filled fields and the share of `need` (area-weighted) that regional_fill had to fill"""
    fields, step = fill_neighbours(fields, have, need)
    rest = (step == -1) & need
    if rest.any():
        fields = regional_fill(fields, need & ~rest & np.isfinite(fields[0]), rest, region, w)
    return fields, float(w[rest].sum() / max(w[need].sum(), 1e-30))
