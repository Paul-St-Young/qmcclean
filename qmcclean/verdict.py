"""Outlier flags per twist and the run verdict (clean / drop_listed / hold)."""
import re
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .config import DROP_FRACTION, ERR_RATIO, FORCE_RATIO_REL, ROBUST_Z
from .stats import load_scalar, read_meta, scalar_name

FORCE_ERR = re.compile(r'^force_\d+_\d+_error$')
RATIO_NOTE_RANGE = (1.5, 3.0)  # healthy run-median VMC/DMC force-error ratio


@dataclass
class Verdict:
    kind: Literal['clean', 'drop_listed', 'hold']
    flagged: dict
    reasons: list = field(default_factory=list)
    ntwist: int = 0


class Flags(dict):
    """twist -> [reasons]; .notes holds run-level remarks that are not flags."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.notes = []


def _robust_z(x, floor):
    """|x - median| / max(1.4826 MAD, floor), NaN ignored in median/MAD; inf if scale is 0."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if not ok.any():
        return np.full(x.shape, np.nan)
    med = np.median(x[ok])
    mad = 1.4826 * np.median(np.abs(x[ok] - med))
    den = max(mad, float(floor) if np.isfinite(floor) else 0.0)
    dev = np.abs(x - med)
    if den <= 1e-9 * abs(med) or den <= 0:  # degenerate scale: any real deviation is an outlier
        return np.where(dev > 1e-9 * max(abs(med), 1e-300), np.inf, 0.0)
    return dev / den


def _med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def _force_ratio(v, d):
    """Per-twist median over force columns of VMC/DMC error; NaN where unavailable.

    A column with VMC error > 0 but DMC error exactly 0 (a frozen DMC force estimator) makes
    the twist's ratio +inf, so it is always flagged; only 0/0 and NaN become NaN."""
    cols = [c for c in v.columns if FORCE_ERR.match(c) and c in d.columns]
    if not cols:
        return None
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.column_stack([v[c].to_numpy(float) / d[c].to_numpy(float) for c in cols])
    frozen = np.isposinf(r).any(axis=1)
    r[~np.isfinite(r)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN rows (short twists)
        med = np.nanmedian(r, axis=1)
    med[frozen] = np.inf
    return med


def flag_twists(stats_path, R):
    """twist -> [reasons] for every outlier twist (against the run's other twists) and every
    structurally flagged twist. Raw twist energies are never tested."""
    out = Flags()

    def add(t, msg):
        out.setdefault(int(t), []).append(msg)

    for src in (R.short, R.missing_files):
        for t, msgs in src.items():
            for m in msgs:
                add(t, m)
    structural = set(out)

    meta = read_meta(stats_path)
    has0 = scalar_name(0) in meta['files']
    d = load_scalar(stats_path, 1)
    # per-row nequil (a hand-replaced row may differ); stats.json value as fallback
    nequil = d['nequil'].to_numpy(int) if 'nequil' in d else int(meta['nequil'].get('1', 0))
    v = load_scalar(stats_path, 0) if has0 else None
    tw = d['twist'].to_numpy()

    def test(name, x, floor, label=None):
        z = _robust_z(x, floor)
        for t, zi, xi in zip(tw, z, x):
            if zi > ROBUST_Z:
                add(t, '%s z=%.0f' % (label or name, zi))

    def ratio_test(x, label):
        # error bars are heavy-tailed across twists: compare to run median (R10), not robust z
        med = _med(x)
        if not med > 0:  # all NaN, or a zero/negative median: the test cannot run
            out.notes.append('%s ratio test skipped: run median %s' % (label, med))
            return
        for t, xi in zip(tw, x):
            if np.isfinite(xi) and not (1 / ERR_RATIO <= xi / med <= ERR_RATIO):
                add(t, '%s ratio %.2g (run median %.2g)' % (label, xi / med, med))

    if 'Variance_mean' in d:
        test('Variance_mean', d['Variance_mean'].to_numpy(float),
             _med(d['Variance_error']) if 'Variance_error' in d else 0.0, 'Variance_mean s001')
    if 'LocalEnergy_error' in d:
        ratio_test(d['LocalEnergy_error'].to_numpy(float), 'LocalEnergy_error s001')
    if v is not None and 'LocalEnergy_mean' in d:
        m = d[['twist', 'LocalEnergy_mean', 'LocalEnergy_error']].merge(
            v[['twist', 'LocalEnergy_mean', 'LocalEnergy_error']], on='twist',
            how='left', suffixes=('_d', '_v'))
        diff = (m['LocalEnergy_mean_d'] - m['LocalEnergy_mean_v']).to_numpy(float)
        err = np.sqrt(m['LocalEnergy_error_d'] ** 2 + m['LocalEnergy_error_v'] ** 2).to_numpy(float)
        test('DMC-VMC', diff, _med(err), 'DMC-VMC LocalEnergy_mean')
    if 'dmc_NumOfWalkers_mean' in d:
        test('dmc_NumOfWalkers_mean', d['dmc_NumOfWalkers_mean'].to_numpy(float),
             _med(d['dmc_NumOfWalkers_error']) if 'dmc_NumOfWalkers_error' in d else 0.0)
    if 'BlockWeight_error' in d:
        ratio_test(d['BlockWeight_error'].to_numpy(float), 'BlockWeight_error s001')

    # unexplained NaN where the twist has enough blocks for a mean (short twists are structural)
    for name in ('LocalEnergy_mean', 'Variance_mean'):
        if name in d:
            bad = ~np.isfinite(d[name].to_numpy(float)) & (d['nblock'].to_numpy() > nequil + 1)
            for t in tw[bad]:
                if int(t) not in structural:
                    add(t, '%s s001 not finite' % name)

    if v is not None:
        ratio = _force_ratio(v.set_index('twist').reindex(tw).reset_index(),
                             d)
        if ratio is not None and np.isfinite(ratio).any():
            rmed = _med(ratio)
            lo, hi = rmed * (1 - FORCE_RATIO_REL), rmed * (1 + FORCE_RATIO_REL)
            for t, r in zip(tw, ratio):
                if not np.isnan(r) and not lo <= r <= hi:  # +inf: frozen DMC force
                    add(t, 'force err ratio %.2g (run median %.2g)' % (r, rmed))
            if not RATIO_NOTE_RANGE[0] <= rmed <= RATIO_NOTE_RANGE[1]:
                out.notes.append('run median force err ratio %.2g outside [%.1f, %.1f]' % (
                    rmed, *RATIO_NOTE_RANGE))
    return out


def decide(L, R, flagged):
    """Run verdict. Critical failure or flagged fraction >= DROP_FRACTION -> hold."""
    n = len(L.twists)
    k = len(flagged)
    reasons = []
    if R.critical:
        reasons.append('critical: missing estimator(s) %s' % ', '.join(R.missing_estimators))
    if n == 0:
        reasons.append('no twists found')
    if n == 0 or k >= DROP_FRACTION * n - 1e-9 and k > 0:
        reasons.append('%d/%d twists flagged (%.1f%%) >= %g%%' % (
            k, n, 100.0 * k / max(n, 1), 100 * DROP_FRACTION))
    if R.critical or reasons:
        kind = 'hold'
    elif k == 0:
        kind = 'clean'
    else:
        kind = 'drop_listed'
        reasons.append('%d/%d twists flagged (< %g%%): flagged twists kept' % (
            k, n, 100 * DROP_FRACTION))
    reasons.extend(getattr(flagged, 'notes', []))
    return Verdict(kind, dict(flagged), reasons, n)
