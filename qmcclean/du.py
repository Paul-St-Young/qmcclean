"""Post-DMC validation of the Jastrow: du(k) from twist-averaged S(k) in dmc.s{SSS}.stat.h5.

Reads only <dmc>/qmcclean/{stats.json,dmc.s000.stat.h5,dmc.s001.stat.h5,record.json} and the
kept dmc.g000.twistnum_0.in.xml, so it works after cleanup. Writes uk.png, du.csv, du.json
next to them.
"""
import json
from math import floor, log10
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from . import record  # noqa: E402
from .config import DU_REL, KEEP, SUBDIR  # noqa: E402
from .jastrow import chc, delta_u, trial_uk, xml  # noqa: E402
from .stats import META, load_scalar, read_meta, stat_name  # noqa: E402
from qharv.inspect import axes_pos  # noqa: E402

KMAX = 3.0
C1, C2, C3, INK, MUT = '#2a78d6', '#eb6834', '#2e9e5b', '#222222', '#8a8a85'


def fmt_err(x, e):
    """x(e) with the error to one significant digit, value rounded to it."""
    if not np.isfinite(e) or e <= 0:
        return '%.3g' % x
    d = -int(floor(log10(e)))
    e1 = round(e, d)
    if e1 >= 10 ** (-d + 1):  # rounding carried into a new digit
        d -= 1
        e1 = round(e, d)
    digits = max(d, 0)
    return '%.*f(%d)' % (digits, x, int(round(e1 * 10 ** digits)) if d >= 0 else int(round(e1)))


def twist_average(stats, series, exclude, nelec):
    """(kvecs, S mean, S err, ntwist_used): equal-weight mean of dsk/N over usable twists.

    stats: path of stats.json; dsk is read from the dmc.s{SSS}.stat.h5 it lists. Only twists
    with a row in that series' scalar table (and not in exclude) are used, so deleting a row
    by hand also removes the twist from S(k)."""
    ms, es, kv = [], [], None
    stats = Path(stats)
    name = stat_name(series)
    if name not in read_meta(stats)['files']:
        raise RuntimeError('%s not listed in %s' % (name, stats))
    h5 = stats.parent / name
    rows = {int(t) for t in load_scalar(stats, series)['twist']}
    with h5py.File(h5, 'r') as f:
        if 'kvecs' in f:
            kv = f['kvecs'][()]  # shared skall k-vectors (one copy per series)
        groups = {n: g['dsk'] for n, g in f.items() if isinstance(g, h5py.Group) and 'dsk' in g}
        if kv is None and any('kvecs' in g for g in groups.values()):
            raise RuntimeError('%s: old stat.h5 layout (per-twist kvecs); recompute stats from '
                               'raw files' % h5)
        if not groups or kv is None:
            raise RuntimeError(
                'no S(k) (dsk) in %s: the run has no skall estimator output '
                '(batched-driver runs have none), so du(k) cannot be computed' % h5)
        for name, g in sorted(groups.items()):
            if int(name[1:]) in exclude or int(name[1:]) not in rows:
                continue
            m = g['mean'][()]
            if not np.all(np.isfinite(m)) or not np.all(np.isfinite(g['error'][()])):
                continue
            if m.shape[0] != len(kv):
                raise RuntimeError('twist %s: dsk has %d values but %d kvecs' % (
                    name, m.shape[0], len(kv)))
            ms.append(m / nelec)
            es.append(g['error'][()] / nelec)
    if not ms:
        raise RuntimeError('no usable twist (all flagged or NaN) in series %d of %s' % (series, stats))
    n = len(ms)
    return kv, np.mean(ms, 0), np.sqrt(np.sum(np.square(es), 0)) / n, n


def _flagged_twists(dmc):
    rec = record.load(dmc)
    return {int(t) for t in ((rec.get('verdict') or {}).get('flagged') or {})}


def run_du(dmc_dir):
    """Write uk.png, du.csv, du.json under <dmc>/qmcclean/; return the png path.

    du.csv columns: k, uT, du, du_err, extrap = uT + 2 du, extrap_err = 2 du_err.
    """
    dmc = Path(dmc_dir)
    out = dmc / SUBDIR
    stats = out / META
    nelec = int(read_meta(stats)['nelec'])
    excl = _flagged_twists(dmc)
    kv0, sv, sve, n0 = twist_average(stats, 0, excl, nelec)
    kv1, sd, sde, n1 = twist_average(stats, 1, excl, nelec)
    if kv0.shape != kv1.shape or not np.allclose(kv0, kv1):
        raise RuntimeError('VMC and DMC skall k-vectors differ in %s' % out)
    ntw = min(n0, n1)
    knorm = np.linalg.norm(kv0, axis=1)
    sel = (knorm > 0) & (knorm < KMAX)
    xp = dmc / KEEP[0]
    # delta_u: S is N*S(k) (as stored); rho from the real cell
    doc = xml.read(str(xp))
    axes = xml.get_axes(doc)
    vol = abs(np.linalg.det(np.asarray(axes, float)))
    k, du, due = delta_u(kv0[sel], nelec * sv[sel], nelec * sd[sel], nelec * sve[sel],
                         nelec * sde[sel], nelec, vol)
    rs = axes_pos.rs(axes, nelec)
    kf = chc.heg_kfermi(rs)
    uT = trial_uk(xp, k, axes)
    kc = float(doc.find('.//jastrow[@type="kSpace"]/correlation').get('kc'))

    with np.errstate(divide='ignore', invalid='ignore'):
        rel = float(du[0] / uT[0])
    finite = bool(np.isfinite(rel))
    flagged = bool(abs(rel) > DU_REL) if finite else True  # cannot be judged: flag it
    note = 'twists used: %d (S(k) equal-weight twist average; %d twists flagged and excluded)' % (
        ntw, len(excl))
    if not finite:
        note += '; first-shell du/u_T is not finite (%r): flagged' % rel
    with open(out / 'du.csv', 'w') as fh:
        fh.write('# %s\n' % note)
        fh.write('k,uT,du,du_err,extrap,extrap_err\n')
        for row in zip(k, uT, du, due, uT + 2 * du, 2 * due):
            fh.write(','.join(repr(float(x)) for x in row) + '\n')
    js = {'du_rel_first_shell': rel if finite else None, 'flagged': flagged, 'ntwist_used': ntw,
          'k_first_shell': float(k[0]), 'time': record.now_et()}
    if not finite:
        js['note'] = 'du_rel_first_shell not finite (%r): flagged' % rel
    (out / 'du.json').write_text(json.dumps(js, indent=1) + '\n')

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = float(abs(due[0] / uT[0]))
    stale = out / 'du.png'  # the figure's name before the uk.png rename
    if stale.exists():
        stale.unlink()
    _plot(out / 'uk.png', dmc, k, uT, du, due, rs, kf, kc, nelec, ntw, rel, rel_err)
    print('du(k): %d twists used; first-shell du/u_T = %s%s' % (
        ntw, fmt_err(rel, rel_err), ('  FLAG: exceeds %g' % DU_REL if finite else
                                     '  FLAG: not finite') if flagged else ''))
    return out / 'uk.png'


def _plot(png, dmc, k, uT, du, due, rs, kf, kc, nelec, ntw, rel, rel_err):
    fig, (a, b) = plt.subplots(2, 1, figsize=(5.6, 6.2), sharex=True,
                               gridspec_kw=dict(height_ratios=[3, 1.2], hspace=0.05))
    kk = np.linspace(0.05, KMAX, 300)
    a.plot(kk, kk ** 2 * chc.gaskell_rpa_uk(kk, rs, kf), c=MUT, lw=1.5, label='Gaskell RPA')
    a.plot(k, k ** 2 * uT, ls='', marker='x', ms=7, mew=1.6, c=C1, label='VMC trial $u_T$')
    a.errorbar(k, k ** 2 * (uT + du), k ** 2 * due, ls='', marker='o', ms=5, mfc='none',
               mew=1.4, c=C2, label=r'DMC $u_T+\delta u$')
    a.errorbar(k, k ** 2 * (uT + 2 * du), k ** 2 * 2 * due, ls='', marker='s', ms=4.5,
               mfc='none', mew=1.3, c=C3, label=r'extrap $u_T+2\delta u$')
    for ax in (a, b):
        ax.axvline(kc, c=MUT, ls=':', lw=1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.25, lw=0.6)
    a.text(kc + 0.03, 0.95, '$k_c$', transform=a.get_xaxis_transform(), color=MUT, va='top')
    a.set_title('%s  (twist-averaged S(k), N=%d, %d twists)\n'
                r'first shell $\delta u/u_T$ = %s' % (dmc.resolve().parent.name, nelec, ntw,
                                                     fmt_err(rel, rel_err)),
                fontsize=10, color=INK)
    b.axhline(0, c=MUT, lw=0.8)
    b.errorbar(k, k ** 2 * du, k ** 2 * due, ls='', marker='o', ms=4, mfc='none', mew=1.2, c=C2)
    b.set_xlabel('k (bohr$^{-1}$)')
    a.set_ylabel('$k^2 u(k)$')
    b.set_ylabel(r'$k^2\,\delta u(k)$')
    a.legend(frameon=False, fontsize=9)
    b.set_xlim(0, KMAX)
    fig.savefig(png, dpi=130, bbox_inches='tight')
    plt.close(fig)
