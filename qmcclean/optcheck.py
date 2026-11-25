"""Opt check: flag an underconverged Jastrow. Advisory only; never blocks DMC cleanup."""
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from qharv.inspect import axes_pos
from qharv.reel import scalar_dat
from qharv.sieve import mean_df

from . import jastrow
from .config import OPT_JK_REL, OPT_K2U_DROP, OPT_LAST, OPT_NSIGMA, OPT_RPA_RATIO, SUBDIR
from .layout import discover_opt
from .record import now_et


@dataclass
class OptReport:
    flagged: bool
    reasons: list = field(default_factory=list)
    last: dict = field(default_factory=dict)


def fmt(x, err):
    """Value with one-significant-digit uncertainty in parentheses: -85.14(2) style.

    Non-finite or non-positive error: no uncertainty can be quoted, so only the value (3 sig. digits).
    """
    if not (np.isfinite(x) and np.isfinite(err)) or err <= 0:
        return '%.3g' % x
    d = -int(math.floor(math.log10(err)))  # decimals of the error's leading digit
    if round(err, d) >= 10 ** (1 - d):  # rounding carried to the next decade (9.6 -> 10)
        d -= 1
    e = round(err, d)
    if d <= 0:
        return '%d(%d)' % (round(x, d), e)
    return '%.*f(%d)' % (d, x, round(e * 10 ** d))




TABLE = 'opt.params.dat'
TABLE_INTS = ('series',)
TABLE_SCALARS = ('LocalEnergy', 'Variance')


def opt_params(doc):
    """{id_i: value} of the optimised kSpace Jastrow coefficients (jk1, jk2, ...): every
    <coefficients> under a <jastrow type="kSpace"> with optimize="yes" (QMCPACK's default when
    the attribute is absent), in document order."""
    out = {}
    for j in doc.iter('jastrow'):
        if (j.get('type') or '').lower() != 'kspace':
            continue
        for c in j.iter('coefficients'):
            if (c.get('optimize') or 'yes').strip().lower() in ('no', 'false', '0'):
                continue
            cid = c.get('id')
            for i, x in enumerate((c.text or '').split()):
                out['%s_%d' % (cid, i)] = float(x)
    return out


def trace(L, nequil=8):
    """(DataFrame, notes, table): per step series, E, E_err, V, V_err, jk2_* (as explore
    opt_trace.py), and the rows of opt.params.dat: series, LocalEnergy_mean/_error,
    Variance_mean/_error, then the optimised kSpace Jastrow parameters ({id}_{i})."""
    from qharv.seed import xml
    from .stats import _mean_error
    rows, notes, table = [], [], []
    for s in L.steps:
        sc = L.files.get((s, '.scalar.dat'))
        xp = L.files.get((s, '.opt.xml'))
        row = {'series': s}
        trow = {'series': s}
        if sc is not None:
            df = scalar_dat.read(str(sc))
            if 'Variance' not in df.columns and {'LocalEnergy', 'LocalEnergy_sq'} <= set(df.columns):
                df = df.assign(Variance=df['LocalEnergy_sq'] - df['LocalEnergy'] ** 2)
            me = _mean_error(df[[c for c in TABLE_SCALARS if c in df.columns]], nequil)
            trow.update(me)
            if len(df) - nequil > 1:
                row.update(E=me.get('LocalEnergy_mean', np.nan), E_err=me.get('LocalEnergy_error', np.nan),
                           V=me.get('Variance_mean', np.nan), V_err=me.get('Variance_error', np.nan))
        if xp is not None:
            try:
                doc = xml.read(str(xp))
                trow.update(opt_params(doc))
            except Exception as ex:
                doc = None
                notes.append('step %d: parameters not read (%s: %s)' % (s, type(ex).__name__, ex))
            try:
                if doc is not None:
                    jk, _ = jastrow.jk2_of(doc)
                    row.update({'jk2_%d' % i: x for i, x in enumerate(jk)})
            except Exception as ex:
                notes.append('step %d: jk2 not read (%s: %s)' % (s, type(ex).__name__, ex))
        rows.append(row)
        table.append(trow)
    return pd.DataFrame(rows), notes, table


def _write_table(path, table):
    """opt.params.dat: one row per step: series, LocalEnergy/Variance mean and error, then the
    parameters in first-seen order; a step lacking a column gets nan. Atomic."""
    from .stats import write_table
    cols = ['series'] + ['%s_%s' % (c, x) for c in TABLE_SCALARS for x in ('mean', 'error')]
    for r in table:
        cols += [c for c in r if c not in cols]
    tmp = path.with_name(path.name + '.tmp%d' % os.getpid())
    write_table(tmp, cols, table, ints=TABLE_INTS)
    os.replace(tmp, path)


def _slope(df, col, n):
    """Change of col per step over the last n steps: (slope, err, nsigma, exceeds OPT_NSIGMA).

    Steps are independent runs, so the mean of the n consecutive differences telescopes to
    (v_last - v_first)/n with error sqrt(e_last^2 + e_first^2)/n. NaNs if not evaluable.
    """
    nan = (np.nan, np.nan, np.nan, False)
    if col not in df or col + '_err' not in df or len(df) < n + 1:
        return nan
    v, e = df[col].to_numpy(float), df[col + '_err'].to_numpy(float)
    m = (v[-1] - v[-n - 1]) / n
    me = np.hypot(e[-1], e[-n - 1]) / n
    if not (np.isfinite(m) and np.isfinite(me)) or me <= 0:
        return nan
    nsig = abs(m) / me
    return m, me, nsig, bool(nsig > OPT_NSIGMA)


def _clean_json(o):
    """Strict JSON: numpy -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): _clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_json(v) for v in o]
    if hasattr(o, 'item'):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


INPUT_XML = ('opt.in.xml', 'opt.xml')  # Nexus-generated, or hand-named input


def input_xml(L):
    """The run's QMCPACK input: the first of INPUT_XML present in the opt folder."""
    for name in INPUT_XML:
        if (L.path / name).is_file():
            return L.path / name
    raise FileNotFoundError('no %s in %s' % (' or '.join(INPUT_XML), L.path))


def _cell(L):
    """(axes, nelec) from the opt input XML (the step XMLs carry no cell)."""
    from qharv.seed import xml
    doc = xml.read(str(input_xml(L)))
    return xml.get_axes(doc), int(np.sum(xml.get_nelec(doc)))


def _atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp%d' % os.getpid())
    with open(tmp, 'w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _rpa_check(L, df, reasons, last):
    from qharv.seed import xml
    axes, nelec = _cell(L)
    rs = axes_pos.rs(axes, nelec)
    kf = jastrow.chc.heg_kfermi(rs)
    xp = L.files[(L.steps[-1], '.opt.xml')]
    _, kc = jastrow.jk2_of(xml.read(str(xp)))
    shells = jastrow.kshells(axes, max(3.0, kc))
    k0 = shells[0]
    uT = jastrow.trial_uk(xp, shells, axes=axes)[0]
    uR = jastrow.rpa_uk([k0], rs, kf)[0]
    ratio = uT / uR  # k^2 cancels
    last.update(k_min=float(k0), k2uT=float(k0 ** 2 * uT), k2uRPA=float(k0 ** 2 * uR),
                rpa_ratio=float(ratio))
    if not np.isfinite(ratio):
        raise ValueError('non-finite ratio')
    if ratio < OPT_RPA_RATIO:
        reasons.append('small-k Jastrow below RPA: k²u_T = %.3g vs RPA %.3g at k = %.3g '
                       '(ratio %.2f < %.2f)' % (k0 ** 2 * uT, k0 ** 2 * uR, k0, ratio, OPT_RPA_RATIO))
    y = shells[shells < kc] ** 2 * jastrow.trial_uk(xp, shells[shells < kc], axes=axes)
    drop = 1 - y[0] / y.max()  # k^2 u should rise toward k -> 0, as RPA does
    last['k2u_drop'] = float(drop)
    if drop > OPT_K2U_DROP:
        reasons.append('k²u_T turns down toward k→0: %.3g at k = %.3g vs peak %.3g at k = %.3g '
                       '(%.0f%% below, limit %.0f%%)' % (y[0], k0, y.max(), shells[int(y.argmax())],
                                                         100 * drop, 100 * OPT_K2U_DROP))


def plot_uk(L, png, kmax=3.0):
    """uk.png for an opt run: k^2 u_T(k) of the initial (input XML) and final step's
    wavefunction on the cell's k-shells, against Gaskell RPA; lower panel u_T/u_RPA with the
    OPT_RPA_RATIO limit. Returns the first-shell ratio of the final step."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from qharv.seed import xml
    C1, C0, INK, MUT, BAD = '#2a78d6', '#9bbfe9', '#222222', '#8a8a85', '#c0392b'
    axes, nelec = _cell(L)
    rs = axes_pos.rs(axes, nelec)
    kf = jastrow.chc.heg_kfermi(rs)
    xp = L.files[(L.steps[-1], '.opt.xml')]
    _, kc = jastrow.jk2_of(xml.read(str(xp)))
    k = jastrow.kshells(axes, kmax)
    uR = jastrow.rpa_uk(k, rs, kf)
    uT = jastrow.trial_uk(xp, k, axes=axes)
    try:
        x0 = input_xml(L)
        u0 = jastrow.trial_uk(x0, k, axes=axes)
    except Exception:  # e.g. initial jk2 longer than the shells below kc: skip it
        u0 = None
    fig, (a, b) = plt.subplots(2, 1, figsize=(5.6, 6.2), sharex=True,
                               gridspec_kw=dict(height_ratios=[3, 1.2], hspace=0.05))
    kk = np.linspace(0.05, kmax, 300)
    a.plot(kk, kk ** 2 * jastrow.rpa_uk(kk, rs, kf), c=MUT, lw=1.5, label='Gaskell RPA')
    if u0 is not None:
        a.plot(k, k ** 2 * u0, ls='', marker='o', ms=5, mfc='none', mew=1.2, c=C0,
               label='initial $u_T$ (%s)' % x0.name)
        b.plot(k, u0 / uR, ls='', marker='o', ms=4, mfc='none', mew=1.1, c=C0)
    a.plot(k, k ** 2 * uT, ls='', marker='x', ms=7, mew=1.6, c=C1,
           label='final $u_T$ (step %d)' % L.steps[-1])
    b.plot(k, uT / uR, ls='', marker='x', ms=6, mew=1.4, c=C1)
    for ax in (a, b):
        ax.axvline(kc, c=MUT, ls=':', lw=1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.25, lw=0.6)
    a.text(kc + 0.03, 0.95, '$k_c$', transform=a.get_xaxis_transform(), color=MUT, va='top')
    ratio = float(uT[0] / uR[0])
    a.set_title('%s opt  (N=%d, %d steps)\nfirst shell $u_T/u_{RPA}$ = %.3f%s'
                % (L.path.resolve().parent.name, nelec, len(L.steps), ratio,
                   '  < %.2f' % OPT_RPA_RATIO if ratio < OPT_RPA_RATIO else ''),
                fontsize=10, color=INK)
    b.axhline(1, c=MUT, lw=0.8)
    b.axhline(OPT_RPA_RATIO, c=BAD, lw=0.9, ls='--')
    b.set_ylabel('$u_T/u_{RPA}$')
    b.set_xlabel('k (bohr$^{-1}$)')
    a.set_ylabel('$k^2 u(k)$')
    a.legend(frameon=False, fontsize=9)
    b.set_xlim(0, kmax)
    fig.savefig(png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return ratio


def plot_ev(L, df, png):
    """ev.png for an opt run: E, V and jk2[0:3] vs series. Series N measures opt.s(N-1).opt.xml
    (series 0 the input), so jk2 of step s is drawn at s+1, the series that measured it. The
    series with the lowest 0.9E+0.1V is marked. Returns that series (None if E/V are missing)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from qharv.seed import xml
    C1, BAD = '#2a78d6', '#c0392b'
    fig, axs = plt.subplots(3, 1, figsize=(5.6, 7.4), sharex=True, gridspec_kw=dict(hspace=0.08))
    best = None
    if {'E', 'V'} <= set(df.columns) and df['E'].notna().any():
        d = df.dropna(subset=['E', 'V'])
        for a, c, lab in ((axs[0], 'E', 'E (Ha)'), (axs[1], 'V', 'V (Ha$^2$)')):
            a.errorbar(d['series'], d[c], d[c + '_err'], c=C1, marker='o', ms=4, capsize=2, lw=1.2)
            a.set_ylabel(lab)
        score = 0.9 * d['E'] + 0.1 * d['V']
        best = int(d['series'].iloc[int(np.argmin(score.to_numpy()))])
    jk = df[['series'] + ['jk2_%d' % i for i in range(3) if 'jk2_%d' % i in df]].copy()
    jk['series'] = jk['series'] + 1
    try:
        j0, _ = jastrow.jk2_of(xml.read(str(input_xml(L))))
        jk = pd.concat([pd.DataFrame([{'series': 0, **{'jk2_%d' % i: x for i, x in enumerate(j0[:3])}}]), jk])
    except Exception:  # input without a kSpace Jastrow: start at series 1
        pass
    # RPA reference: the jk2 that makes u_T = FT[u_sr] - jk2 equal Gaskell RPA, i.e.
    # FT[u_sr](k) - u_RPA(k), with FT[u_sr] of the final step's B-spline J2 (~+5 at p325)
    try:
        axes, nelec = _cell(L)
        rs = axes_pos.rs(axes, nelec)
        xp = L.files[(L.steps[-1], '.opt.xml')]
        doc = xml.read(str(xp))
        k = jastrow.kshells(axes, jastrow.jk2_of(doc)[1])[:3]
        jref = jastrow._usrk_with_axes(k, doc, axes) - jastrow.rpa_uk(k, rs, jastrow.chc.heg_kfermi(rs))
    except Exception:  # no cell or no J2: plot without the reference
        jref = []
    for i in range(3):
        if 'jk2_%d' % i in jk:
            ln, = axs[2].plot(jk['series'], jk['jk2_%d' % i], marker='o', ms=4, lw=1.2, label='jk2[%d]' % i)
            if i < len(jref):
                axs[2].axhline(jref[i], c=ln.get_color(), ls=':', lw=1.2)
    if len(jref):
        axs[2].plot([], [], c='k', ls=':', lw=1.2, label='RPA')
    axs[2].set_ylabel('jk2 (measured wf)')
    axs[2].legend(frameon=False, fontsize=8, ncol=3)
    axs[2].set_xlabel('series')
    for a in axs:
        if best is not None:
            a.axvline(best, c=BAD, ls='--', lw=1)
        a.spines[['top', 'right']].set_visible(False)
        a.grid(alpha=0.25, lw=0.6)
    p = L.path.resolve()
    axs[0].set_title('%s/%s/%s  (%d steps)%s' % (p.parent.parent.name, p.parent.name, p.name, len(L.steps),
                     '' if best is None else '\nbest 0.9E+0.1V: series %d (dashed)' % best), fontsize=10)
    axs[2].xaxis.get_major_locator().set_params(integer=True)
    fig.savefig(png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return best


def check_opt(opt_dir, nequil=8):
    opt_dir = Path(opt_dir)
    L = discover_opt(opt_dir)
    df, notes, table = trace(L, nequil)
    reasons, last = [], {}
    n = len(df)
    last['nsteps'] = n
    if n < 2:
        reasons.append('too few optimisation steps to judge convergence (%d)' % n)
    else:
        nn = min(OPT_LAST, n - 1)
        for col, name in (('V', 'variance'), ('E', 'energy')):
            m, me, nsig, exceeds = _slope(df, col, nn)
            last[col + '_slope'], last[col + '_slope_err'], last[col + '_nsigma'] = m, me, nsig
            if not np.isfinite(m):
                reasons.append('could not evaluate %s trend (missing or non-finite data; '
                               'is nequil=%d too large?)' % (name, nequil))
            elif exceeds:
                reasons.append('%s still %s: %s per step, %.0fσ'
                               % (name, 'falling' if m < 0 else 'rising', fmt(m, me), nsig))
        if 'jk2_0' not in df or df['jk2_0'].isna().iloc[-2:].any():
            notes.append('jk2 not available in the last two steps: drift check skipped')
        else:
            j0 = df['jk2_0'].to_numpy(float)
            step = j0[-1] - j0[-2]
            last['jk2_0'], last['jk2_0_step'] = j0[-1], step
            if j0[-2] == 0 or not np.isfinite(step):
                reasons.append('could not evaluate jk2[0] drift (previous value %g)' % j0[-2])
            else:
                rel = step / abs(j0[-2])
                last['jk2_0_rel'] = rel
                if abs(rel) > OPT_JK_REL:
                    reasons.append('jk2[0] still moving: %+.3g per step (%.0f%% of its value, limit %.0f%%)'
                                   % (step, 100 * abs(rel), 100 * OPT_JK_REL))
        try:
            _rpa_check(L, df, reasons, last)
        except Exception as ex:  # a check that could not run is not a pass
            reasons.append('could not evaluate RPA check: %s: %s' % (type(ex).__name__, ex))
    if notes:
        last['notes'] = notes
    rep = OptReport(flagged=bool(reasons), reasons=reasons, last=last)
    out = opt_dir / SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    _write_table(out / TABLE, table)
    if n >= 1:
        try:
            plot_uk(L, out / 'uk.png')
        except Exception as ex:  # the figure is a convenience: never fails the check
            notes.append('uk.png not written: %s: %s' % (type(ex).__name__, ex))
            last['notes'] = notes
        try:
            plot_ev(L, df, out / 'ev.png')
        except Exception as ex:  # the figure is a convenience: never fails the check
            notes.append('ev.png not written: %s: %s' % (type(ex).__name__, ex))
            last['notes'] = notes
    for old in ('opt_trace.csv', 'opt.scalar.dat'):  # earlier qmcclean names of this table
        if (out / old).exists():
            (out / old).unlink()
    rec = {'opt_dir': str(opt_dir), 'time': now_et(), 'nequil': nequil,
           'flagged': rep.flagged, 'reasons': reasons, 'last': last}
    _atomic_write(out / 'opt_record.json', json.dumps(_clean_json(rec), indent=1, allow_nan=False))
    return rep
