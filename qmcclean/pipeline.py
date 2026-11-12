"""One dmc folder, start to finish: structure -> stats -> verdict -> record -> cleanup."""
from dataclasses import asdict
from pathlib import Path

from . import record
from .cleanup import _g2t, _reconcile, execute, plan_cleanup
from .config import NEQUIL, SUBDIR
from .layout import discover_dmc
from .stats import META, DataLossRefused, compute_stats, old_layout, read_meta, verify_stats
from .structure import check_structure
from .verdict import decide, flag_twists

RESUMED = ('cleaned', 'partially_cleaned')


def _nequil(nequil):
    if nequil is None:
        return dict(NEQUIL)
    if isinstance(nequil, int):
        return {k: nequil for k in NEQUIL}
    return {**NEQUIL, **nequil}


def _key(nequil):
    return {str(k): int(v) for k, v in nequil.items()}


def _stored_nequil(path):
    """nequil saved in stats.json (str-keyed dict), or None if unreadable."""
    try:
        nq = read_meta(path)['nequil']
        return nq if isinstance(nq, dict) else None
    except Exception:
        return None


def run_dmc(dmc_dir, skip_opt=False, clean=False, nequil=None, executor=None):
    """Process one dmc folder and return its record (also saved in <dmc>/qmcclean/).

    clean=False is a dry run: the record is updated, no run file is touched.
    A run already (partially) cleaned keeps its stored verdict; only newly released twists
    are planned. Any unexpected exception ends in state 'error' (raw files untouched
    unless the failure came after deletion began, which the history shows).
    executor: optional concurrent.futures executor for the per-twist raw-file reads (structure
    line counts, statistics, verify line counts); None = serial.
    Holds <dmc>/qmcclean/lock throughout; if another qmcclean process holds it, nothing is
    changed and the returned (unsaved) record says 'locked by another qmcclean process'.
    """
    dmc_dir = Path(dmc_dir)
    try:
        with record.lock(dmc_dir):
            return _run_dmc(dmc_dir, skip_opt, clean, nequil, executor)
    except record.Locked as e:
        rec = record.load(dmc_dir)
        rec['state'] = 'error'
        record.log(rec, 'error: %s; nothing changed' % e)
        return rec  # not saved: the other process owns the record


def _run_dmc(dmc_dir, skip_opt, clean, nequil, executor=None):
    rec = record.load(dmc_dir)
    prev_state = rec['state']
    resumed = bool(rec['verdict']) and (rec['state'] in RESUMED or bool(
        rec['deleted'] or rec['archives'] or rec['pending']))
    try:
        L = discover_dmc(dmc_dir)
        rec['prefix'] = L.prefix
        if resumed:
            _reconcile(rec, dmc_dir.resolve())
            if rec['state'] not in RESUMED:  # deletion began earlier: say so
                rec['state'] = 'partially_cleaned'
            if nequil is not None and rec.get('nequil') and _key(_nequil(nequil)) != rec['nequil']:
                record.log(rec, 'nequil %s ignored: raw files already deleted, stats keep %s' % (
                    _key(_nequil(nequil)), rec['nequil']))
        if not resumed:
            rec.update(verdict=None, structure=None, overrides=[])
            # remembered (merged with any earlier one) so a rerun, and excluded_twists for a
            # stopped or failed run, still know twists whose in.xml is gone
            old = rec.get('layout') or {}
            rec['layout'] = {
                'twists': sorted({int(t) for t in L.twists} | {int(t) for t in old.get('twists', [])}),
                'g2t': {**{str(g): int(t) for g, t in old.get('g2t', {}).items()},
                        **{str(g): t for g, t in _g2t(dmc_dir).items()}}}
            R = check_structure(L, skip_opt=skip_opt, executor=executor)
            rec['structure'] = asdict(R)
            rec['state'] = 'checked'
            if R.overridden:
                rec['overrides'].append('skip_opt: missing %s' % ', '.join(R.missing_estimators))
            if R.critical:
                rec['state'] = 'stopped_critical'
                record.log(rec, 'critical: missing estimator(s) %s; stopped, nothing deleted' %
                           ', '.join(R.missing_estimators))
                record.save(dmc_dir, rec)
                return rec
            record.log(rec, 'structure checked')
            want = _nequil(nequil)
            final = dmc_dir / SUBDIR / META  # a legacy stats.h5 is ignored
            why = None
            existed = final.exists()
            if existed:
                stored = _stored_nequil(final)
                if prev_state == 'error':
                    why = 'last state was error'
                elif stored != _key(want):
                    why = 'nequil %s -> %s' % (stored, _key(want))
                elif old_layout(final):  # pre-kvecs stat.h5: stale format, recompute now
                    why = 'old stat.h5 layout'
            try:
                stats = compute_stats(L, R, want, overwrite=why is not None, executor=executor)
            except DataLossRefused as e:  # _refuse_data_loss: existing stats protect gone raw files
                rec['state'] = 'error'
                record.log(rec, 'stats not recomputed (%s), nothing deleted: %s' % (why, e))
                record.save(dmc_dir, rec)
                return rec
            if why:
                record.log(rec, 'stats recomputed (%s)' % why)
            elif existed:
                record.log(rec, 'existing stats reused (nequil %s)' % _key(want))
            else:
                record.log(rec, 'stats computed (nequil %s)' % _key(want))
            rec['nequil'] = _stored_nequil(stats) or _key(want)  # what the stats were made with
            rec['state'] = 'stats_saved'
            bad = verify_stats(stats, L, R, executor=executor)
            if bad:
                rec['state'] = 'error'
                record.log(rec, 'stats not verified, nothing deleted: ' + '; '.join(bad[:3]) +
                           (' (+%d more)' % (len(bad) - 3) if len(bad) > 3 else ''))
                record.save(dmc_dir, rec)
                return rec
            record.log(rec, 'stats saved and verified')
            V = decide(L, R, flag_twists(stats, R))
            rec['verdict'] = asdict(V)
            rec['state'] = 'verdict'
            record.log(rec, 'verdict %s (%d/%d twists flagged)' % (V.kind, len(V.flagged), V.ntwist))
            record.save(dmc_dir, rec)
        plan = plan_cleanup(L, rec)
        res = execute(plan, dmc_dir, rec, dry_run=not clean)
        rec['plan'] = {'files': res['files'], 'bytes': res['bytes'],
                       'archive': res['archive'], 'ok': res['ok'], 'dry_run': res['dry_run']}
        record.save(dmc_dir, rec)
    except Exception as e:
        if rec['deleted'] or rec['pending'] or rec['archives'] or rec['state'] in RESUMED:
            rec['state'] = rec['state'] if rec['state'] in RESUMED else 'partially_cleaned'
        else:
            rec['state'] = 'error'
        record.log(rec, 'error: %s: %s' % (type(e).__name__, e))
        record.save(dmc_dir, rec)
    return rec
