"""Command line: qmcclean run | review | du | survey.

`run` processes calculations one after another; within each, the per-twist raw-file reads
are spread over one process pool shared by the whole invocation (--nproc; 1 = serial).
"""
import argparse
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path

from . import record
from .layout import classify
from .pipeline import run_dmc


def _nequil(vmc, dmc):
    out = {}
    if vmc is not None:
        out[0] = vmc
    if dmc is not None:
        out[1] = dmc
    return out or None


def _stages(path):
    """(opt_dir or None, dmc_dir or None, notes) for one PATH, or raises ValueError."""
    p = Path(path)
    kind = classify(p) if p.is_dir() else 'unknown'
    if kind == 'unknown':
        raise ValueError('not an opt, dmc or calc folder: %s' % path)
    if kind == 'opt':
        return p, None, []
    if kind == 'dmc':
        return None, p, []
    notes = []
    opt = p / 'opt' if (p / 'opt').is_dir() else None
    dmc = p / 'dmc' if (p / 'dmc').is_dir() else None
    for name, d in (('opt', opt), ('dmc', dmc)):
        if d is None:
            notes.append('no %s/' % name)
    return opt, dmc, notes


def _work(args, executor=None):
    """Process one calculation; never raises. Returns a summary dict (with wall 'seconds')."""
    t0 = time.monotonic()
    out = _work1(args, executor)
    out['seconds'] = time.monotonic() - t0
    return out


def _work1(args, executor):
    path, skip_opt, clean, nequil = args
    out = {'path': str(path), 'opt': 'absent', 'state': None, 'verdict': None, 'flagged': [],
           'bytes': None, 'dry_run': not clean, 'error': None, 'notes': [], 'stopped': False}
    try:
        opt, dmc, out['notes'] = _stages(path)
    except Exception as e:
        out['error'] = str(e)
        return out
    if opt is not None:
        try:
            # imported here: a missing chiesa_correction (QMCCLEAN_PYMODS) fails the opt
            # check of this calculation only; the dmc stage still runs
            from .optcheck import check_opt
            rep = check_opt(opt)
            out['opt'] = 'flagged' if rep.flagged else 'ok'
            out['opt_reasons'] = list(rep.reasons)
        except Exception as e:
            out['opt'] = 'error'
            out['error'] = 'opt: %s: %s' % (type(e).__name__, e)
    if dmc is not None:
        try:
            rec = run_dmc(dmc, skip_opt=skip_opt, clean=clean, nequil=nequil, executor=executor)
        except Exception as e:  # run_dmc catches its own; this is the last line of defence
            msg = '%s: %s' % (type(e).__name__, e)
            out['error'] = msg
            out['state'] = 'error'
            if dmc.is_dir():
                try:
                    rec = record.load(dmc)
                    began = bool(rec['deleted'] or rec['pending'] or rec['archives']
                                 or rec['state'] in ('cleaned', 'partially_cleaned'))
                    if began:  # deletion began earlier: keep the state, only note the error
                        out['state'] = rec['state']
                    else:
                        rec['state'] = 'error'
                    record.log(rec, 'error: ' + msg)
                    record.save(dmc, rec)
                except Exception:
                    out['error'] += ' (record not saved: %s)' % traceback.format_exc(limit=1).strip()
            return out
        out['prefix'] = rec.get('prefix')
        out['state'] = rec['state']
        v = rec.get('verdict')
        if v:
            out['verdict'] = v['kind']
            out['flagged'] = sorted(int(t) for t in v['flagged'])
        plan = rec.get('plan')
        if plan:
            out['bytes'] = plan['bytes']
            out['dry_run'] = plan['dry_run']
        out['stopped'] = rec['state'] == 'stopped_critical'
        if rec['state'] == 'error':
            last = rec['history'][-1]['action'] if rec['history'] else 'error'
            out['error'] = (out['error'] + '; ' if out['error'] else '') + last
    return out


def _human(n):
    n = float(n)
    for u in ('B', 'kB', 'MB', 'GB', 'TB'):
        if round(n, 3 if u == 'B' else 2) < 999.5 or u == 'TB':
            return '%.3g %s' % (n, u)
        n /= 1000


def _line(r):
    parts = [r.get('prefix') or r['path'], 'opt=' + r['opt']]
    if r['state'] is not None:
        parts.append('%s%s' % (('verdict=%s ' % r['verdict']) if r['verdict'] else '',
                               'state=' + r['state']))
        if r['flagged']:
            parts.append('flagged=' + ','.join(map(str, r['flagged'])))
        if r['bytes'] is not None:
            parts.append(('would free ' if r['dry_run'] else 'freed ') + _human(r['bytes']))
    if r['notes']:
        parts.append('(%s)' % '; '.join(r['notes']))
    if r.get('seconds') is not None:
        parts.append('in %.0f s' % r['seconds'])
    if r['error']:
        parts.append('ERROR: ' + r['error'])
    return '  '.join(parts)


def _new_pool(nproc):
    from concurrent.futures import ProcessPoolExecutor
    # spawn: h5py/HDF5 state does not survive fork reliably
    return ProcessPoolExecutor(nproc, mp_context=multiprocessing.get_context('spawn'))


def _healthy(ex):
    """False if the pool is broken (a worker was killed) or unusable."""
    try:
        return ex.submit(int, 0).result(timeout=120) == 0
    except Exception:
        return False


def _kill_pool(ex, grace=10):
    """Shut a broken or stuck pool down without blocking: terminate (then kill) its workers.

    Workers only read raw files, so killing them loses nothing.
    """
    procs = list((getattr(ex, '_processes', None) or {}).values())
    ex.shutdown(wait=False, cancel_futures=True)
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(grace)
        if p.is_alive():
            p.kill()
            p.join(grace)


def cmd_run(a):
    """Runs one after another; each fans its per-twist reads out over one shared pool."""
    nproc = max(1, a.nproc or len(os.sched_getaffinity(0)))
    nq = _nequil(a.nequil_vmc, a.nequil_dmc)
    if not a.clean:
        print('dry run: no file will be removed (use --clean to delete)')
    ex = _new_pool(nproc) if nproc > 1 else None  # --nproc 1: no pool, all in-process
    bad = 0
    try:
        for p in a.paths:
            r = _work((p, a.skip_opt, a.clean, nq), ex)
            print(_line(r), flush=True)
            bad += bool(r['error'] or r['stopped'])
            if ex is not None and not _healthy(ex):  # killed worker or stuck: fresh pool
                _kill_pool(ex)  # never more than nproc live workers, never wait on a stuck one
                ex = _new_pool(nproc)
    finally:
        if ex is not None:
            ex.shutdown(cancel_futures=True)
    return 1 if bad else 0


def cmd_review(a):
    if a.release == 'all':
        tw = 'all'
    else:
        try:
            tw = [int(x) for x in a.release.split(',') if x.strip()]
        except ValueError:
            print('--release takes "all" or a comma list of twist numbers', file=sys.stderr)
            return 2
    d = Path(a.dmc_dir)
    if not d.is_dir():
        print('ERROR: not a directory: %s' % d, file=sys.stderr)
        return 1
    try:
        out = record.release(d, tw, note=a.note)
    except ValueError as e:
        print('ERROR: %s' % e, file=sys.stderr)
        return 1
    if out.get('refused') or not record.path_of(d).exists():  # no record / verdict / locked
        for m in out['messages']:
            print('ERROR: ' + m, file=sys.stderr)
        return 1
    for m in out['messages']:
        print(m)
    print('released: %s' % (out['released'] or 'nothing'))
    if out['released']:
        print('nothing deleted yet: run `python -m qmcclean run %s --clean` to act' % a.dmc_dir)
    return 0


def cmd_du(a):
    from .du import run_du
    print(run_du(a.dmc_dir))
    return 0


def cmd_survey(a):
    from .survey import survey
    df = survey(a.root)
    df.to_csv(a.out, index=False)
    print('%d dmc folders -> %s (%d with errors)' % (len(df), a.out, int((df['error'] != '').sum())))
    return 0


def _posint(x):
    n = int(x)
    if n < 1:
        raise argparse.ArgumentTypeError('must be >= 1')
    return n


def build_parser():
    ap = argparse.ArgumentParser(prog='qmcclean', description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='check, save statistics and (with --clean) delete raw files')
    r.add_argument('paths', nargs='+', metavar='PATH', help='<calc>/opt, <calc>/dmc or <calc>')
    r.add_argument('--skip-opt', action='store_true', help='override critical structure failures for every run')
    r.add_argument('--clean', action='store_true', help='actually delete (default: dry run)')
    r.add_argument('--nproc', type=_posint, default=None, help='worker processes for the per-twist reads of each run (default: all allocated cores; 1 = serial, no pool)')
    r.add_argument('--nequil-vmc', type=int, default=None, help='equilibration blocks, series 0 (default 32)')
    r.add_argument('--nequil-dmc', type=int, default=None, help='equilibration blocks, series 1 (default 16)')
    r.set_defaults(fn=cmd_run)
    v = sub.add_parser('review', help='release flagged twists for the next `run --clean`')
    v.add_argument('dmc_dir')
    v.add_argument('--release', required=True, metavar='all|TWIST[,TWIST...]')
    v.add_argument('--note', default='')
    v.set_defaults(fn=cmd_review)
    d = sub.add_parser('du', help='delta-u(k) validation: uk.png plot, du.csv, du.json')
    d.add_argument('dmc_dir')
    d.set_defaults(fn=cmd_du)
    s = sub.add_parser('survey', help='read-only inventory of all dmc folders under ROOT')
    s.add_argument('root')
    s.add_argument('--out', default='survey.csv')
    s.set_defaults(fn=cmd_survey)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    return a.fn(a)
