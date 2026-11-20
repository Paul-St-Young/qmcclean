"""Per-run record <dmc>/qmcclean/record.json: state, verdict, overrides, deletions, history."""
import contextlib
import fcntl
import getpass
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import SUBDIR

TZ = ZoneInfo('America/New_York')
STATES = ('checked', 'stats_saved', 'verdict', 'cleaned', 'partially_cleaned',
          'stopped_critical', 'error')


def now_et():
    return datetime.now(TZ).isoformat(timespec='seconds')


def path_of(dmc_dir):
    return Path(dmc_dir) / SUBDIR / 'record.json'


class Locked(RuntimeError):
    """Another qmcclean process holds the run's lock."""


@contextlib.contextmanager
def lock(dmc_dir):
    """Hold an exclusive flock on <dmc>/qmcclean/lock; raise Locked at once if it is busy.

    The lock is released when the process exits, however it exits (no stale lock files).
    """
    d = Path(dmc_dir) / SUBDIR
    if not Path(dmc_dir).is_dir():
        raise FileNotFoundError('not a directory: %s' % dmc_dir)
    d.mkdir(exist_ok=True)
    fd = os.open(d / 'lock', os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Locked('locked by another qmcclean process (%s)' % (d / 'lock')) from None
        yield
    finally:
        os.close(fd)


def new(prefix=''):
    return {'prefix': prefix, 'state': 'checked', 'structure': None, 'overrides': [], 'nequil': None,
            'verdict': None, 'released': [], 'deleted': [], 'pending': [], 'archives': [], 'history': []}


def _fix_keys(d):
    """JSON turns int dict keys into strings; turn them back for twist-keyed dicts."""
    return {int(k) if str(k).lstrip('-').isdigit() else k: v for k, v in d.items()}


def load(dmc_dir):
    """The stored record, or a fresh one if none exists. Raises on an unreadable record."""
    p = path_of(dmc_dir)
    rec = new()
    if p.exists():
        with open(p) as f:
            rec.update(json.load(f))
        if rec.get('structure'):
            for k in ('short', 'short_ok', 'missing_files'):
                rec['structure'][k] = _fix_keys(rec['structure'].get(k, {}))
        if rec.get('verdict'):
            rec['verdict']['flagged'] = _fix_keys(rec['verdict'].get('flagged', {}))
    return rec


def _default(o):
    if hasattr(o, 'item'):  # numpy scalar
        return o.item()
    if hasattr(o, 'tolist'):
        return o.tolist()
    return str(o)


def save(dmc_dir, rec):
    """Atomic write: tmp file in the same directory, fsync, os.replace."""
    p = path_of(dmc_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name('record.json.tmp.%d' % os.getpid())
    try:
        with open(tmp, 'w') as f:
            json.dump(rec, f, indent=1, default=_default)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        fd = os.open(p.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def log(rec, action):
    rec['history'].append({'at_ET': now_et(), 'action': action})


def release(dmc_dir, twists, note=''):
    """Record a human decision to let cleanup delete flagged twists (list of ints, or 'all').

    Deletes nothing; the next `run --clean` does. Returns {'released': twists actually added,
    'messages': why anything was skipped, 'refused': True when nothing could be considered}.
    Refused (no-op) when there is no record, the record has no verdict (e.g. stopped_critical:
    run with skip_opt first to obtain one), or another qmcclean process holds the run's lock.
    """
    if twists != 'all' and (isinstance(twists, (str, bytes)) or not hasattr(twists, '__iter__')):
        raise ValueError("twists must be a list of ints or 'all', got %r" % (twists,))
    out = {'released': [], 'messages': [], 'refused': False}
    if not path_of(dmc_dir).exists():
        out['messages'].append('no record: run qmcclean first')
        out['refused'] = True
        return out
    try:
        with lock(dmc_dir):
            return _release(dmc_dir, twists, note, out)
    except Locked as e:
        out['messages'].append(str(e))
        out['refused'] = True
        return out


def _release(dmc_dir, twists, note, out):
    rec = load(dmc_dir)
    v = rec.get('verdict')
    if not v:
        out['messages'].append('no verdict in record (state %s): run with --skip-opt (dry) '
                               'to get one, then release' % rec['state'])
        out['refused'] = True
        return out
    if any(e['twists'] == 'all' for e in rec['released']):
        out['messages'].append("already released: 'all'")
        return out
    flagged = {int(t) for t in v['flagged']}
    done = {int(t) for e in rec['released'] for t in e['twists']}
    if twists == 'all':
        if v['kind'] == 'clean':
            out['messages'].append('run is clean: nothing to release')
            return out
        left = sorted(flagged - done)
        if not left and v['kind'] != 'hold':
            out['messages'].append('all flagged twists already released')
            return out
        add = 'all'
        out['released'] = left
    else:
        add = []
        for t in sorted({int(t) for t in twists}):
            if t not in flagged:
                out['messages'].append('twist %d is not flagged: not released' % t)
            elif t in done:
                out['messages'].append('twist %d already released' % t)
            else:
                add.append(t)
        out['released'] = list(add)
        if not add:
            return out
        if v['kind'] == 'hold' and flagged <= done | set(add):
            add = 'all'  # a hold run whose flagged twists are all released is cleaned whole
    user = os.environ.get('USER') or getpass.getuser()
    rec['released'].append({'twists': add, 'by': user, 'at_ET': now_et(), 'note': note})
    log(rec, 'released %s by %s: %s' % ('all' if add == 'all' else
                                         'twists ' + ','.join(map(str, add)), user, note))
    save(dmc_dir, rec)
    return out


def excluded_twists(dmc_dir):
    """Twists to leave out of twist averages, as a sorted list of ints.

    - verdict 'clean' / 'drop_listed': the flagged twists.
    - verdict 'hold', or a record without a verdict (stopped_critical, error, interrupted):
      ALL twists of the run (from the stored layout, else from the dmc.gNNN.twistnum_T.in.xml
      names in the folder), since nothing about the run has been accepted.
    - no record (never processed): [] -- nothing is known.
    Independent of release: releasing a twist (OK to delete its raw files) does not put it
    back into averages.
    """
    if not path_of(dmc_dir).exists():
        return []
    rec = load(dmc_dir)
    v = rec.get('verdict')
    if v and v.get('kind') != 'hold':
        return sorted(int(t) for t in v.get('flagged', {}))
    tw = {int(t) for t in (rec.get('layout') or {}).get('twists', [])}
    tw |= {int(t) for t in ((rec.get('layout') or {}).get('g2t') or {}).values()}
    if v:
        tw |= {int(t) for t in v.get('flagged', {})}
    from .layout import TWIST_XML  # local: layout imports qharv
    try:
        for n in os.listdir(dmc_dir):
            m = TWIST_XML.match(n)
            if m:
                tw.add(int(m.group(2)))
    except OSError:
        pass
    return sorted(tw)
