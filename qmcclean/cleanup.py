"""Plan and execute the cleanup of a dmc folder. Every deletion is gated and recorded.

Order (each step gated on the previous): verify_stats empty -> plan still allowed by the record
-> write archive to <name>.tmp, reopen and compare members -> rename -> delete archived originals
-> delete big files -> record. When in doubt nothing is deleted and the record says why.
"""
import os
import re
import tarfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import record
from .config import BIG_SUFFIXES, KEEP, SUBDIR
from .layout import TWIST_XML, discover_dmc
from .stats import META, verify_stats
from .structure import StructureReport

TWIST_FILE = re.compile(r'^dmc\.g(\d+)\.')
BANDINFO = re.compile(r'\.g(\d+)\.bandinfo\.dat$')
RUN_LEVEL = re.compile(r'^(dmc\.(in|out|err|info\.xml|qsub\.in)|dmc\.struct\..*|Jk.*\.dat|'
                       r'local_hostfile\..*|sim_dmc)$')
ARCHIVE = re.compile(r'^dmc\.small(\.release-\d+)?\.tgz(\.tmp)?$')


@dataclass
class CleanupPlan:
    delete: list = field(default_factory=list)   # big files of cleanable twists
    archive: list = field(default_factory=list)  # small files/dirs to put in the new archive
    keep: list = field(default_factory=list)     # left in place
    archive_name: str = ''
    stale: list = field(default_factory=list)    # small originals already inside an archive
    remaining: list = field(default_factory=list)  # twists that stay (not cleanable)
    orphans: list = field(default_factory=list)  # groups with files but no twist xml: kept

    def empty(self):
        return not (self.delete or self.archive or self.stale)


def _cleanable(twists, rec):
    """Twists whose raw files the record allows to delete."""
    twists = set(twists)
    v = rec.get('verdict')
    ok = set()
    if v:
        flagged = {int(t) for t in v.get('flagged', {})}
        if v.get('kind') == 'clean':
            ok |= twists
        elif v.get('kind') == 'drop_listed':
            ok |= twists - flagged
    for e in rec.get('released', []):
        t = e.get('twists')
        ok |= twists if t == 'all' else twists & {int(x) for x in t}
    return ok


def _entries(p):
    """Relative-to-p names of p and everything below it (symlinked dirs are leaves)."""
    out = [p]
    if p.is_dir() and not p.is_symlink():
        for d, dirs, files in os.walk(p):
            out += [Path(d) / n for n in dirs + files]
    return out


def _archive_members(path):
    try:
        with tarfile.open(path, 'r:gz') as t:
            return {m.name: (m.size, int(m.mtime)) for m in t.getmembers() if not m.isdir()}
    except Exception:
        return {}


def _g2t(root):
    return {int(m.group(1)): int(m.group(2)) for f in root.iterdir()
            if (m := TWIST_XML.match(f.name))}


def _in_archive(root, p, members):
    files = [e for e in _entries(p) if not e.is_dir() or e.is_symlink()]
    for e in files:
        m = members.get(str(e.relative_to(root)))
        st = e.lstat()
        if m is None or m[1] != int(st.st_mtime) or (not e.is_symlink() and m[0] != st.st_size):
            return False
    return bool(files)


def plan_cleanup(L, rec):
    root = Path(L.path)
    stored = rec.get('layout') or {}
    g2t = {int(g): int(t) for g, t in stored.get('g2t', {}).items()}
    g2t.update(_g2t(root))
    universe = sorted(set(L.twists) | {int(t) for t in stored.get('twists', [])})
    cleanable = _cleanable(universe, rec)
    plan = CleanupPlan(remaining=sorted(set(universe) - cleanable))
    members = {}
    for a in sorted(root.iterdir()):
        if ARCHIVE.match(a.name) and not a.name.endswith('.tmp'):
            members.update(_archive_members(a))
    # files of a group with no known twist xml are never touched (their stats were never saved)
    orphans = set()
    for f in root.iterdir():
        m = TWIST_FILE.match(f.name) or BANDINFO.search(f.name)
        if m and int(m.group(1)) not in g2t and f.name.endswith(BIG_SUFFIXES):
            orphans.add(int(m.group(1)))
    plan.orphans = sorted(orphans)
    # run-level files ride the archive after which no flagged twist is left unreleased
    final = bool(cleanable) and not plan.remaining and not plan.orphans
    for f in sorted(root.iterdir()):
        if f.name == SUBDIR or f.name in KEEP or ARCHIVE.match(f.name):
            plan.keep.append(f)
            continue
        m = TWIST_FILE.match(f.name) or BANDINFO.search(f.name)
        if m:
            g = int(m.group(1))
            t = g2t.get(g, g)
            if g in orphans or t not in cleanable or f.is_dir():
                plan.keep.append(f)
            elif f.name.endswith(BIG_SUFFIXES):
                plan.delete.append(f)
            elif _in_archive(root, f, members):
                plan.stale.append(f)
            else:
                plan.archive.append(f)
        elif RUN_LEVEL.match(f.name):
            if _in_archive(root, f, members):
                plan.stale.append(f)
            elif final:
                plan.archive.append(f)
            else:
                plan.keep.append(f)
        else:
            plan.keep.append(f)  # unknown file: never touched
    if plan.archive:
        name, n = 'dmc.small.tgz', 0
        while (root / name).exists():
            n += 1
            name = 'dmc.small.release-%d.tgz' % n
        plan.archive_name = name
    return plan


def _inside(root, p):
    """True if p (not following a final symlink) lies strictly inside root."""
    try:
        return (Path(p).parent.resolve() / Path(p).name).relative_to(root) != Path('.')
    except ValueError:
        return False


def _direct_child(root, p):
    """Every plan path is an entry directly inside the run folder."""
    p = Path(p)
    return p.name not in ('', '.', '..') and p.parent.resolve() == root


def _rebase(plan, root):
    """The plan with every path expressed under the resolved run folder."""
    f = lambda ps: [root / Path(p).name for p in ps]
    return replace(plan, delete=f(plan.delete), archive=f(plan.archive), stale=f(plan.stale),
                   keep=f(plan.keep))


def _summary(plan):
    files = [e for p in plan.delete + plan.archive + plan.stale for e in _entries(p)
             if not e.is_dir() or e.is_symlink()]
    return len(files), sum(e.lstat().st_size for e in files)


def _structure(rec):
    s = rec.get('structure')
    if not s:
        return None
    return StructureReport(**s)


def _blocked(rec, dmc_dir, why, res):
    record.log(rec, 'cleanup blocked: ' + why)
    record.save(dmc_dir, rec)
    res.update(ok=False, reasons=[why])
    return res


def _gate(plan, root, rec):
    """Reasons the plan must not run now ([] = allowed)."""
    R = _structure(rec)
    if R is None:
        return ['no structure report in record']
    try:
        L = discover_dmc(root)
    except Exception as e:
        return ['rediscovery failed: %s: %s' % (type(e).__name__, e)]
    stats = root / SUBDIR / META
    if not stats.exists():
        return ['stats.json missing']
    bad = verify_stats(stats, L, R)
    if bad:
        return ['stats not verified: ' + '; '.join(bad[:3]) +
                (' (+%d more)' % (len(bad) - 3) if len(bad) > 3 else '')]
    fresh = plan_cleanup(L, rec)
    for k in ('delete', 'archive', 'stale'):
        extra = {p.name for p in getattr(plan, k)} - {p.name for p in getattr(fresh, k)}
        if extra:
            return ['plan not allowed by record (%d paths): %s' % (
                len(extra), ', '.join(sorted(extra)[:3]))]
    return []


def _write_archive(plan, root, snap=None):
    """Write <name>.tmp from the snapshotted entries, reopen, compare members; return tmp path."""
    final = root / plan.archive_name
    tmp = root / (plan.archive_name + '.tmp')
    if final.exists():
        raise FileExistsError('%s exists' % final)
    if tmp.exists():  # our own stale tmp from an interrupted run
        tmp.unlink()
    snap = snap or {p: _entries(p) for p in plan.archive}
    want = {}
    try:
        with tarfile.open(tmp, 'w:gz') as t:
            for p in plan.archive:
                for e in snap[p]:
                    t.add(e, arcname=str(e.relative_to(root)), recursive=False)
                    if not e.is_dir() or e.is_symlink():
                        want[str(e.relative_to(root))] = e.lstat().st_size
        got = {n: s for n, (s, _) in _archive_members(tmp).items()}
        if set(got) != set(want):
            raise RuntimeError('archive members differ from plan: %s' % sorted(
                set(got) ^ set(want))[:5])
        bad = [n for n in want if not os.path.islink(root / n) and got[n] != want[n]]
        if bad:
            raise RuntimeError('archive member sizes differ: %s' % bad[:5])
        with open(tmp, 'rb') as fh:
            os.fsync(fh.fileno())
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return tmp


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_entries(entries, root):
    """Unlink exactly these verified entries; a directory goes only if empty afterwards."""
    gone = []
    for e in sorted(entries, key=lambda x: -len(x.parts)):
        assert _inside(root, e), e
        if e.is_dir() and not e.is_symlink():
            try:
                e.rmdir()
            except OSError:
                continue
        else:
            e.unlink(missing_ok=True)
        gone.append(e)
    return gone


def _reconcile(rec, root):
    """Pending entries whose files are gone count as deleted; the rest are re-planned."""
    if not rec.get('pending'):
        return False
    for rel in rec['pending']:
        if not os.path.lexists(root / rel) and rel not in rec['deleted']:
            rec['deleted'].append(rel)
    rec['pending'] = []
    return True


def execute(plan, dmc_dir, rec, dry_run):
    """Carry out (or, with dry_run, only count) the plan. Returns a result dict; never raises
    for a blocked cleanup (see rec['history'])."""
    root = Path(dmc_dir).resolve()
    res = {'ok': True, 'dry_run': dry_run, 'files': 0, 'bytes': 0,
           'archive': plan.archive_name, 'reasons': []}
    outside = [p for p in plan.delete + plan.archive + plan.stale + ([root / plan.archive_name] if plan.archive_name else [])
               if not _direct_child(root, p)]
    if outside:
        return _blocked(rec, dmc_dir, 'path outside run folder: %s' % outside[0], res)
    plan = _rebase(plan, root)
    res['files'], res['bytes'] = _summary(plan)
    if dry_run:
        record.log(rec, 'dry-run plan: %d files, %d bytes' % (res['files'], res['bytes']))
        why = _gate(plan, root, rec) if not plan.empty() else []
        if why:
            record.log(rec, 'dry-run: cleanup would be blocked: ' + why[0])
            res.update(ok=False, reasons=why)
        record.save(dmc_dir, rec)
        return res
    if _reconcile(rec, root):
        record.save(dmc_dir, rec)
    if plan.empty():
        return res
    why = _gate(plan, root, rec)
    if why:
        return _blocked(rec, dmc_dir, why[0], res)

    snap = {p: _entries(p) for p in plan.archive + plan.stale}
    if plan.archive:
        try:
            tmp = _write_archive(plan, root, snap)
            os.replace(tmp, root / plan.archive_name)
            _fsync_dir(root)
        except Exception as e:
            return _blocked(rec, dmc_dir, 'archive failed: %s: %s' % (type(e).__name__, e), res)
        rec['archives'].append(plan.archive_name)
        record.log(rec, 'archived %d files into %s' % (
            sum(1 for p in plan.archive for e in snap[p] if not e.is_dir()),
            plan.archive_name))
        record.save(dmc_dir, rec)

    rec['state'] = 'partially_cleaned'  # from here on an interruption must not look unclean
    record.log(rec, 'deleting raw files')
    record.save(dmc_dir, rec)
    # big files first: the archive is verified, and the small originals (incl. each twist's
    # in.xml, which discovery needs) are the last thing to go
    phases = ((plan.delete, [e for p in plan.delete for e in _entries(p)], 'big files'),
              (plan.archive + plan.stale, [e for p in plan.archive + plan.stale for e in snap[p]],
               'archived originals'))
    try:
        for _, entries, label in phases:
            rec['pending'] = [str(e.relative_to(root)) for e in entries]
            record.save(dmc_dir, rec)  # intent first: survives a kill during the unlinks
            try:
                _remove_entries(entries, root)
            finally:
                n = len(rec['pending'])
                _reconcile(rec, root)
                record.log(rec, 'deleted %d %s' % (n, label))
                record.save(dmc_dir, rec)
    except OSError as e:
        return _blocked(rec, dmc_dir, 'delete failed: %s: %s' % (type(e).__name__, e), res)
    rec['state'] = 'partially_cleaned' if plan.remaining or plan.orphans else 'cleaned'
    record.log(rec, 'cleanup done: %s%s%s' % (rec['state'], (
        ' (twists kept: %s)' % ', '.join(map(str, plan.remaining[:20]))) if plan.remaining else '',
        (' (groups without input xml kept: %s)' % ', '.join(
            'g%03d' % g for g in plan.orphans)) if plan.orphans else ''))
    record.save(dmc_dir, rec)
    return res
