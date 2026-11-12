"""Structure checks of a dmc run: block counts, required files, estimators."""
from dataclasses import dataclass, field

import h5py

from .config import MIN_BLOCK_FRACTION

OPTIONAL = ('.config.h5', '.random.h5')


@dataclass
class StructureReport:
    short: dict = field(default_factory=dict)  # twist -> ['s001: 10/30 blocks']
    short_ok: dict = field(default_factory=dict)  # twist -> ['s001: 28/30 blocks (accepted)']
    missing_files: dict = field(default_factory=dict)  # twist -> [reason]
    missing_estimators: list = field(default_factory=list)
    absent_optional: list = field(default_factory=list)
    critical: bool = False
    overridden: bool = False


def _count_lines(path):
    """Data lines of a scalar.dat (comment/blank lines excluded)."""
    n = 0
    with open(path, 'rb') as f:
        for line in f:
            s = line.lstrip()
            if s and not s.startswith(b'#'):
                n += 1
    return n


def _read_h5(path):
    """(nblock, set of top-level group names), cheap: shape only."""
    with h5py.File(path, 'r') as f:
        return f['LocalEnergy/value'].shape[0], set(f.keys())


def _probe(kind, path):
    """(value, None) or (None, error text); never raises. kind: 'scalar' line count or 'h5'."""
    try:
        return (_count_lines(path) if kind == 'scalar' else _read_h5(path)), None
    except OSError as e:
        if kind == 'scalar':
            return None, str(e)
        return None, '%s: %s' % (type(e).__name__, e)
    except Exception as e:  # any h5 problem: unreadable
        return None, '%s: %s' % (type(e).__name__, e)


def _probe_all(L, executor):
    """{(twist, series, kind): (value, error)} for every scalar.dat and stat.h5 of the run."""
    keys = [(t, s.index, kind) for t in L.twists for s in L.series
            for kind, suf in (('scalar', '.scalar.dat'), ('h5', '.stat.h5'))
            if (t, s.index, suf) in L.files]
    paths = [L.files[(t, i, '.scalar.dat' if k == 'scalar' else '.stat.h5')] for t, i, k in keys]
    kinds = [k for _, _, k in keys]
    res = map(_probe, kinds, paths) if executor is None else executor.map(
        _probe, kinds, paths, chunksize=8)
    return dict(zip(keys, res))


def check_structure(L, skip_opt=False, executor=None):
    """executor: optional concurrent.futures executor for the per-file reads (None = serial)."""
    R = StructureReport()
    probes = _probe_all(L, executor)
    missing_at = {}  # estimator -> [(twist, series)] whose readable stat.h5 lacks it
    nread = 0  # (twist, series) pairs with a readable stat.h5

    def flag(d, twist, msg):
        d.setdefault(twist, []).append(msg)

    for t in L.twists:
        for s in L.series:
            tag = 's%03d' % s.index
            nb = []
            sc = L.files.get((t, s.index, '.scalar.dat'))
            if sc is None:
                flag(R.missing_files, t, '%s: scalar.dat missing' % tag)
            else:
                n, err = probes[(t, s.index, 'scalar')]
                if err is None:
                    nb.append(n)
                else:
                    flag(R.missing_files, t, '%s: scalar.dat unreadable (%s)' % (tag, err))
            st = L.files.get((t, s.index, '.stat.h5'))
            if st is None:
                flag(R.missing_files, t, '%s: stat.h5 missing' % tag)
            else:
                v, err = probes[(t, s.index, 'h5')]
                if err is None:
                    n, groups = v
                    nb.append(n)
                    nread += 1
                    for e in L.estimators:
                        if e not in groups:
                            missing_at.setdefault(e, []).append((t, s.index))
                else:  # any h5 problem: unreadable, never raise
                    flag(R.missing_files, t, '%s: stat.h5 unreadable (%s)' % (tag, err))
            if nb and min(nb) < s.blocks:
                if min(nb) >= MIN_BLOCK_FRACTION * s.blocks - 1e-9:  # e.g. cut by walltime
                    flag(R.short_ok, t, '%s: %d/%d blocks (accepted, >= %g)' % (
                        tag, min(nb), s.blocks, MIN_BLOCK_FRACTION))
                else:
                    flag(R.short, t, '%s: %d/%d blocks' % (tag, min(nb), s.blocks))
            elif nb and max(nb) > s.blocks:  # e.g. a restart appended to the same files
                flag(R.short, t, '%s: %d/%d blocks (more than requested)' % (
                    tag, max(nb), s.blocks))
    for t, fs in L.unexpected.items():
        flag(R.missing_files, t, 'ambiguous: ' + ', '.join(sorted(f.name for f in fs)))
    for g, fs in sorted(L.no_xml.items()):
        flag(R.missing_files, g, 'no input xml for g%03d: %s' % (
            g, ', '.join(sorted(f.name for f in fs))))

    # an estimator absent from every readable stat.h5 is a run-level (critical, overridable)
    # failure; absent from only some twists, it flags those twists (never overridden)
    R.missing_estimators = [e for e in L.estimators
                            if nread and len(missing_at.get(e, ())) == nread]
    for e in L.estimators:
        if e in R.missing_estimators:
            continue
        for t, i in missing_at.get(e, ()):
            flag(R.missing_files, t, 'missing estimator %s in s%03d' % (e, i))

    for suffix, series in [('.dmc.dat', [1])] + [(x, [s.index for s in L.series])
                                                  for x in OPTIONAL]:
        n = sum((t, i, suffix) not in L.files for t in L.twists for i in series)
        if n:
            R.absent_optional.append('%s absent: %d/%d files' % (
                suffix[1:], n, len(L.twists) * len(series)))
    R.critical = bool(R.missing_estimators)
    if skip_opt and R.critical:
        R.critical, R.overridden = False, True
    return R
