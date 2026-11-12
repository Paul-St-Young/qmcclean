"""Compact per-twist statistics and per-block traces saved in <dmc>/qmcclean/.

Files (S = series index, T = twist index, 3 digits each):
  dmc.s{SSS}.scalar.dat   QMCPACK scalar.dat-style text table, one row per twist (sorted):
                          twist nblock nequil, then {col}_mean {col}_error for every column of
                          the twist's scalar.dat in order (minus index, plus derived Variance)
  dmc.s001.dmc.dat        same format for the per-block reduced dmc.dat of the DMC series;
                          load_scalar exposes its columns with a 'dmc_' prefix
  dmc.s{SSS}.stat.h5      /kvecs                   skall k-vectors, stored once (present iff any
                                                   twist has dsk/rhok); every twist's raw
                                                   skall/kpoints/value must equal it exactly
                          /g{TTT}/dsk/{mean,error}  N*S(k) minus |rho(k)|^2 (qharv dsk_from_skall)
                          /g{TTT}/rhok/{mean,error} complex rho(k) (qharv rhok_from_skall)
                          /g{TTT}/{obs}/{mean,error[,kvecs]}  every other stat.h5 group with a
                          'value' dataset that is not a scalar.dat column (e.g. nofk, gofr);
                          kvecs = the group's own 'kpoints' (twist-dependent) if it has one;
                          error is qharv me2d (autocorrelation-corrected, as the scalar table)
                          root attrs: nequil, series
  dmc.s{SSS}.trace.h5     /g{TTT} (nblock, k) per-block traces; attr 'columns' names the k
                          columns; s001 dmc.dat columns are per-block means of its per-step rows
                          (attrs dmc_dat_steps, dmc_dat_rows, dmc_dat_len_mismatch, dmc_dat_irregular)
  stats.json              prefix, driver, nelec, nequil, created (ET), qmcclean_version, files.
                          Written LAST: the statistics exist iff stats.json exists, and only the
                          files it lists belong to them.

The text tables may be edited by hand (replace a rerun twist's row, append a twist);
verify_stats checks them again before anything is deleted.

These qmcclean/ tables share QMCPACK's file names (dmc.s001.scalar.dat, ...): external `find`
scripts looking for raw QMCPACK output must exclude qmcclean/.
"""
import json
import os
from concurrent import futures
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import pandas as pd

from qharv.reel import scalar_dat, stat_h5
from qharv.sieve import mean_df

from . import __version__
from .config import SUBDIR
from .structure import _probe

TRACE_COLS = ['LocalEnergy', 'Variance', 'Kinetic', 'BlockWeight']
TRACE_DMC_COLS = ['NumOfWalkers', 'Weight', 'TrialEnergy']
TZ = ZoneInfo('America/New_York')

META = 'stats.json'
META_KEYS = ('prefix', 'driver', 'nelec', 'nequil', 'created', 'qmcclean_version', 'files')
INT_COLS = ('twist', 'nblock', 'nequil')
CANCEL_WAIT = 600  # s: max wait for running tasks of a failed compute_stats
FLOAT_FMT = '%.16e'  # 17 significant digits: float64 round-trips exactly


def scalar_name(s):
    return 'dmc.s%03d.scalar.dat' % s


def dmcdat_name(s):
    return 'dmc.s%03d.dmc.dat' % s


def stat_name(s):
    return 'dmc.s%03d.stat.h5' % s


def trace_name(s):
    return 'dmc.s%03d.trace.h5' % s


# --- text tables ---------------------------------------------------------------------------

def _cell(c, v, ints=INT_COLS):
    if c in ints:
        return '%d' % int(v)
    return FLOAT_FMT % float(v)


def write_table(path, cols, rows, ints=INT_COLS):
    """Write rows (list of dicts) as a scalar.dat-style table; missing entries are NaN.
    Columns named in ints are written as integers."""
    cells = [[_cell(c, r.get(c, np.nan), ints) for c in cols] for r in rows]
    width = [max([len(c)] + [len(x[i]) for x in cells]) for i, c in enumerate(cols)]
    lines = ['#' + ''.join(' ' + c.rjust(w) for c, w in zip(cols, width))]
    lines += [' ' + ''.join(' ' + x.rjust(w) for x, w in zip(row, width)) for row in cells]
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
        fh.flush()
        os.fsync(fh.fileno())


def read_table(path):
    """A table written by write_table (or edited by hand) as a DataFrame.

    Strict: ValueError unless every data line has one field per header column (pandas would
    silently fill a truncated line with NaN)."""
    with open(path) as fh:
        lines = [x.split() for x in fh.read().splitlines() if x.strip()]
    if not lines or lines[0][:1] != ['#']:
        raise ValueError('%s: no "# name ..." header line' % path)
    ncol = len(lines[0]) - 1
    for i, x in enumerate(lines[1:], 2):
        if len(x) != ncol:
            raise ValueError('%s: line %d has %d fields, header has %d' % (path, i, len(x), ncol))
    df = scalar_dat.read(str(path))
    if len(df) == 0:  # header only: give the columns their dtypes
        df = df.astype({c: (np.int64 if c in INT_COLS else np.float64) for c in df.columns})
    return df


def _table_problems(df, name):
    """Format problems of a parsed table ([] = fine)."""
    out = []
    cols = list(df.columns)
    if cols[:3] != list(INT_COLS):
        out.append('%s: columns start with %s, expected twist nblock nequil' % (name, cols[:3]))
        return out
    for i, c in enumerate(cols[3:], 3):
        if c.endswith('_mean'):
            if i + 1 >= len(cols) or cols[i + 1] != c[:-5] + '_error':
                out.append('%s: %s not followed by %s_error' % (name, c, c[:-5]))
        elif c.endswith('_error'):
            if cols[i - 1] != c[:-6] + '_mean':
                out.append('%s: %s not preceded by %s_mean' % (name, c, c[:-6]))
        else:
            out.append('%s: column %s is neither _mean nor _error' % (name, c))
    if len(set(cols)) != len(cols):
        out.append('%s: duplicate column names' % name)
    for c in INT_COLS:
        v = pd.to_numeric(df[c], errors='coerce').to_numpy(float)
        if not (np.isfinite(v).all() and (v == np.round(v)).all() and (v >= 0).all()):
            out.append('%s: column %s is not all non-negative integers' % (name, c))
    tw = df['twist'].tolist()
    dup = sorted({t for t in tw if tw.count(t) > 1})
    if dup:
        out.append('%s: duplicate twists %s' % (name, dup))
    return out


def read_meta(stats_path):
    with open(stats_path) as fh:
        meta = json.load(fh)
    if not isinstance(meta, dict):
        raise ValueError('%s is not a JSON object' % stats_path)
    return meta


# --- statistics ----------------------------------------------------------------------------

def _skipped(R, twist, series):
    """True if the structure check excluded this (twist, series)."""
    tag = 's%03d:' % series
    return any(m.startswith(tag) or m.startswith('ambiguous')
               for m in R.missing_files.get(twist, []))


def _clean(df):
    """Drop the block-index column, duplicated columns; make sure Variance exists once (R2)."""
    df = df.drop(columns=[c for c in df.columns if c in ('index', 'Index')])
    df = df.loc[:, ~df.columns.duplicated()]
    if 'Variance' not in df.columns and {'LocalEnergy', 'LocalEnergy_sq'} <= set(df.columns):
        df = df.assign(Variance=df['LocalEnergy_sq'] - df['LocalEnergy'] ** 2)
    return df


def _dmc_blocks(path, steps):
    """dmc.dat (one row per DMC step) reduced to per-block means.

    Returns (df, nrow, irregular): df has one row per complete block; nrow is the number of
    step rows read; irregular counts non-final blocks whose row count is not `steps` (a
    malformed file; verify_stats rejects it). Block of a row = (Index - first Index) // steps,
    so an Index starting at 0 or 1 both work. A trailing partial block (fewer than `steps`
    rows, as when the run was killed) is dropped, like an incomplete scalar.dat block.
    """
    raw = scalar_dat.read(str(path))
    steps = max(int(steps), 1)
    col = next((c for c in ('Index', 'index') if c in raw.columns), None)
    idx = raw[col].to_numpy(int) if col else np.arange(len(raw))
    dd = _clean(raw)
    if len(dd) == 0:
        return dd, 0, 0
    blk = (idx - idx[0]) // steps
    count = pd.Series(blk).value_counts().sort_index()
    means = dd.groupby(blk).mean()
    if count.iloc[-1] < steps:  # trailing partial block: drop
        means = means.iloc[:-1]
        count = count.iloc[:-1]
    irregular = int((count.to_numpy() != steps).sum())
    return means.reset_index(drop=True), len(raw), irregular


def _mean_error(df, nequil):
    """{col_mean, col_error} over blocks >= nequil, in column order (error right after its
    mean); NaN if nothing (or too little) is left."""
    sel = df.iloc[nequil:]
    me = mean_df.create(sel).iloc[0].to_dict() if len(sel) > 1 else {}
    out = {}
    for c in df.columns:
        out[c + '_mean'] = float(me.get(c + '_mean', np.nan))
        err = float(me.get(c + '_error', np.nan))
        if len(sel) > 1 and np.isfinite(err) and sel[c].nunique(dropna=False) == 1:
            err = 0.0  # constant column: qharv's correlation time is 0/0 (gives e.g. 4.7e-310)
        out[c + '_error'] = err
    return out


def _dsk(fp, nequil):
    """(kvecs, dsk_mean, dsk_err, rhok_mean, rhok_err) as returned by qharv (N*S(k)); NaN if too short."""
    nblock = fp['skall/rhok_e_e/value'].shape[0]
    if nblock - nequil <= 1:  # same cut as _mean_error
        kv = fp['skall/kpoints/value'][()]
        nan = np.full(len(kv), np.nan)
        return kv, nan, nan.copy(), nan + 0j, nan + 0j
    kv, dm, de = stat_h5.dsk_from_skall(fp, 'skall', nequil)
    _, rm, re = stat_h5.rhok_from_skall(fp, 'skall', nequil)
    return kv, dm, de, rm, re


def _obs_mean_error(v, nequil):
    """mean, error over blocks >= nequil along axis 0 (qharv me2d, real and imaginary parts
    separately); NaN of the per-block shape if <= 1 block is left."""
    v = np.asarray(v)
    cplx = np.iscomplexobj(v)
    shape = v.shape[1:]
    n = v.shape[0] - nequil
    if n <= 1:
        nan = np.full(shape, np.nan)
        return (nan + 0j, nan + 0j) if cplx else (nan, nan.copy())
    e = v[nequil:].reshape(n, -1)
    def me(a):  # constant columns: error exactly 0 (qharv's 0/0 correlation time)
        m, err = stat_h5.me2d(a)
        err = np.where(np.ptp(a, axis=0) == 0, 0.0, err)
        return m, err
    m, err = me(e.real.astype(float))
    if cplx:
        mi, ei = me(e.imag.astype(float))
        m, err = m + 1j * mi, err + 1j * ei
    return m.reshape(shape), err.reshape(shape)


def _obs_kpoints(obj):
    """An observable group's own k-vectors ('kpoints' dataset, or 'kpoints/value'), or None."""
    kp = obj.get('kpoints')
    if isinstance(kp, h5py.Group):
        kp = kp.get('value')
    return kp[()] if isinstance(kp, h5py.Dataset) else None


def _extra_obs(fp, scalar_cols):
    """Names of top-level stat.h5 groups with a 'value' dataset that are not scalar.dat columns."""
    out = []
    for name, obj in fp.items():
        if name == 'skall' or name in scalar_cols or not isinstance(obj, h5py.Group):
            continue
        if isinstance(obj.get('value'), h5py.Dataset):
            out.append(name)
    return out


def _raw_scalar_cols(path):
    """Column names of a raw scalar.dat / dmc.dat header minus index/Index, plus the derived
    Variance where _clean computes it."""
    with open(path) as fh:
        head = fh.readline().split()
    cols = set(head[1:] if head[:1] == ['#'] else head) - {'index', 'Index'}
    if {'LocalEnergy', 'LocalEnergy_sq'} <= cols:
        cols.add('Variance')
    return cols


def _missing_pairs(table_cols, raw_path):
    """Raw columns without both {c}_mean and {c}_error in the table; None if raw is unreadable."""
    try:
        raw = _raw_scalar_cols(raw_path)
    except OSError:
        return None
    have = set(table_cols)
    return sorted(c for c in raw if c + '_mean' not in have or c + '_error' not in have)


def _union(rows, first):
    cols = list(first)
    for r in rows:
        cols += [c for c in r if c not in cols]
    return cols


# --- workers (top-level, picklable: run in a spawn process pool or in-process) --------------

def _scalar_task(scalar_path, dmc_path, nequil, steps):
    """Statistics of one twist-series scalar.dat (and its dmc.dat, if given). Reads only."""
    df = _clean(scalar_dat.read(str(scalar_path)))
    nb = len(df)
    out = {'row': {'nblock': nb, 'nequil': int(nequil)}, 'drow': None, 'attrs': {}}
    out['row'].update(_mean_error(df, nequil))
    tcols = [c for c in TRACE_COLS if c in df.columns]
    trace = [df[c].to_numpy(float) for c in tcols]
    if dmc_path is not None:
        dd, dmc_rows, dmc_irregular = _dmc_blocks(dmc_path, steps)
        dmc_len = len(dd)  # complete blocks
        out['drow'] = {'nblock': dmc_len, 'nequil': int(nequil)}
        out['drow'].update(_mean_error(dd, nequil))
        for c in TRACE_DMC_COLS:
            if c in dd.columns:
                tcols.append(c)
                trace.append(dd[c].reindex(range(nb)).to_numpy(float))
        out['attrs']['dmc_dat_steps'] = int(steps)
        out['attrs']['dmc_dat_rows'] = int(dmc_rows)
        if dmc_len != nb:  # judged in blocks
            out['attrs']['dmc_dat_len_mismatch'] = dmc_len
        if dmc_irregular:
            out['attrs']['dmc_dat_irregular'] = dmc_irregular
    out['trace'] = np.column_stack(trace) if trace else np.empty((nb, 0))
    out['columns'] = tcols
    return out


def _stat_task(stat_path, scalar_path, nequil):
    """dsk/rhok and the other observables of one twist-series stat.h5. Reads only.

    Returns {'kp': raw skall k-vectors or None, 'sk': [(name, mean, error)],
    'obs': [(name, mean, error, kvecs or None)]}."""
    out = {'kp': None, 'sk': [], 'obs': []}
    with h5py.File(stat_path, 'r') as fp:
        if 'skall' in fp:
            kp = fp['skall/kpoints/value'][()]
            kv, dm, de, rm, re = _dsk(fp, nequil)
            if not np.array_equal(kv, kp):
                raise RuntimeError('%s: qharv dsk k-vectors differ from skall/kpoints/value'
                                   % stat_path)
            out['kp'] = kp
            out['sk'] = [('dsk', dm, de), ('rhok', rm, re)]
        for name in _extra_obs(fp, _raw_scalar_cols(scalar_path)):
            m, e = _obs_mean_error(fp[name + '/value'][()], nequil)
            out['obs'].append((name, m, e, _obs_kpoints(fp[name])))
    return out


class _Now:
    """A lazy in-process stand-in for a Future (executor=None): runs on .result()."""

    def __init__(self, fn, *a):
        self.fn, self.a = fn, a

    def result(self):
        return self.fn(*self.a)

    def cancel(self):
        return True


def _submit(executor, fn, *a):
    return _Now(fn, *a) if executor is None else executor.submit(fn, *a)


def _submit_series(L, R, s, nequil, executor, jobs):
    """Append (twist, scalar future, stat future or None) in twist order for one series to the
    caller-owned list jobs, so a failure mid-submit still lets the caller cancel what was sent."""
    for t in sorted(L.twists):
        if _skipped(R, t, s.index):
            continue
        sc = L.files[(t, s.index, '.scalar.dat')]
        dpath = L.files.get((t, s.index, '.dmc.dat')) if s.index == 1 else None
        fsc = _submit(executor, _scalar_task, sc, dpath, nequil, s.steps)
        sth5 = L.files.get((t, s.index, '.stat.h5'))
        fst = _submit(executor, _stat_task, sth5, sc, nequil) if sth5 is not None else None
        jobs.append((t, fsc, fst))


def _write_twist(s, t, fsc, fst, ft, fs, rows, drows, shared):
    """Write one twist's task results; return the (twist, kpoints) holder of /kvecs."""
    r = fsc.result()
    rows.append({'twist': t, **r['row']})
    if r['drow'] is not None:
        drows.append({'twist': t, **r['drow']})
    ds = ft.create_dataset('g%03d' % t, data=r['trace'])
    ds.attrs['columns'] = r['columns']
    for k, v in r['attrs'].items():
        ds.attrs[k] = v
    if fst is None:
        return shared
    st = fst.result()
    if st['kp'] is not None:
        if shared is None:
            shared = (t, st['kp'])
            fs.create_dataset('kvecs', data=st['kp'])
        elif not np.array_equal(st['kp'], shared[1]):
            raise RuntimeError(
                's%03d twist %d: skall kpoints differ from twist %d: cannot share /kvecs'
                % (s.index, t, shared[0]))
    for name, m, e in st['sk']:
        g = fs.require_group('g%03d/%s' % (t, name))
        g.create_dataset('mean', data=m)
        g.create_dataset('error', data=e)
    for name, m, e, kv in st['obs']:
        g = fs.require_group('g%03d/%s' % (t, name))
        g.create_dataset('mean', data=m)
        g.create_dataset('error', data=e)
        if kv is not None:
            g.create_dataset('kvecs', data=kv)
    return shared


def _write_series(s, nequil, jobs, tmp):
    """Write the per-series files to temp paths (tmp(name) -> path) from the task results,
    consumed in twist order; return the written names."""
    rows, drows = [], []
    shared = None  # (twist, skall kpoints) stored once as /kvecs
    with h5py.File(tmp(trace_name(s.index)), 'w') as ft, \
            h5py.File(tmp(stat_name(s.index)), 'w') as fs:
        for f in (ft, fs):
            f.attrs['nequil'] = int(nequil)
            f.attrs['series'] = int(s.index)
        for i in range(len(jobs)):
            t, fsc, fst = jobs[i]
            shared = _write_twist(s, t, fsc, fst, ft, fs, rows, drows, shared)
            jobs[i] = None  # written: release the futures and their results (M3)
            del fsc, fst
    names = [scalar_name(s.index), stat_name(s.index), trace_name(s.index)]
    write_table(tmp(scalar_name(s.index)), _union(rows, INT_COLS), rows)
    if drows:
        write_table(tmp(dmcdat_name(s.index)), _union(drows, INT_COLS), drows)
        names.append(dmcdat_name(s.index))
    return names


def _fsync(path):
    with open(path, 'rb') as fh:
        os.fsync(fh.fileno())


def _fsync_dir(d):
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class DataLossRefused(RuntimeError):
    """Existing statistics hold data whose raw files are gone; they were not replaced."""


def _refuse_data_loss(out, L):
    """DataLossRefused if an existing scalar table holds a (series, twist) that can no longer be
    recomputed, or cannot be read (so what it holds is unknown)."""
    lost = []
    for s in L.series:
        p = out / scalar_name(s.index)
        if not p.exists():
            continue
        try:
            twists = read_table(p)['twist'].tolist()
        except Exception as e:
            raise DataLossRefused('refusing to overwrite %s: unreadable (%s: %s)' % (
                p, type(e).__name__, e))
        for t in twists:
            raw = L.files.get((int(t), s.index, '.scalar.dat'))
            if raw is None or not raw.exists():
                lost.append('s%03d twist %d' % (s.index, t))
    if lost:
        raise DataLossRefused('refusing to overwrite %s: raw scalar.dat gone for %s' % (
            out, ', '.join(lost)))


def compute_stats(L, R, nequil, overwrite=False, executor=None):
    """Write the statistics files into <dmc>/qmcclean/ and return the path of stats.json.

    Every file is written under <name>.tmp.<pid>, fsynced and renamed into place; stats.json
    (the commit marker) goes last. When replacing, the old stats.json is removed before any
    file it lists is replaced, so an interruption never leaves a stats.json over mixed or
    missing files. Existing stats are returned untouched unless overwrite=True; existing
    tables are never replaced if they hold a twist/series whose scalar.dat is gone
    (DataLossRefused). A legacy stats.h5 is never read or touched.

    executor: a concurrent.futures executor (e.g. a spawn ProcessPoolExecutor) that runs the
    per-(series, twist, scalar|stat) read tasks; None = serial, in-process. Only this process
    writes, in twist order, so the output does not depend on the executor. Any task failure
    (including a killed worker: BrokenProcessPool) aborts the write like a serial exception.
    """
    out = Path(L.path) / SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    final = out / META
    if final.exists() and not overwrite:
        return final
    _refuse_data_loss(out, L)
    pid = os.getpid()
    temps = {}

    def tmp(name):
        p = out / ('%s.tmp.%d' % (name, pid))
        if p.exists():  # own stale tmp only
            p.unlink()
        temps[name] = p
        return p

    jobs = []
    try:
        for s in L.series:  # every series and source in flight at once
            sj = []
            jobs.append((s, sj))
            _submit_series(L, R, s, nequil.get(s.index, 0), executor, sj)
        files = []
        for s, sj in jobs:
            files += _write_series(s, nequil.get(s.index, 0), sj, tmp)
        for name in files:
            _fsync(temps[name])
        meta = {'prefix': L.prefix, 'driver': L.driver, 'nelec': int(L.nelec),
                'nequil': {str(k): int(v) for k, v in nequil.items()},
                'created': datetime.now(TZ).isoformat(timespec='seconds'),
                'qmcclean_version': __version__, 'files': files}
        mt = tmp(META)
        with open(mt, 'w') as fh:
            fh.write(json.dumps(meta, indent=1) + '\n')
            fh.flush()
            os.fsync(fh.fileno())
        if final.exists():  # uncommit before touching any file the old marker lists
            final.unlink()
            _fsync_dir(out)
        for name in files:
            os.replace(temps[name], out / name)
        _fsync_dir(out)
        os.replace(mt, final)
        _fsync_dir(out)
    except BaseException:
        live = []
        for _, sj in jobs:
            for job in sj:
                if job is None:
                    continue
                for f in job[1:]:
                    if f is not None and not f.cancel() and not isinstance(f, _Now):
                        live.append(f)  # already running: cannot be cancelled
        if live:  # bounded wait so a failed run leaves no reader of its files behind
            futures.wait(live, timeout=CANCEL_WAIT)
        for p in temps.values():
            if p.exists():
                p.unlink()
        raise
    return final


def load_scalar(stats_path, series):
    """The per-twist scalar table of one series as a DataFrame: twist, nblock, nequil,
    <col>_mean/_error, plus the dmc.dat columns as dmc_<col>_mean/_error merged on twist."""
    stats_path = Path(stats_path)
    files = read_meta(stats_path)['files']
    name = scalar_name(series)
    if name not in files:
        raise KeyError('%s not listed in %s' % (name, stats_path))
    df = read_table(stats_path.parent / name)
    dn = dmcdat_name(series)
    if dn in files:
        dd = read_table(stats_path.parent / dn)
        keep = [c for c in dd.columns if c not in INT_COLS]
        dd = dd[['twist'] + keep].rename(columns={c: 'dmc_' + c for c in keep})
        df = df.merge(dd, on='twist', how='left')
    return df


# --- verification --------------------------------------------------------------------------

def _expects_skall(L, R, t, s):
    sth5 = L.files.get((t, s, '.stat.h5'))
    if sth5 is not None:
        try:
            with h5py.File(sth5, 'r') as fp:
                return 'skall' in fp
        except Exception:
            pass
    if 'missing estimator skall in s%03d' % s in R.missing_files.get(t, []):
        return False  # this twist alone lacked skall (structure check)
    return 'skall' in L.estimators and 'skall' not in R.missing_estimators


def _first_raw_kpoints(L, R, s):
    """(twist, skall/kpoints/value) of the first twist whose raw stat.h5 has skall; None if
    no raw stat.h5 is left (or readable)."""
    for t in sorted(L.twists):
        if _skipped(R, t, s):
            continue
        p = L.files.get((t, s, '.stat.h5'))
        if p is None or not p.exists():
            continue
        try:
            with h5py.File(p, 'r') as fp:
                if 'skall' in fp:
                    return t, fp['skall/kpoints/value'][()]
        except Exception:
            continue
    return None


def old_layout(stats_path):
    """True if a listed stat.h5 has the pre-kvecs layout (no root /kvecs, per-twist
    dsk/rhok kvecs). Such stats must be recomputed; False if unreadable or current."""
    try:
        stats_path = Path(stats_path)
        for name in read_meta(stats_path)['files']:
            if not name.endswith('.stat.h5'):
                continue
            with h5py.File(stats_path.parent / name, 'r') as f:
                if 'kvecs' in f:
                    continue
                for g in f.values():
                    if isinstance(g, h5py.Group) and any(
                            isinstance(g.get(o), h5py.Group) and 'kvecs' in g[o]
                            for o in ('dsk', 'rhok')):
                        return True
    except Exception:
        return False
    return False


def _expected_obs(L, t, s):
    """{name: has own kpoints} of extra stat.h5 observables the raw files say must be saved;
    None if raw files are gone."""
    sth5 = L.files.get((t, s, '.stat.h5'))
    sc = L.files.get((t, s, '.scalar.dat'))
    if sth5 is None or sc is None or not sth5.exists() or not sc.exists():
        return None
    try:
        with h5py.File(sth5, 'r') as fp:
            return {n: _obs_kpoints(fp[n]) is not None
                    for n in _extra_obs(fp, _raw_scalar_cols(sc))}
    except Exception:
        return None


def _open_files(d, files, reasons):
    """Parse every listed file; return {name: DataFrame} for tables, {name: h5py.File} for h5."""
    tables, h5 = {}, {}
    for name in files:
        if not isinstance(name, str) or '/' in name or name.startswith('.'):
            reasons.append('stats.json: bad file name %r' % (name,))
            continue
        p = d / name
        if not p.is_file():
            reasons.append('%s listed in stats.json but missing' % name)
            continue
        try:
            if name.endswith('.h5'):
                h5[name] = h5py.File(p, 'r')
            else:
                df = read_table(p)
                bad = _table_problems(df, name)
                reasons.extend(bad)
                if not bad:
                    tables[name] = df
        except Exception as e:
            reasons.append('%s unreadable: %s: %s' % (name, type(e).__name__, e))
    return tables, h5


def _verify_series(L, R, s, tables, h5, reasons, want_nequil, counts=None):
    tag = 's%03d' % s.index
    sname, tname, hname, dname = (scalar_name(s.index), trace_name(s.index),
                                  stat_name(s.index), dmcdat_name(s.index))
    if sname not in tables:
        reasons.append('%s: scalar table missing (%s)' % (tag, sname))
        return
    for n in (tname, hname):
        if n not in h5:
            reasons.append('%s: %s missing' % (tag, n))
    ft, fs = h5.get(tname), h5.get(hname)
    kv = fs.get('kvecs') if fs is not None else None
    if kv is not None and not isinstance(kv, h5py.Dataset):
        reasons.append('%s: %s:/kvecs is not a dataset' % (tag, hname))
        kv = None
    if kv is not None and not (kv.ndim == 2 and kv.shape[1] == 3):
        reasons.append('%s: %s:/kvecs has shape %s, expected (n, 3)' % (tag, hname, kv.shape))
    nk = kv.shape[0] if kv is not None and kv.ndim >= 1 else None
    if kv is not None:
        ref = _first_raw_kpoints(L, R, s.index)
        if ref is not None and not (ref[1].shape == kv.shape and np.array_equal(ref[1], kv[()])):
            reasons.append('%s: %s:/kvecs differs from twist %d raw skall/kpoints/value' % (
                tag, hname, ref[0]))
    any_sk = False
    sc = tables[sname].set_index('twist')
    dm = tables[dname].set_index('twist') if dname in tables else None
    for t in L.twists:
        if _skipped(R, t, s.index):
            continue
        what = '%s twist %d' % (tag, t)
        if t not in sc.index:
            reasons.append('%s: no scalar row in %s' % (what, sname))
            continue
        nb = int(sc.at[t, 'nblock'])
        nequil = int(sc.at[t, 'nequil'])
        if want_nequil is not None and nequil != want_nequil:
            reasons.append('%s: row nequil %d differs from stats.json nequil %d' % (
                what, nequil, want_nequil))
        if ft is not None:
            g = 'g%03d' % t
            tr = ft.get(g)
            if not isinstance(tr, h5py.Dataset):
                reasons.append('%s: trace missing (%s:/%s)' % (what, tname, g))
            else:
                if tr.shape[0] != nb:
                    reasons.append('%s: trace has %d rows, expected %d blocks' % (
                        what, tr.shape[0], nb))
                if 'dmc_dat_len_mismatch' in tr.attrs:
                    reasons.append('%s: dmc.dat has %d blocks but scalar.dat has %d' % (
                        what, tr.attrs['dmc_dat_len_mismatch'], nb))
                if 'dmc_dat_irregular' in tr.attrs:
                    reasons.append('%s: dmc.dat has %d blocks without %d steps' % (
                        what, tr.attrs['dmc_dat_irregular'], tr.attrs['dmc_dat_steps']))
                raw_dmc = L.files.get((t, s.index, '.dmc.dat'))
                if s.index == 1 and raw_dmc is not None and raw_dmc.exists() \
                        and 'dmc_dat_steps' not in tr.attrs:
                    reasons.append('%s: dmc.dat exists but its trace columns were not saved' % what)
                if 'dmc_dat_steps' in tr.attrs:
                    want = int(tr.attrs.get('dmc_dat_len_mismatch', nb))
                    if dm is None:
                        reasons.append('%s: dmc.dat statistics missing (%s)' % (what, dname))
                    elif t not in dm.index:
                        reasons.append('%s: no row in %s' % (what, dname))
                    elif int(dm.at[t, 'nblock']) != want:
                        reasons.append('%s: %s has %d blocks, trace says %d' % (
                            what, dname, int(dm.at[t, 'nblock']), want))
                    if dm is not None and raw_dmc is not None and raw_dmc.exists():
                        miss = _missing_pairs(dm.columns, raw_dmc)
                        if miss:
                            reasons.append('%s: %s lacks %s for raw dmc.dat columns %s' % (
                                what, dname, '_mean/_error', ', '.join(miss)))
        le = float(sc.at[t, 'LocalEnergy_mean']) if 'LocalEnergy_mean' in sc.columns else np.nan
        if nb > nequil + 1 and not np.isfinite(le):
            reasons.append('%s: LocalEnergy_mean is NaN with %d blocks (nequil %d)' % (
                what, nb, nequil))
        for c in ('LocalEnergy_error', 'Variance_error'):
            if nb > nequil + 1 and c in sc.columns and not np.isfinite(float(sc.at[t, c])):
                reasons.append('%s: %s is NaN with %d blocks (nequil %d)' % (what, c, nb, nequil))
        raw = L.files.get((t, s.index, '.scalar.dat'))
        if raw is not None and raw.exists():
            miss = _missing_pairs(sc.columns, raw)
            if miss:
                reasons.append('%s: %s lacks %s for raw scalar.dat columns %s' % (
                    what, sname, '_mean/_error', ', '.join(miss)))
            n, err = (counts or {}).get(raw) or _probe('scalar', raw)
            if err is not None:  # fail closed: the last gate before deletion
                reasons.append('%s: raw scalar.dat line count failed (%s)' % (what, err))
            elif n != nb:
                reasons.append('%s: %d blocks saved but scalar.dat has %d' % (what, nb, n))
        if fs is None:
            continue
        g = fs.get('g%03d' % t)
        want = _expects_skall(L, R, t, s.index)
        for name in ('dsk', 'rhok'):
            p = g.get(name) if g is not None else None
            any_sk = any_sk or p is not None
            have = isinstance(p, h5py.Group) and all(
                isinstance(p.get(k), h5py.Dataset) and p[k].ndim >= 1 for k in ('mean', 'error'))
            if isinstance(p, h5py.Group) and 'kvecs' in p:
                reasons.append('%s: stray per-twist %s/kvecs (k-vectors belong in /kvecs)' % (
                    what, name))
            if want and not have:
                reasons.append('%s: %s missing' % (what, name))
            elif have and nk is None:
                reasons.append('%s: %s present but %s:/kvecs missing' % (what, name, hname))
            elif have and len({nk, p['mean'].shape[0], p['error'].shape[0]}) != 1:
                reasons.append('%s: %s mean/error lengths %d/%d differ from %d kvecs' % (
                    what, name, p['mean'].shape[0], p['error'].shape[0], nk))
            elif not want and p is not None:
                reasons.append('%s: %s present but no skall' % (what, name))
        for name, own_k in (_expected_obs(L, t, s.index) or {}).items():
            p = g.get(name) if g is not None else None
            if not (isinstance(p, h5py.Group) and all(
                    isinstance(p.get(k), h5py.Dataset) for k in ('mean', 'error'))):
                reasons.append('%s: %s missing in %s' % (what, name, hname))
            elif own_k and not isinstance(p.get('kvecs'), h5py.Dataset):
                reasons.append('%s: %s kvecs missing in %s' % (what, name, hname))
        if g is not None:
            for name, p in g.items():
                if name in ('dsk', 'rhok') or not isinstance(p, h5py.Group):
                    continue
                if not all(isinstance(p.get(k), h5py.Dataset) for k in ('mean', 'error')):
                    reasons.append('%s: %s incomplete in %s' % (what, name, hname))
                elif p['mean'].shape != p['error'].shape:
                    reasons.append('%s: %s mean/error shapes differ' % (what, name))
                elif 'kvecs' in p and not (
                        isinstance(p['kvecs'], h5py.Dataset) and p['mean'].ndim >= 1
                        and p['kvecs'].shape[0] == p['mean'].shape[0]):
                    reasons.append('%s: %s kvecs inconsistent with its mean/error' % (what, name))
    if nk is not None and not any_sk:
        reasons.append('%s: %s:/kvecs present but no twist has dsk/rhok' % (tag, hname))


def _raw_counts(L, executor):
    """{raw scalar.dat path: (line count, error)} counted on the executor; {} on any failure
    (the caller then counts serially)."""
    if executor is None:
        return {}
    try:
        paths = [p for k, p in L.files.items() if k[2] == '.scalar.dat' and p.exists()]
        return dict(zip(paths, executor.map(_probe, ['scalar'] * len(paths), paths, chunksize=8)))
    except Exception:
        return {}


def verify_stats(stats_path, L, R, executor=None):
    """Reasons the statistics are incomplete or unreadable; [] means they can replace the raw
    files. Never raises. executor: optional, counts raw scalar.dat lines in parallel."""
    reasons = []
    h5 = {}
    try:
        stats_path = Path(stats_path)
        try:
            meta = read_meta(stats_path)
        except Exception as e:
            return ['stats.json unreadable: %s: %s' % (type(e).__name__, e)]
        for k in META_KEYS:
            if k not in meta:
                reasons.append('stats.json: %s missing' % k)
        nq = meta.get('nequil')
        if not (isinstance(nq, dict) and all(isinstance(v, int) for v in nq.values())):
            reasons.append('stats.json: nequil is not a dict of integers')
        files = meta.get('files')
        if not (isinstance(files, list) and all(isinstance(f, str) for f in files)):
            reasons.append('stats.json: files is not a list of names')
            return reasons
        if len(set(files)) != len(files):
            reasons.append('stats.json: duplicate file names')
        tables, h5 = _open_files(stats_path.parent, files, reasons)
        counts = _raw_counts(L, executor)
        for s in L.series:
            wn = nq.get(str(s.index), 0) if isinstance(nq, dict) else None
            _verify_series(L, R, s, tables, h5, reasons, wn if isinstance(wn, int) else None,
                           counts)
    except Exception as e:
        reasons.append('stats unreadable: %s: %s' % (type(e).__name__, e))
    finally:
        for f in h5.values():
            try:
                f.close()
            except Exception:
                pass
    return reasons
