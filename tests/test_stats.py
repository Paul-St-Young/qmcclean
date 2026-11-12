import json

import h5py
import numpy as np
import pandas as pd
import pytest

from qharv.reel import scalar_dat
from qharv.sieve.scalar_df import mean_error_scalar_df

from qmcclean import stats
from qmcclean.config import NEQUIL
from qmcclean.layout import discover_dmc
from qmcclean.structure import check_structure

from .fakes import edit_table, make_dmc


def run(tmp_path, skip_opt=False, **kw):
    dmc = make_dmc(tmp_path, **kw)
    L = discover_dmc(dmc)
    R = check_structure(L, skip_opt=skip_opt)
    p = stats.compute_stats(L, R, NEQUIL)
    return dmc, L, R, p


def test_stats_roundtrip(tmp_path):
    dmc, L, R, p = run(tmp_path)
    assert p == dmc / 'qmcclean' / 'stats.json'
    assert not list((dmc / 'qmcclean').glob('*.tmp*'))
    df = stats.load_scalar(p, 1)
    assert len(df) == 20
    ref = mean_error_scalar_df(scalar_dat.read(str(dmc / 'dmc.g000.s001.scalar.dat')), 16)
    row = df[df.twist == 0].iloc[0]
    assert abs(row['LocalEnergy_mean'] - ref['LocalEnergy_mean'].iloc[0]) < 1e-12
    assert abs(row['LocalEnergy_error'] - ref['LocalEnergy_error'].iloc[0]) < 1e-12
    assert abs(row['Variance_mean'] - ref['Variance_mean'].iloc[0]) < 1e-12
    assert list(df.columns).count('Variance_mean') == 1
    assert list(df.columns).count('Variance_error') == 1
    assert 'dmc_LocalEnergy_mean' in df.columns and 'dmc_Weight_mean' in df.columns
    assert 'dmc_LocalEnergy_mean' not in stats.load_scalar(p, 0).columns
    assert df.nblock.iloc[0] == 30
    meta = json.loads(p.read_text())
    assert meta['driver'] == 'legacy' and meta['nelec'] == 8
    assert meta['nequil'] == {'0': 32, '1': 16}
    assert meta['created'].endswith(('-04:00', '-05:00'))
    assert 'qmcclean_version' in meta and 'prefix' in meta
    assert meta['files'] == ['dmc.s000.scalar.dat', 'dmc.s000.stat.h5', 'dmc.s000.trace.h5',
                             'dmc.s001.scalar.dat', 'dmc.s001.stat.h5', 'dmc.s001.trace.h5',
                             'dmc.s001.dmc.dat']
    assert all((p.parent / n).is_file() for n in meta['files'])
    assert stats.verify_stats(p, L, R) == []


def test_trace_lengths(tmp_path):
    _, L, R, p = run(tmp_path, short={5: 10})
    with h5py.File(p.parent / 'dmc.s001.trace.h5', 'r') as f, \
            h5py.File(p.parent / 'dmc.s000.trace.h5', 'r') as f0:
        assert f['g005'].shape[0] == 10
        assert f['g004'].shape[0] == 30
        assert f0['g005'].shape[0] == 40
        cols = list(f['g004'].attrs['columns'])
        assert cols[:4] == ['LocalEnergy', 'Variance', 'Kinetic', 'BlockWeight']
        assert 'TrialEnergy' in cols
    df = stats.load_scalar(p, 1)
    r = df[df.twist == 5].iloc[0]
    assert r.nblock == 10 and np.isnan(r.LocalEnergy_mean) and np.isnan(r.LocalEnergy_error)
    assert stats.verify_stats(p, L, R) == []


def test_dsk_present_iff_skall(tmp_path):
    _, L, R, p = run(tmp_path / 'a', skall=True)
    with h5py.File(p.parent / 'dmc.s001.stat.h5', 'r') as f:
        g = f['g003/dsk']
        assert set(g) == {'mean', 'error'}
        assert f['kvecs'].shape == (6, 3)
        assert 'g003/rhok/mean' in f
        assert np.iscomplexobj(f['g003/rhok/mean'][()])
        assert f.attrs['nequil'] == 16 and f.attrs['series'] == 1
    assert stats.verify_stats(p, L, R) == []
    _, L, R, p = run(tmp_path / 'b', skall=False, skip_opt=True)
    assert R.overridden
    with h5py.File(p.parent / 'dmc.s001.stat.h5', 'r') as f:
        assert not any('dsk' in g or 'rhok' in g for g in f.values())
    assert stats.verify_stats(p, L, R) == []


def test_verify_detects_truncation(tmp_path):
    _, L, R, p = run(tmp_path)
    with h5py.File(p.parent / 'dmc.s001.trace.h5', 'a') as f:
        del f['g007']
    assert stats.verify_stats(p, L, R) == ['s001 twist 7: trace missing (dmc.s001.trace.h5:/g007)']


def test_verify_detects_short_trace_and_missing_dsk(tmp_path):
    _, L, R, p = run(tmp_path)
    with h5py.File(p.parent / 'dmc.s000.trace.h5', 'a') as f:
        d = f['g002'][:10]
        del f['g002']
        f['g002'] = d
    with h5py.File(p.parent / 'dmc.s001.stat.h5', 'a') as f:
        del f['g004/dsk']
    rs = stats.verify_stats(p, L, R)
    assert len(rs) == 2
    assert 'trace has 10 rows, expected 40' in rs[0] and 'dsk missing' in rs[1]


def test_verify_missing_or_garbage(tmp_path):
    dmc, L, R, p = run(tmp_path)
    assert stats.verify_stats(dmc / 'nope.json', L, R)
    bad = dmc / 'bad.json'
    bad.write_bytes(b'garbage')
    assert 'unreadable' in stats.verify_stats(bad, L, R)[0]


def test_missing_twist_skipped(tmp_path):
    dmc = make_dmc(tmp_path)
    (dmc / 'dmc.g003.s001.scalar.dat').unlink()
    L = discover_dmc(dmc)
    R = check_structure(L)
    p = stats.compute_stats(L, R, NEQUIL)
    assert 3 not in set(stats.load_scalar(p, 1).twist)
    assert 3 in set(stats.load_scalar(p, 0).twist)
    assert 3 not in set(stats.load_scalar(p, 1).twist)
    assert stats.verify_stats(p, L, R) == []


def test_failure_leaves_no_final(tmp_path, monkeypatch):
    dmc = make_dmc(tmp_path)
    L = discover_dmc(dmc)
    R = check_structure(L)
    def boom(*a, **k):
        raise RuntimeError('boom')
    monkeypatch.setattr(stats, '_dsk', boom)
    with pytest.raises(RuntimeError):
        stats.compute_stats(L, R, NEQUIL)
    out = dmc / 'qmcclean'
    assert not (out / 'stats.json').exists() and not list(out.glob('*.tmp*'))


def test_second_call_keeps_existing(tmp_path):
    _, L, R, p = run(tmp_path)
    before, mt = p.read_bytes(), p.stat().st_mtime_ns
    assert stats.compute_stats(L, R, NEQUIL) == p
    assert p.stat().st_mtime_ns == mt and p.read_bytes() == before


def test_overwrite_refuses_data_loss(tmp_path):
    dmc, L, R, p = run(tmp_path)
    before = p.read_bytes()
    (dmc / 'dmc.g004.s001.scalar.dat').unlink()
    L2 = discover_dmc(dmc)
    R2 = check_structure(L2)
    with pytest.raises(RuntimeError, match='s001 twist 4'):
        stats.compute_stats(L2, R2, NEQUIL, overwrite=True)
    assert p.read_bytes() == before
    assert not list(p.parent.glob('*.tmp*'))


def test_overwrite_allowed_when_complete(tmp_path):
    dmc, L, R, p = run(tmp_path)
    assert stats.compute_stats(L, R, NEQUIL, overwrite=True) == p
    assert stats.verify_stats(p, L, R) == []


def test_verify_rejects_nan_energy_on_long_twist(tmp_path):
    dmc, L, R, p = run(tmp_path)
    def nan3(df):
        df.loc[df.twist == 3, 'LocalEnergy_mean'] = np.nan
    edit_table(p, 'dmc.s001.scalar.dat', nan3)
    rs = stats.verify_stats(p, L, R)
    assert len(rs) == 1 and 'twist 3' in rs[0] and 'NaN' in rs[0]


def test_nblock_nequil_plus_one_is_nan_not_crash(tmp_path):
    _, L, R, p = run(tmp_path, short={5: 17})
    r = stats.load_scalar(p, 1).query('twist == 5').iloc[0]
    assert r.nblock == 17 and np.isnan(r.LocalEnergy_mean)
    with h5py.File(p.parent / 'dmc.s001.stat.h5', 'r') as f:
        assert np.isnan(f['g005/dsk/mean'][()]).all()
        assert np.isfinite(f['g006/dsk/mean'][()]).all()
    assert stats.verify_stats(p, L, R) == []
    _, L, R, p = run(tmp_path / 'b', short={5: 18})
    r = stats.load_scalar(p, 1).query('twist == 5').iloc[0]
    assert np.isfinite(r.LocalEnergy_mean) and np.isfinite(r.LocalEnergy_error)
    assert stats.verify_stats(p, L, R) == []


def test_dmc_dat_length_mismatch_recorded_and_reported(tmp_path):
    # dmc.dat has steps=4 rows per block; cut 5 rows: 28 full blocks + a 3-row partial (dropped)
    dmc = make_dmc(tmp_path)
    f = dmc / 'dmc.g002.s001.dmc.dat'
    lines = f.read_text().splitlines()
    assert len(lines) == 1 + 30 * 4
    f.write_text('\n'.join(lines[:-5]) + '\n')
    L = discover_dmc(dmc)
    R = check_structure(L)
    p = stats.compute_stats(L, R, NEQUIL)
    with h5py.File(p.parent / 'dmc.s001.trace.h5', 'r') as h:
        assert h['g002'].attrs['dmc_dat_len_mismatch'] == 28
    rs = stats.verify_stats(p, L, R)
    assert len(rs) == 1 and 'dmc.dat has 28 blocks but scalar.dat has 30' in rs[0]


@pytest.mark.parametrize('steps', [1, 4, 7])
def test_dmc_dat_per_step_rows_reduced_to_blocks(tmp_path, steps):
    """C1: real dmc.dat has one row per DMC step; stats must work in blocks."""
    dmc, L, R, p = run(tmp_path, steps=(98, steps))
    assert L.series[1].steps == steps
    nrow = len((dmc / 'dmc.g003.s001.dmc.dat').read_text().splitlines()) - 1
    assert nrow == 30 * steps
    assert stats.verify_stats(p, L, R) == []
    raw = scalar_dat.read(str(dmc / 'dmc.g003.s001.dmc.dat'))
    blk = raw.drop(columns='Index').groupby(np.arange(nrow) // steps).mean()
    sc = scalar_dat.read(str(dmc / 'dmc.g003.s001.scalar.dat'))
    with h5py.File(p.parent / 'dmc.s001.trace.h5', 'r') as f:
        tr = f['g003']
        cols = list(tr.attrs['columns'])
        assert tr.shape[0] == 30 and tr.attrs['dmc_dat_rows'] == nrow
        assert 'dmc_dat_len_mismatch' not in tr.attrs
        w = tr[:, cols.index('Weight')]
        assert np.allclose(w, blk['Weight'].to_numpy())
        assert np.allclose(w * steps, sc['BlockWeight'].to_numpy(), rtol=1e-5)
    row = stats.load_scalar(p, 1).query('twist == 3').iloc[0]
    ref = mean_error_scalar_df(blk.rename_axis('index').reset_index(), NEQUIL[1])
    assert row['dmc_LocalEnergy_mean'] == pytest.approx(ref['LocalEnergy_mean'].iloc[0], abs=1e-12)
    assert row['dmc_LocalEnergy_error'] == pytest.approx(ref['LocalEnergy_error'].iloc[0], abs=1e-12)


def test_dmc_dat_short_twist_with_partial_trailing_block(tmp_path):
    """A killed twist: 23 full blocks plus a partial one in dmc.dat; scalar.dat has 23 blocks."""
    dmc, L, R, p = run(tmp_path, short={4: 23}, partial={4: 3}, steps=(98, 5))
    assert len((dmc / 'dmc.g004.s001.dmc.dat').read_text().splitlines()) - 1 == 23 * 5 + 3
    assert stats.verify_stats(p, L, R) == []
    with h5py.File(p.parent / 'dmc.s001.trace.h5', 'r') as f:
        tr = f['g004']
        assert tr.shape[0] == 23 and 'dmc_dat_len_mismatch' not in tr.attrs
    row = stats.load_scalar(p, 1).query('twist == 4').iloc[0]
    assert row.nblock == 23 and np.isfinite(row['dmc_LocalEnergy_mean'])


def test_dmc_dat_index_from_one(tmp_path):
    dmc = make_dmc(tmp_path, steps=(98, 4))
    f = dmc / 'dmc.g001.s001.dmc.dat'
    lines = f.read_text().splitlines()
    out = [lines[0]] + [str(int(r.split()[0]) + 1).rjust(10) + r[10:] for r in lines[1:]]
    f.write_text('\n'.join(out) + '\n')
    L = discover_dmc(dmc)
    R = check_structure(L)
    p = stats.compute_stats(L, R, NEQUIL)
    assert stats.verify_stats(p, L, R) == []


def test_dmc_dat_irregular_block_rejected(tmp_path):
    dmc = make_dmc(tmp_path, steps=(98, 4))
    f = dmc / 'dmc.g001.s001.dmc.dat'
    lines = f.read_text().splitlines()
    f.write_text('\n'.join(lines[:10] + lines[11:]) + '\n')  # a step missing mid-file
    L = discover_dmc(dmc)
    R = check_structure(L)
    p = stats.compute_stats(L, R, NEQUIL)
    rs = stats.verify_stats(p, L, R)
    assert any('twist 1' in r and 'dmc.dat' in r for r in rs)


def test_verify_dsk_length_mismatch(tmp_path):
    dmc, L, R, p = run(tmp_path)
    with h5py.File(p.parent / 'dmc.s001.stat.h5', 'a') as f:
        d = f['g001/dsk/error'][:3]
        del f['g001/dsk/error']
        f['g001/dsk/error'] = d
    rs = stats.verify_stats(p, L, R)
    assert len(rs) == 1 and 'dsk mean/error lengths 6/3 differ from 6 kvecs' in rs[0]


def test_verify_ok_after_raw_files_deleted(tmp_path):
    dmc, L, R, p = run(tmp_path)
    for f in dmc.glob('dmc.g005.*'):
        if f.suffix in ('.dat', '.h5'):
            f.unlink()
    assert stats.verify_stats(p, L, R) == []


# --- split layout: text tables, stat.h5 extras, commit marker, legacy stats.h5 ---

def test_scalar_dat_round_trip(tmp_path):
    dmc, L, R, p = run(tmp_path, short={5: 10})
    f = p.parent / 'dmc.s001.scalar.dat'
    head = f.read_text().splitlines()[0].split()
    assert head[0] == '#' and head[1:4] == ['twist', 'nblock', 'nequil']
    raw = [c for c in scalar_dat.read(str(dmc / 'dmc.g000.s001.scalar.dat')).columns if c != 'index']
    assert head[4:] == [c + s for c in raw for s in ('_mean', '_error')]
    df = scalar_dat.read(str(f))
    assert list(df.columns) == head[1:]
    for c in ('twist', 'nblock', 'nequil'):
        assert df[c].dtype.kind == 'i'
    assert df.twist.tolist() == sorted(df.twist.tolist()) == list(range(20))
    r = df[df.twist == 5].iloc[0]
    assert r.nblock == 10 and r.nequil == 16 and np.isnan(r.LocalEnergy_mean)
    # full precision: identical to the in-memory value
    ref = mean_error_scalar_df(scalar_dat.read(str(dmc / 'dmc.g000.s001.scalar.dat')), 16)
    assert df.LocalEnergy_mean.iloc[0] == ref['LocalEnergy_mean'].iloc[0]
    # columns aligned: every line has the same length
    assert len({len(x) for x in f.read_text().splitlines()}) == 1


def test_hand_deleted_row_reported_and_dropped(tmp_path):
    _, L, R, p = run(tmp_path)
    edit_table(p, 'dmc.s001.scalar.dat', lambda df: df[df.twist != 6])
    rs = stats.verify_stats(p, L, R)
    assert rs == ['s001 twist 6: no scalar row in dmc.s001.scalar.dat']
    tw = set(stats.load_scalar(p, 1).twist)
    assert 6 not in tw and len(tw) == 19
    assert 6 in set(stats.load_scalar(p, 0).twist)


def test_hand_appended_duplicate_or_bad_columns_rejected(tmp_path):
    _, L, R, p = run(tmp_path)
    edit_table(p, 'dmc.s000.scalar.dat', lambda df: pd.concat([df, df.iloc[[2]]]))
    assert any('duplicate twists [2]' in r for r in stats.verify_stats(p, L, R))
    _, L, R, p = run(tmp_path / 'b')
    edit_table(p, 'dmc.s001.scalar.dat', lambda df: df.drop(columns='Kinetic_error'))
    assert any('Kinetic_mean not followed by Kinetic_error' in r
               for r in stats.verify_stats(p, L, R))
    _, L, R, p = run(tmp_path / 'c')
    f = p.parent / 'dmc.s001.scalar.dat'
    f.write_text(f.read_text()[:-200])  # truncated last line
    assert stats.verify_stats(p, L, R)


def test_listed_file_missing_rejected(tmp_path):
    _, L, R, p = run(tmp_path)
    (p.parent / 'dmc.s000.stat.h5').unlink()
    rs = stats.verify_stats(p, L, R)
    assert 'dmc.s000.stat.h5 listed in stats.json but missing' in rs
    _, L, R, p = run(tmp_path / 'b')
    meta = json.loads(p.read_text())
    del meta['nelec']
    meta['files'].remove('dmc.s001.trace.h5')
    p.write_text(json.dumps(meta))
    rs = stats.verify_stats(p, L, R)
    assert 'stats.json: nelec missing' in rs and 's001: dmc.s001.trace.h5 missing' in rs


def test_dmc_dat_table(tmp_path):
    _, L, R, p = run(tmp_path)
    df = scalar_dat.read(str(p.parent / 'dmc.s001.dmc.dat'))
    assert list(df.columns[:3]) == ['twist', 'nblock', 'nequil']
    assert 'NumOfWalkers_mean' in df and 'NumOfWalkers_error' in df
    assert list(df.columns).index('NumOfWalkers_error') == list(df.columns).index('NumOfWalkers_mean') + 1
    assert not (p.parent / 'dmc.s000.dmc.dat').exists()
    d = stats.load_scalar(p, 1)
    row = d[d.twist == 2].iloc[0]
    assert row['dmc_NumOfWalkers_mean'] == df[df.twist == 2].NumOfWalkers_mean.iloc[0]
    assert 'dmc_nblock' not in d and 'dmc_twist' not in d
    edit_table(p, 'dmc.s001.dmc.dat', lambda df: df[df.twist != 8])
    assert stats.verify_stats(p, L, R) == ['s001 twist 8: no row in dmc.s001.dmc.dat']


def test_stat_h5_extra_observable(tmp_path):
    dmc = make_dmc(tmp_path)
    for t in range(20):
        for s in (0, 1):
            with h5py.File(dmc / ('dmc.g%03d.s%03d.stat.h5' % (t, s)), 'a') as f:
                nb = f['LocalEnergy/value'].shape[0]
                f['nofk/value'] = np.arange(nb * 5, dtype=float).reshape(nb, 5) + t
                f['cplx/value'] = (1 + 2j) * np.linspace(0, 1, nb * 2).reshape(nb, 2)
    L = discover_dmc(dmc)
    R = check_structure(L)
    p = stats.compute_stats(L, R, NEQUIL)
    with h5py.File(p.parent / 'dmc.s001.stat.h5', 'r') as f:
        g = f['g003']
        assert set(g) == {'dsk', 'rhok', 'nofk', 'cplx'}  # LocalEnergy, Kinetic, force skipped
        assert g['nofk/mean'].shape == (5,) and g['nofk/error'].shape == (5,)
        v = np.arange(30 * 5, dtype=float).reshape(30, 5) + 3
        assert np.allclose(g['nofk/mean'][()], v[16:].mean(0))
        assert np.all(g['nofk/error'][()] > 0)
        assert np.iscomplexobj(g['cplx/mean'][()]) and np.iscomplexobj(g['cplx/error'][()])
    assert stats.verify_stats(p, L, R) == []
    with h5py.File(p.parent / 'dmc.s001.stat.h5', 'a') as f:
        del f['g004/nofk']
    assert stats.verify_stats(p, L, R) == ['s001 twist 4: nofk missing in dmc.s001.stat.h5']


def test_extra_observable_too_short_is_nan(tmp_path):
    dmc = make_dmc(tmp_path, short={5: 17})
    with h5py.File(dmc / 'dmc.g005.s001.stat.h5', 'a') as f:
        f['nofk/value'] = np.ones((17, 5))
    L = discover_dmc(dmc)
    R = check_structure(L)
    p = stats.compute_stats(L, R, NEQUIL)
    with h5py.File(p.parent / 'dmc.s001.stat.h5', 'r') as f:
        assert np.isnan(f['g005/nofk/mean'][()]).all() and f['g005/nofk/mean'].shape == (5,)


@pytest.mark.parametrize('target', ['write_table', '_fsync_dir', 'replace'])
def test_crash_leaves_no_marker_and_no_temps(tmp_path, monkeypatch, target):
    dmc = make_dmc(tmp_path)
    L = discover_dmc(dmc)
    R = check_structure(L)
    def boom(*a, **k):
        raise RuntimeError('boom')
    if target == 'replace':  # dies after some files were renamed into place
        real, n = stats.os.replace, []
        def boom(a, b):  # noqa: F811
            n.append(b)
            if len(n) == 3:
                raise RuntimeError('boom')
            return real(a, b)
        monkeypatch.setattr(stats.os, 'replace', boom)
    else:
        monkeypatch.setattr(stats, target, boom)
    with pytest.raises(RuntimeError, match='boom'):
        stats.compute_stats(L, R, NEQUIL)
    out = dmc / 'qmcclean'
    assert not (out / 'stats.json').exists() and not list(out.glob('*.tmp*'))
    monkeypatch.undo()
    p = stats.compute_stats(L, R, NEQUIL)  # next run recomputes from scratch
    assert stats.verify_stats(p, L, R) == []


def test_legacy_stats_h5_untouched(tmp_path):
    dmc = make_dmc(tmp_path)
    out = dmc / 'qmcclean'
    out.mkdir()
    with h5py.File(out / 'stats.h5', 'w') as f:
        f.create_group('meta').attrs['nequil'] = '{"0": 1, "1": 1}'
    before = (out / 'stats.h5').read_bytes()
    L = discover_dmc(dmc)
    R = check_structure(L)
    p = stats.compute_stats(L, R, NEQUIL)
    assert p == out / 'stats.json' and stats.verify_stats(p, L, R) == []
    stats.compute_stats(L, R, NEQUIL, overwrite=True)
    assert (out / 'stats.h5').read_bytes() == before


def test_orphan_tables_without_marker_still_protected(tmp_path):
    # stats.json removed (e.g. crash during an overwrite) after raw files of twist 4 are gone
    dmc, L, R, p = run(tmp_path)
    p.unlink()
    for f in dmc.glob('dmc.g004.s001.*'):
        f.unlink()
    L2 = discover_dmc(dmc)
    R2 = check_structure(L2)
    before = (p.parent / 'dmc.s001.scalar.dat').read_bytes()
    with pytest.raises(RuntimeError, match='s001 twist 4'):
        stats.compute_stats(L2, R2, NEQUIL)
    assert (p.parent / 'dmc.s001.scalar.dat').read_bytes() == before


# --- fix round 1 ---

def test_verify_requires_every_raw_column(tmp_path):
    _, L, R, p = run(tmp_path)
    edit_table(p, 'dmc.s001.scalar.dat',
               lambda df: df.drop(columns=['Kinetic_mean', 'Kinetic_error']))
    rs = stats.verify_stats(p, L, R)
    assert len(rs) == 20 and all('raw scalar.dat columns Kinetic' in r for r in rs)
    _, L, R, p = run(tmp_path / 'b')
    edit_table(p, 'dmc.s001.dmc.dat',
               lambda df: df.drop(columns=['TrialEnergy_mean', 'TrialEnergy_error']))
    rs = stats.verify_stats(p, L, R)
    assert len(rs) == 20 and all('raw dmc.dat columns TrialEnergy' in r for r in rs)


def test_verify_rejects_row_nequil_mismatch(tmp_path):
    _, L, R, p = run(tmp_path)
    def n0(df):
        df.loc[df.twist == 2, 'nequil'] = 0
    edit_table(p, 'dmc.s001.scalar.dat', n0)
    assert stats.verify_stats(p, L, R) == [
        's001 twist 2: row nequil 0 differs from stats.json nequil 16']


def test_verify_rejects_nan_error_on_long_twist(tmp_path):
    _, L, R, p = run(tmp_path)
    def nan(df):
        df.loc[df.twist == 4, 'LocalEnergy_error'] = np.nan
        df.loc[df.twist == 5, 'Variance_error'] = np.nan
    edit_table(p, 'dmc.s001.scalar.dat', nan)
    rs = stats.verify_stats(p, L, R)
    assert len(rs) == 2
    assert 'twist 4: LocalEnergy_error is NaN' in rs[0] and 'twist 5: Variance_error is NaN' in rs[1]


def test_mean_error_constant_columns():
    n = 40
    rng = np.random.default_rng(3)
    near = np.full(n, 3.0)
    near[25] += 1e-15
    df = pd.DataFrame({'const': np.full(n, -2.0), 'near': near,
                       'nan': np.full(n, np.nan), 'noise': rng.standard_normal(n)})
    me = stats._mean_error(df, 16)
    assert me['const_mean'] == -2.0 and me['const_error'] == 0.0
    assert me['near_error'] > 0 and np.isfinite(me['near_error'])
    assert np.isnan(me['nan_mean']) and np.isnan(me['nan_error'])
    assert me['noise_error'] > 0
    v = np.column_stack([np.full(n, 5.0), near, np.full(n, np.nan), rng.standard_normal(n)])
    m, e = stats._obs_mean_error(v, 16)
    assert e[0] == 0.0 and m[0] == 5.0
    assert e[1] > 0 and np.isfinite(e[1])
    assert np.isnan(m[2]) and np.isnan(e[2])
    assert e[3] > 0
    m, e = stats._obs_mean_error(v * (1 + 1j), 16)
    assert e[0] == 0.0 and np.iscomplexobj(e)


def test_crash_during_overwrite_removes_old_marker(tmp_path, monkeypatch):
    _, L, R, p = run(tmp_path)
    real, n = stats.os.replace, []
    def boom(a, b):
        n.append(b)
        if len(n) == 2:
            raise RuntimeError('boom')
        return real(a, b)
    monkeypatch.setattr(stats.os, 'replace', boom)
    with pytest.raises(RuntimeError):
        stats.compute_stats(L, R, {0: 30, 1: 10}, overwrite=True)
    monkeypatch.undo()
    assert not p.exists() and not list(p.parent.glob('*.tmp*'))
    assert stats.verify_stats(p, L, R)  # no marker: not verified, nothing may be deleted
    q = stats.compute_stats(L, R, {0: 30, 1: 10})  # no marker: computed afresh
    assert q == p and stats.verify_stats(p, L, R) == []
    assert json.loads(p.read_text())['nequil'] == {'0': 30, '1': 10}
    assert set(stats.load_scalar(p, 1).nequil) == {10}


# --- shared skall kvecs ---

def test_skall_kvecs_stored_once(tmp_path):
    from .fakes import KPOINTS
    _, L, R, p = run(tmp_path)
    for s in (0, 1):
        with h5py.File(p.parent / ('dmc.s%03d.stat.h5' % s), 'r') as f:
            assert np.array_equal(f['kvecs'][()], KPOINTS)
            for name, g in f.items():
                if name == 'kvecs':
                    continue
                for obs in ('dsk', 'rhok'):
                    assert set(g[obs]) == {'mean', 'error'}
                    assert len(g[obs]['mean']) == len(KPOINTS)
            n = []
            f.visit(lambda x: n.append(x) if x.endswith('kvecs') else None)
            assert n == ['kvecs']
    assert stats.verify_stats(p, L, R) == []


def test_mismatched_skall_kpoints_refused(tmp_path):
    dmc = make_dmc(tmp_path)
    with h5py.File(dmc / 'dmc.g006.s001.stat.h5', 'a') as f:
        d = f['skall/kpoints/value']
        a = d[()]
        a[2, 0] += 1e-12
        d[...] = a
    L = discover_dmc(dmc)
    R = check_structure(L)
    with pytest.raises(RuntimeError, match='s001 twist 6: skall kpoints differ from twist 0'):
        stats.compute_stats(L, R, NEQUIL)
    out = dmc / 'qmcclean'
    assert not (out / 'stats.json').exists() and not list(out.glob('*.tmp*'))


def test_verify_root_kvecs(tmp_path):
    _, L, R, p = run(tmp_path)
    h = p.parent / 'dmc.s001.stat.h5'
    with h5py.File(h, 'a') as f:
        kv = f['kvecs'][:4]
        del f['kvecs']
        f['kvecs'] = kv
    rs = stats.verify_stats(p, L, R)
    assert rs[0] == 's001: dmc.s001.stat.h5:/kvecs differs from twist 0 raw skall/kpoints/value'
    assert len(rs) == 41 and all('differ from 4 kvecs' in r for r in rs[1:])
    with h5py.File(h, 'a') as f:
        del f['kvecs']
    rs = stats.verify_stats(p, L, R)
    assert len(rs) == 40 and all('dmc.s001.stat.h5:/kvecs missing' in r for r in rs)
    _, L, R, p = run(tmp_path / 'b')
    with h5py.File(p.parent / 'dmc.s000.stat.h5', 'a') as f:
        f['g003/dsk/kvecs'] = f['kvecs'][()]
    assert stats.verify_stats(p, L, R) == ['s000 twist 3: stray per-twist dsk/kvecs '
                                           '(k-vectors belong in /kvecs)']


def test_nofk_keeps_own_kvecs(tmp_path):
    dmc = make_dmc(tmp_path)
    for t in range(20):
        for s in (0, 1):
            with h5py.File(dmc / ('dmc.g%03d.s%03d.stat.h5' % (t, s)), 'a') as f:
                nb = f['LocalEnergy/value'].shape[0]
                f['nofk/value'] = np.ones((nb, 5)) + np.arange(nb)[:, None] % 3
                f['nofk/kpoints'] = np.arange(15.0).reshape(5, 3) + 0.01 * t  # twist-dependent
    L = discover_dmc(dmc)
    R = check_structure(L)
    p = stats.compute_stats(L, R, NEQUIL)
    with h5py.File(p.parent / 'dmc.s001.stat.h5', 'r') as f:
        assert np.allclose(f['g007/nofk/kvecs'][()], np.arange(15.0).reshape(5, 3) + 0.07)
        assert f['g007/nofk/mean'].shape == (5,)
        assert f['kvecs'].shape == (6, 3)  # skall still shared
    assert stats.verify_stats(p, L, R) == []
    with h5py.File(p.parent / 'dmc.s001.stat.h5', 'a') as f:
        del f['g002/nofk/kvecs']
        del f['g003/nofk/kvecs']
        f['g003/nofk/kvecs'] = np.zeros((4, 3))
    rs = stats.verify_stats(p, L, R)
    assert rs == ['s001 twist 2: nofk kvecs missing in dmc.s001.stat.h5',
                  's001 twist 3: nofk kvecs inconsistent with its mean/error']


# --- fix round (kvecs+parallel) ---

def to_old_layout(stat_h5):
    """Rewrite a stat.h5 into the pre-kvecs layout: per-twist dsk/rhok kvecs, no root /kvecs."""
    with h5py.File(stat_h5, 'a') as f:
        kv = f['kvecs'][()]
        del f['kvecs']
        for g in f.values():
            for o in ('dsk', 'rhok'):
                if o in g:
                    g[o]['kvecs'] = kv


def test_old_layout_recomputed_by_pipeline(tmp_path):
    from qmcclean.pipeline import run_dmc
    dmc = make_dmc(tmp_path)
    run_dmc(dmc)
    out = dmc / 'qmcclean'
    for s in (0, 1):
        to_old_layout(out / ('dmc.s%03d.stat.h5' % s))
    assert stats.old_layout(out / 'stats.json')
    rec = run_dmc(dmc)  # same nequil, last state verdict: recomputed on this invocation
    assert rec['state'] == 'verdict'
    assert any(h['action'] == 'stats recomputed (old stat.h5 layout)' for h in rec['history'])
    assert not stats.old_layout(out / 'stats.json')
    # the data-loss guard still applies: raw files of a listed twist gone -> refused
    for s in (0, 1):
        to_old_layout(out / ('dmc.s%03d.stat.h5' % s))
    for f in dmc.glob('dmc.g004.s001.*'):
        f.unlink()
    rec = run_dmc(dmc)
    assert rec['state'] == 'error' and 'refusing to overwrite' in rec['history'][-1]['action']
    assert stats.old_layout(out / 'stats.json')  # untouched


def test_verify_fails_closed_on_raw_count_error(tmp_path, monkeypatch):
    _, L, R, p = run(tmp_path)
    real = stats._probe
    def bad(kind, path):
        if kind == 'scalar' and 'dmc.g003.s000' in str(path):
            return None, 'ValueError: boom'
        return real(kind, path)
    monkeypatch.setattr(stats, '_probe', bad)
    assert stats.verify_stats(p, L, R) == [
        's000 twist 3: raw scalar.dat line count failed (ValueError: boom)']


def test_verify_kvecs_value_against_raw(tmp_path):
    _, L, R, p = run(tmp_path)
    with h5py.File(p.parent / 'dmc.s001.stat.h5', 'a') as f:
        f['kvecs'][1, 2] += 1e-13
    assert stats.verify_stats(p, L, R) == [
        's001: dmc.s001.stat.h5:/kvecs differs from twist 0 raw skall/kpoints/value']
