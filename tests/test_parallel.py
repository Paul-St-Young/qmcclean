"""Within-run parallel statistics: identical output for any executor, failure handling."""
import json
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import h5py
import numpy as np
import pytest

from qmcclean import cli, record, stats
from qmcclean.pipeline import run_dmc

from .fakes import make_dmc

FAST = ['--nequil-vmc', '4', '--nequil-dmc', '4']


@pytest.fixture(scope='module')
def pool():
    ex = ProcessPoolExecutor(4, mp_context=multiprocessing.get_context('spawn'))
    yield ex
    ex.shutdown()


def _h5_items(path):
    out = {}
    with h5py.File(path, 'r') as f:
        out['/'] = dict(f.attrs)
        def visit(name, obj):
            attrs = {k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in obj.attrs.items()}
            out[name] = (obj[()] if isinstance(obj, h5py.Dataset) else None, attrs)
        f.visititems(visit)
    return out


def assert_same_stats(a, b):
    """Two qmcclean/ folders hold the same statistics (stats.json equal except 'created')."""
    ma, mb = json.loads((a / 'stats.json').read_text()), json.loads((b / 'stats.json').read_text())
    for m in (ma, mb):
        m.pop('created')
    ma.pop('prefix'), mb.pop('prefix')  # path-derived
    assert ma == mb
    for name in ma['files']:
        if name.endswith('.h5'):
            ia, ib = _h5_items(a / name), _h5_items(b / name)
            assert ia.keys() == ib.keys(), name
            for k in ia:
                if k == '/':
                    assert ia[k] == ib[k]
                    continue
                (da, aa), (db, ab) = ia[k], ib[k]
                assert aa == ab, (name, k)
                if da is not None:
                    assert da.dtype == db.dtype and np.array_equal(da, db, equal_nan=True), (name, k)
        else:
            assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_serial_equals_pool(tmp_path, pool):
    da = make_dmc(tmp_path / 'a', short={5: 10})
    db = make_dmc(tmp_path / 'b', short={5: 10})
    for t in range(20):  # an extra observable with its own k-vectors
        for d in (da, db):
            with h5py.File(d / ('dmc.g%03d.s001.stat.h5' % t), 'a') as f:
                nb = f['LocalEnergy/value'].shape[0]
                f['nofk/value'] = np.cos(np.arange(nb * 3).reshape(nb, 3) + t)
                f['nofk/kpoints'] = np.eye(3) * (1 + t)
    ra = run_dmc(da)
    rb = run_dmc(db, executor=pool)
    assert ra['state'] == rb['state'] == 'verdict'
    assert ra['verdict'] == rb['verdict']
    assert_same_stats(da / 'qmcclean', db / 'qmcclean')


def _boom(*a, **k):
    raise RuntimeError('worker boom')


def test_worker_exception_errors_run(tmp_path, pool, monkeypatch):
    dmc = make_dmc(tmp_path)
    before = sorted(os.listdir(dmc))
    monkeypatch.setattr(stats, '_stat_task', _boom)  # submitted by name: runs in the pool
    rec = run_dmc(dmc, clean=True, executor=pool)
    assert rec['state'] == 'error' and 'worker boom' in rec['history'][-1]['action']
    assert not (dmc / 'qmcclean' / 'stats.json').exists()
    assert not list((dmc / 'qmcclean').glob('*.tmp*'))
    assert sorted(os.listdir(dmc)) == sorted(before + ['qmcclean'])


def calc(tmp, name):
    c = tmp / name
    c.mkdir(parents=True)
    make_dmc(c, ntwist=6, blocks=(10, 12))
    return c


def test_batch_nproc2_matches_serial(tmp_path, capsys):
    cs = [calc(tmp_path / 'p', 'i%d' % i) for i in range(3)]
    ss = [calc(tmp_path / 's', 'i%d' % i) for i in range(3)]
    assert cli.main(['run', *map(str, cs), '--nproc', '2', *FAST]) == 0
    out = capsys.readouterr().out
    lines = [x for x in out.splitlines() if 'state=' in x]
    assert len(lines) == 3 and all('state=verdict' in x and x.rstrip().endswith(' s')
                                   and '  in ' in x for x in lines)
    assert cli.main(['run', *map(str, ss), '--nproc', '1', *FAST]) == 0
    for c, s in zip(cs, ss):
        assert record.load(c / 'dmc')['verdict'] == record.load(s / 'dmc')['verdict']
        assert_same_stats(c / 'dmc' / 'qmcclean', s / 'dmc' / 'qmcclean')


def test_nproc1_builds_no_pool(tmp_path, monkeypatch):
    import concurrent.futures
    def no(*a, **k):
        raise AssertionError('ProcessPoolExecutor constructed')
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', no)
    monkeypatch.setattr(cli, '_new_pool', no)
    c = calc(tmp_path, 'i0')
    assert cli.main(['run', str(c), '--nproc', '1', *FAST]) == 0


def _kill_k0(stat_path, scalar_path, nequil):
    if '/k0/' in str(stat_path):
        os._exit(9)
    # runs only in a spawned worker, where qmcclean.stats is freshly imported (unpatched)
    return stats._stat_task(stat_path, scalar_path, nequil)


def test_killed_worker_errors_run_and_batch_continues(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(stats, '_stat_task', _kill_k0)
    cs = [calc(tmp_path, 'k0'), calc(tmp_path, 'k1')]
    before = sorted(os.listdir(cs[0] / 'dmc'))
    rc = cli.main(['run', *map(str, cs), '--clean', '--nproc', '2', *FAST])
    assert rc == 1
    r0, r1 = record.load(cs[0] / 'dmc'), record.load(cs[1] / 'dmc')
    assert r0['state'] == 'error' and 'BrokenProcessPool' in r0['history'][-1]['action']
    assert not (cs[0] / 'dmc' / 'qmcclean' / 'stats.json').exists()
    assert sorted(os.listdir(cs[0] / 'dmc')) == sorted(before + ['qmcclean'])
    assert r1['state'] == 'cleaned'  # fresh pool for the next run


# --- fix round (kvecs+parallel) ---

def _die():
    os._exit(3)


def _broken_pool():
    ex = ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn'))
    try:
        ex.submit(_die).result()
    except Exception:
        pass
    return ex


def test_broken_pool_in_verify_falls_back(tmp_path):
    from qmcclean.layout import discover_dmc
    from qmcclean.structure import check_structure
    from qmcclean.config import NEQUIL
    dmc = make_dmc(tmp_path)
    L = discover_dmc(dmc)
    R = check_structure(L)
    p = stats.compute_stats(L, R, NEQUIL)
    ex = _broken_pool()
    try:
        assert stats.verify_stats(p, L, R, executor=ex) == []
    finally:
        ex.shutdown()


def test_broken_pool_in_structure_check_errors_run(tmp_path):
    dmc = make_dmc(tmp_path)
    before = sorted(os.listdir(dmc))
    ex = _broken_pool()
    try:
        rec = run_dmc(dmc, clean=True, executor=ex)
    finally:
        ex.shutdown()
    assert rec['state'] == 'error' and 'BrokenProcessPool' in rec['history'][-1]['action']
    assert not (dmc / 'qmcclean' / 'stats.json').exists()
    assert sorted(os.listdir(dmc)) == sorted(before + ['qmcclean'])


def test_kvecs_mismatch_through_pool_deletes_nothing(tmp_path, pool):
    dmc = make_dmc(tmp_path)
    with h5py.File(dmc / 'dmc.g006.s001.stat.h5', 'a') as f:
        d = f['skall/kpoints/value']
        a = d[()]
        a[0, 1] += 1e-9
        d[...] = a
    before = sorted(os.listdir(dmc))
    rec = run_dmc(dmc, clean=True, executor=pool)
    assert rec['state'] == 'error'
    assert 's001 twist 6: skall kpoints differ from twist 0' in rec['history'][-1]['action']
    assert not (dmc / 'qmcclean' / 'stats.json').exists()
    assert not list((dmc / 'qmcclean').glob('*.tmp*'))
    assert sorted(os.listdir(dmc)) == sorted(before + ['qmcclean'])
