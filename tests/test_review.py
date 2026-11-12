import tarfile

from qmcclean import record
from qmcclean.pipeline import run_dmc

from .test_cleanup import bigs, fake, listing


def test_release_then_clean(tmp_path):
    dmc = fake(tmp_path, short={5: 10})
    run_dmc(dmc, clean=True)
    assert len(bigs(dmc, 5)) == 9
    r = record.release(dmc, [5], 'reviewed')
    assert r['released'] == [5] and r['messages'] == []
    e = record.load(dmc)['released'][0]
    assert e['twists'] == [5] and e['note'] == 'reviewed' and e['by'] and e['at_ET']
    assert len(bigs(dmc, 5)) == 9  # release itself deletes nothing
    rec = run_dmc(dmc, clean=True)
    assert rec['state'] == 'cleaned' and bigs(dmc) == []
    assert rec['archives'] == ['dmc.small.tgz', 'dmc.small.release-1.tgz']
    with tarfile.open(dmc / 'dmc.small.release-1.tgz') as t:
        assert any('g005' in m.name for m in t.getmembers())
    assert any(h['action'].startswith('released twists 5') for h in rec['history'])


def test_release_all_on_hold(tmp_path):
    dmc = fake(tmp_path, short={i: 10 for i in range(6)})
    before = listing(dmc)
    run_dmc(dmc, clean=True)
    assert listing(dmc) == before
    assert record.release(dmc, 'all', 'ok')['released'] == [0, 1, 2, 3, 4, 5]
    assert listing(dmc) == before
    rec = run_dmc(dmc, clean=True)
    assert rec['state'] == 'cleaned' and bigs(dmc) == []


def test_release_unflagged_is_noop(tmp_path):
    dmc = fake(tmp_path, short={5: 10})
    run_dmc(dmc, clean=True)
    r = record.release(dmc, [7], 'x')
    assert r['released'] == [] and 'not flagged' in r['messages'][0]
    assert record.load(dmc)['released'] == []
    r = record.release(dmc, [5, 7], 'x')
    assert r['released'] == [5] and len(r['messages']) == 1


def test_release_all_on_clean_is_noop(tmp_path):
    dmc = fake(tmp_path, ntwist=20)
    run_dmc(dmc, clean=False)
    r = record.release(dmc, 'all', 'x')
    assert r['released'] == [] and r['messages'] and record.load(dmc)['released'] == []


def test_release_without_record(tmp_path):
    dmc = fake(tmp_path, ntwist=5)
    assert record.release(dmc, [1], 'x')['messages']
    assert not record.path_of(dmc).exists()


def test_release_bad_argument_raises(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        record.release(tmp_path, 'al', 'x')


def test_release_all_after_all_flagged_released_individually(tmp_path):
    dmc = fake(tmp_path, short={5: 10})
    run_dmc(dmc, clean=False)
    record.release(dmc, [5], 'a')
    r = record.release(dmc, 'all', 'b')
    assert r['released'] == [] and r['messages'] and len(record.load(dmc)['released']) == 1


def test_release_twice_idempotent_and_third_run_noop(tmp_path):
    dmc = fake(tmp_path, short={5: 10})
    run_dmc(dmc, clean=True)
    record.release(dmc, [5], 'a')
    r2 = record.release(dmc, [5], 'a')
    assert r2['released'] == [] and 'already' in r2['messages'][0]
    assert len(record.load(dmc)['released']) == 1
    run_dmc(dmc, clean=True)
    after = listing(dmc)
    rec = run_dmc(dmc, clean=True)
    assert listing(dmc) == after and not (dmc / 'dmc.small.release-2.tgz').exists()
    assert rec['archives'] == ['dmc.small.tgz', 'dmc.small.release-1.tgz']
    assert rec['state'] == 'cleaned'


def test_release_on_stopped_critical_refused_then_works_after_skip_opt(tmp_path):
    dmc = fake(tmp_path, skall=False, short={1: 10})
    run_dmc(dmc, clean=True)
    before = listing(dmc)
    r = record.release(dmc, [1], 'try')
    assert r['released'] == [] and 'no verdict' in r['messages'][0]
    assert record.load(dmc)['released'] == []
    rec = run_dmc(dmc, clean=True)
    assert rec['state'] == 'stopped_critical' and listing(dmc) == before
    assert not (dmc / 'qmcclean' / 'stats.json').exists()
    rec = run_dmc(dmc, skip_opt=True, clean=False)  # dry run obtains a verdict
    assert rec['verdict']['kind'] == 'drop_listed' and rec['released'] == []
    assert listing(dmc) == before
    r = record.release(dmc, [1], 'ok')
    assert r['released'] == [1] and record.load(dmc)['released'][0]['twists'] == [1]
    assert len(bigs(dmc, 1)) == 9
    run_dmc(dmc, skip_opt=True, clean=True)
    assert bigs(dmc, 1) == [] and len(bigs(dmc)) == 0
