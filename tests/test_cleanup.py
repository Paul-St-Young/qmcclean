import os
import tarfile
from pathlib import Path

import pytest

from qmcclean import cleanup, record
from qmcclean.config import BIG_SUFFIXES, KEEP
from qmcclean.layout import discover_dmc
from qmcclean.pipeline import run_dmc

from .fakes import make_dmc

RUNLEVEL = {'dmc.in', 'dmc.out', 'dmc.err', 'dmc.info.xml', 'dmc.qsub.in', 'dmc.struct.h5',
            'Jk2.dat', 'local_hostfile.7'}
BLK = (100, 60)  # verdict-clean at realistic block counts (see test_verdict)


def fake(tmp_path, ntwist=108, runlevel=True, **kw):
    kw.setdefault('blocks', BLK)
    dmc = make_dmc(tmp_path, ntwist=ntwist, **kw)
    if runlevel:
        for n in ('dmc.in', 'dmc.info.xml', 'dmc.qsub.in', 'dmc.struct.h5', 'Jk2.dat',
                  'local_hostfile.7'):
            (dmc / n).write_text('x\n')
        (dmc / 'sim_dmc').mkdir()
        (dmc / 'sim_dmc' / 'a.dat').write_text('a\n')
    return dmc


def listing(dmc):
    return sorted(str(p.relative_to(dmc)) for p in dmc.rglob('*')
                  if 'qmcclean' not in p.relative_to(dmc).parts)


def bigs(dmc, twist=None):
    pre = '' if twist is None else 'dmc.g%03d.' % twist
    return [p for p in dmc.iterdir() if p.name.startswith(pre) and p.name.endswith(BIG_SUFFIXES)]


def test_clean_deletes_all_big(tmp_path):
    dmc = fake(tmp_path, ntwist=20)
    before = listing(dmc)
    isfile = {n for n in before if (dmc / n).is_file()}
    rec = run_dmc(dmc, clean=True)
    assert rec['state'] == 'cleaned' and rec['verdict']['kind'] == 'clean'
    assert bigs(dmc) == []
    for k in KEEP:
        assert (dmc / k).exists()
    assert (dmc / 'qmcclean' / 'stats.json').exists()
    assert rec['archives'] == ['dmc.small.tgz']
    with tarfile.open(dmc / 'dmc.small.tgz') as t:
        names = {m.name for m in t.getmembers() if not m.isdir()}
    expect = {n for n in before if n in isfile and not n.endswith(BIG_SUFFIXES)
              and n not in KEEP and not n.startswith('dmc.small')}
    assert names == expect
    assert 'sim_dmc/a.dat' in names and 'dmc.in' in names and 'dmc.out' in names
    assert not (dmc / 'dmc.out').exists() and not (dmc / 'sim_dmc').exists()
    assert not list(dmc.glob('*.tmp'))
    assert sorted(os.listdir(dmc)) == sorted(list(KEEP) + ['dmc.small.tgz', 'qmcclean'])
    assert record.load(dmc)['state'] == 'cleaned'
    assert len(rec['deleted']) >= 20 * 9


def test_drop_listed_keeps_flagged(tmp_path):
    dmc = fake(tmp_path, short={5: 10})
    five = sorted(p.name for p in dmc.iterdir() if p.name.startswith('dmc.g005.'))
    rec = run_dmc(dmc, clean=True)
    assert rec['verdict']['kind'] == 'drop_listed' and rec['state'] == 'partially_cleaned'
    assert sorted(p.name for p in dmc.iterdir() if p.name.startswith('dmc.g005.')) == five
    assert len(bigs(dmc, 5)) == 9 and len(bigs(dmc)) == 9
    assert bigs(dmc, 6) == [] and not (dmc / 'dmc.g006.s001.qmc.xml').exists()
    with tarfile.open(dmc / 'dmc.small.tgz') as t:
        assert not any('g005' in m.name for m in t.getmembers())
        assert not any(m.name.startswith(('dmc.out', 'dmc.in', 'sim_dmc')) for m in t.getmembers())
    for n in ('dmc.out', 'dmc.in', 'dmc.err', 'sim_dmc/a.dat'):  # reviewers keep them
        assert (dmc / n).exists()


def test_hold_deletes_nothing(tmp_path):
    dmc = fake(tmp_path, short={i: 10 for i in range(6)})
    before = listing(dmc)
    rec = run_dmc(dmc, clean=True)
    assert rec['verdict']['kind'] == 'hold' and rec['state'] == 'verdict'
    assert listing(dmc) == before


def test_critical_deletes_nothing(tmp_path):
    dmc = fake(tmp_path, ntwist=5, skall=False)
    before = listing(dmc)
    rec = run_dmc(dmc, clean=True)
    assert rec['state'] == 'stopped_critical' and listing(dmc) == before
    assert not (dmc / 'qmcclean' / 'stats.json').exists()
    assert record.load(dmc)['state'] == 'stopped_critical'


def test_skip_opt_hold_still_deletes_nothing(tmp_path):
    dmc = fake(tmp_path, skall=False, short={i: 10 for i in range(6)})
    before = listing(dmc)
    rec = run_dmc(dmc, skip_opt=True, clean=True)
    assert rec['overrides'] == ['skip_opt: missing skall']
    assert rec['verdict']['kind'] == 'hold' and listing(dmc) == before


def test_dry_run_changes_nothing(tmp_path):
    dmc = fake(tmp_path, ntwist=20)
    before = listing(dmc)
    mt = {p: p.stat().st_mtime_ns for p in dmc.iterdir() if p.is_file()}
    rec = run_dmc(dmc, clean=False)
    assert rec['state'] == 'verdict' and rec['verdict']['kind'] == 'clean'
    assert listing(dmc) == before
    assert {p: p.stat().st_mtime_ns for p in dmc.iterdir() if p.is_file()} == mt
    assert any(h['action'].startswith('dry-run plan: ') and ' files, ' in h['action']
               for h in rec['history'])
    assert rec['plan']['files'] > 0 and rec['plan']['dry_run']


def _prepared(tmp_path, **kw):
    dmc = fake(tmp_path, ntwist=20, **kw)
    rec = run_dmc(dmc, clean=False)
    return dmc, rec


def test_unverified_stats_blocks_delete(tmp_path):
    dmc, rec = _prepared(tmp_path)
    (dmc / 'qmcclean' / 'stats.json').write_bytes(b'garbage')
    before = listing(dmc)
    L = discover_dmc(dmc)
    res = cleanup.execute(cleanup.plan_cleanup(L, rec), dmc, rec, dry_run=False)
    assert not res['ok'] and listing(dmc) == before
    assert rec['state'] == 'verdict'
    assert 'stats not verified' in rec['history'][-1]['action']
    assert 'unreadable' in rec['history'][-1]['action']
    # through the pipeline: the unreadable stats.json (nequil unknown) is recomputed from the
    # intact raw files and verified before anything is deleted (I2). A reused-but-bad stats.json
    # is covered by test_pipeline_reused_bad_stats_then_recomputed.
    rec2 = run_dmc(dmc, clean=True)
    acts = [h['action'] for h in rec2['history']]
    i = next(i for i, a in enumerate(acts) if a.startswith('stats recomputed (nequil None'))
    assert acts[i + 1] == 'stats saved and verified'
    assert rec2['state'] == 'cleaned'


def test_truncated_stats_blocks_delete(tmp_path):
    import h5py
    dmc, rec = _prepared(tmp_path)
    with h5py.File(dmc / 'qmcclean' / 'dmc.s001.trace.h5', 'r+') as f:
        del f['g003']
    before = listing(dmc)
    res = cleanup.execute(cleanup.plan_cleanup(discover_dmc(dmc), rec), dmc, rec, False)
    assert not res['ok'] and listing(dmc) == before
    assert 'trace missing' in rec['history'][-1]['action']


def test_idempotent(tmp_path):
    dmc = fake(tmp_path, ntwist=20)
    rec = run_dmc(dmc, clean=True)
    arc = dmc / 'dmc.small.tgz'
    mt, after = arc.stat().st_mtime_ns, listing(dmc)
    nhist = len(rec['history'])
    L = discover_dmc(dmc)
    plan = cleanup.plan_cleanup(L, rec)
    assert plan.empty() and plan.archive_name == ''
    rec2 = run_dmc(dmc, clean=True)
    assert rec2['state'] == 'cleaned' and rec2['verdict'] == rec['verdict']
    assert arc.stat().st_mtime_ns == mt and listing(dmc) == after
    assert rec2['archives'] == ['dmc.small.tgz'] and len(rec2['history']) == nhist
    run_dmc(dmc, clean=False)
    assert listing(dmc) == after and record.load(dmc)['state'] == 'cleaned'


def test_rerun_keeps_stored_verdict_and_releases_only_new(tmp_path):
    dmc = fake(tmp_path, short={5: 10})
    rec = run_dmc(dmc, clean=True)
    arc = dmc / 'dmc.small.tgz'
    mt = arc.stat().st_mtime_ns
    # rerun: shrunk layout must not change verdict/state, nothing more deleted
    after = listing(dmc)
    rec = run_dmc(dmc, clean=True)
    assert rec['state'] == 'partially_cleaned' and rec['verdict']['kind'] == 'drop_listed'
    assert set(rec['verdict']['flagged']) == {5} and listing(dmc) == after
    # release twist 5: only its files go, first archive untouched, new release archive
    rec['released'].append({'twists': [5], 'by': 'me', 'at_ET': record.now_et(), 'note': 'ok'})
    record.save(dmc, rec)
    rec = run_dmc(dmc, clean=True)
    assert rec['state'] == 'cleaned' and rec['archives'] == [
        'dmc.small.tgz', 'dmc.small.release-1.tgz']
    assert arc.stat().st_mtime_ns == mt and bigs(dmc) == []
    with tarfile.open(dmc / 'dmc.small.release-1.tgz') as t:
        names = {m.name for m in t.getmembers()}
    assert 'dmc.out' in names and 'sim_dmc/a.dat' in names
    assert all('g005' in n or '.g5.' in n or n in RUNLEVEL or n.startswith('sim_dmc')
               for n in names)
    assert not (dmc / 'dmc.out').exists() and not (dmc / 'sim_dmc').exists()
    assert rec['verdict']['kind'] == 'drop_listed'
    assert not [p for p in dmc.iterdir() if 'g005' in p.name or '.g5.' in p.name]


def test_release_on_hold_run(tmp_path):
    dmc = fake(tmp_path, short={i: 10 for i in range(6)})
    rec = run_dmc(dmc, clean=True)
    rec['released'].append({'twists': [0, 1], 'by': 'me', 'at_ET': 'x', 'note': ''})
    record.save(dmc, rec)
    rec = run_dmc(dmc, clean=True)
    assert rec['state'] == 'partially_cleaned' and rec['verdict']['kind'] == 'hold'
    assert bigs(dmc, 0) == [] and bigs(dmc, 1) == [] and len(bigs(dmc, 2)) == 9
    assert (dmc / 'dmc.g000.twistnum_0.in.xml').exists()  # KEEP
    assert (dmc / 'dmc.in').exists()  # flagged twists remain: run-level files stay


def test_release_all(tmp_path):
    dmc = fake(tmp_path, ntwist=108, short={i: 10 for i in range(6)})
    rec = run_dmc(dmc, clean=True)
    rec['released'].append({'twists': 'all'})
    record.save(dmc, rec)
    rec = run_dmc(dmc, clean=True)
    assert rec['state'] == 'cleaned' and bigs(dmc) == []


def test_crash_after_archive_before_delete(tmp_path):
    dmc, rec = _prepared(tmp_path)
    plan = cleanup._rebase(cleanup.plan_cleanup(discover_dmc(dmc), rec), dmc.resolve())
    tmp = cleanup._write_archive(plan, dmc.resolve())
    os.replace(tmp, dmc / plan.archive_name)  # archive exists, originals still there
    rec2 = run_dmc(dmc, clean=True)
    assert rec2['state'] == 'cleaned' and bigs(dmc) == []
    assert not (dmc / 'dmc.small.release-1.tgz').exists()
    assert not (dmc / 'dmc.out').exists()


def test_never_overwrite_existing_archive(tmp_path):
    dmc = fake(tmp_path, ntwist=20)
    (dmc / 'dmc.small.tgz').write_bytes(b'precious')
    rec = run_dmc(dmc, clean=True)
    assert (dmc / 'dmc.small.tgz').read_bytes() == b'precious'
    assert rec['archives'] == ['dmc.small.release-1.tgz']
    assert (dmc / 'dmc.small.release-1.tgz').exists()
    assert bigs(dmc) == []
    assert not (dmc / 'dmc.out').exists()  # run-level files ride the final archive


def test_unknown_files_and_keep_untouched(tmp_path):
    dmc = fake(tmp_path, ntwist=20)
    (dmc / 'notes.txt').write_text('mine')
    (dmc / 'data.stat.h5').write_bytes(b'0')
    run_dmc(dmc, clean=True)
    assert (dmc / 'notes.txt').read_text() == 'mine' and (dmc / 'data.stat.h5').exists()


def test_path_outside_run_blocks(tmp_path):
    dmc, rec = _prepared(tmp_path)
    victim = tmp_path / 'victim.scalar.dat'
    victim.write_text('v')
    plan = cleanup.plan_cleanup(discover_dmc(dmc), rec)
    plan.delete.append(victim)
    res = cleanup.execute(plan, dmc, rec, dry_run=False)
    assert not res['ok'] and victim.exists() and len(bigs(dmc)) == 20 * 9
    # a path that is inside the folder but not in what the record allows is refused too
    plan = cleanup.plan_cleanup(discover_dmc(dmc), rec)
    rec['verdict']['kind'] = 'hold'
    res = cleanup.execute(plan, dmc, rec, dry_run=False)
    assert not res['ok'] and len(bigs(dmc)) == 20 * 9
    assert 'not allowed' in rec['history'][-1]['action']


def test_record_roundtrip(tmp_path):
    dmc = fake(tmp_path, ntwist=20, short={5: 10}, blocks=BLK)
    rec = run_dmc(dmc)
    r2 = record.load(dmc)
    assert r2['structure']['short'] == {5: rec['structure']['short'][5]}
    assert set(r2['verdict']['flagged']) == {5}
    assert all(h['at_ET'].endswith(('-05:00', '-04:00')) for h in r2['history'])
    assert not list((dmc / 'qmcclean').glob('*.tmp*'))


def test_symlinked_run_path(tmp_path):
    dmc = fake(tmp_path / 'real', ntwist=20)
    link = tmp_path / 'link'
    link.symlink_to(dmc)
    rec = run_dmc(link, clean=True)
    assert rec['state'] == 'cleaned' and bigs(dmc) == []
    assert (dmc / 'dmc.small.tgz').exists()


def _interrupt_second_phase(monkeypatch, exc):
    real = cleanup._remove_entries
    calls = []

    def fake_remove(entries, root):
        calls.append(len(entries))
        if len(calls) == 2:
            raise exc
        return real(entries, root)
    monkeypatch.setattr(cleanup, '_remove_entries', fake_remove)
    return calls


@pytest.mark.parametrize('exc', [KeyboardInterrupt(), SystemExit(1)])
def test_kill_between_delete_phases(tmp_path, monkeypatch, exc):
    dmc = fake(tmp_path, ntwist=20)
    _interrupt_second_phase(monkeypatch, exc)
    with pytest.raises(type(exc)):
        run_dmc(dmc, clean=True)
    monkeypatch.undo()
    mid = record.load(dmc)
    assert bigs(dmc) == [] and mid['state'] == 'partially_cleaned'  # big files go first
    assert (dmc / 'dmc.g003.twistnum_3.in.xml').exists()
    assert mid['archives'] == ['dmc.small.tgz']
    rec = run_dmc(dmc, clean=True)
    assert rec['state'] == 'cleaned' and bigs(dmc) == [] and rec['pending'] == []
    assert sorted(os.listdir(dmc)) == sorted(list(KEEP) + ['dmc.small.tgz', 'qmcclean'])
    assert any(d.endswith('.scalar.dat') for d in rec['deleted'])
    assert rec['archives'] == ['dmc.small.tgz']


def test_rerun_knows_twists_whose_xml_is_gone(tmp_path):
    dmc, rec = _prepared(tmp_path)
    assert set(rec['layout']['twists']) == set(range(20))
    for t in (3, 4):  # as after a hand clean or an older crash: in.xml gone, big files left
        (dmc / ('dmc.g%03d.twistnum_%d.in.xml' % (t, t))).unlink()
    rec['deleted'].append('x')  # deletion had begun
    record.save(dmc, rec)
    rec = run_dmc(dmc, clean=True)
    assert bigs(dmc) == [] and rec['state'] == 'cleaned'


def test_error_after_deletion_began_keeps_verdict(tmp_path, monkeypatch):
    dmc = fake(tmp_path, ntwist=20)
    _interrupt_second_phase(monkeypatch, RuntimeError('disk'))
    rec = run_dmc(dmc, clean=True)
    monkeypatch.undo()
    assert rec['state'] == 'partially_cleaned' and rec['verdict']['kind'] == 'clean'
    rec = run_dmc(dmc, clean=True)
    assert rec['state'] == 'cleaned' and rec['verdict']['kind'] == 'clean' and bigs(dmc) == []


def test_pending_counts_as_deleted_when_gone(tmp_path):
    dmc, rec = _prepared(tmp_path)
    rec['pending'] = ['dmc.g000.s000.scalar.dat', 'gone.dat']
    (dmc / 'dmc.g000.s000.scalar.dat').unlink()
    record.save(dmc, rec)
    rec = run_dmc(dmc, clean=True)
    assert 'gone.dat' in rec['deleted'] and 'dmc.g000.s000.scalar.dat' in rec['deleted']
    assert rec['pending'] == [] and bigs(dmc) == [] and rec['state'] == 'cleaned'


def test_stale_tmp_archive_ignored(tmp_path):
    dmc = fake(tmp_path, ntwist=20)
    (dmc / 'dmc.small.tgz.tmp').write_bytes(b'half written')
    rec = run_dmc(dmc, clean=True)
    assert rec['archives'] == ['dmc.small.tgz'] and rec['state'] == 'cleaned'
    assert not (dmc / 'dmc.small.tgz.tmp').exists()


def test_directory_removed_only_if_empty(tmp_path):
    dmc = fake(tmp_path, ntwist=20)
    real = cleanup._write_archive

    def sneaky(plan, root, snap=None):
        tmp = real(plan, root, snap)
        (root / 'sim_dmc' / 'late.dat').write_text('new')  # appears after archive verification
        return tmp
    import pytest as _p
    mp = _p.MonkeyPatch()
    mp.setattr(cleanup, '_write_archive', sneaky)
    try:
        run_dmc(dmc, clean=True)
    finally:
        mp.undo()
    assert (dmc / 'sim_dmc' / 'late.dat').read_text() == 'new'
    assert not (dmc / 'sim_dmc' / 'a.dat').exists()


# --- final-review fixes: I2 (nequil / error recompute), M2 (group without xml), M5 (lock) ---

def _meta_nequil(dmc):
    import json
    return json.loads((dmc / 'qmcclean' / 'stats.json').read_text())['nequil']


def test_pipeline_nequil_change_recomputes(tmp_path):
    dmc = fake(tmp_path, ntwist=20)
    rec = run_dmc(dmc)  # dry run, default nequil
    assert rec['nequil'] == {'0': 32, '1': 16} == _meta_nequil(dmc)
    rec = run_dmc(dmc)  # same nequil: reused
    assert any(h['action'].startswith('existing stats reused') for h in rec['history'])
    rec = run_dmc(dmc, nequil={1: 20})
    assert _meta_nequil(dmc) == {'0': 32, '1': 20} == rec['nequil']
    assert any(h['action'].startswith('stats recomputed (nequil') for h in rec['history'])
    import h5py
    with h5py.File(dmc / 'qmcclean' / 'dmc.s001.stat.h5', 'r') as f:
        assert f.attrs['nequil'] == 20
    from qmcclean import stats
    assert set(stats.load_scalar(dmc / 'qmcclean' / 'stats.json', 1).nequil) == {20}


def test_pipeline_reused_bad_stats_then_recomputed(tmp_path):
    import h5py
    dmc = fake(tmp_path, ntwist=20)
    run_dmc(dmc)
    with h5py.File(dmc / 'qmcclean' / 'dmc.s001.trace.h5', 'r+') as f:  # same nequil, but incomplete
        del f['g003']
    rec = run_dmc(dmc, clean=True)  # last state 'verdict': reused, fails verify, nothing deleted
    assert rec['state'] == 'error' and len(bigs(dmc)) == 20 * 9
    assert 'trace missing' in rec['history'][-1]['action']
    rec = run_dmc(dmc)  # last state 'error': recomputed and verified (dry)
    assert rec['state'] == 'verdict'
    assert any(h['action'] == 'stats recomputed (last state was error)' for h in rec['history'])


def test_pipeline_error_recompute_refused_when_raw_gone(tmp_path):
    dmc = fake(tmp_path, ntwist=20)
    run_dmc(dmc)
    rec = record.load(dmc)
    rec['state'] = 'error'
    record.save(dmc, rec)
    for f in bigs(dmc, 4):  # raw files of a twist whose stats exist are gone
        f.unlink()
    (dmc / 'dmc.g004.twistnum_4.in.xml').unlink()
    before = listing(dmc)
    rec = run_dmc(dmc, clean=True)
    assert rec['state'] == 'error' and listing(dmc) == before
    assert 'refusing to overwrite' in rec['history'][-1]['action']


def test_group_without_xml_never_deleted(tmp_path):
    dmc = fake(tmp_path, ntwist=40)  # 1/40 flagged: drop_listed
    orphan = []
    for suf in ('s000.scalar.dat', 's000.stat.h5', 's001.scalar.dat', 's001.stat.h5'):
        p = dmc / ('dmc.g150.' + suf)
        p.write_bytes((dmc / ('dmc.g001.' + suf)).read_bytes())
        orphan.append(p)
    rec = run_dmc(dmc, clean=True)
    assert rec['verdict']['flagged'] == {150: [rec['verdict']['flagged'][150][0]]}
    assert rec['verdict']['flagged'][150][0].startswith('no input xml for g150')
    assert all(p.exists() for p in orphan)
    assert rec['state'] == 'partially_cleaned'  # never 'cleaned' while g150 is there
    assert 'groups without input xml kept: g150' in rec['history'][-1]['action']
    assert record.release(dmc, 'all')['released'] == [150]  # even a human release keeps them
    rec = run_dmc(dmc, clean=True)
    assert all(p.exists() for p in orphan) and rec['state'] == 'partially_cleaned'


def test_locked_run_changes_nothing(tmp_path):
    import fcntl
    dmc = fake(tmp_path, ntwist=20)
    (dmc / 'qmcclean').mkdir()
    before = listing(dmc)
    with open(dmc / 'qmcclean' / 'lock', 'w') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        rec = run_dmc(dmc, clean=True)
        assert rec['state'] == 'error'
        assert 'locked by another qmcclean process' in rec['history'][-1]['action']
        assert listing(dmc) == before and not record.path_of(dmc).exists()
        out = record.release(dmc, 'all')  # no record yet -> refused anyway
        assert out['refused']
    rec = run_dmc(dmc)
    assert rec['state'] == 'verdict'
    with open(dmc / 'qmcclean' / 'lock', 'w') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        saved = record.path_of(dmc).read_bytes()
        out = record.release(dmc, 'all')
        assert out['refused'] and 'locked by another qmcclean process' in out['messages'][0]
        assert record.path_of(dmc).read_bytes() == saved


def test_hand_edited_inconsistent_table_blocks_delete(tmp_path):
    # M6: a hand edit that drops a column pair fails verify; nothing is deleted
    from .fakes import edit_table
    dmc, rec = _prepared(tmp_path)
    edit_table(dmc / 'qmcclean' / 'stats.json', 'dmc.s001.scalar.dat',
               lambda df: df.drop(columns=['Kinetic_mean', 'Kinetic_error']))
    before = listing(dmc)
    res = cleanup.execute(cleanup.plan_cleanup(discover_dmc(dmc), rec), dmc, rec, dry_run=False)
    assert not res['ok'] and listing(dmc) == before
    assert 'raw scalar.dat columns Kinetic' in rec['history'][-1]['action']
    rec = run_dmc(dmc, clean=True)  # reused (same nequil), fails verify, nothing deleted
    assert rec['state'] == 'error' and listing(dmc) == before
    assert 'stats not verified' in rec['history'][-1]['action']


def test_slightly_short_twists_do_not_hold(tmp_path):
    # 10 of 108 twists end at 57/60 DMC blocks (>= 90%): accepted, run is not held, stats verify
    dmc = fake(tmp_path, short={i: 57 for i in range(10)})
    rec = run_dmc(dmc, clean=False)
    assert rec['state'] == 'verdict', rec['history'][-1]
    assert rec['verdict']['kind'] != 'hold'
    assert not set(range(10)) & set(rec['verdict']['flagged'])
    assert sorted(record.load(dmc)['structure']['short_ok']) == list(range(10))
