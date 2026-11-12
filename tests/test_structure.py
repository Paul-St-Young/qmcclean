import h5py

from qmcclean.layout import discover_dmc
from qmcclean.structure import check_structure
from tests.fakes import make_dmc


def _R(d, **kw):
    return check_structure(discover_dmc(d), **kw)


def test_clean_run(tmp_path):
    R = _R(make_dmc(tmp_path, ntwist=6))
    assert not R.short and not R.missing_files and not R.missing_estimators
    assert not R.critical and not R.overridden
    assert not R.absent_optional


def test_short_twist(tmp_path):
    R = _R(make_dmc(tmp_path, ntwist=8, short={5: 10}))
    assert R.short == {5: ['s001: 10/30 blocks']}
    assert not R.critical


def test_slightly_short_twist_accepted(tmp_path):
    # >= MIN_BLOCK_FRACTION (0.9) of the requested blocks: noted, not flagged; below: flagged
    R = _R(make_dmc(tmp_path, ntwist=8, short={5: 27, 6: 26}))
    assert R.short_ok == {5: ['s001: 27/30 blocks (accepted, >= 0.9)']}
    assert R.short == {6: ['s001: 26/30 blocks']}
    assert not R.critical


def test_missing_skall_critical(tmp_path):
    R = _R(make_dmc(tmp_path, ntwist=4, skall=False))
    assert R.missing_estimators == ['skall']
    assert R.critical


def test_one_twist_missing_skall(tmp_path):
    # I1: skall present elsewhere -> that twist is flagged (not run-level, not overridable)
    d = make_dmc(tmp_path, ntwist=4)
    p = d / 'dmc.g002.s001.stat.h5'
    with h5py.File(p, 'a') as f:
        del f['skall']
    for skip in (False, True):
        R = _R(d, skip_opt=skip)
        assert R.missing_estimators == [] and not R.critical and not R.overridden
        assert R.missing_files == {2: ['missing estimator skall in s001']}


def test_skall_missing_in_one_series_everywhere(tmp_path):
    # absent from every s000 but present in s001: not run-level; every twist flagged
    d = make_dmc(tmp_path, ntwist=3)
    for t in range(3):
        with h5py.File(d / ('dmc.g%03d.s000.stat.h5' % t), 'a') as f:
            del f['skall']
    R = _R(d, skip_opt=True)
    assert R.missing_estimators == [] and not R.overridden
    assert R.missing_files == {t: ['missing estimator skall in s000'] for t in range(3)}


def test_skall_missing_everywhere_is_run_level(tmp_path):
    d = make_dmc(tmp_path, ntwist=3, skall=False)
    R = _R(d)
    assert R.missing_estimators == ['skall'] and R.critical and not R.missing_files


def test_more_blocks_than_requested_flagged(tmp_path):
    # M1: e.g. a restart appended to the same scalar.dat
    d = make_dmc(tmp_path, ntwist=4, blocks=(40, 30))
    f = d / 'dmc.g001.s001.scalar.dat'
    lines = f.read_text().splitlines()
    f.write_text('\n'.join(lines + lines[1:6]) + '\n')
    R = _R(d)
    assert R.short == {1: ['s001: 35/30 blocks (more than requested)']}


def test_group_without_xml_flagged(tmp_path):
    # M2: big files of g007 but no dmc.g007.twistnum_*.in.xml
    d = make_dmc(tmp_path, ntwist=4)
    for suf in ('s001.scalar.dat', 's001.stat.h5'):
        (d / ('dmc.g007.' + suf)).write_bytes((d / ('dmc.g001.' + suf)).read_bytes())
    L = discover_dmc(d)
    assert set(L.no_xml) == {7} and not any(k[0] == 7 for k in L.files)
    R = check_structure(L)
    assert list(R.missing_files) == [7]
    assert R.missing_files[7][0].startswith('no input xml for g007: dmc.g007.s001.scalar.dat')


def test_skip_opt_overrides(tmp_path):
    R = _R(make_dmc(tmp_path, ntwist=4, skall=False), skip_opt=True)
    assert not R.critical and R.overridden
    assert R.missing_estimators == ['skall']


def test_hand_cleaned(tmp_path):
    d = make_dmc(tmp_path, ntwist=4, dmcdat=False)
    for f in d.glob('*.random.h5'):
        f.unlink()
    R = _R(d)
    assert any('dmc.dat' in s for s in R.absent_optional)
    assert any('random.h5' in s for s in R.absent_optional)
    assert not R.short and not R.missing_files and not R.critical


def test_twist_missing_series(tmp_path):
    d = make_dmc(tmp_path, ntwist=5)
    (d / 'dmc.g003.s001.scalar.dat').unlink()
    (d / 'dmc.g003.s001.stat.h5').unlink()
    R = _R(d)
    assert 3 in R.missing_files
    assert not R.critical


def test_unreadable_stat_h5(tmp_path):
    d = make_dmc(tmp_path, ntwist=4)
    (d / 'dmc.g001.s000.stat.h5').write_bytes(b'garbage')
    R = _R(d)
    assert 1 in R.missing_files
    assert any('stat.h5' in s for s in R.missing_files[1])


def test_duplicate_scalar_flagged(tmp_path):
    d = make_dmc(tmp_path, ntwist=4)
    (d / 'dmc.g002.s002.scalar.dat').write_text('# index\n')
    R = _R(d)
    assert 2 in R.missing_files
    assert any(s.startswith('ambiguous: ') and 'dmc.g002.s002.scalar.dat' in s
               for s in R.missing_files[2])


def test_stat_h5_odd_contents(tmp_path):
    import numpy as np
    d = make_dmc(tmp_path, ntwist=4)
    with h5py.File(d / 'dmc.g001.s000.stat.h5', 'w') as f:
        f.create_dataset('Kinetic/value', data=np.zeros(3))  # no LocalEnergy
    with h5py.File(d / 'dmc.g002.s001.stat.h5', 'w') as f:
        f.create_dataset('LocalEnergy/value', data=1.0)  # 0-d
    with h5py.File(d / 'dmc.g003.s001.stat.h5', 'w') as f:
        f.create_group('LocalEnergy/value')  # wrong type
    (d / 'dmc.g000.s001.stat.h5').write_bytes(b'')  # empty file
    R = _R(d)
    assert set(R.missing_files) == {0, 1, 2, 3}


def test_single_optional_file_removed(tmp_path):
    d = make_dmc(tmp_path, ntwist=4)
    (d / 'dmc.g001.s000.random.h5').unlink()
    R = _R(d)
    assert R.absent_optional == ['random.h5 absent: 1/8 files']
    assert not R.missing_files


def test_true_duplicate_key(tmp_path):
    d = make_dmc(tmp_path, ntwist=4)
    # a second group g050 that maps to twist 2 -> same (twist, series, suffix) key
    (d / 'dmc.g050.twistnum_2.in.xml').write_text('<simulation/>')
    dup = d / 'dmc.g050.s001.scalar.dat'
    dup.write_text('# index\n')
    L = discover_dmc(d)
    assert L.unexpected == {2: [dup]}
    R = check_structure(L)
    assert any(s.startswith('ambiguous: ') and dup.name in s
               for s in R.missing_files[2])
