import numpy as np
import pytest

from qmcclean import stats, verdict
from qmcclean.config import NEQUIL
from qmcclean.layout import discover_dmc
from qmcclean.structure import check_structure

from .fakes import edit_table, make_dmc


def run(tmp_path, skip_opt=False, **kw):
    kw.setdefault('blocks', (100, 60))  # realistic block count; 14 post-equil blocks give heavy-tailed error bars
    dmc = make_dmc(tmp_path, **kw)
    L = discover_dmc(dmc)
    R = check_structure(L, skip_opt=skip_opt)
    p = stats.compute_stats(L, R, NEQUIL)
    flagged = verdict.flag_twists(p, R)
    return L, R, p, flagged, verdict.decide(L, R, flagged)


def test_clean_run(tmp_path):
    L, R, p, fl, v = run(tmp_path, ntwist=108)
    assert fl == {} and v.kind == 'clean' and v.ntwist == 108


def test_short_twist_drop_listed(tmp_path):
    _, _, _, fl, v = run(tmp_path, ntwist=108, short={5: 10})
    assert v.kind == 'drop_listed' and fl.keys() == {5}
    assert any('10/60 blocks' in r for r in fl[5])


@pytest.mark.parametrize('kind,key', [('variance', 'Variance_mean'),
                                      ('stuck', 'BlockWeight_error s001 ratio'),
                                      ('force_vmc', 'force err ratio')])
def test_outliers(tmp_path, kind, key):
    _, _, _, fl, v = run(tmp_path, ntwist=108, outlier={7: kind})
    assert fl.keys() == {7}, fl
    assert any(key in r for r in fl[7]), fl[7]
    assert v.kind == 'drop_listed'


def test_variance_only_variance(tmp_path):
    _, _, _, fl, _ = run(tmp_path, ntwist=108, outlier={7: 'variance'})
    assert len(fl[7]) == 1 and fl[7][0].startswith('Variance_mean s001 z=')


def test_stuck_walkers_reason(tmp_path):
    _, _, _, fl, _ = run(tmp_path, ntwist=108, outlier={7: 'stuck'})
    assert any(r.startswith('dmc_NumOfWalkers_mean') for r in fl[7])


def test_threshold_5pct(tmp_path):
    _, _, _, fl, v = run(tmp_path, ntwist=108, short={i: 10 for i in range(5)})
    assert len(fl) == 5 and v.kind == 'drop_listed'
    _, _, _, fl, v = run(tmp_path / 'b', ntwist=108, short={i: 10 for i in range(6)})
    assert len(fl) == 6 and v.kind == 'hold'
    assert any('5%' in r for r in v.reasons)


def test_skip_opt_still_holds(tmp_path):
    _, R, _, fl, v = run(tmp_path, skip_opt=True, ntwist=108, skall=False,
                         short={i: 10 for i in range(6)})
    assert R.overridden and not R.critical
    assert v.kind == 'hold'


def test_critical_holds(tmp_path):
    _, R, _, fl, v = run(tmp_path, skall=False)
    assert R.critical and v.kind == 'hold'
    assert any('skall' in r for r in v.reasons)


def test_no_raw_energy_flag(tmp_path):
    # raw twist energies differ by 0.4 Ha, DMC-VMC constant: still no flags
    dmc = make_dmc(tmp_path, ntwist=30, blocks=(100, 60))
    L = discover_dmc(dmc)
    R = check_structure(L)
    p = stats.compute_stats(L, R, NEQUIL)
    for s in (0, 1):
        def shift(df):
            df['LocalEnergy_mean'] += 0.4 * np.linspace(0, 1, len(df))
        edit_table(p, stats.scalar_name(s), shift)
    assert verdict.flag_twists(p, R) == {}


def test_nan_rows_do_not_distort(tmp_path):
    _, _, _, fl, _ = run(tmp_path, ntwist=30, short={i: 5 for i in range(1, 4)})
    assert fl.keys() == {1, 2, 3}


def test_force_ratio_note(tmp_path):
    L, R, p, fl, v = run(tmp_path, ntwist=20)
    assert not any('force' in r for r in v.reasons)
    def ten(df):  # make VMC force errors 10x: run ratio far above 3
        for c in df.columns:
            if c.startswith('force_') and c.endswith('_error'):
                df[c] *= 10
    edit_table(p, 'dmc.s000.scalar.dat', ten)
    fl = verdict.flag_twists(p, R)
    assert fl == {}
    v = verdict.decide(L, R, fl)
    assert v.kind == 'clean' and any('force err ratio' in r for r in v.reasons)


def test_long_blocks_plain_clean_and_err_outlier(tmp_path):
    _, _, p, fl, v = run(tmp_path, ntwist=108)
    assert fl == {} and v.kind == 'clean'
    def x4(df):
        df.loc[9, 'LocalEnergy_error'] *= 4
    edit_table(p, 'dmc.s001.scalar.dat', x4)
    from qmcclean.structure import StructureReport
    fl = verdict.flag_twists(p, StructureReport())
    assert fl.keys() == {9} and 'LocalEnergy_error s001 ratio' in fl[9][0]


def test_ratio_test_skipped_is_noted(tmp_path):
    # M3: a run median <= 0 / NaN cannot be ratio-tested; say so in the verdict
    L, R, p, _, _ = run(tmp_path, ntwist=20)
    def z(df):
        df['BlockWeight_error'] = 0.0
        df['LocalEnergy_error'] = np.nan
    edit_table(p, 'dmc.s001.scalar.dat', z)
    fl = verdict.flag_twists(p, R)
    assert 'BlockWeight_error s001 ratio test skipped: run median 0.0' in fl.notes
    assert 'LocalEnergy_error s001 ratio test skipped: run median nan' in fl.notes
    v = verdict.decide(L, R, fl)
    assert any('BlockWeight_error s001 ratio test skipped' in r for r in v.reasons)


def test_frozen_dmc_force_flagged(tmp_path):
    # I2: a DMC force column constant in one twist has error exactly 0 -> ratio +inf -> flagged
    from qharv.reel import scalar_dat
    from .fakes import _write_table
    dmc = make_dmc(tmp_path, ntwist=30, blocks=(100, 60))
    f = dmc / 'dmc.g007.s001.scalar.dat'
    df = scalar_dat.read(str(f)).drop(columns=['index', 'Variance'])
    df['force_0_0'] = 0.25
    _write_table(f, list(df.columns), df.to_numpy())
    L = discover_dmc(dmc)
    R = check_structure(L)
    p = stats.compute_stats(L, R, NEQUIL)
    d = stats.load_scalar(p, 1)
    assert d.loc[d.twist == 7, 'force_0_0_error'].iloc[0] == 0.0
    fl = verdict.flag_twists(p, R)
    assert fl.keys() == {7} and any('force err ratio inf' in r for r in fl[7]), fl
