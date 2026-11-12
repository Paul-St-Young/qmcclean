import numpy as np

from qmcclean.layout import classify, discover_dmc, discover_opt
from tests.fakes import make_dmc, make_opt


def test_discover_dmc_counts(tmp_path):
    d = make_dmc(tmp_path, ntwist=20)
    L = discover_dmc(d)
    assert len(L.twists) == 20
    assert [s.blocks for s in L.series] == [40, 30]
    assert [s.method for s in L.series] == ['vmc', 'dmc']
    assert set(L.estimators) >= {'force', 'skall'}
    assert L.driver == 'legacy'
    assert L.nelec == 8
    assert L.axes.shape == (3, 3)
    assert (3, 1, '.scalar.dat') in L.files
    assert (3, 1, '.dmc.dat') in L.files
    assert (3, 0, '.stat.h5') in L.files
    assert L.unexpected == {}
    assert L.prefix  # non-empty


def test_discover_batched_driver(tmp_path):
    L = discover_dmc(make_dmc(tmp_path, ntwist=3, driver='batched'))
    assert L.driver == 'batched'


def test_discover_missing_optional(tmp_path):
    L = discover_dmc(make_dmc(tmp_path, ntwist=5, dmcdat=False))
    assert not any(k[2] == '.dmc.dat' for k in L.files)
    assert (0, 1, '.scalar.dat') in L.files


def test_unexpected_series_file(tmp_path):
    d = make_dmc(tmp_path, ntwist=5)
    stray = d / 'dmc.g003.s002.scalar.dat'
    stray.write_text('# index\n')
    L = discover_dmc(d)
    assert L.unexpected == {3: [stray]}
    assert (3, 2, '.scalar.dat') not in L.files


def test_duplicate_file_is_unexpected(tmp_path):
    d = make_dmc(tmp_path, ntwist=5)
    dup = d / 'old.g002.s001.scalar.dat'
    dup.write_text('# index\n')
    L = discover_dmc(d)
    assert list(L.unexpected) == [2]
    assert dup in L.unexpected[2]
    assert L.files[(2, 1, '.scalar.dat')].name == 'dmc.g002.s001.scalar.dat'


def test_discover_opt(tmp_path):
    L = discover_opt(make_opt(tmp_path, nstep=12, converged=True))
    assert L.steps == list(range(12))
    assert L.files[(11, '.scalar.dat')].name == 'opt.s011.scalar.dat'
    assert L.files[(11, '.opt.xml')].name == 'opt.s011.opt.xml'


def test_classify(tmp_path):
    d = make_dmc(tmp_path, ntwist=3)
    o = make_opt(tmp_path, nstep=4, converged=True)
    assert classify(d) == 'dmc'
    assert classify(o) == 'opt'
    assert classify(tmp_path) == 'calc'
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert classify(empty) == 'unknown'


def test_fakes_are_qharv_readable(tmp_path):
    from qharv.reel import scalar_dat
    d = make_dmc(tmp_path, ntwist=2)
    df = scalar_dat.read(str(d / 'dmc.g001.s001.scalar.dat'))
    assert len(df) == 30
    # real scalar.dat has no Variance column on disk (qharv derives one on read)
    assert 'Variance' not in (d / 'dmc.g001.s001.scalar.dat').read_text().splitlines()[0]
    assert {'LocalEnergy', 'LocalEnergy_sq', 'BlockWeight', 'force_3_2'} <= set(df.columns)
    assert np.all(df['LocalEnergy_sq'] > df['LocalEnergy'] ** 2)


def test_make_opt_drift_vs_converged(tmp_path):
    import re
    from qharv.reel import scalar_dat

    def jk0(f):
        txt = f.read_text()
        return float(re.search(r'id="jk2"[^>]*>\s*(\S+)', txt).group(1))

    out = {}
    for conv in (False, True):
        o = make_opt(tmp_path / str(conv), nstep=12, converged=conv)
        L = discover_opt(o)
        df = scalar_dat.read(str(L.files[(11, '.scalar.dat')]))
        out[conv] = (jk0(L.files[(11, '.opt.xml')]) / jk0(L.files[(10, '.opt.xml')]),
                     df['Variance'].mean())
    assert abs(out[False][0] - 1.06) < 1e-6
    assert abs(out[True][0] - 1) < 0.01
    assert abs(out[False][1] - (2.0 - 0.03 * 11)) < 0.02
    assert abs(out[True][1] - 1.5) < 0.02


def test_prefix_last_five_segments(tmp_path):
    base = tmp_path / 'n96-runs' / 'pbe-e240-k6s1' / 'p300' / 'i129'
    base.mkdir(parents=True)
    L = discover_dmc(make_dmc(base))
    assert L.prefix == 'n96-runs_pbe-e240-k6s1_p300_i129_dmc'


def test_prefix_short_path():
    from qmcclean.layout import _prefix
    assert _prefix('/a/b') == 'a_b'
