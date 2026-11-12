import json

import h5py
import numpy as np
import pytest

from qmcclean import du, jastrow
from qmcclean.pipeline import run_dmc

from .fakes import LATTICE, make_dmc

BLK = (100, 60)


def _common_dsk(dmc):
    """Fakes draw different k-vectors per twist; real runs share them. Make them shared and sane."""
    shells = jastrow.kshells(LATTICE, 3.0)[:12]
    kv = np.array([[k, 0, 0] for k in shells] + [[0, shells[0], 0]])
    n = json.loads((dmc / 'qmcclean' / 'stats.json').read_text())['nelec']
    for s, scale in (('s000', 0.5), ('s001', 0.55)):
        with h5py.File(dmc / 'qmcclean' / ('dmc.%s.stat.h5' % s), 'r+') as f:
            del f['kvecs']
            f['kvecs'] = kv  # shared, once per series
            for t, g0 in f.items():
                if not isinstance(g0, h5py.Group):
                    continue
                for obs in ('dsk', 'rhok'):
                    g = g0[obs]
                    for d in ('mean', 'error'):
                        del g[d]
                    m = n * scale * np.linspace(0.6, 1, len(kv))
                    g['mean'] = m + 0j if obs == 'rhok' else m
                    g['error'] = np.full(len(kv), 0.01 * n)


def test_du_after_cleanup(tmp_path):
    dmc = make_dmc(tmp_path, ntwist=10, blocks=BLK)
    rec = run_dmc(dmc, clean=True)
    assert rec['state'] == 'cleaned'
    _common_dsk(dmc)  # after cleanup: verify compares /kvecs with the raw kpoints while they exist
    assert not list(dmc.glob('*.stat.h5'))
    png = du.run_du(dmc)
    assert png == dmc / 'qmcclean' / 'uk.png' and png.stat().st_size > 1000
    assert not (dmc / 'qmcclean' / 'du.png').exists()
    csv = dmc / 'qmcclean' / 'du.csv'
    lines = [x for x in csv.read_text().splitlines()]
    assert lines[0].startswith('#') and 'twists used: 10' in lines[0]
    assert lines[1] == 'k,uT,du,du_err,extrap,extrap_err'
    a = np.array([[float(x) for x in r.split(',')] for r in lines[2:]])
    assert np.allclose(a[:, 4], a[:, 1] + 2 * a[:, 2], rtol=0, atol=1e-14)
    assert np.allclose(a[:, 5], 2 * a[:, 3], rtol=0, atol=1e-14)
    assert len(lines) == 2 + 12  # 12 distinct shells (13th k-vector repeats shell 1)
    js = json.loads((dmc / 'qmcclean' / 'du.json').read_text())
    assert js['ntwist_used'] == 10 and isinstance(js['flagged'], bool)
    assert js['flagged'] == (abs(js['du_rel_first_shell']) > 0.10)
    assert js['time'].endswith(('-04:00', '-05:00'))


def test_du_excludes_flagged(tmp_path):
    dmc = make_dmc(tmp_path, ntwist=10, blocks=BLK)
    run_dmc(dmc)
    _common_dsk(dmc)
    p = dmc / 'qmcclean' / 'record.json'
    r = json.loads(p.read_text())
    r['verdict']['flagged'] = {'3': ['x'], '4': ['y']}
    p.write_text(json.dumps(r))
    du.run_du(dmc)
    assert json.loads((dmc / 'qmcclean' / 'du.json').read_text())['ntwist_used'] == 8


def test_du_no_skall(tmp_path):
    dmc = make_dmc(tmp_path, ntwist=3, blocks=BLK, skall=False)
    run_dmc(dmc, skip_opt=True)
    with pytest.raises(RuntimeError, match='skall'):
        du.run_du(dmc)


def test_du_non_finite_flagged(tmp_path, monkeypatch, capsys):
    # M6: u_T = 0 at the first shell -> du/u_T is not finite -> flagged with a note
    dmc = make_dmc(tmp_path, ntwist=10, blocks=BLK)
    run_dmc(dmc)
    _common_dsk(dmc)
    monkeypatch.setattr(du, 'trial_uk', lambda xp, k, axes: np.zeros(len(k)))
    du.run_du(dmc)
    js = json.loads((dmc / 'qmcclean' / 'du.json').read_text())
    assert js['flagged'] is True and js['du_rel_first_shell'] is None
    assert 'not finite' in js['note']
    assert 'not finite' in (dmc / 'qmcclean' / 'du.csv').read_text().splitlines()[0]
    assert 'FLAG: not finite' in capsys.readouterr().out


def test_du_skips_twists_without_scalar_row(tmp_path):
    # M3: hand-deleting a twist's row from the scalar table also drops it from S(k)
    from qmcclean.stats import read_meta
    from .fakes import edit_table
    dmc = make_dmc(tmp_path, ntwist=10, blocks=BLK)
    run_dmc(dmc)
    _common_dsk(dmc)
    p = dmc / 'qmcclean' / 'stats.json'
    edit_table(p, 'dmc.s001.scalar.dat', lambda df: df[df.twist != 3])
    n = read_meta(p)['nelec']
    assert du.twist_average(p, 1, set(), n)[3] == 9
    assert du.twist_average(p, 0, set(), n)[3] == 10
    assert du.twist_average(p, 1, {4}, n)[3] == 8


def test_du_old_layout_message(tmp_path):
    from .test_stats import to_old_layout
    dmc = make_dmc(tmp_path, ntwist=3, blocks=BLK)
    run_dmc(dmc)
    to_old_layout(dmc / 'qmcclean' / 'dmc.s000.stat.h5')
    with pytest.raises(RuntimeError, match=r'old stat.h5 layout \(per-twist kvecs\); recompute '
                                           'stats from raw files') as e:
        du.run_du(dmc)
    assert 'batched' not in str(e.value)


def test_du_removes_stale_du_png(tmp_path):
    dmc = make_dmc(tmp_path, ntwist=10, blocks=BLK)
    run_dmc(dmc)
    _common_dsk(dmc)
    (dmc / 'qmcclean' / 'du.png').write_bytes(b'old figure')
    du.run_du(dmc)
    assert (dmc / 'qmcclean' / 'uk.png').exists() and not (dmc / 'qmcclean' / 'du.png').exists()
