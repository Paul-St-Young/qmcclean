import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qmcclean import record
from qmcclean.pipeline import run_dmc

from .test_cleanup import fake

C_TABC = Path('/home/pyang/scratch/malpes/009_jiuyang/d_batch/gather/c_tabc.py')


def load_ctabc():
    spec = importlib.util.spec_from_file_location('c_tabc', C_TABC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_excluded_twists(tmp_path):
    dmc = fake(tmp_path, short={5: 10})
    assert record.excluded_twists(dmc) == []  # no record
    run_dmc(dmc)
    assert record.excluded_twists(dmc) == [5]
    record.release(dmc, [5], 'ok')
    assert record.excluded_twists(dmc) == [5]  # release does not re-include


def test_excluded_twists_no_verdict(tmp_path):
    # M7: a record without a verdict (stopped_critical, error) excludes every twist
    rec = record.new()
    record.save(tmp_path, rec)
    assert record.excluded_twists(tmp_path) == []  # no twist known at all
    for g, t in ((0, 0), (1, 4), (2, 9)):
        (tmp_path / ('dmc.g%03d.twistnum_%d.in.xml' % (g, t))).write_text('<simulation/>')
    assert record.excluded_twists(tmp_path) == [0, 4, 9]
    rec['layout'] = {'twists': [0, 4, 9, 11], 'g2t': {'3': 11}}  # 11: xml already gone
    record.save(tmp_path, rec)
    assert record.excluded_twists(tmp_path) == [0, 4, 9, 11]


def test_excluded_twists_stopped_critical_and_hold(tmp_path):
    dmc = fake(tmp_path / 'a', ntwist=20, skall=False)
    assert run_dmc(dmc)['state'] == 'stopped_critical'
    assert record.excluded_twists(dmc) == list(range(20))
    dmc = fake(tmp_path / 'b', ntwist=20, short={5: 10})  # 1/20 = 5%: hold
    assert run_dmc(dmc)['verdict']['kind'] == 'hold'
    assert record.excluded_twists(dmc) == list(range(20))


def test_tabc_exclude_held_run(tmp_path):
    c = load_ctabc()
    proj = tmp_path / 'proj'
    dmc = fake(proj / 'runH', ntwist=20, short={5: 10})
    assert run_dmc(dmc)['verdict']['kind'] == 'hold'
    vals = np.arange(20, dtype=float)
    df = pd.concat([_rows('runH/dmc', vals), _rows('runB/dmc', vals)])
    meta = df.fdat.apply(c.mole.interpret_qmcpack_fname).apply(pd.Series)
    df = pd.concat([meta, df], axis=1)
    out = c.drop_excluded(df, str(proj))
    assert len(out[out.path == 'runH/dmc']) == 0  # held run: nothing enters the average
    assert len(out[out.path == 'runB/dmc']) == 20


def _rows(path, vals):
    return pd.DataFrame({
        'path': path, 'series': 1,
        'fdat': ['dmc.g%03d.s001.scalar.dat' % g for g in range(len(vals))],
        'LocalEnergy_mean': vals, 'LocalEnergy_error': 0.1})


def test_tabc_exclude(tmp_path):
    c = load_ctabc()
    proj = tmp_path / 'proj'
    dmc = fake(proj / 'runA', short={5: 10})  # fake makes <dir>/dmc
    run_dmc(dmc)
    vals = np.arange(20, dtype=float)
    vals[5] = 1000.0
    df = pd.concat([_rows('runA/dmc', vals), _rows('runB/dmc', vals)])
    meta = df.fdat.apply(c.mole.interpret_qmcpack_fname).apply(pd.Series)
    df = pd.concat([meta, df], axis=1)
    out = c.drop_excluded(df, str(proj))
    a = out[out.path == 'runA/dmc']
    assert len(a) == 19 and 5 not in set(a.group)
    assert a.LocalEnergy_mean.mean() == pytest.approx(np.delete(vals, 5).mean())
    assert len(out[out.path == 'runB/dmc']) == 20  # no record: untouched


def test_group_to_twist_map(tmp_path):
    c = load_ctabc()
    dmc = tmp_path / 'p' / 'dmc'
    rec = record.new()
    rec['verdict'] = {'kind': 'drop_listed', 'flagged': {'7': ['short']}}
    rec['layout'] = {'twists': [0, 1, 7], 'g2t': {'0': 0, '1': 1, '2': 7}}
    record.save(dmc, rec)
    df = pd.DataFrame({'path': 'p/dmc', 'group': [0, 1, 2, 7]})
    out = c.drop_excluded(df, str(tmp_path))
    assert list(out.group) == [0, 1]  # group 2 is twist 7; group 7 -> twist 7 via default


def test_refuse_cta_target():
    c = load_ctabc()
    with pytest.raises(ValueError):
        c.check_target('data/cta.json')
    with pytest.raises(ValueError):
        c.check_target('./data/../data/cta.json')
    c.check_target('data/cta_excl.json')
