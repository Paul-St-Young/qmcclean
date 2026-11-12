import json
import os

import pytest

from qmcclean import cli, record

from .fakes import make_dmc, make_opt

FAST = ['--nequil-vmc', '4', '--nequil-dmc', '4']


def calc(tmp, name, opt=True, dmc=True, **kw):
    c = tmp / name
    c.mkdir()
    if opt:
        make_opt(c)
    if dmc:
        make_dmc(c, ntwist=6, blocks=(10, 12), **kw)
    return c


def bigs(d):
    return sorted(p.name for p in d.iterdir() if p.name.endswith(('.scalar.dat', '.stat.h5')))


def test_dispatch_calc(tmp_path):
    c = calc(tmp_path, 'i1')
    assert cli.main(['run', str(c), '--nproc', '1', *FAST]) == 0
    orec = c / 'opt' / 'qmcclean' / 'opt_record.json'
    drec = record.load(c / 'dmc')
    assert orec.exists() and drec['history']
    # seconds-resolution strings can tie, so the order is checked on mtimes too
    assert json.loads(orec.read_text())['time'] <= drec['history'][0]['at_ET']
    assert orec.stat().st_mtime_ns < (c / 'dmc' / 'qmcclean' / 'stats.json').stat().st_mtime_ns


def test_dispatch_opt_and_dmc_paths(tmp_path):
    c = calc(tmp_path, 'i1')
    assert cli.main(['run', str(c / 'opt')]) == 0
    assert not (c / 'dmc' / 'qmcclean').exists()
    assert cli.main(['run', str(c / 'dmc'), *FAST]) == 0
    assert (c / 'dmc' / 'qmcclean' / 'record.json').exists()


def test_unknown_path_exit1(tmp_path, capsys):
    (tmp_path / 'x').mkdir()
    assert cli.main(['run', str(tmp_path / 'x')]) == 1
    assert 'ERROR' in capsys.readouterr().out


def test_calc_missing_dmc_noted(tmp_path, capsys):
    c = calc(tmp_path, 'i1', dmc=False)
    assert cli.main(['run', str(c)]) == 0
    assert 'no dmc/' in capsys.readouterr().out


def test_default_dry_run(tmp_path, capsys):
    c = calc(tmp_path, 'i1')
    before = set(os.listdir(c / 'dmc'))
    assert cli.main(['run', str(c), '--nproc', '1', *FAST]) == 0
    assert set(os.listdir(c / 'dmc')) == before | {'qmcclean'}
    assert len(bigs(c / 'dmc')) == 24
    assert 'would free' in capsys.readouterr().out
    assert cli.main(['run', str(c / 'dmc'), '--clean', *FAST]) == 0
    assert bigs(c / 'dmc') == []
    assert record.load(c / 'dmc')['state'] == 'cleaned'


def test_batch_error_isolated(tmp_path, capsys):
    cs = [calc(tmp_path, 'i%d' % i, opt=False) for i in range(3)]
    (cs[1] / 'dmc' / 'dmc.g000.twistnum_0.in.xml').write_text('<broken')
    rc = cli.main(['run', *map(str, cs), '--clean', '--nproc', '2', *FAST])
    assert rc == 1
    st = [record.load(c / 'dmc')['state'] for c in cs]
    assert st == ['cleaned', 'error', 'cleaned']
    assert capsys.readouterr().out.count('ERROR') == 1


def test_skip_opt_batch(tmp_path):
    cs = [calc(tmp_path, 'i%d' % i, opt=False, skall=False) for i in range(2)]
    assert cli.main(['run', *map(str, cs), '--skip-opt', '--nproc', '1', *FAST]) == 0
    for c in cs:
        r = record.load(c / 'dmc')
        assert r['verdict'] and any('skip_opt' in o for o in r['overrides'])
    assert cli.main(['run', str(calc(tmp_path, 'j', opt=False, skall=False)), *FAST]) == 1


def test_review_subcommand(tmp_path, capsys):
    c = calc(tmp_path, 'i1', opt=False)
    cli.main(['run', str(c), *FAST])
    assert cli.main(['review', str(c / 'dmc'), '--release', 'all']) == 0
    assert 'run' in capsys.readouterr().out
    assert cli.main(['review', str(c / 'dmc'), '--release', 'x']) == 2


# ---- review fixes ----
import re
from pathlib import Path

import h5py

from .test_cleanup import fake


def test_fallback_no_dmc_dir(tmp_path, monkeypatch, capsys):
    c = calc(tmp_path, 'i1', opt=False)
    monkeypatch.setattr(cli, 'run_dmc', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    monkeypatch.setattr(cli, '_stages', lambda p: (None, tmp_path / 'nodmc', []))
    assert cli.main(['run', str(c), '--nproc', '1']) == 1
    assert 'boom' in capsys.readouterr().out
    assert not (tmp_path / 'nodmc').exists()


def test_fallback_keeps_partially_cleaned(tmp_path, monkeypatch):
    c = calc(tmp_path, 'i1', opt=False)
    cli.main(['run', str(c), '--nproc', '1', *FAST])
    rec = record.load(c / 'dmc')
    rec['state'] = 'partially_cleaned'
    rec['pending'] = ['x']
    record.save(c / 'dmc', rec)
    monkeypatch.setattr(cli, 'run_dmc', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    assert cli.main(['run', str(c), '--nproc', '1']) == 1
    r = record.load(c / 'dmc')
    assert r['state'] == 'partially_cleaned' and 'boom' in r['history'][-1]['action']


def test_fallback_sets_error_before_deletion(tmp_path, monkeypatch):
    c = calc(tmp_path, 'i1', opt=False)
    monkeypatch.setattr(cli, 'run_dmc', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    assert cli.main(['run', str(c), '--nproc', '1']) == 1
    assert record.load(c / 'dmc')['state'] == 'error'


def test_review_comma_list(tmp_path, capsys):
    dmc = fake(tmp_path, short={5: 10})
    assert cli.main(['run', str(dmc)]) == 0  # default nequil: twist 5 short and flagged
    capsys.readouterr()
    assert cli.main(['review', str(dmc), '--release', '5,7', '--note', 'ok']) == 0
    out = capsys.readouterr().out
    assert 'twist 7 is not flagged' in out and 'released: [5]' in out and '--clean' in out
    assert [e['twists'] for e in record.load(dmc)['released']] == [[5]]


def test_review_errors(tmp_path, capsys):
    assert cli.main(['review', str(tmp_path / 'nope'), '--release', 'all']) == 1
    (tmp_path / 'd').mkdir()
    assert cli.main(['review', str(tmp_path / 'd'), '--release', 'all']) == 1
    assert 'no record' in capsys.readouterr().err


def test_summary_line_and_nequil(tmp_path, capsys):
    dmc = fake(tmp_path, short={5: 10})
    assert cli.main(['run', str(dmc), '--nequil-vmc', '7', '--nequil-dmc', '3']) == 0
    out = capsys.readouterr().out
    line = [x for x in out.splitlines() if 'verdict=' in x][0]
    assert 'verdict=drop_listed' in line and 'flagged=5' in line
    m = re.search(r'would free (\S+ \S+)', line)
    assert m
    rec = record.load(dmc)
    assert cli._human(rec['plan']['bytes']) == m.group(1)
    assert json.loads((dmc / 'qmcclean' / 'stats.json').read_text())['nequil'] == {'0': 7, '1': 3}
    cli.main(['run', str(dmc), '--clean'])
    line = [x for x in capsys.readouterr().out.splitlines() if 'state=' in x][0]
    assert ' freed ' in line and 'would free' not in line


def test_human():
    assert cli._human(999) == '999 B'
    assert cli._human(999600) == '1 MB'
    assert cli._human(1500) == '1.5 kB'


def test_nproc_validated():
    with pytest.raises(SystemExit):
        cli.main(['run', 'x', '--nproc', '0'])


def test_slurm_template():
    s = (Path(__file__).parent.parent / 'sbatch' / 'qmcclean_batch.slurm').read_text()
    for x in ('--nproc "$SLURM_CPUS_PER_TASK"', '#SBATCH -p qmchamm', '/PATH/TO/qmcclean-venv',
              'QMCCLEAN_PYMODS=/PATH/TO/pymods', 'QMCCLEAN_ROOT=/PATH/TO/qmcclean',
              'export PYTHONPATH="$QMCCLEAN_ROOT', "python -c 'import qmcclean.cli, qharv' ||"):
        assert x in s
    # the preflight comes before the run
    assert s.index("import qmcclean.cli") < s.rindex('python -m qmcclean run')


# ---- final-review fixes: I3 (opt import isolated), M4 (refused review exits 1) ----

def test_opt_import_failure_does_not_stop_dmc(tmp_path, monkeypatch, capsys):
    import sys
    # chiesa_correction not importable (QMCCLEAN_PYMODS wrong on the cluster)
    monkeypatch.setitem(sys.modules, 'chiesa_correction', None)
    import qmcclean
    for m in ('optcheck', 'jastrow'):  # force a fresh import of both
        monkeypatch.delitem(sys.modules, 'qmcclean.' + m, raising=False)
        monkeypatch.delattr(qmcclean, m, raising=False)
    c = calc(tmp_path, 'i1')
    assert cli.main(['run', str(c), '--nproc', '1', *FAST]) == 1
    line = [x for x in capsys.readouterr().out.splitlines() if 'opt=' in x][0]
    assert 'opt=error' in line and 'state=verdict' in line and 'chiesa_correction' in line
    assert record.load(c / 'dmc')['state'] == 'verdict'


def test_review_refused_without_verdict_exit1(tmp_path, capsys):
    dmc = fake(tmp_path, ntwist=20, skall=False)
    from qmcclean.pipeline import run_dmc
    assert run_dmc(dmc)['state'] == 'stopped_critical'
    assert cli.main(['review', str(dmc), '--release', 'all']) == 1
    assert 'no verdict' in capsys.readouterr().err
