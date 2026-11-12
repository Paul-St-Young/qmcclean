import csv

from qmcclean import cli
from qmcclean.survey import survey

from .fakes import make_dmc


def listing(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.name != 'survey.csv')


def tree(tmp):
    a = tmp / 'proj' / 'p300' / 'i1'
    b = tmp / 'proj' / 'p250' / 'i2'
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    make_dmc(a, ntwist=3, blocks=(10, 12))
    make_dmc(b, ntwist=4, blocks=(10, 12), skall=False, driver='batched', dmcdat=False)
    return tmp / 'proj'


def test_survey_fake_tree(tmp_path):
    root = tree(tmp_path)
    before = listing(tmp_path)
    df = survey(root)
    assert listing(tmp_path) == before
    assert len(df) == 2
    r = df.set_index('path')
    la, lb = r.loc[str(root / 'p300/i1/dmc')], r.loc[str(root / 'p250/i2/dmc')]
    assert la.driver == 'legacy' and lb.driver == 'batched'
    assert la.ntwist == 3 and lb.ntwist == 4
    assert la.series_blocks == '10,12'
    assert 'skall' in la.est_requested and 'skall' in la.est_present_g000
    assert 'skall' in lb.est_requested and 'skall' not in lb.est_present_g000
    assert 'LocalEnergy' in la.est_present_g000
    assert la.n_scalar_per_twist_series == 1 and la.n_unexpected == 0
    assert 'dmc.dat' in lb.absent_optional and la.absent_optional == ''
    assert la.error == '' and lb.error == ''


def test_survey_error_row_and_state(tmp_path):
    root = tree(tmp_path)
    bad = root / 'p250' / 'i2' / 'dmc' / 'dmc.g000.twistnum_0.in.xml'
    bad.write_text('<simulation><oops></simulation>')
    good = root / 'p300' / 'i1' / 'dmc'
    (good / 'qmcclean').mkdir()
    (good / 'qmcclean' / 'record.json').write_text('{"state": "cleaned"}')
    df = survey(root).set_index('path')
    assert df.loc[str(bad.parent), 'error'] != ''
    assert df.loc[str(good), 'error'] == ''
    assert df.loc[str(good), 'state'] == 'cleaned'
    assert len(df) == 2


def test_survey_prunes_and_duplicates(tmp_path):
    root = tree(tmp_path)
    d = root / 'p300' / 'i1' / 'dmc'
    (d / 'dmc.g050.twistnum_2.in.xml').write_text('<simulation/>')
    (d / 'dmc.g050.s001.scalar.dat').write_text('# index\n')
    (d / 'weird.scalar.dat').write_text('x')
    decoy = d / 'qmcclean' / 'old'
    decoy.mkdir(parents=True)
    make_dmc(decoy, ntwist=2, blocks=(10, 12))
    df = survey(root)
    assert len(df) == 2
    r = df.set_index('path').loc[str(d)]
    assert r.n_scalar_per_twist_series == 2 and r.n_unexpected == 1
    assert r.n_unparsed == 1
    assert not any('qmcclean' in p for p in df.path)


def test_survey_cli(tmp_path, monkeypatch):
    root = tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert cli.main(['survey', str(root)]) == 0
    rows = list(csv.DictReader(open(tmp_path / 'survey.csv')))
    assert len(rows) == 2 and {r['driver'] for r in rows} == {'legacy', 'batched'}
    out = tmp_path / 'o' / 'x.csv'
    out.parent.mkdir()
    assert cli.main(['survey', str(root), '--out', str(out)]) == 0
    assert out.exists()
