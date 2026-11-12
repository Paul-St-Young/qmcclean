from qmcclean.config import OPT_RPA_RATIO
import json
import re

import numpy as np
import pytest

from qmcclean.optcheck import check_opt, fmt
from tests.fakes import make_opt


def test_fmt_one_digit():
    assert fmt(-85.144, 0.016) == '-85.14(2)'
    assert fmt(0.08559, 0.00012) == '0.0856(1)'
    assert fmt(-0.0312, 0.0031) == '-0.031(3)'


def test_opt_converged(tmp_path):
    opt = make_opt(tmp_path, converged=True)
    rep = check_opt(opt, nequil=8)
    assert not rep.flagged, rep.reasons
    assert not rep.reasons
    assert 'notes' not in rep.last
    assert np.isfinite(rep.last['rpa_ratio']) and rep.last['rpa_ratio'] > 0.8


def test_opt_unconverged(tmp_path):
    opt = make_opt(tmp_path, converged=False)
    rep = check_opt(opt, nequil=8)
    assert rep.flagged
    txt = ' | '.join(rep.reasons)
    assert 'variance still falling' in txt and 'σ' in txt
    assert 'jk2[0]' in txt


def test_opt_outputs(tmp_path):
    opt = make_opt(tmp_path, converged=False)
    check_opt(opt, nequil=8)
    rec = json.loads((opt / 'qmcclean' / 'opt_record.json').read_text())
    assert rec['flagged'] is True and rec['time'].endswith(('-05:00', '-04:00'))
    from qmcclean.stats import read_table
    tab = opt / 'qmcclean' / 'opt.params.dat'
    head = tab.read_text().splitlines()[0].split()
    assert head[:6] == ['#', 'series', 'LocalEnergy_mean', 'LocalEnergy_error',
                        'Variance_mean', 'Variance_error']
    assert 'jk2_0' in head and all('_mean' not in c and '_error' not in c for c in head[6:])
    df = read_table(tab)
    assert len(df) == 12 and list(df['series']) == list(range(12))
    assert not (opt / 'qmcclean' / 'opt_trace.csv').exists()
    assert (opt / 'qmcclean' / 'uk.png').exists() or 'uk.png not written' in str(rec['last'].get('notes'))
    assert not list((opt / 'qmcclean').glob('*.tmp*'))


def test_opt_too_few_steps(tmp_path):
    opt = make_opt(tmp_path, nstep=1)
    rep = check_opt(opt, nequil=8)
    assert rep.flagged and 'too few' in rep.reasons[0]


def test_slope_sigma_endpoint(tmp_path):
    import pandas as pd
    from qmcclean.optcheck import _slope
    df = pd.DataFrame({'V': [3.0, 2.0, 1.0, 0.5, 0.3], 'V_err': [9, 9, 0.1, 0.2, 0.1]})
    m, me, nsig, ex = _slope(df, 'V', 3)
    assert np.isclose(m, (0.3 - 2.0) / 3)
    assert np.isclose(me, np.hypot(0.1, 9) / 3)  # endpoints only: interior errors do not enter
    assert np.isclose(nsig, abs(m) / me)


def test_low_rpa_ratio_flags(tmp_path):
    opt = make_opt(tmp_path, converged=True)
    x = opt / 'opt.s011.opt.xml'
    old = x.read_text()
    t = re.sub(r'(id="jk2"[^>]*>) *[^ <]+', r'\g<1> -1.0', old)
    assert t != old
    x.write_text(t)
    rep = check_opt(opt, nequil=8)
    assert rep.flagged and any('below RPA' in r for r in rep.reasons)
    assert rep.last['rpa_ratio'] < OPT_RPA_RATIO


def test_energy_flag(tmp_path):
    opt = make_opt(tmp_path, converged=True, e_slope=0.1)
    rep = check_opt(opt, nequil=8)
    assert any(r.startswith('energy still falling') for r in rep.reasons)
    assert not any(r.startswith('variance') for r in rep.reasons)


def test_noise_larger_than_drift_not_flagged(tmp_path):
    opt = make_opt(tmp_path, converged=False, vnoise=40.0)
    rep = check_opt(opt, nequil=8)
    assert not any(r.startswith('variance') for r in rep.reasons), rep.reasons
    assert rep.last['V_nsigma'] < 3


def test_unevaluable_rpa_is_flagged(tmp_path):
    opt = make_opt(tmp_path, converged=True)
    (opt / 'opt.in.xml').write_text('<?xml version="1.0"?>\n<simulation/>\n')
    rep = check_opt(opt, nequil=8)
    assert rep.flagged and any('could not evaluate RPA check' in r for r in rep.reasons)


def test_nequil_too_large_flags(tmp_path):
    opt = make_opt(tmp_path, converged=True)
    rep = check_opt(opt, nequil=50)
    assert rep.flagged and any('could not evaluate variance' in r for r in rep.reasons)
    rec = json.loads((opt / 'qmcclean' / 'opt_record.json').read_text(),
                     parse_constant=lambda c: pytest.fail('non-strict JSON constant ' + c))
    assert rec['flagged'] is True


def test_missing_jk2_noted_not_crash(tmp_path):
    opt = make_opt(tmp_path, converged=True)
    for x in opt.glob('opt.s*.opt.xml'):
        x.write_text('<qmcsystem/>')
    rep = check_opt(opt, nequil=8)
    assert any('jk2' in n for n in rep.last['notes'])
    assert rep.flagged  # RPA check could not run


def test_opt_params_only_optimised_kspace():
    from lxml import etree
    from qmcclean.optcheck import opt_params
    doc = etree.fromstring(
        '<w><jastrow type="Two-Body"><correlation>'
        '<coefficients id="uu" type="Array" optimize="yes"> 1 2 </coefficients></correlation></jastrow>'
        '<jastrow type="kSpace"><correlation>'
        '<coefficients id="jk1" type="Array" optimize="no"> 9 </coefficients></correlation><correlation>'
        '<coefficients id="jk2" type="Array"> -3 -4 </coefficients></correlation></jastrow></w>')
    assert opt_params(doc) == {'jk2_0': -3.0, 'jk2_1': -4.0}


def test_input_named_opt_xml(tmp_path):
    from tests.fakes import make_opt as _mk
    opt = _mk(tmp_path, converged=True)
    (opt / 'opt.in.xml').rename(opt / 'opt.xml')
    rep = check_opt(opt, nequil=8)
    assert not any('opt.in.xml' in r for r in rep.reasons), rep.reasons
    assert np.isfinite(rep.last['rpa_ratio'])
