import numpy as np
import pytest

from qmcclean import jastrow
from tests.fakes import JK2, JK2_KC, LATTICE, make_dmc


def _xml(tmp_path):
    make_dmc(tmp_path, ntwist=1, blocks=(5, 5))
    return tmp_path / 'dmc' / 'dmc.g000.twistnum_0.in.xml'


def test_fake_shell_count_matches_jk2():
    k = jastrow.kshells(LATTICE, 3.0)
    assert (k < JK2_KC).sum() == len(JK2)


def test_trial_uk_continuity(tmp_path):
    p = _xml(tmp_path)
    k = jastrow.kshells(LATTICE, 3.0)
    u = jastrow.trial_uk(p, k)
    n = len(JK2)
    assert abs(u[n] / u[n - 1] - 1) < 0.2  # last shell below kc -> first above


def test_trial_uk_subtracts_jk2(tmp_path):
    p = _xml(tmp_path)
    k = jastrow.kshells(LATTICE, 3.0)
    full = jastrow.trial_uk(p, k)
    import chiesa_correction as chc
    from qharv.seed import xml
    usr = chc.calc_usrk(k, xml.read(str(p)))
    n = len(JK2)
    assert np.allclose(full[:n], usr[:n] - np.array(JK2))
    assert np.allclose(full[n:], usr[n:])


def test_rpa_uk_vector():
    u = jastrow.rpa_uk([0.6, 1.0], 1.4, 1.0)
    assert u.shape == (2,) and np.all(u > 0)


def test_delta_u_normalization():
    k = np.array([1.0, 1.0, 2.0, 2.0])
    N, vol = 8, 100.0
    S = np.array([0.5, 0.5, 0.8, 0.8])
    z = np.zeros(4)
    ks, du, err = jastrow.delta_u(k, N * S, N * S, z + .01, z + .01, N, vol)
    assert np.allclose(ks, [1.0, 2.0]) and np.allclose(du, 0)
    Sd = S.copy()
    Sd[:2] *= 0.9
    ks, du, err = jastrow.delta_u(k, N * S, N * Sd, z + .01, z + .01, N, vol)
    rho = N / vol
    assert np.isclose(2 * rho * du[0], (1 / 0.9 - 1) / 0.5)
    assert np.isclose(du[1], 0)
    assert np.all(err > 0)


def test_trial_uk_axes_override(tmp_path):
    p = _xml(tmp_path)
    k = jastrow.kshells(LATTICE, 3.0)
    assert np.allclose(jastrow.trial_uk(p, k), jastrow.trial_uk(p, k, axes=LATTICE))


def test_trial_uk_shell_mismatch_raises(tmp_path):
    p = _xml(tmp_path)
    k = jastrow.kshells(LATTICE, 3.0)[:3]  # fewer than len(jk2) shells below kc
    with pytest.raises(ValueError):
        jastrow.trial_uk(p, k)


def test_delta_u_error_hand_computed():
    # one k vector: S_v = 0.5, S_d = 0.45, errors 0.01 on both (S units), N = 8, V = 100
    N, vol = 8, 100.0
    ks, du, err = jastrow.delta_u([1.0], [N * 0.5], [N * 0.45], [N * 0.01], [N * 0.01], N, vol)
    rho = N / vol
    assert np.isclose(du[0], (1 / 0.45 - 1 / 0.5) / (2 * rho))
    expect = np.sqrt((0.01 / 0.45 ** 2) ** 2 + (0.01 / 0.5 ** 2) ** 2) / (2 * rho)
    assert np.isclose(err[0], expect) and np.isclose(err[0], 0.397, atol=1e-3)
    # two equivalent k vectors in the shell: independent errors average down by sqrt(2)
    ks, du2, err2 = jastrow.delta_u([1.0, 1.0], [N * 0.5] * 2, [N * 0.45] * 2,
                                    [N * 0.01] * 2, [N * 0.01] * 2, N, vol)
    assert np.isclose(du2[0], du[0]) and np.isclose(err2[0], err[0] / np.sqrt(2))


def test_pymod_paths_from_env(monkeypatch):
    monkeypatch.setenv('QMCCLEAN_PYMODS', '/some/where')
    assert jastrow.pymod_paths() == ['/some/where', '/some/where/qharv_db']
    monkeypatch.delenv('QMCCLEAN_PYMODS')
    assert jastrow.pymod_paths()[0] == jastrow.PYMODS_DEFAULT
