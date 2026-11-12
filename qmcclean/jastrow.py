"""Jastrow u(k): trial FT[B-spline J2] + (-jk2), RPA reference, and the DMC correction du(k).

Conventions (verified in explore): 2 rho u = 1/S - 1/S0; dsk from qharv is N*S(k).
"""
import os
import sys

import numpy as np

PYMODS_DEFAULT = '/home/pyang/soft/pymods/solid_hydrogen'


def pymod_paths():
    """sys.path entries for chiesa_correction and qharv_db; base dir from env QMCCLEAN_PYMODS."""
    base = os.environ.get('QMCCLEAN_PYMODS', PYMODS_DEFAULT)
    return [base, os.path.join(base, 'qharv_db')]


for _p in pymod_paths():
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chiesa_correction as chc  # noqa: E402
from qharv.seed import xml  # noqa: E402

SHELL_DECIMALS = 4


def kshells(axes, kmax):
    """Unique reciprocal-lattice |G| with 0 < |G| < kmax, rounded to 1e-4."""
    rax = 2 * np.pi * np.linalg.inv(np.asarray(axes, float)).T
    n = int(np.ceil(kmax / np.linalg.norm(rax, axis=1).min())) + 1
    r = range(-n, n + 1)
    g = np.array([i * rax[0] + j * rax[1] + k * rax[2] for i in r for j in r for k in r])
    m = np.round(np.linalg.norm(g, axis=1), SHELL_DECIMALS)
    return np.unique(m[(m > 0) & (m < kmax)])


def jk2_of(doc):
    """(jk2 coefficients, kc) of the kSpace Jastrow in a parsed XML doc."""
    jk = np.array(doc.find('.//coefficients[@id="jk2"]').text.split(), float)
    kc = float(doc.find('.//jastrow[@type="kSpace"]/correlation').get('kc'))
    return jk, kc


def trial_uk(xml_path, k, axes=None):
    """Trial u(k) at shells k: FT of the B-spline J2 plus (-jk2) on shells below kc.

    `axes` overrides the cell from the XML (opt.sNNN.opt.xml has no cell; use the
    run's opt.in.xml/dmc XML for it). The XML must contain the J2 and kSpace Jastrows.
    """
    doc = xml.read(str(xml_path))
    k = np.asarray(k, float)
    if axes is None:
        usr = chc.calc_usrk(k, doc)
    else:
        usr = _usrk_with_axes(k, doc, axes)
    jk, kc = jk2_of(doc)
    below = int((k < kc).sum())
    if below != len(jk):
        raise ValueError(f'{below} shells below kc={kc} but len(jk2)={len(jk)}')
    u = np.array(usr, float)
    u[:len(jk)] += -jk
    return u


def _usrk_with_axes(k, doc, axes):
    from qharv.inspect import axes_pos
    axes = np.asarray(axes, float)
    rcut = axes_pos.rwsc(axes)
    return chc.evaluate_ft(k, chc.get_fusr(doc, rcut), rcut)


def rpa_uk(k, rs, kf):
    """Gaskell RPA u(k); vector input."""
    return chc.gaskell_rpa_uk(np.atleast_1d(np.asarray(k, float)), rs, kf)


def delta_u(k_dsk, dsk_vmc, dsk_dmc, err_vmc, err_dmc, nelec, volume):
    """Shell-averaged du = (1/S_DMC - 1/S_VMC)/(2 rho) with its error.

    k_dsk: |k| (n,) or k vectors (n,3); dsk_* are N*S(k) (divided by nelec here).
    Returns (k_shell, du, du_err).
    """
    k = np.asarray(k_dsk, float)
    if k.ndim == 2:
        k = np.linalg.norm(k, axis=1)
    kr = np.round(k, SHELL_DECIMALS)
    shells = np.unique(kr)
    rho = nelec / volume
    sv, sd, ev, ed = (np.asarray(a, float) / nelec for a in (dsk_vmc, dsk_dmc, err_vmc, err_dmc))
    out = []
    for kk in shells:
        s = kr == kk
        n = s.sum()
        out.append((sv[s].mean(), sd[s].mean(),
                    np.sqrt((ev[s] ** 2).sum()) / n, np.sqrt((ed[s] ** 2).sum()) / n))
    sv, sd, ev, ed = (np.array(a) for a in zip(*out))
    du = (1 / sd - 1 / sv) / (2 * rho)
    err = np.sqrt((ed / sd ** 2) ** 2 + (ev / sv ** 2) ** 2) / (2 * rho)
    return shells, du, err
