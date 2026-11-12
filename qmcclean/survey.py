"""Read-only inventory of a run tree: one row per dmc folder."""
import json
import os
from collections import Counter
from pathlib import Path

import h5py
import pandas as pd

from qharv.reel import mole

from . import config
from .layout import TWIST_XML, discover_dmc
from .structure import OPTIONAL

COLUMNS = ['prefix', 'path', 'driver', 'ntwist', 'series_blocks', 'est_requested',
           'est_present_g000', 'n_scalar_per_twist_series', 'n_unexpected', 'n_unparsed',
           'absent_optional', 'state', 'error']


def find_dmc(root):
    """Yield (path, names) of every dmc folder under root; no descent into one or into qmcclean/."""
    for dp, dns, fns in os.walk(root):
        if any(TWIST_XML.match(n) for n in fns):
            dns[:] = []
            yield Path(dp), fns
        else:
            dns[:] = sorted(d for d in dns if d != config.SUBDIR)


def _n_scalar(L, names):
    """(max scalar.dat count per (twist, series), number of unparsable scalar-like names)."""
    c = Counter(k[:2] for k in L.files if k[2] == '.scalar.dat')
    for t, fs in L.unexpected.items():
        for f in fs:
            if f.name.endswith('.scalar.dat'):
                try:
                    c[(t, mole.interpret_qmcpack_fname(f.name)['series'])] += 1
                except (RuntimeError, ValueError, IndexError, KeyError):
                    pass
    bad = 0
    for n in names:
        if n.endswith('.scalar.dat'):
            try:
                ok = mole.interpret_qmcpack_fname(n)['grouped']
            except (RuntimeError, ValueError, IndexError, KeyError):
                ok = False
            bad += not ok
    return (max(c.values()) if c else 0), bad


def _row(path, names):
    r = dict.fromkeys(COLUMNS, '')
    r['path'] = str(path)
    try:
        rec = path / config.SUBDIR / 'record.json'
        if rec.exists():
            try:
                r['state'] = json.loads(rec.read_text()).get('state', '')
            except Exception as e:
                r['state'] = 'unreadable record (%s)' % type(e).__name__
        L = discover_dmc(path, names)
        r['prefix'] = L.prefix
        r['driver'] = L.driver
        r['ntwist'] = len(L.twists)
        r['series_blocks'] = ','.join(str(s.blocks) for s in L.series)
        r['est_requested'] = ','.join(L.estimators)
        r['n_scalar_per_twist_series'], r['n_unparsed'] = _n_scalar(L, names)
        r['n_unexpected'] = len(L.unexpected)
        absent = []
        for suffix, series in [('.dmc.dat', [1])] + [(x, [s.index for s in L.series])
                                                      for x in OPTIONAL]:
            n = sum((t, i, suffix) not in L.files for t in L.twists for i in series)
            if n:
                absent.append('%s:%d' % (suffix[1:], n))
        r['absent_optional'] = ';'.join(absent)
        if L.twists:
            g0 = path / 'dmc.g000.s000.stat.h5'
            if g0.exists():
                with h5py.File(g0, 'r') as f:
                    r['est_present_g000'] = ','.join(sorted(f.keys()))
    except Exception as e:
        r['error'] = '%s: %s' % (type(e).__name__, e)
    return r


def survey(root):
    rows = [_row(p, names) for p, names in find_dmc(root)]
    return pd.DataFrame(rows, columns=COLUMNS)
