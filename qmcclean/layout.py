"""Discover the files and structure of a QMCPACK dmc or opt run folder."""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from qharv.reel import mole
from qharv.seed import xml

from . import config

TWIST_XML = re.compile(r'^dmc\.g(\d+)\.twistnum_(\d+)\.in\.xml$')
OPT_FILE = re.compile(r'^opt\.s(\d+)(\..+)$')


@dataclass
class SeriesSpec:
    index: int
    method: str
    blocks: int
    steps: int = 1  # MC steps per block; dmc.dat has one row per step


@dataclass
class DmcLayout:
    path: Path
    prefix: str
    twists: list
    series: list
    estimators: list
    driver: str
    nelec: int
    axes: np.ndarray
    files: dict  # (twist, series, suffix) -> Path
    unexpected: dict = field(default_factory=dict)  # twist -> [Path]
    no_xml: dict = field(default_factory=dict)  # group -> [Path] big files with no twist xml


@dataclass
class OptLayout:
    path: Path
    steps: list
    files: dict  # (step, suffix) -> Path


def _prefix(path):
    """Path segments joined by '_' (qharv_db.meta.get_prefix_from_path style)."""
    parts = [p for p in Path(path).resolve().parts if p != '/']
    return '_'.join(parts[-5:])


def discover_dmc(path, names=None):
    """names: optional pre-listed file names of path (e.g. from os.walk); then no listing is done."""
    path = Path(path)
    if names is None:
        with os.scandir(path) as it:  # one pass; DirEntry.is_file() uses d_type, no stat
            names = [e.name for e in it if e.is_file()]
    names = sorted(names)
    g2t = {}
    for n in names:
        m = TWIST_XML.match(n)
        if m:
            g2t[int(m.group(1))] = int(m.group(2))
    ref = path / config.KEEP[0]
    if not ref.exists():
        raise FileNotFoundError('%s not found' % ref)
    doc = xml.read(str(ref))
    series = []
    for i, q in enumerate(doc.findall('.//qmc')):
        nb = xml.get_param(q, 'blocks')
        ns = xml.get_param(q, 'steps')
        series.append(SeriesSpec(i, q.get('method'), int(nb), int(ns) if ns else 1))
    ests = [e.get('name') for e in doc.findall('.//hamiltonian/estimator')
            if e.get('name')]
    driver = (xml.get_param(doc, 'driver_version') or 'legacy').strip()
    nseries = len(series)

    files = {}
    unexpected = {}
    no_xml = {}
    for name in names:
        f = path / name
        if name.endswith('.xml') and '.in.' in name:
            continue
        try:
            meta = mole.interpret_qmcpack_fname(name)
        except (RuntimeError, ValueError, IndexError):
            continue
        if not meta['grouped']:
            continue
        suffix = '.%s.%s' % (meta['category'], meta['ext'])
        big = suffix in config.BIG_SUFFIXES
        if meta['group'] not in g2t:  # no dmc.gNNN.twistnum_T.in.xml: twist unknown
            if big:
                no_xml.setdefault(meta['group'], []).append(f)
            continue
        twist = g2t[meta['group']]
        key = (twist, meta['series'], suffix)
        if meta['series'] >= nseries and big:
            unexpected.setdefault(twist, []).append(f)
        elif key in files:
            if big:
                unexpected.setdefault(twist, []).append(f)
        else:
            files[key] = f
    return DmcLayout(
        path=path, prefix=_prefix(path),
        twists=sorted(g2t.values()), series=series, estimators=ests,
        driver=driver, nelec=int(xml.get_nelec(doc)), axes=xml.get_axes(doc),
        files=files, unexpected=unexpected, no_xml=no_xml)


def discover_opt(path):
    path = Path(path)
    files = {}
    for f in sorted(path.iterdir()):
        m = OPT_FILE.match(f.name)
        if m and f.is_file():
            files[(int(m.group(1)), m.group(2))] = f
    steps = sorted({k[0] for k in files})
    return OptLayout(path=path, steps=steps, files=files)


def classify(path) -> Literal['opt', 'dmc', 'calc', 'unknown']:
    path = Path(path)
    if (path / 'opt').is_dir() or (path / 'dmc').is_dir():
        return 'calc'
    names = [f.name for f in path.iterdir()] if path.is_dir() else []
    if any(TWIST_XML.match(n) for n in names):
        return 'dmc'
    if any(OPT_FILE.match(n) for n in names):
        return 'opt'
    return 'unknown'
