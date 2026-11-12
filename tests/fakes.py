"""Synthetic QMCPACK run folders in the real file formats (small, deterministic)."""
from pathlib import Path

import h5py
import numpy as np

NELEC = 8  # 4 up + 4 down
NION = 4
NK = 6
SCALARS = ['LocalEnergy', 'LocalEnergy_sq', 'LocalPotential', 'Kinetic',
           'ElecElec', 'IonIon', 'ElecIon']
LATTICE = [[8.965, 0.0, 0.0], [0.0, 10.352, 0.0], [0.0, 0.0, 8.984]]
JK2_KC = 1.0836
UU = [0.3603964576, 0.2546065956, 0.1731321408, 0.1122921693,
      0.06890854308, 0.03903202899, 0.01869993342, 0.007595590367]
UD = [0.545807856, 0.347254223, 0.2135760916, 0.12844203,
      0.07630226587, 0.04312403204, 0.0208865499, 0.008666999129]
KPOINTS = np.random.default_rng(42).random((NK, 3))  # skall k-vectors shared by all twists
JK2 = [0.0] * 6  # derived from the fake cell at the end of this module (_derive_jk2)


def _arr(a):
    return ' '.join('%.10g' % x for x in a)


def _jastrows(uu, ud, jk2, indent='      '):
    return f'''<jastrow type="Two-Body" name="J2" function="Bspline" print="yes" optimize="yes">
{indent}  <correlation speciesA="u" speciesB="u" size="8">
{indent}    <coefficients id="uu" type="Array" optimize="yes"> {_arr(uu)}</coefficients>
{indent}  </correlation>
{indent}  <correlation speciesA="u" speciesB="d" size="8">
{indent}    <coefficients id="ud" type="Array" optimize="yes"> {_arr(ud)}</coefficients>
{indent}  </correlation>
{indent}</jastrow>
{indent}<jastrow type="kSpace" name="Jk" source="ion0">
{indent}  <correlation kc="{JK2_KC}" type="Two-Body" symmetry="isotropic">
{indent}    <coefficients id="jk2" type="Array" optimize="yes"> {_arr(jk2)}</coefficients>
{indent}  </correlation>
{indent}</jastrow>'''


def _ion_positions():
    rng = np.random.default_rng(7)
    return rng.random((NION, 3)) * np.array([8.9, 10.3, 8.9])


def _twist_xml(twist, blocks, driver, steps=(98, 107)):
    lat = '\n'.join('                  %.8f %.8f %.8f' % tuple(r) for r in LATTICE)
    pos = '\n'.join('                  %.8f %.8f %.8f' % tuple(r) for r in _ion_positions())
    return f'''<?xml version="1.0"?>
<simulation>
   <project id="dmc" series="0">
      <parameter name="driver_version"      >    {driver}          </parameter>
      <application name="qmcpack" role="molecu" class="serial" version="1.0"/>
   </project>
   <qmcsystem>
      <simulationcell>
         <parameter name="lattice" units="bohr">
{lat}
         </parameter>
         <parameter name="bconds">
            p p p
         </parameter>
         <parameter name="LR_dim_cutoff"       >    30                 </parameter>
         <parameter name="LR_handler"          >    ewald              </parameter>
      </simulationcell>
      <particleset name="e" random="yes" randomsrc="ion0">
         <group name="u" size="{NELEC // 2}" mass="1.0">
            <parameter name="charge"              >    -1                    </parameter>
         </group>
         <group name="d" size="{NELEC // 2}" mass="1.0">
            <parameter name="charge"              >    -1                    </parameter>
         </group>
      </particleset>
      <particleset name="ion0">
         <group name="H" size="{NION}" mass="1837.3622193391143">
            <parameter name="charge"              >    1                     </parameter>
            <parameter name="valence"             >    1                     </parameter>
            <attrib name="position" datatype="posArray" condition="0">
{pos}
            </attrib>
         </group>
      </particleset>
      <wavefunction name="psi0" target="e">
         <sposet_builder type="bspline" href="../scf/pwscf_output/pwscf.pwscf.h5" tilematrix="1 0 0 0 1 0 0 0 1" twistnum="{twist}" source="ion0" version="0.10" meshfactor="1.0" precision="float" truncate="no">
            <sposet type="bspline" name="spo_ud" size="{NELEC // 2}" spindataset="0"/>
         </sposet_builder>
         <determinantset>
            <slaterdeterminant>
               <determinant id="updet" group="u" sposet="spo_ud" size="{NELEC // 2}"/>
               <determinant id="downdet" group="d" sposet="spo_ud" size="{NELEC // 2}"/>
            </slaterdeterminant>
         </determinantset>
         {_jastrows(UU, UD, JK2, indent='         ')}
      </wavefunction>
      <hamiltonian name="h0" type="generic" target="e">
         <pairpot type="coulomb" name="ElecElec" source="e" target="e"/>
         <pairpot type="coulomb" name="IonIon" source="ion0" target="ion0"/>
         <pairpot type="coulomb" name="ElecIon" source="ion0" target="e"/>
         <estimator type="Force" name="force" mode="cep" addionion="yes">
            <parameter name="rcut"                >    0.4                   </parameter>
            <parameter name="nbasis"              >    4                     </parameter>
            <parameter name="weightexp"           >    2                     </parameter>
         </estimator>
         <estimator name="skall" type="skall" hdf5="yes" source="ion0" target="e"/>
      </hamiltonian>
   </qmcsystem>
   <qmc method="vmc" move="pbyp" checkpoint="0">
      <parameter name="blocks"              >    {blocks[0]}             </parameter>
      <parameter name="steps"               >    {steps[0]}              </parameter>
      <parameter name="timestep"            >    0.8             </parameter>
      <parameter name="samples"             >    1632            </parameter>
      <parameter name="walkers"             >    16              </parameter>
   </qmc>
   <qmc method="dmc" move="pbyp" checkpoint="16">
      <parameter name="blocks"              >    {blocks[1]}             </parameter>
      <parameter name="steps"               >    {steps[1]}             </parameter>
      <parameter name="timestep"            >    0.015           </parameter>
      <parameter name="targetwalkers"       >    1632            </parameter>
   </qmc>
</simulation>
'''


STATE_XML = ('<?xml version="1.0" encoding="utf-8"?>\n<state><branchmode>0000001100'
             '</branchmode><vparam><tau>0.8</tau><etrial>-4.7</etrial></vparam></state>')


def _write_table(path, cols, data, first='index'):
    """Write a scalar.dat-style table: '#' header line, fixed-width rows."""
    hdr = '#' + first.rjust(8) + ''.join(c.rjust(16) for c in cols)
    lines = [hdr]
    for i, row in enumerate(data):
        lines.append(str(i).rjust(10) + ''.join('%16.6e' % x for x in row))
    Path(path).write_text('\n'.join(lines) + '\n')


def _block_data(rng, nb, series, kind, twist):
    """Per-block observables for one (twist, series). Returns dict of arrays."""
    vmc = series == 0
    e0 = -4.7 + 0.01 * (twist % 7) / 7
    sd_e = 0.30 if vmc else 0.15
    var0 = 2.0 if vmc else 1.6
    e = e0 + sd_e * rng.standard_normal(nb)
    var = var0 + 0.05 * rng.standard_normal(nb)
    w = (1568.0 if vmc else 174624.0) * (1 + 0.01 * rng.standard_normal(nb))
    if not vmc and kind == 'variance':
        var = var * 6
    if not vmc and kind == 'stuck':
        e = np.full(nb, e0)
        w = w * 5
    sd_f = 0.5 if vmc else 0.2  # error ratio ~2 with 68 VMC / 44 DMC post-equil blocks (100,60)
    if vmc and kind == 'force_vmc':
        sd_f *= 0.25
    force = sd_f * rng.standard_normal((nb, NION, 3))
    kin = 3.0 + 0.1 * rng.standard_normal(nb)
    d = {'LocalEnergy': e, 'LocalEnergy_sq': e ** 2 + var,
         'Kinetic': kin, 'LocalPotential': e - kin,
         'ElecElec': -1.5 + 0.05 * rng.standard_normal(nb),
         'IonIon': np.full(nb, -2.0),
         'ElecIon': -1.2 + 0.05 * rng.standard_normal(nb),
         'BlockWeight': w, 'force': force}
    return d


def _write_series(dmc, g, s, d, skall, dmcdat, rng, steps=1, partial=0):
    nb = len(d['LocalEnergy'])
    pre = dmc / f'dmc.g{g:03d}.s{s:03d}'
    fcols = ['force_%d_%d' % (i, j) for i in range(NION) for j in range(3)]
    cols = SCALARS + ['force'] + fcols + ['BlockWeight', 'BlockCPU', 'AcceptRatio']
    fl = d['force'].reshape(nb, -1)
    rows = np.column_stack(
        [d[c] for c in SCALARS] + [np.zeros(nb)] + [fl] +
        [d['BlockWeight'], np.full(nb, 0.56), np.full(nb, 0.62)])
    _write_table(str(pre) + '.scalar.dat', cols, rows)
    with h5py.File(str(pre) + '.stat.h5', 'w', libver='latest') as f:
        for c in SCALARS:
            f.create_dataset(c + '/value', data=d[c][:, None])
        f.create_dataset('force/value', data=d['force'])
        if skall:
            rng.random((NK, 3))  # keep the random stream of earlier fakes
            # real runs share the skall k-vectors across twists
            f.create_dataset('skall/kpoints/value', data=KPOINTS)
            f.create_dataset('skall/rhok_e_e/value',
                             data=NELEC * 0.8 + 0.1 * rng.standard_normal((nb, NK)))
            f.create_dataset('skall/rhok_e_r/value', data=0.1 * rng.standard_normal((nb, NK)))
            f.create_dataset('skall/rhok_e_i/value', data=0.1 * rng.standard_normal((nb, NK)))
    for ext in ('config.h5', 'random.h5'):
        Path(f'{pre}.{ext}').write_bytes(b'\0' * 64)
    Path(f'{pre}.qmc.xml').write_text('<?xml version="1.0"?>\n<simulation/>\n')
    if s == 1:
        Path(f'{pre}.cont.xml').write_text('<?xml version="1.0"?>\n<simulation/>\n')
        if dmcdat:
            # one row per DMC step, as QMCPACK writes it: `steps` rows per block whose mean
            # is the block value (zero-sum jitter), then `partial` rows of an unfinished block
            var = d['LocalEnergy_sq'] - d['LocalEnergy'] ** 2
            cols = ['LocalEnergy', 'Variance', 'Weight', 'NumOfWalkers',
                    'AvgSentWalkers', 'TrialEnergy', 'DiffEff', 'LivingFraction']
            blk = np.column_stack([
                d['LocalEnergy'], var, d['BlockWeight'] / steps,
                d['BlockWeight'] / steps, np.zeros(nb), d['LocalEnergy'] - 0.1,
                np.full(nb, 0.99), np.ones(nb)])
            jit = np.linspace(-1, 1, steps) * 1e-3 if steps > 1 else np.zeros(1)
            rows = np.repeat(blk, steps, axis=0)
            rows[:, [0, 5]] += np.tile(jit, nb)[:, None]
            if partial:
                rows = np.vstack([rows, np.repeat(blk[-1:], partial, axis=0)])
            _write_table(str(pre) + '.dmc.dat', cols, rows, first='Index')


def make_dmc(tmp, ntwist=20, blocks=(40, 30), short=None, skall=True, dmcdat=True,
             outlier=None, driver='legacy', steps=(98, 4), partial=None):
    """Write a fake dmc folder tmp/dmc and return its path.

    steps: MC steps per block (VMC, DMC) written to the XML; dmc.dat gets steps[1] rows per
    block. partial: twist -> extra dmc.dat rows of an unfinished trailing block.
    """
    partial = partial or {}
    short = short or {}
    outlier = outlier or {}
    tmp = Path(tmp)
    dmc = tmp / 'dmc'
    dmc.mkdir(parents=True, exist_ok=True)
    for t in range(ntwist):
        rng = np.random.default_rng(1000 + t)
        (dmc / f'dmc.g{t:03d}.twistnum_{t}.in.xml').write_text(
            _twist_xml(t, blocks, driver, steps))
        (dmc / f'dmc.g{t:03d}.twistnum_{t}.in.g{t:03d}.qmc').write_text(STATE_XML)
        (dmc / f'dmc.g{t:03d}.twist_info.dat').write_text(
            '         2.000000    0.116805964355   0.101156947542   0.116558475972\n')
        (dmc / f'spo_ud.tile_100010001.spin_0.tw_{t}.g{t}.bandinfo.dat').write_text(
            '#  Band    State   TwistIndex BandIndex Energy\n       0        0        0        0    -0.599657\n')
        for s in (0, 1):
            nb = min(blocks[s], short.get(t, blocks[s])) if s == 1 else blocks[s]
            d = _block_data(rng, nb, s, outlier.get(t), t)
            _write_series(dmc, t, s, d, skall, dmcdat, rng, steps[1], partial.get(t, 0))
    (dmc / 'dmc.out').write_text('QMCPACK fake output\n')
    (dmc / 'dmc.err').write_text('')
    return dmc


def _opt_xml(step, converged, rng):
    if converged:
        f = 1 + 0.002 * rng.standard_normal(len(JK2))
        jk2 = np.array(JK2) * f
        uu, ud = np.array(UU), np.array(UD)
    else:
        jk2 = np.array(JK2) * 1.06 ** step
        uu, ud = np.array(UU) * 1.02 ** step, np.array(UD) * 1.02 ** step
    return f'''<?xml version="1.0"?>
<qmcsystem>
  <wavefunction name="psi0" target="e">
         <determinantset>
            <slaterdeterminant>
               <determinant id="updet" group="u" sposet="spo_ud" size="{NELEC // 2}"/>
               <determinant id="downdet" group="d" sposet="spo_ud" size="{NELEC // 2}"/>
            </slaterdeterminant>
         </determinantset>
         {_jastrows(uu, ud, jk2, indent='         ')}
      <override_variational_parameters href="opt.s{step:03d}.vp.h5"/></wavefunction>
</qmcsystem>
'''


def make_opt(tmp, nstep=12, converged=True, e_slope=0.0, vnoise=1.0):
    """Write a fake opt folder tmp/opt and return its path.

    Unconverged: variance falls 0.03/step (error ~0.003), jk2[0] drifts 6%/step.
    Converged: variance and jk2 flat within errors.
    e_slope: extra energy drop per step toward the end; vnoise: scale of the variance noise.
    """
    opt = Path(tmp) / 'opt'
    opt.mkdir(parents=True, exist_ok=True)
    nrow = 20
    for step in range(nstep):
        rng = np.random.default_rng(5000 + step)
        v = 1.5 if converged else 2.0 - 0.03 * step
        e0 = (-4.7 if converged else -4.7 + 0.003 * (nstep - step)) + e_slope * (nstep - step)
        e = e0 + 0.05 * rng.standard_normal(nrow)
        var = v + vnoise * 0.003 * np.sqrt(nrow) * rng.standard_normal(nrow)
        kin = 3.0 + 0.1 * rng.standard_normal(nrow)
        cols = SCALARS + ['BlockWeight', 'BlockCPU', 'AcceptRatio']
        rows = np.column_stack([
            e, e ** 2 + var, e - kin, kin, np.full(nrow, -1.5), np.full(nrow, -2.0),
            np.full(nrow, -1.2), np.full(nrow, 9384.0), np.full(nrow, 0.56),
            np.full(nrow, 0.62)])
        _write_table(opt / f'opt.s{step:03d}.scalar.dat', cols, rows)
        (opt / f'opt.s{step:03d}.opt.xml').write_text(_opt_xml(step, converged, rng))
        (opt / f'opt.s{step:03d}.stat.h5').write_bytes(b'\0' * 64)
        (opt / f'opt.s{step:03d}.vp.h5').write_bytes(b'\0' * 64)
    (opt / 'opt.in.xml').write_text(_twist_xml(0, (nrow, nrow), 'legacy'))
    (opt / 'opt.out').write_text('QMCPACK fake output\n')
    return opt


def _derive_jk2():
    """jk2 for the fake cell from its own physics, not hand-tuned numbers.

    Shells 0..4: -0.9 u_RPA(k) (so k^2 u_T / k^2 u_RPA ~ 1 at the smallest shell);
    last shell below kc: usr(k5) - usr(k6), so u_T is continuous across kc.
    """
    import tempfile
    from chiesa_correction import calc_usrk, gaskell_rpa_uk, heg_kfermi
    from qharv.inspect import axes_pos
    from qharv.seed import xml
    from qmcclean import jastrow
    axes = np.array(LATTICE)
    k = jastrow.kshells(axes, 3.0)
    n = len(JK2)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'a.xml'
        p.write_text(_twist_xml(0, (1, 1), 'legacy'))
        usr = calc_usrk(k, xml.read(str(p)))
    rs = axes_pos.rs(axes, NELEC)
    rpa = gaskell_rpa_uk(k[:n], rs, heg_kfermi(rs))
    jk = -0.9 * rpa
    jk[n - 1] = usr[n - 1] - usr[n]
    return jk


JK2[:] = _derive_jk2().tolist()


def edit_table(stats_path, name, fn):
    """Hand-edit a qmcclean text table next to stats_path: df = fn(df) (or fn edits in place)."""
    from qmcclean import stats
    p = Path(stats_path).parent / name
    df = stats.read_table(p)
    out = fn(df)
    df = df if out is None else out
    stats.write_table(p, list(df.columns), df.to_dict('records'))
