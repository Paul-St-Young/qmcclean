"""Tunable defaults. CLI options may override NEQUIL."""

NEQUIL = {0: 32, 1: 16}  # equilibration blocks per series (VMC, DMC)
MIN_BLOCK_FRACTION = 0.9  # a twist with >= this fraction of requested blocks is accepted (noted, not flagged)
DROP_FRACTION = 0.05  # flagged fraction of twists at/above which the run is held
ROBUST_Z = 5.0  # robust z-score cut for twist outliers
ERR_RATIO = 2.5  # LocalEnergy_error / BlockWeight_error: flag if x/median outside [1/ERR_RATIO, ERR_RATIO]
FORCE_RATIO_REL = 0.5  # flag twist if median VMC/DMC force-error ratio is outside run median * [1-0.5, 1+0.5]

OPT_LAST = 3  # last optimization steps examined
OPT_NSIGMA = 3.0  # variance-trend significance
OPT_JK_REL = 0.02  # max relative drift of jk2 coefficients
OPT_RPA_RATIO = 0.72  # first-shell u_T/u_RPA floor
OPT_K2U_DROP = 0.03  # max fractional dip of k^2 u_T at the first shell below its peak under kc
DU_REL = 0.10  # relative u(k) correction regarded as significant

BIG_SUFFIXES = ('.scalar.dat', '.dmc.dat', '.stat.h5', '.config.h5', '.random.h5')
KEEP = ('dmc.g000.twistnum_0.in.xml', 'dmc.g000.twistnum_0.in.g000.qmc')
SUBDIR = 'qmcclean'
