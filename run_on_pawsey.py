NO_GLEAMX = True
from transient_search import Filter, TransientSearch
import filters as fil
from os.path import expanduser

filters = [
    Filter('tcg'  , 5.5, 7.0, True , fil.Correlator, (1,1,1), (125,1,1)),
    Filter('spike', 5.5, 7.5, False, fil.Spike, 3),
    Filter('rms'  , 2.0, 2.25, True , fil.RMS)]

for obsid in [1062014144, 1062015584]:
    cands, table_fname = TransientSearch(expanduser('~/{0}/{0}_{1}'), obsid, filters, '5_RFI_obs', True, True, max_plots=100)