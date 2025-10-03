import numpy as np, polars as pl
from .war_calculator import WARCalculator
cl = pl.col

class xBaseRunsWAR(WARCalculator):
    def calc_xra(self):
        xRA = cl('xRA')+cl('xRA_sprp_adj')
        framing = cl('framing_runs')
        return cl('PPF_custom')*xRA/100 - framing + cl('xRA_extras_adj')

    def calc_ra(self): 
        return cl(self.RA)

