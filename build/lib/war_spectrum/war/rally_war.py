import numpy as np, polars as pl
from .war_calculator import WARCalculator
cl = pl.col

class RallyWAR(WARCalculator):
    def calc_xra(self):
        xRA = cl('xRA')-cl('xRA_def_pitcher')+cl('xRA_sprp_adj')
        return cl('PPF_custom')*xRA/100 + cl('xRA_extras_adj')

    def calc_ra(self): 
        return cl(self.RA)

