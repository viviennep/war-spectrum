import numpy as np, polars as pl
from .war_calculator import WARCalculator
cl = pl.col

class RunsAllowedWAR(WARCalculator):
    def calc_xra(self):
        return cl('xRA') + cl('xRA_sprp_adj') + cl('xRA_extras_adj')

    def calc_ra(self): 
        return cl(self.RA)


