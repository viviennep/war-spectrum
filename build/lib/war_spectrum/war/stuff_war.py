import numpy as np, polars as pl
from .war_calculator import WARCalculator
cl = pl.col

class StuffWAR(WARCalculator):
    def calc_xra(self):
        lg_RAperTBF = cl('runs_allowed').sum()/cl(self.PA).sum() 
        xRA = lg_RAperTBF*cl(self.PA) + cl('xRA_sprp_adj')
        return xRA

    def calc_ra(self): 
        return cl(self.RA)


