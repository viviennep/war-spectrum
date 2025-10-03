import numpy as np, polars as pl
from .war_calculator import WARCalculator
cl = pl.col

class OutsAboveAverageWAR(WARCalculator):
    def calc_xra(self):
        team_bip = cl(self.BIP).over('season','team_abbr').sum()
        xRA = cl('xRA')+cl('xRA_sprp_adj')-cl('positioning_runs')*cl(self.BIP)/team_bip
        defence = cl(self.FRV)
        framing = cl('framing_runs')
        return cl('PPF_custom')*xRA/100 - defence - framing + cl('xRA_extras_adj')

    def calc_ra(self): 
        return cl(self.RA)


