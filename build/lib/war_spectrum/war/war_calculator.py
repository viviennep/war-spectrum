import numpy as np, polars as pl
cl = pl.col

class WARCalculator():
    '''
    Base class for implementing the WAR calculation
    Specific WARs will just extend this and provide their own RA calculation
    '''
    def __init__(
            self, 
            name, 
            PA_col = 'stint_pa',
            G_col = 'stint_g',
            BIP_col = 'stint_bip',
            RA_col = 'runs_allowed',
            FRV_col = 'stint_frp',
            sum_target = 'rWAR',
        ):
        self.name = name
        self.PA = PA_col
        self.G = G_col
        self.BIP = BIP_col
        self.RA = RA_col
        self.FRV = FRV_col
        self.sum_target = sum_target

    def calc_xra(self):
        return NotImplementedError()

    def calc_ra(self):
        return NotImplementedError()

    def pythagenpat(self):
        rs = cl('teamRpG')
        g  = cl(self.G)
        ra = (rs - cl('RAA')/g).clip(0,None)
        x  = (rs+ra)**0.285
        wp = 1./(1. + (ra/rs)**x)
        return wp

    def adjust_war_to_target(self, df):
        target_war = cl(self.sum_target).sum().over('season')
        current_war = cl('WAR').sum().over('season')
        correction = target_war - current_war
        return df.with_columns(
            WAR = cl('WAR') + cl('pitcher_stint_weight')*correction
        )

    def calculate(self, df):
        # Calculate deviation from expected runs allowed
        xRA = self.calc_xra()
        RA  = self.calc_ra()
        df = df.with_columns(RAA = xRA-RA)

        # Center so lg average is truly 0
        lg_raa = cl('RAA').sum().over(['season','lg'])
        lg_pa  = cl(self.PA).sum().over(['season','lg'])
        pitcher_stint_weight = cl(self.PA)/lg_pa
        df = df.with_columns(
            pitcher_stint_weight = pitcher_stint_weight,
            RAA = cl('RAA') - lg_raa*pitcher_stint_weight,
        )

        # Add in WAA
        wp = self.pythagenpat()
        df = df.with_columns(
            WinPerc = wp,
            WAA = (wp-0.5)*cl(self.G)
        )

        # Center WAA to zero
        lg_waa = cl('WAA').sum().over(['season','lg'])
        df = df.with_columns(
            WAA = cl('WAA') - lg_waa*pitcher_stint_weight
        )

        # Calculate WAR (adding in rep level, taken from bref)
        df = df.with_columns(
            WAR = cl('WAA') + cl('WAR_rep')
        )

        df = self.adjust_war_to_target(df)

        return df.rename({'WAR':self.name})


