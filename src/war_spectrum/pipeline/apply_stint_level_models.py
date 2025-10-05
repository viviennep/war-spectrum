import numpy as np, polars as pl
from war_spectrum.models.baseruns import estimate_baseruns, b_terms
cl = pl.col

def attach_iffip(stints):
    hr = cl('stint_hr').cast(pl.Float64)
    bb = cl('stint_bb').cast(pl.Float64)
    hbp = cl('stint_hbp').cast(pl.Float64)
    k = cl('stint_k').cast(pl.Float64)
    iffb = cl('stint_iffb').cast(pl.Float64)
    ip = cl('stint_out').cast(pl.Float64)/3
    iffip = (13*hr+3*(bb+hbp)-2*(k+iffb))/ip
    lg_iffip = (13*hr.sum() + 3*(bb+hbp).sum() - 2*(k+iffb).sum())/ip.sum()
    lg_ra9 = 9*cl('runs_allowed').sum()/ip.sum()
    return stints.with_columns(
        iffip = iffip + lg_ra9 - lg_iffip
    )

def attach_dips_baseruns(stints):
    non_hr_bip = cl('stint_bip')-cl('stint_hr')
    lg_babip = (cl('stint_h')-cl('stint_hr')).sum()/(cl('stint_bip')-cl('stint_hr')).sum()
    stints = stints.with_columns(
        stint_hpred = non_hr_bip*lg_babip,
        zero = pl.lit(0.),
    )
    b_terms = {
        'bb'  : 'stint_bb',
        'k'   : 'stint_k',
        '1b'  : 'stint_hpred',
        '2b'  : 'zero',
        '3b'  : 'zero',
        'hr'  : 'stint_hr',
        'sf'  : 'zero',
        'gidp': 'zero',
        'out' : 'stint_out',
        'sb'  : 'zero',
        'cs'  : 'zero',
        'ra'  : 'runs_allowed',
    }
    return estimate_baseruns(stints, b_terms, name='dips_baseruns')

def attach_baseruns(stints):
    b_terms = {
        'bb'  : 'stint_bb',
        'k'   : 'stint_k',
        '1b'  : 'stint_1b',
        '2b'  : 'stint_2b',
        '3b'  : 'stint_3b',
        'hr'  : 'stint_hr',
        'sf'  : 'stint_sf',
        'gidp': 'stint_gidp',
        'out' : 'stint_out',
        'sb'  : 'n_sb',
        'cs'  : 'n_cs',
        'ra'  : 'runs_allowed',
    }
    return estimate_baseruns(stints, b_terms,name='baseruns',cutoff=1.)

def attach_xbaseruns(stints, b_terms=b_terms):
    b_terms = {
        'bb'  : 'stint_bb',
        'k'   : 'stint_k',
        '1b'  : 'stint_x1b',
        '2b'  : 'stint_x2b',
        '3b'  : 'stint_x3b',
        'hr'  : 'stint_xhr',
        'sf'  : 'stint_xsf',
        'gidp': 'stint_xgidp',
        'out' : 'stint_xout',
        'sb'  : 'xsb',
        'cs'  : 'xcs',
        'ra'  : 'runs_allowed',
    }
    return estimate_baseruns(stints, b_terms, name='xbaseruns', cutoff=1.02)

def attach_xbaseruns_stuff(stints, b_terms=b_terms):
    b_terms = {
        'bb'  : 'xBB_stuff',
        'k'   : 'xK_stuff',
        '1b'  : 'stint_1b_stuff',
        '2b'  : 'stint_2b_stuff',
        '3b'  : 'stint_3b_stuff',
        'hr'  : 'stint_hr_stuff',
        'sf'  : 'stint_sf_stuff',
        'gidp': 'stint_gidp_stuff',
        'out' : 'stint_out_stuff',
        'sb'  : 'xsb',
        'cs'  : 'xcs',
        'ra'  : 'runs_allowed',
    }
    return estimate_baseruns(stints, b_terms, name='xbaseruns_stuff',cutoff=1.5)

def attach_xbaseruns_pitch(stints, b_terms=b_terms):
    b_terms = {
        'bb'  : 'xBB_pitch',
        'k'   : 'xK_pitch',
        '1b'  : 'stint_1b_pitch',
        '2b'  : 'stint_2b_pitch',
        '3b'  : 'stint_3b_pitch',
        'hr'  : 'stint_hr_pitch',
        'sf'  : 'stint_sf_pitch',
        'gidp': 'stint_gidp_pitch',
        'out' : 'stint_out_pitch',
        'sb'  : 'xsb',
        'cs'  : 'xcs',
        'ra'  : 'runs_allowed',
    }
    return estimate_baseruns(stints, b_terms, name='xbaseruns_pitch',cutoff=1.)

