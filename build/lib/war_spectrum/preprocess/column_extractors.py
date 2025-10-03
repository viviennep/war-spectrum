import numpy as np, polars as pl
cl = pl.col

def extract_baseball_reference(df):
    return df.select(
        'pitcher','season','team_abbr',
        'RA',
        'IPouts',
        'PPF',
        'IPouts_start',
        'IPouts_relief',
        'xRA',
        'xRA_sprp_adj',
        'xRA_extras_adj',
        'xRA_def_pitcher',
        'PPF_custom',
        'xRA_final',
        'BIP',
        'BIP_perc',
        'RS_def_total',
        'runs_above_avg',
        'runs_above_avg_adj',
        'runs_above_rep',
        'RpO_replacement',
        'GR_leverage_index_avg',
        'teamRpG',
        'oppRpG',
        'pyth_exponent',
        'waa_win_perc',
        'WAA',
        'WAA_adj',
        'oppRpG_rep',
        'pyth_exponent_rep',
        'waa_win_perc_rep',
        'WAR_rep',
        'ERA_plus',
        'ER_lg',
        rWAR = 'WAR',
    )

def extract_fangraphs(df):
    return df.select(
        'pitcher','season','team_abbr',
        W_fg       = 'W',
        L_fg       = 'L',
        ERA_fg     = 'ERA',
        G_fg       = 'G',
        GS_fg      = 'GS',
        IP_fg      = 'IP',
        TBF_fg     = 'TBF',
        H_fg       = 'H',
        R_fg       = 'R',
        ER_fg      = 'ER',
        HR_fg      = 'HR',
        BB_fg      = 'BB',
        IBB_fg     = 'IBB',
        HBP_fg     = 'HBP',
        SO_fg      = 'SO',
        IFFB_fg    = 'IFFB',
        Pitches_fg = 'Pitches',
        Balls_fg   = 'Balls',
        Strikes_fg = 'Strikes',
        xERA_fg    = 'xERA',
        fWAR       = 'WAR',
    )

def extract_pitcher_run_game(df):
    sba = cl('n_sb')+cl('n_cs')
    sba_per_opp = sba.sum()/cl('n_init').sum()
    sb_per_sba = cl('n_sb').sum()/sba.sum()
    cs_per_sba = cl('n_cs').sum()/sba.sum()
    avg_sb = sb_per_sba*sba_per_opp*cl('n_init')
    avg_cs = cs_per_sba*sba_per_opp*cl('n_init')
    return df.select(
        'season',
        'pitcher',
        'team_abbr',
        'n_sb',
        'n_cs',
        xsb = avg_sb - cl('net_attr_plus'),
        xcs = avg_cs + cl('net_attr_minus'),
    )

def extract_pitcher_defence(df):
    return (
        df
        .group_by('pitcher','season')
        .agg(
            season_oaa = cl('outs_above_average').sum(),
            season_frp = cl('fielding_runs_prevented').sum()
        )
    )

def extract_sprint_speed(df):
    return df.select(
        'season',
        'sprint_speed',
        batter = 'runner',
    )

def extract_player_bios(df):
    feet = cl('height').str.extract(r"(\d+)'").cast(pl.Int32)
    inch = cl('height').str.extract(r"(\d+)\"").cast(pl.Int32)
    return df.select(
        'pitcher',
        full_name = 'fullName',
        birth_date = cl('birthDate').str.to_date(),
        height = feet+inch/12,
        primary_position = cl('primaryPosition').struct.field('abbreviation')
    )



