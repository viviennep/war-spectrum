import polars as pl
from war_spectrum.scrapers.statcast import (
    get_season_start_end_dates,
    get_statcast,
)
from war_spectrum.scrapers.utils import (
    retrieve_sprint_speed_data,
    retrieve_baseball_reference_war,
    retrieve_fangraphs_data,
    retrieve_pitcher_oaa,
    retrieve_pitcher_run_game
)
from war_spectrum.scrapers.bio import retrieve_player_bios
from war_spectrum.scrapers.retrieve_defensive_positioning import retrieve_positioning_runs
from war_spectrum.preprocess.utils import *
from war_spectrum.preprocess.standardize import *
from war_spectrum.preprocess.column_extractors import *
from war_spectrum.preprocess.inning_info import *
from war_spectrum.preprocess.run_responsibility import *
cl = pl.col

def load_raw_statcast(data_dir, start_year, end_year, daily=True, use_file=False):
    if use_file:
        path = data_dir / 'sc-2021-2024.parquet'
        sc_df = pl.read_parquet(path)
    else:
        try:
            start_date = (
                pl.scan_parquet(data_dir/'play_by_play')
                .filter(cl('season').is_between(start_year, end_year))
                .select(cl('game_date')-pl.duration(days=1))
                .max()
                .collect()
            ).item().strftime('%Y-%m-%d')
        except:
            start_date = '2021-03-28'
        if daily:
            sc_df = get_statcast(data_dir,start_date,None)
        else:
            _, end_date = get_season_start_end_dates(start_year,end_year)
            sc_df = get_statcast(data_dir,start_date,end_date)
    return sc_df.filter(cl('game_type').eq('R'))

def retrieve_raw_data(data_dir, years, daily=False, use_file=True):
    start_year, end_year = years[0], years[-1]
    if daily:
        sc_df  = load_raw_statcast(
            data_dir, None, None, daily=daily, use_file=use_file
        )
    else:
        sc_df  = load_raw_statcast(
            data_dir, start_year, end_year, daily=daily, use_file=use_file
        )
    sp_df  = pl.concat([retrieve_sprint_speed_data(y) for y in years])
    pd_df  = retrieve_pitcher_oaa(start_year, end_year)
    pr_df  = retrieve_pitcher_run_game(start_year, end_year)
    br_df  = retrieve_baseball_reference_war()
    fg_df  = pl.concat([retrieve_fangraphs_data(y) for y in years])
    pos_df = retrieve_positioning_runs()
    bio_df = retrieve_player_bios(sc_df.select('pitcher').unique().to_numpy().ravel())
    return {
        'sc': sc_df,
        'sp': sp_df,
        'pd': pd_df,
        'pr': pr_df,
        'br': br_df,
        'fg': fg_df,
        'pos': pos_df,
        'bio': bio_df,
    }

def standardize_raw_data(raw_dfs):
    sc_df  = standardize_statcast(raw_dfs['sc'])
    sp_df  = standardize_sprint_speed(raw_dfs['sp'])
    pd_df  = standardize_pitcher_defence(raw_dfs['pd'])
    pr_df  = standardize_pitcher_run_game(raw_dfs['pr'])
    br_df  = standardize_baseball_reference(raw_dfs['br'])
    fg_df  = standardize_fangraphs(raw_dfs['fg'])
    pos_df = standardize_positioning_runs(raw_dfs['pos'])
    bio_df = standardize_player_bios(raw_dfs['bio'])
    return {
        'sc': sc_df,
        'sp': sp_df,
        'pd': pd_df,
        'pr': pr_df,
        'br': br_df,
        'fg': fg_df,
        'pos': pos_df,
        'bio': bio_df,
    }

def preprocess_event_data(raw_dfs):
    sp_extract = extract_sprint_speed(raw_dfs['sp'])
    bio_extract = extract_player_bios(raw_dfs['bio'])
    sc_df = (
        raw_dfs['sc']
        .join(sp_extract,  on=['season','batter'], how='left')
        .join(bio_extract, on=['pitcher'],         how='left')
        .pipe(add_spray_angle)
        .pipe(add_lefty_indicator)
        .pipe(add_pa_indicators)
        .pipe(add_bbe_indicators)
        .pipe(add_event_indicators)
        .pipe(add_sprint_speed_indicator)
        .pipe(add_runs_scored)
        .pipe(add_outs_made)
        .pipe(add_reached_on_error_and_dropped_strike_three)
        .pipe(add_pickoffs_and_caught_stealing)
        .pipe(add_catcher_and_pitcher_years)
        .pipe(add_auto_strikes)
        .pipe(add_other_walks)
        .pipe(add_pitch_quality_features)
        .pipe(add_pitch_quality_indicators)
        .pipe(add_pitch_type_features)
        .pipe(add_league_tag)
    )
    return sc_df

def join_team_context_tables(stints, raw_dfs):
    br_extract = extract_baseball_reference(raw_dfs['br'])
    pr_extract = extract_pitcher_run_game(raw_dfs['pr'])
    fg_extract = extract_fangraphs(raw_dfs['fg'])
    pd_extract = extract_pitcher_defence(raw_dfs['pd'])
    pos_df = raw_dfs['pos']
    return (
        stints
        .join(br_extract,  on=['pitcher','season','team_abbr'], how='left')
        .join(pr_extract,  on=['pitcher','season','team_abbr'], how='left')
        .join(fg_extract,  on=['pitcher','season','team_abbr'], how='left')
        .join(pd_extract,  on=['pitcher','season'],             how='left')
        .join(pos_df,      on=['season','team_abbr'],           how='left')
        .with_columns(
            stint_oaa = cl('season_oaa')*cl('stint_bip_weight'),
            stint_frp = cl('season_frp')*cl('stint_bip_weight'),
        )
    )


