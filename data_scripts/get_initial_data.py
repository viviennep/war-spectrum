import numpy as np, polars as pl, pathlib
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

data_dir = pathlib.Path(__file__).resolve().parent.parent / 'data'

# Years
years = list(range(2021,2025))
start_year = years[0]
end_year = years[-1]

## Data Retrieval 
# Statcast data
start_date, end_date = get_season_start_end_dates(start_year,end_year)
#sc_df = get_statcast(start_date,end_date)
sc_df = pl.read_parquet(data_dir / 'sc-2021-2024.parquet')

# Sprint speed
sp_df = pl.concat([retrieve_sprint_speed_data(year) for year in years])

# Pitcher OAA - season aggregate 
pd_df = retrieve_pitcher_oaa(start_year=start_year, end_year=end_year)

# Pitcher run game - split by team stint
pr_df = retrieve_pitcher_run_game(start_year=start_year, end_year=end_year)

# Baseball reference - split by team stint
br_df = retrieve_baseball_reference_war()

# Fangraphs - season aggregate
fg_dfs = [retrieve_fangraphs_data(year) for year in years]
fg_df = pl.concat([i.cast(fg_dfs[-1].schema) for i in fg_dfs])

# Player bio info
pitchers = sc_df.select('pitcher').unique().to_numpy().ravel()
bio_df = retrieve_player_bios(pitchers)

# Positional runs from SIS
pos_df = retrieve_positioning_runs()

## Standardize data:
sc_df  = standardize_statcast(sc_df)
sp_df  = standardize_sprint_speed(sp_df)
pd_df  = standardize_pitcher_defence(pd_df)
pr_df  = standardize_pitcher_run_game(pr_df)
br_df  = standardize_baseball_reference(br_df)
fg_df  = standardize_fangraphs(fg_df)
bio_df = standardize_player_bios(bio_df)
pos_df = standardize_positioning_runs(pos_df)

## Merge data sources
# Play by play merge
bio_extract = extract_player_bios(bio_df)
sp_extract = extract_sprint_speed(sp_df)

sc_df = (
    sc_df
    .join(sp_extract, on=['season','batter'], how='left')
    .join(bio_extract, on=['pitcher'], how='left')
)

## Process statcast data
sc_df = add_spray_angle(sc_df)
sc_df = add_lefty_indicator(sc_df)
sc_df = add_pa_indicators(sc_df)
sc_df = add_bbe_indicators(sc_df)
sc_df = add_event_indicators(sc_df)
sc_df = add_sprint_speed_indicator(sc_df)
sc_df = add_runs_scored(sc_df)
sc_df = add_outs_made(sc_df)
sc_df = add_reached_on_error_and_dropped_strike_three(sc_df)
sc_df = add_pickoffs_and_caught_stealing(sc_df)
sc_df = add_catcher_and_pitcher_years(sc_df)
sc_df = add_auto_strikes(sc_df)
sc_df = add_other_walks(sc_df)
sc_df = add_pitch_quality_features(sc_df)
sc_df = add_pitch_quality_indicators(sc_df)
sc_df = add_pitch_type_features(sc_df)
sc_df = add_league_tag(sc_df)


# Find per-team weights for each season stint with team
pitcher_stints = (
    sc_df
    .group_by('pitcher','season','team_abbr')
    .agg(
        stint_pa  = pl.struct(['game_pk','at_bat_number']).n_unique(),
        stint_bip = cl('description').eq('hit_into_play').sum(),
        stint_g = cl('game_pk').n_unique(),
        primary_position = cl('primary_position').first(),
        lg = cl('lg').first(),
        name = cl('full_name').first(),
    )
)
pitcher_seasons = (
    pitcher_stints
    .group_by('pitcher','season')
    .agg(
        season_pa  = cl('stint_pa').sum(),
        season_bip = cl('stint_bip').sum(),
        n_stints = pl.len()
    )
)
pitcher_stints = (
    pitcher_stints.join(
        pitcher_seasons,
        on=['pitcher','season'],
        how='left'
    )
    .with_columns(
        stint_pa_weight  = cl('stint_pa')/cl('season_pa'),
        stint_bip_weight = cl('stint_bip')/cl('season_bip')
    )
)

# Pre-split sources
br_extract = extract_baseball_reference(br_df)
pr_extract = extract_pitcher_run_game(pr_df)
fg_extract = extract_fangraphs(fg_df)

pitcher_stints = (
    pitcher_stints
    .join(br_extract, on=['pitcher','season','team_abbr'], how='left')
    .join(pr_extract, on=['pitcher','season','team_abbr'], how='left')
    .join(fg_extract, on=['pitcher','season','team_abbr'], how='left')
    .join(pos_df,     on=['season','team_abbr'],           how='left')
)

# Prorate aggregated sources
pd_extract = extract_pitcher_defence(pd_df)
pitcher_stints = (
    pitcher_stints
    .join(pd_extract, on=['pitcher','season'], how='left')
    .with_columns(
        stint_oaa = cl('season_oaa')*cl('stint_bip_weight'),
        stint_frp = cl('season_frp')*cl('stint_bip_weight'),
    )
)

## Find pitcher runs allowed
pa_df = create_pa_df(sc_df)
pitcher_runs_by_inning = (
    pa_df
    .group_by('game_pk','inning','inning_topbot')
    .map_groups(assign_runs_in_half_inning)
)
pitcher_runs_allowed_stints = (
    pitcher_runs_by_inning
    .group_by('pitcher','season','team_abbr')
    .agg(
        runs_allowed = cl('charged_runs').sum(),
    )
)
pitcher_stints = (
    pitcher_stints
    .join(pitcher_runs_allowed_stints, on=['pitcher','season','team_abbr'],how='left')
    .with_columns(
        runs_allowed = cl('runs_allowed').fill_null(0)
    )
)

## Write datasets
# Hive partitioned, allows easy access from both polars & duckdb 
def write_hive_dataset(df, base_dir, partition_cols, mode='delete_matching'):
    df.write_parquet(
        base_dir,
        use_pyarrow=True,
        pyarrow_options={
            'partition_cols': partition_cols,
            'existing_data_behavior': mode,
            'compression': 'zstd',
        }
    )

# Play-by-play
write_hive_dataset(sc_df, data_dir / 'curated/events', ['season','game_date'])

# Pitcher stint level
write_hive_dataset(pitcher_stints, data_dir / 'curated/stints', ['season'])



