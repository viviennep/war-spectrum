import polars as pl, pathlib, logging, duckdb
from war_spectrum.pipeline.data_retrieval import (
    retrieve_raw_data,
    standardize_raw_data,
    preprocess_event_data,
    join_team_context_tables,
)
from war_spectrum.pipeline.apply_event_level_models import (
    apply_framing,
    apply_bbe_classifier,
    apply_stuff,
    apply_pitching,
)
from war_spectrum.pipeline.aggregate_to_stints import (
    build_pitcher_stints,
    compute_runs_allowed_from_events,
)
from war_spectrum.pipeline.apply_stint_level_models import (
    attach_baseruns,
    attach_dips_baseruns,
    attach_xbaseruns,
    attach_xbaseruns_stuff,
    attach_xbaseruns_pitch,
)
from war_spectrum.pipeline.calculate_markov_expected_kbb import (
    attach_markov_expected_kbb
)
from war_spectrum.war import (
    RunsAllowedWAR,
    RallyWAR,
    OutsAboveAverageWAR,
    DIPSWAR,
    BaseRunsWAR,
    xBaseRunsWAR,
    StuffWAR,
    PitchWAR,
)
cl = pl.col

logger = logging.getLogger(__name__)

data_dir  = pathlib.Path(__file__).resolve().parent.parent / 'data'
model_dir = pathlib.Path(__file__).resolve().parent.parent / 'models'

years = list(range(2021,2026))
#years = [2025]

## Data retrieval and preprocessing
raw_dfs = retrieve_raw_data(data_dir, years, daily=False, use_file=False)
raw_dfs = standardize_raw_data(raw_dfs)
sc_df   = preprocess_event_data(raw_dfs).lazy()

sc_df = pl.concat([
    pl.scan_parquet(data_dir / 'play_by_play')
    .filter(cl('season').is_in(years))
    .select(i for i in sc_df.collect_schema()),
    sc_df
]).collect()

## Apply event level models
print('bbe')
sc_df = apply_bbe_classifier(sc_df, model_dir/'bbe/bbe_classifier.joblib')
print('stuff')
sc_df = apply_stuff(sc_df, model_dir/'pitch_quality/stuff.joblib')
print('pitching')
sc_df = apply_pitching(sc_df, model_dir/'pitch_quality/pitching.joblib')
print('framing')
sc_df = apply_framing(sc_df, model_dir/'framing', years=years) # directory this time

# Write daily
sc_df.write_parquet(
    f"{data_dir}/play_by_play/",
    use_pyarrow=True,
    pyarrow_options={
        'partition_cols':['game_year','game_date'],
        'existing_data_behavior':'delete_matching'
    }
)
con = duckdb.connect(data_dir / 'leaderboard.duckdb')
con.execute(f"""
    create or replace view all_plays as
    select *
    from read_parquet('{data_dir}/play_by_play/*/*/*.parquet',
                      hive_partitioning=True);
""")

# Aggregate
stints = build_pitcher_stints(sc_df)
stints = join_team_context_tables(stints, raw_dfs)
stints = compute_runs_allowed_from_events(stints, sc_df)

# Figure out expected K and BB from pitch quality models
stints = attach_markov_expected_kbb(sc_df,stints,model='stuff')
stints = attach_markov_expected_kbb(sc_df,stints,model='pitch')

# Apply baseruns
stints = attach_baseruns(stints)
stints = attach_dips_baseruns(stints)
stints = attach_xbaseruns(stints)
stints = attach_xbaseruns_stuff(stints)
stints = attach_xbaseruns_pitch(stints)

# Calculate WARs
ra_war = RunsAllowedWAR(
    name = 'RA_WAR', 
    PA_col = 'stint_pa',
    G_col = 'stint_g',
    BIP_col = 'stint_bip',
    RA_col = 'runs_allowed',
    sum_target = 'rWAR',   # Add a slight nudge to all WAR calcs to sum to rWAR
)
rally_war = RallyWAR(
    name = 'Rally_WAR', 
    PA_col = 'stint_pa',
    G_col = 'stint_g',
    BIP_col = 'stint_bip',
    RA_col = 'runs_allowed',
    sum_target = 'rWAR',
)
oaa_war = OutsAboveAverageWAR(
    name = 'OAA_WAR', 
    PA_col = 'stint_pa',
    G_col = 'stint_g',
    BIP_col = 'stint_bip',
    RA_col = 'runs_allowed',
    FRV_col = 'stint_frp',
    sum_target = 'rWAR',
)
dips_war = DIPSWAR(
    name = 'DIPS_WAR', 
    PA_col = 'stint_pa',
    G_col = 'stint_g',
    RA_col = 'dips_baseruns',
    sum_target = 'rWAR',
)
bsr_war = BaseRunsWAR(
    name = 'BsR_WAR', 
    PA_col = 'stint_pa',
    G_col = 'stint_g',
    BIP_col = 'stint_bip',
    RA_col = 'baseruns',
    FRV_col = 'stint_frp',
    sum_target = 'rWAR',
)
xbsr_war = xBaseRunsWAR(
    name = 'xBsR_WAR', 
    PA_col = 'stint_pa',
    G_col = 'stint_g',
    BIP_col = 'stint_bip',
    RA_col = 'xbaseruns',
    FRV_col = 'stint_frp',
    sum_target = 'rWAR',
)
stuff_war = StuffWAR(
    name = 'Stuff_WAR', 
    PA_col = 'stint_pa',
    G_col = 'stint_g',
    BIP_col = 'stint_bip',
    RA_col = 'xbaseruns_stuff',
    sum_target = 'rWAR',
)
pitch_war = PitchWAR(
    name = 'Pitch_WAR', 
    PA_col = 'stint_pa',
    G_col = 'stint_g',
    BIP_col = 'stint_bip',
    RA_col = 'xbaseruns_pitch',
    sum_target = 'rWAR',
)

def attach_war(stints,war_calc):
    join_columns = ['pitcher','season','team_abbr']
    return stints.join(
        war_calc.calculate(stints).select(*join_columns,war_calc.name),
        on=join_columns,
        how='left'
    )

stints = attach_war(stints,ra_war)
stints = attach_war(stints,rally_war)
stints = attach_war(stints,oaa_war)
stints = attach_war(stints,dips_war)
stints = attach_war(stints,bsr_war)
stints = attach_war(stints,xbsr_war)
stints = attach_war(stints,stuff_war)
stints = attach_war(stints,pitch_war)

# Write stints
stints.write_parquet(
    f"{data_dir}/stints/",
    use_pyarrow=True,
    pyarrow_options={
        'partition_cols':['season'],
        'existing_data_behavior':'delete_matching'
    }
)
con.execute(f"""
    create or replace table stints as
    select *
    from read_parquet('{data_dir}/stints/*/*.parquet',
                      hive_partitioning=True);
""")

