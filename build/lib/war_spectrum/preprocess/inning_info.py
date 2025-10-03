import numpy as np, polars as pl
cl = pl.col

def add_runs_scored(df):
    return (
        df.with_columns(
            runs_scored_play = cl('post_bat_score')-cl('bat_score')
        )
        .with_columns(
            runs_scored_inning = (
                cl('runs_scored_play')
                .sum()
                .over('game_pk','inning','inning_topbot')
            )
        )
        .with_columns(
            rest_inning_runs = (
                cl('runs_scored_inning')-
                cl('runs_scored_play')
                .cum_sum()
                .over('game_pk','inning','inning_topbot')
                .sort_by('at_bat_number','pa_pitch_number')
            )
        )
    )

def add_outs_made(df):
    out_map = {
        'sac_fly_double_play': 2,
        'triple_play': 3,
        'strikeout_double_play': 2,
        'double_play': 2,
        'sac_bunt_double_play': 2, 
        'field_out': 1,
        'fielders_choice': 1,
        'force_out': 1,
        'sac_fly': 1,
        'fielders_choice_out': 1,
        'grounded_into_double_play': 2,
        'strikeout': 1,
        'sac_bunt': 1,
    }
    return (
        df
        .sort('game_pk','inning','inning_topbot','at_bat_number','pa_pitch_number')
        .with_columns(
            outs_on_play = (
                cl('outs_when_up')
                .shift(-1)
                .over('game_pk','inning','inning_topbot')
                .fill_null(3)
                -
                cl('outs_when_up')
            ).clip(0,3),
            is_next_ab_in_game = (
                cl('at_bat_number').shift(-1).over('game_pk').is_not_null()
            ),
            is_next_ab_in_inning = (
                cl('at_bat_number')
                .shift(-1)
                .over('game_pk','inning','inning_topbot')
                .is_not_null()
            )
        )
        .with_columns(
            outs_from_event = pl.when(cl('events').is_in(out_map))
                                .then(cl('events').replace(out_map))
                                .otherwise('outs_on_play')
                                .cast(pl.Int64),
            is_walkoff = (
                ( cl('inning_topbot').eq('Bot')) &
                (~cl('is_next_ab_in_game')) &
                ( cl('post_home_score') > cl('post_away_score'))
            )
        )
        .with_columns(
            outs_made_on_play = pl.when('is_walkoff')
                                  .then('outs_from_event')
                                  .otherwise('outs_on_play')
                                  .cast(pl.Int64)
        )
    )



