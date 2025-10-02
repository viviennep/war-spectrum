import polars as pl
from war_spectrum.preprocess.run_responsibility import (
    create_pa_df,
    assign_runs_in_half_inning
)
cl = pl.col

def build_pitcher_stints(sc_df):
    stints = (
        sc_df
        .group_by('pitcher','season','team_abbr')
        .agg(
            stint_pa  = pl.struct(['game_pk','at_bat_number']).n_unique(),
            stint_bip = cl('description').eq('hit_into_play').sum(),
            stint_g   = cl('game_pk').n_unique(),
            primary_position = cl('primary_position').first(),
            lg = cl('lg').first(),
            name = cl('full_name').first(),
            stint_ci = cl('is_catcher_interf').sum(),
            stint_bb = cl('is_bb').sum(),
            stint_k = cl('is_k').sum(),
            stint_h = cl('is_hit').sum(),
            stint_1b = cl('is_1b').sum(),
            stint_2b = cl('is_2b').sum(),
            stint_3b = cl('is_3b').sum(),
            stint_hr = cl('is_hr').sum(),
            stint_sf = cl('is_sf').sum(),
            stint_gidp = cl('is_gidp').sum(),
            stint_sbu = cl('is_sbu').sum(),
            stint_iffb = cl('is_iffb').sum(),
            stint_balls = cl('is_ball').sum(),
            stint_strikes = cl('is_strike').sum(),
            stint_called_strikes = cl('is_called_strike').sum(),
            stint_out = cl('is_out').sum(),
            stint_roe = cl('is_reached_on_error').sum(),
            stint_cs = cl('is_caught_steal').sum(),
            stint_hbp = cl('is_hbp').sum(),
            stint_foul = cl('is_foul').sum(),
            stint_whiff = cl('is_whiff').sum(),
            stint_x1b = cl('pred_1b').sum(),
            stint_x2b = cl('pred_2b').sum(),
            stint_x3b = cl('pred_3b').sum(),
            stint_xhr = cl('pred_hr').sum(),
            stint_xsf = cl('pred_sf').sum(),
            stint_xgidp = cl('pred_gidp').sum(),
            stint_xout = cl('pred_out').sum(),
            stint_ball_stuff = cl('p_ball_stuff').sum(),
            stint_called_strike_stuff = cl('p_called_strike_stuff').sum(),
            stint_hbp_stuff = cl('p_hbp_stuff').sum(),
            stint_whiff_stuff = cl('p_whiff_stuff').sum(),
            stint_foul_stuff = cl('p_foul_stuff').sum(),
            stint_1b_stuff = cl('p_1b_stuff').sum(),
            stint_2b_stuff = cl('p_2b_stuff').sum(),
            stint_3b_stuff = cl('p_3b_stuff').sum(),
            stint_hr_stuff = cl('p_hr_stuff').sum(),
            stint_sf_stuff = cl('p_sf_stuff').sum(),
            stint_gidp_stuff = cl('p_gidp_stuff').sum(),
            stint_out_stuff = cl('p_out_stuff').sum(),
            stint_ball_pitch = cl('p_ball_pitch').sum(),
            stint_called_strike_pitch = cl('p_called_strike_pitch').sum(),
            stint_hbp_pitch = cl('p_hbp_pitch').sum(),
            stint_whiff_pitch = cl('p_whiff_pitch').sum(),
            stint_foul_pitch = cl('p_foul_pitch').sum(),
            stint_1b_pitch = cl('p_1b_pitch').sum(),
            stint_2b_pitch = cl('p_2b_pitch').sum(),
            stint_3b_pitch = cl('p_3b_pitch').sum(),
            stint_hr_pitch = cl('p_hr_pitch').sum(),
            stint_sf_pitch = cl('p_sf_pitch').sum(),
            stint_gidp_pitch = cl('p_gidp_pitch').sum(),
            stint_out_pitch = cl('p_out_pitch').sum(),
            framing_runs = (
                pl.when('is_called')
                  .then(cl('p_called_strike_catcher') - cl('p_called_strike_avg'))
                  .otherwise(0.)
                  .sum()
                  *0.125
            )
        )
    )
    seasons = (
        stints.group_by('pitcher','season')
              .agg(
                  season_pa  = cl('stint_pa').sum(),
                  season_bip = cl('stint_bip').sum(),
                  n_stints   = pl.len(),
              )
    )
    return (
        stints.join(seasons, on=['pitcher','season'], how='left')
        .with_columns(
            stint_pa_weight  = cl('stint_pa')/cl('season_pa'),
            stint_bip_weight = cl('stint_bip')/cl('season_bip'),
            pitcher_year = cl('pitcher').cast(str)+'-'+cl('season').cast(str)
        )
    )

def compute_runs_allowed_from_events(stints, sc_df):
    pa_df = create_pa_df(sc_df)
    by_half_inning = (
        pa_df
        .group_by('game_pk','inning','inning_topbot')
        .map_groups(assign_runs_in_half_inning)
    )
    return stints.join(
        by_half_inning
        .group_by('pitcher','season','team_abbr')
        .agg(runs_allowed = cl('charged_runs').sum()),
        on=['pitcher','season','team_abbr'],
        how='left'
    )


