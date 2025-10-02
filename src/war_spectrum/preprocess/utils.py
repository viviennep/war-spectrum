import numpy as np, polars as pl
from .abbr_standardization import *
cl = pl.col

def add_spray_angle(df):
    return (
        df.with_columns(
            hc_x_ft = 2.495671*( cl('hc_x')-125.42), 
            hc_y_ft = 2.495671*(-cl('hc_y')+198.27),
        )
        .with_columns(
            theta = pl.arctan2('hc_x_ft','hc_y_ft'),
            hc_dist = (cl('hc_x_ft')**2+cl('hc_y_ft')**2)**0.5,
        )
    )

def add_lefty_indicator(df):
    return (
        df.with_columns(
            is_lhh = cl('stand').eq('L')
        )
        .with_columns(
            flip_lefty = pl.when('is_lhh').then(-1).otherwise(1)
        )
    )

def add_pa_indicators(df):
    return (
        df.sort('game_date','game_pk','at_bat_number','pa_pitch_number')
        .with_columns(
            is_last_pitch_pa = (
                cl('pa_pitch_number')
                .eq(cl('pa_pitch_number').max().over('game_pk','at_bat_number'))
            ),
            is_last_pitch_inning = (
                cl('pa_pitch_number')
                .shift(-1)
                .over('game_pk','inning','inning_topbot')
                .is_null()
            ),
            is_last_pitch_game = (
                cl('pa_pitch_number')
                .shift(-1)
                .over('game_pk')
                .is_null()
            ),
        )
    )

def add_bbe_indicators(df):
    return (
        df.with_columns(
            is_hit_into_play = cl('description').eq('hit_into_play'),
            is_catcher_interf = cl('events').eq('catcher_interf'),
            is_strung = cl('hc_x').is_not_null() & cl('hc_y').is_not_null(),
            is_tracked = (
                cl('launch_speed').is_not_null() & cl('launch_angle').is_not_null()
            ),
            is_to_left_field = cl('theta')<0,
            is_ground_ball = (cl('launch_angle')<10) & (cl('hit_distance_sc')<133),
        )
        .with_columns(
            is_bbe = cl('is_tracked') & cl('is_strung') & cl('is_hit_into_play'),
            is_pulled = pl.when('is_lhh')
                          .then(cl('theta') > np.pi/6)
                          .otherwise(cl('theta') < -np.pi/6),
            is_middle = cl('theta').is_between(-np.pi/6,np.pi/6),
        )
        .with_columns(
            is_oppo = ~(cl('is_pulled') | cl('is_middle')),
        )
    )

def add_event_indicators(df):
    ab_events = [
        'single',
        'field_out',
        'strikeout',
        'grounded_into_double_play',
        'double',
        'home_run',
        'field_error',
        'strikeout_double_play',
        'fielders_choice',
        'fielders_choice_out',
        'force_out',
        'triple',
        'double_play',
        'other_out',
        'triple_play'
    ]
    bb_events  = ['walk','hit_by_pitch']
    k_events   = ['strikeout','strikeout_double_play']
    hit_events = ['single','double','triple','home_run']
    balls = ['ball','blocked_ball','pitchout']           # treating hbp separately 
    strikes = [
        'bunt_foul_tip',
        'called_strike',
        'foul',
        'foul_bunt',
        'foul_pitchout',
        'foul_tip',
        'hit_into_play',
        'missed_bunt',
        'swinging_strike',
        'swinging_strike_blocked',
    ]
    return (
        df.with_columns(
            is_at_bat = cl('events').is_in(ab_events),
            is_pa = cl('events').is_in(ab_events),
            is_bb = cl('events').is_in(bb_events),
            is_k = cl('events').is_in(k_events),
            is_hit = cl('events').is_in(hit_events),
            is_1b = cl('events').eq('single'),
            is_2b = cl('events').eq('double'),
            is_3b = cl('events').eq('triple'),
            is_hr = cl('events').eq('home_run'),
            is_gidp = cl('events').eq('grounded_into_double_play'),
            is_sf = cl('events').is_in(['sac_fly', 'sac_fly_double_play']),
            is_sbu = cl('events').is_in(['sac_bunt', 'sac_bunt_double_play']),
            is_iffb = cl('hit_location').is_in([1,2,3,4,5,6]) & (cl('launch_angle')>=50),
            is_on_1b = cl('on_1b').is_not_null(),
            is_on_2b = cl('on_2b').is_not_null(),
            is_on_3b = cl('on_3b').is_not_null(),
            is_ball = cl('description').is_in(balls),
            is_strike = cl('description').is_in(strikes),
            is_called_strike = cl('description').eq('called_strike')
        )
        .with_columns(
            is_out = (cl('is_at_bat') ^ cl('is_hit')) | cl('is_sbu'),
            is_called = cl('is_called_strike') | cl('is_ball'),
        )
    )

def add_reached_on_error_and_dropped_strike_three(df):
    des = cl('des').str.to_lowercase()
    has_throwing_error = des.str.contains('throwing error',literal=False)
    has_fielding_error = des.str.contains('fielding error',literal=False)
    has_missed_catch_error = des.str.contains('missed catch error',literal=False)
    has_interference_error = des.str.contains('interference error',literal=False)
    reaches = des.str.contains(r"\breaches\b|\bsafe at first\b",literal=False)
    out_occured = des.str.contains(
        r"\bout at (first|1st)\b|\btagged out\b|"
        r"\bdouble play\b|\bforce out\b|\btriple play\b",
        literal=False
    )
    dropped_strike_three = des.str.contains(
        'dropped third strike|swinging strike \\(blocked\\)', 
        literal=False
    )

    has_error = (
        has_throwing_error | has_fielding_error | has_missed_catch_error |
        has_interference_error
    )
    reached_on_error = reaches & has_error & ~out_occured
    reached_on_dropped_strike_three = reaches & dropped_strike_three & ~out_occured

    final_pitch = cl('events').is_not_null()

    return df.with_columns(
        is_reached_on_error = reached_on_error & final_pitch,
        is_reached_on_dropped_strike_three = reached_on_dropped_strike_three & final_pitch
    ) 

def add_pickoffs_and_caught_stealing(df):
    des = cl('des').str.to_lowercase()
    has_pickoff = des.str.contains(r'pick.+ off',literal=False)
    has_caught = des.str.contains('caught', literal=False)
    return df.with_columns(
        is_pickoff = has_pickoff & cl('is_last_pitch_pa'),
        is_caught_steal = has_caught & cl('is_last_pitch_pa'),
    )

def add_auto_strikes(df):
    auto_strike = cl('des').str.to_lowercase().str.contains('automatic strike')
    return df.with_columns(
        is_auto_strikeout = auto_strike & cl('is_last_pitch_pa')
    )

def add_intentional_walks(df):
    intentional = cl('des').str.to_lowercase().str.contains('intentional')
    return df.with_columns(
        is_intentional_walk = intentional & cl('is_last_pitch_pa')
    )

def add_other_walks(df):
    des = cl('des').str.to_lowercase()
    missed_plays = cl('is_bb').is_null()
    is_other_bb = des.str.contains('walks') & cl('is_last_pitch_pa') & missed_plays
    return df.with_columns(
        is_bb = cl('is_bb') | is_other_bb
    )

def add_sprint_speed_indicator(df):
    return df.with_columns(has_sprint_speed=cl('sprint_speed').is_not_null())

def add_league_tag(df):
    return df.with_columns(lg = cl('team_abbr').replace(abbr_to_lg))

# Framing model work
def add_catcher_and_pitcher_years(df):
    return df.with_columns(
        catcher_year = cl('fielder_2').cast(str) + '-' + cl('season').cast(str),
        pitcher_year = cl('pitcher').cast(str) + '-' + cl('season').cast(str),
    )

# Pitch model work
def add_pitch_quality_features(df):
    spin_axis_inferred = (
        pl.arctan2(-cl('pfx_x'), cl('pfx_z')).degrees() + 180
    )
    spin_axis_diff1 = (cl('spin_axis') - spin_axis_inferred).mod(360)
    spin_axis_diff2 = (spin_axis_inferred - cl('spin_axis')).mod(360)
    spin_axis_diff = (
        pl.when(spin_axis_diff1 <= spin_axis_diff2)
          .then(spin_axis_diff1)
          .otherwise(-spin_axis_diff2)
    )
    movement_norm = (cl('pfx_x')**2 + cl('pfx_z')**2).sqrt()
    spin_efficiency = movement_norm/cl('release_spin_rate')
    return (
        df.with_columns(
            log_extension = cl('release_extension').log(),
            spin_axis_diff = spin_axis_diff,
            spin_efficiency = spin_efficiency,
            a_mag = (cl('ax')**2 + cl('ay')**2 + cl('az')**2).sqrt(),
        )
    )

def add_pitch_quality_indicators(df):
    swings = [
        'foul_bunt',
        'swinging_strike',
        'swinging_strike_blocked',
        'missed_bunt',
        'foul_tip',
        'foul_pitchout',
        'bunt_foul_tip',
        'foul',
        'hit_into_play',
    ]
    fouls = [
        'foul_bunt',
        'foul_tip',
        'foul_pitchout',
        'foul',
        'bunt_foul_tip',
    ]
    whiffs = [
        'swinging_strike',
        'swinging_strike_blocked',
        'missed_bunt',
    ]

    return df.with_columns(
        is_swing = cl('description').is_in(swings),
        is_hbp = cl('description').eq('hit_by_pitch'),
        is_foul = cl('description').is_in(fouls),
        is_whiff = cl('description').is_in(whiffs),
        is_1b = cl('is_1b').fill_null(False),
        is_2b = cl('is_2b').fill_null(False),
        is_3b = cl('is_3b').fill_null(False),
        is_hr = cl('is_hr').fill_null(False),
        is_sf = cl('is_sf').fill_null(False),
        is_gidp = cl('is_gidp').fill_null(False),
        is_out = cl('is_out').fill_null(False),
    )

def add_pitch_type_features(df):
    norm_map = {
        'FA': 'FF', 'FT': 'SI',
        'SV': 'SL', 'ST': 'SL', 'CL': 'SL',
        'KC': 'CU',
    }
    df = df.with_columns(
        pitch_type_norm = cl('pitch_type').replace(norm_map).fill_null(cl('pitch_type'))
    )

    slow_cut = cl('pitch_type_norm').eq('FC') & (cl('release_speed') < 89.)
    hard_cut = cl('pitch_type_norm').eq('FC') & (cl('release_speed') >= 89.)

    fast_list = ['FF', 'SI', 'FC']
    slow_list = ['CH', 'FS', 'SC']
    bend_list = ['CU', 'SL', 'ST', 'KC', 'SV', 'FC', 'CS', 'KN', 'EP', 'FO', 'PO']

    df = df.with_columns(
        is_fast = cl('pitch_type_norm').is_in(fast_list) ^ slow_cut,
        is_slow = cl('pitch_type_norm').is_in(slow_list),
        is_bend = cl('pitch_type_norm').is_in(bend_list) ^ hard_cut,
    )

    # Idenfity primary/secondary
    usage = (
        df
        .group_by(['pitcher', 'season', 'pitch_type_norm'])
        .agg(
            n = pl.len().alias('n'),
            v_mean = cl('release_speed').mean(),
        )
    )
    primary = (
        usage
        .group_by(['pitcher', 'season'])
        .agg(
            primary_offer = (
                cl('pitch_type_norm')
                .sort_by([cl('n'), cl('v_mean')], descending=[True, True])
                .first()
            )
        )
    )

    # Join them back in
    df = (
        df
        .join(primary, on=['pitcher', 'season'], how='left')
        .with_columns(
            is_primary = cl('pitch_type_norm').eq(cl('primary_offer'))
        )
    )

    prim_means = (
        df
        .filter('is_primary')
        .group_by(['pitcher', 'season'])
        .agg(
            primary_release_speed = cl('release_speed').mean(),
            primary_pfx_x         = cl('pfx_x').mean(),
            primary_pfx_z         = cl('pfx_z').mean(),
        )
    )
    df = (
        df
        .join(prim_means, on=['pitcher', 'season'], how='left')
        .with_columns(
            diff_release_speed = cl('release_speed') - cl('primary_release_speed'),
            diff_pfx_x         = cl('pfx_x') - cl('primary_pfx_x'),
            diff_pfx_z         = cl('pfx_z') - cl('primary_pfx_z'),
        )
    )

    df = df.with_columns(
        pitch_type_classification = (
            pl
            .when('is_primary')
            .then(
                pl
                .when('is_fast').then(pl.lit('primary_fast'))
                .when('is_slow').then(pl.lit('primary_slow'))
                .when('is_bend').then(pl.lit('primary_bend'))
                .otherwise(pl.lit('primary_bend'))
            )
            .otherwise(
                pl
                .when('is_fast').then(pl.lit('secondary_fast'))
                .when('is_slow').then(pl.lit('secondary_slow'))
                .when('is_bend').then(pl.lit('secondary_bend'))
                .otherwise(pl.lit('secondary_bend'))
            )
        )
    )
    return df



