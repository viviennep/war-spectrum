import numpy as np, polars as pl
cl = pl.col

def create_pa_df(df):
    return (
        df
        .sort('game_date','game_pk','at_bat_number','pa_pitch_number')
        .filter(
            cl('pa_pitch_number').eq(
                cl('pa_pitch_number').max().over('game_pk','at_bat_number')
            )
        )
        .select(
            'pitcher', 'season', 'team_abbr', 'lg',
            'game_date','game_pk','inning','inning_topbot',
            'at_bat_number','pa_pitch_number',
            'events',
            'runs_scored_play',                             
            batter_reached = (
                (cl('is_hit') | cl('is_bb') | 
                 cl('is_catcher_interf') | cl('is_reached_on_error')) 
                & ~cl('is_hr')
            ).fill_null(False)
        )
    )

def test(df):
    print(len(df))
    return pl.DataFrame({'a':len(df)})

def assign_runs_in_half_inning(df):
    runner_queue = []      # holds pitcher ids responsible for baserunners 
    charged = {}    # holds running tally of pitcher charged runs
    for row in df.iter_rows(named=True):
        pitcher          = int(row['pitcher'])
        runs_scored_play = int(row['runs_scored_play'])
        batter_reached   = int(row['batter_reached'])
        while runs_scored_play>0 and runner_queue:
            # if runs are scored, charge them to the resp pitchers
            resp_pitcher = runner_queue.pop(0)
            charged[resp_pitcher] = charged.get(resp_pitcher,0) + 1
            runs_scored_play -= 1
        if runs_scored_play > 0:
            # if runs remain assign them to the current pitcher
            charged[pitcher] = charged.get(pitcher,0) + runs_scored_play
        if batter_reached:
            # if the batter reached, add him to the queue
            runner_queue.append(pitcher)
    if charged:
        out = pl.DataFrame({
            'game_pk'      : [df['game_pk'][0]]*len(charged),
            'season'       : [df['season'][0]]*len(charged),
            'team_abbr'    : [df['team_abbr'][0]]*len(charged),
            'inning'       : [df['inning'][0]]*len(charged),
            'inning_topbot': [df['inning_topbot'][0]]*len(charged),
            'pitcher'      : list(charged),
            'charged_runs' : list(charged.values())
        })
    else:
        out = pl.DataFrame(
            schema = {
                'game_pk'      : pl.Int64,
                'season'       : pl.Int64,
                'team_abbr'    : pl.Utf8,
                'inning'       : pl.Int64,
                'inning_topbot': pl.Utf8,
                'pitcher'      : pl.Int64,
                'charged_runs' : pl.Int64
            }
        )
    return out





