import polars as pl, requests, json, pandas as pd
import pybaseball as pb
from war_spectrum.preprocess.standardize import insane_fg_schema 
from bs4 import BeautifulSoup
cl = pl.col

headers = {'User-Agent': 'Mozilla/5.0'}

prg_dict = {
    'game_type'     : 'Regular',
    'n'             : '1',
    'pitch_hand'    : 'all',
    'runner_moved'  : 'All',
    'prior_pk'      : 'All',
    'season_end'    : '2025',
    'season_start'  : '2016',
    'sortColumn'    : 'simple_prevented_on_running_attr',
    'sortDirection' : 'desc',
    'split'         : 'yes',
    'team'          : 'split',
    'type'          : 'Pit',
    'with_team_only': '1',
    'expanded'      : '0'
}

def retrieve_pitcher_run_game(
        start_year=None,
        end_year=None,
        prg_dict=prg_dict,
    ):
    prg_url   = "https://baseballsavant.mlb.com/leaderboard/pitcher-running-game?"
    prg_url  += '&'.join(f"{k}={v}" for k,v in prg_dict.items())
    res       = requests.get(prg_url,headers=headers)
    content   = res.content.decode('utf-8')
    start_ind = content.index('const data = [')+13
    end_ind   = content[start_ind:].index(';')+start_ind
    json_data = content[start_ind:end_ind]
    prg_df    = pl.DataFrame(json.loads(json_data))
    return prg_df

oaa_dict = {
    'type'     : 'Pitcher',
    'startYear': '2021',
    'endYear'  : '2024',
    'split'    : 'yes',
    'team'     : '',
    'range'    : 'year',
    'min'      : '1',
    'pos'      : '',
    'roles'    : '',
    'viz'      : ''
}

def retrieve_pitcher_oaa(
        start_year=None,
        end_year=None,
        oaa_dict=oaa_dict,
    ):
    if start_year is not None:
        oaa_dict['startYear'] = str(start_year)
    if end_year is not None:
        oaa_dict['endYear'] = str(end_year)
    oaa_url  = "https://baseballsavant.mlb.com/leaderboard/outs_above_average?"
    oaa_url  += '&'.join(f"{k}={v}" for k,v in oaa_dict.items())
    res       = requests.get(oaa_url,headers=headers)
    content   = res.content.decode('utf-8')
    start_ind = content.index('var data = [')+11
    end_ind   = content[start_ind:].index(';')+start_ind
    json_data = content[start_ind:end_ind]
    oaa_df    = pl.DataFrame(json.loads(json_data))
    return oaa_df

def retrieve_baseball_reference_war():
    rwar_df = pd.read_csv("https://www.baseball-reference.com/data/war_daily_pitch.txt")
    return pl.from_pandas(rwar_df)

def retrieve_sprint_speed_data(year):
    df = pl.from_pandas(pb.statcast_sprint_speed(year,1))
    return df.with_columns(year = year)

def retrieve_fangraphs_data(year):
    fg_data_url  = f"https://www.fangraphs.com/api/leaders/major-league/data?"
    fg_data_url += f"age=&pos=all&stats=&lg=all&qual=1&season=&season1=&"
    fg_data_url += f"startdate=&enddate=&month=0&hand=&team=&"
    fg_data_url += f"pageitems=2000000000&pagenum=1&ind=0&rost=0&players=&"
    fg_data_url += f"type=c,0,1,6,7,8,13,16,18,19,20,21,24,28,45,55,57,58,59,62,66,70,410&"
    fg_data_url += f"sortdir=default&sortstat=Throws"
    pit_url = fg_data_url.replace('stats=','stats=pit').replace('season=',f'season={year}')
    res = requests.get(pit_url).content
    pitchers = json.loads(res)['data']
    df = pl.from_pandas(pd.DataFrame(pitchers)).cast(insane_fg_schema)
    return df


