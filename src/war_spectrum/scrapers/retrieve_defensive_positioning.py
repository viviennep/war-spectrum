import polars as pl, requests, json, pandas as pd
from bs4 import BeautifulSoup
cl = pl.col
headers = {'User-Agent': 'Mozilla/5.0'}

'''
Old fielding run bible team DRS
'''

name_fixer = {
    'Cleveland Guardians'  :'CLE',
    'Toronto Blue Jays'    :'TOR',
    'Kansas City Royals'   :'KCR',
    'Milwaukee Brewers'    :'MIL',
    'Los Angeles Dodgers'  :'LAD',
    'St Louis Cardinals'   :'STL',
    'New York Yankees'     :'NYY',
    'Texas Rangers'        :'TEX',
    'Arizona Diamondbacks' :'ARI',
    'Boston Red Sox'       :'BOS',
    'Atlanta Braves'       :'ATL',
    'Seattle Mariners'     :'SEA',
    'Colorado Rockies'     :'COL',
    'Philadelphia Phillies':'PHI',
    'Detroit Tigers'       :'DET',
    'San Francisco Giants' :'SFG',
    'Houston Astros'       :'HOU',
    'Chicago Cubs'         :'CHC',
    'Los Angeles Angels'   :'LAA',
    'New York Mets'        :'NYM',
    'Minnesota Twins'      :'MIN',
    'Baltimore Orioles'    :'BAL',
    'San Diego Padres'     :'SDP',
    'Tampa Bay Rays'       :'TBR',
    'Pittsburgh Pirates'   :'PIT',
    'Miami Marlins'        :'MIA',
    'Oakland Athletics'    :'OAK',
    'Washington Nationals' :'WSN',
    'Cincinnati Reds'      :'CIN',
    'Chicago White Sox'    :'CHW'
}

team_pos = {}
team_pos[2023] = {'TOR':  sum((1,14,7)), 'MIL':  sum((0,13,-4)), 'LAD':  sum((7,19,6)), 'ARI':  sum((-2,9,-6)),
                  'SDP':  sum((1,10,7)), 'BAL':  sum((-4,1,5)), 'TEX':  sum((-10,11,-3)), 'CHC':  sum((-1,7,6)),
                  'MIN':  sum((4,16,0)), 'TBR':  sum((1,2,-2)), 'NYY':  sum((0,17,-1)), 'DET':  sum((3,15,18)),
                  'CLE':  sum((3,3,11)), 'COL':  sum((-8,5,3)), 'ATL':  sum((2,12,-10)), 'SEA':  sum((-2,18,3)),
                  'HOU':  sum((-2,9,8)), 'PIT':  sum((2,12,-5)), 'LAA':  sum((-1,7,9)), 'STL':  sum((-1,33,-1)),
                  'MIA':  sum((7,20,-13)), 'SFG':  sum((-3,12,2)), 'KCR':  sum((4,2,-7)), 'BOS':  sum((2,10,-6)),
                  'PHI':  sum((-4,-1,2)), 'NYM':  sum((-4,5,-4)), 'CIN':  sum((7,11,11)), 'WSN':  sum((5,1,-22)),
                  'CHW':  sum((0,9,-3)), 'OAK':  sum((-1,4,-10))}
team_pos[2022] = {'NYY':  sum((7,17,4)), 'LAD':  sum((8,37,5)), 'CLE':  sum((9,9,1)), 'HOU':  sum((0,34,5)),
                  'STL':  sum((-4,18,-8)), 'MIL':  sum((9,22,-2)), 'ARI':  sum((-4,30,5)), 'TOR':  sum((4,15,21)),
                  'BAL':  sum((-3,3,6)), 'SEA':  sum((-5,26,10)), 'ATL':  sum((-4,28,3)), 'DET':  sum((-4,24,11)),
                  'MIN':  sum((-1,24,-16)), 'LAA':  sum((4,20,5)), 'MIA':  sum((-2,22,-10)), 'NYM':  sum((8,12,-4)),
                  'COL':  sum((-3,10,-6)), 'TBR':  sum((1,11,-2)), 'TEX':  sum((1,20,-8)), 'SDP':  sum((-5,21,-4)),
                  'CHC':  sum((0,16,-1)), 'PIT':  sum((-4,18,7)), 'BOS':  sum((0,27,10)), 'OAK':  sum((-9,9,2)),
                  'PHI':  sum((-3,10,-5)), 'CHW':  sum((2,11,1)), 'CIN':  sum((-6,13,-8)), 'KCR':  sum((-3,13,-4)),
                  'WSN':  sum((5,17,-12)), 'SFG':  sum((7,24,-1))}
team_pos[2021] = {'STL':  sum((0,13,-14)), 'TEX':  sum((-7,15,6)), 'TBR':  sum((0,14,3)), 'COL':  sum((-2,9,-3)),
                  'HOU':  sum((0,14,6)), 'MIL':  sum((4,24,5)), 'NYM':  sum((8,22,9)), 'ATL':  sum((0,32,6)),
                  'MIA':  sum((-2,23,-5)), 'LAD':  sum((9,23,2)), 'SFG':  sum((1,34,-1)), 'CHC':  sum((5,27,-7)),
                  'MIN':  sum((4,22,-5)), 'KCR':  sum((5,2,-4)), 'TOR':  sum((-4,10,-2)), 'SDP':  sum((-1,20,-14)),
                  'WSN':  sum((-2,21,-1)), 'BOS':  sum((7,13,8)), 'CLE':  sum((-1,17,-9)), 'PIT':  sum((-2,8,-8)),
                  'SEA':  sum((-11,13,7)), 'OAK':  sum((-6,10,1)), 'BAL':  sum((-3,12,8)), 'ARI':  sum((-6,11,11)),
                  'CHW':  sum((4,13,-3)), 'NYY':  sum((-4,16,7)), 'CIN':  sum((2,12,-4)), 'LAA':  sum((8,31,-7)),
                  'DET':  sum((5,29,7)), 'PHI':  sum((-8,9,-5))}

team_pos[2024] = {
    'Toronto Blue Jays'   :  sum((-4,29,1)),
    'Cleveland Guardians'  : sum(( 2,21,5)),
    'Los Angeles Dodgers'  : sum(( 6,28,17)),
    'Milwaukee Brewers'    : sum((-2,16,-2)),
    'Detroit Tigers'       : sum(( 5,26,2)),
    'Boston Red Sox'       : sum(( 1,17,5)),
    'Kansas City Royals'   : sum((-6,8,-3)),
    'Seattle Mariners'     : sum(( 0,20,8)),
    'Chicago Cubs'         : sum(( 8,11,1)),
    'Atlanta Braves'       : sum(( 9,13,1)),
    'Texas Rangers'        : sum(( 0,15,-5)),
    'New York Yankees'     : sum(( 7,17,5)),
    'St Louis Cardinals'   : sum(( 4,19,10)),
    'Colorado Rockies'     : sum((11,11,-8)),
    'New York Mets'        : sum(( 5,12,11)),
    'Los Angeles Angels'   : sum(( 0,25,-4)),
    'Arizona Diamondbacks' : sum((-6,14,-10)),
    'Philadelphia Phillies': sum((-8,11,-5)),
    'Houston Astros'       : sum((-4,21,2)),
    'Baltimore Orioles'    : sum((-9,15,8)),
    'San Francisco Giants' : sum((-5,6,-1)),
    'Pittsburgh Pirates'   : sum(( 1,11,-1)),
    'Tampa Bay Rays'       : sum(( 1,15,-16)),
    'San Diego Padres'     : sum(( 1,22,-5)),
    'Minnesota Twins'      : sum(( 1,30,-4)),
    'Washington Nationals' : sum((-2,9,-9)),
    'Miami Marlins'        : sum((-4,12,-1)),
    'Cincinnati Reds'      : sum(( 2,19,1)),
    'Oakland Athletics'    : sum((-6,17,4)),
    'Chicago White Sox'    : sum(( 0,10,-6))
}
team_pos[2024] = {name_fixer[k]: v for k,v in team_pos[2024].items()}

def retrieve_positioning_runs(team_pos=team_pos):
    trs_url = "https://archive.fieldingbible.com/TeamDefensiveRunsSaved"
    res     = requests.get(trs_url,headers=headers)
    content = res.content.decode('utf-8')
    soup    = BeautifulSoup(content,features='lxml')

    names     = {i: soup.find(id=f'cphContent_rptLeaders_lblTeam_{i}').contents[0] for i in range(30)}
    nonshifts = {i: int(soup.find(id=f'cphContent_rptLeaders_lblNonShifts_{i}').contents[0]) for i in range(30)}
    shifts    = {i: int(soup.find(id=f'cphContent_rptLeaders_lblShifts_{i}').contents[0]) for i in range(30)}
    ofpos     = {i: int(soup.find(id=f'cphContent_rptLeaders_lblOFPos_{i}').contents[0]) for i in range(30)}

    team_pos[2025] = {name_fixer[names[i]]: nonshifts[i]+shifts[i]+ofpos[i] for i in range(30)}

    positioning_df = (
        pl.from_pandas(
            pd.DataFrame(team_pos)
            .stack()
            .rename_axis(['Team','Year'])
            .reset_index(name='Positioning')
        )
    )
    return positioning_df

