import numpy as np, polars as pl, requests, json
cl = pl.col
headers = {'User-Agent': 'Mozilla/5.0'}

def retrieve_player_bios(ids):
    people = []
    for i in np.array_split(ids,max(1,len(ids)//300)):
        api_url = f"https://statsapi.mlb.com/api/v1/people?personIds={','.join(map(str,i))}"
        res = requests.get(api_url,headers={'UserAgent':'Mozilla'}).json()
        people += res['people']
    return pl.DataFrame(people)

