import streamlit as st, numpy as np, polars as pl, pandas as pd, plotly.express as px
import pathlib
import plotly.graph_objects as go
import plotly.figure_factory as ff
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode, JsCode
cl = pl.col

data_dir  = pathlib.Path(__file__).resolve().parent / 'data'

stints = pl.read_parquet(data_dir / 'stints')

war_convert = {
    'RA_WAR'   : 'Runs Allowed WAR',
    'Rally_WAR': 'Rally WAR',
    'OAA_WAR'  : 'OAA WAR',
    'BsR_WAR'  : 'BaseRuns WAR',
    'xBsR_WAR' : 'xBaseRuns WAR',
    'DIPS_WAR' : 'FIP WAR',
    'Stuff_WAR': 'Stuff+ WAR',
    'Pitch_WAR': 'Pitch+ WAR',
}

wars = (
    stints.group_by('pitcher','season')
    .agg(
        cl('name').first(),
        *(cl(i).sum() for i in war_convert),
        IP      = cl('stint_out').sum()/3,
        RA      = cl('runs_allowed').sum(),
        ER      = (
            pl.when(cl('R_fg')>0)
            .then(cl('runs_allowed')*cl('ER_fg')/cl('R_fg'))
            .otherwise(0.)
            .sum()
        ),
        BsR     = cl('baseruns').sum(),
        xBsR    = cl('xbaseruns').sum(),
        dipsBsR = cl('dips_baseruns').sum(),
        piBsR   = cl('xbaseruns_pitch').sum(),
        stBsR   = cl('xbaseruns_stuff').sum(),
        K       = cl('stint_k').sum(),
        PA      = cl('stint_pa').sum(),
        xERA    = (cl('stint_pa')*cl('xERA_fg')).sum()/cl('stint_pa').sum(),
        team=(
            pl.when(cl('team_abbr').len().eq(1))
            .then(cl('team_abbr').first())
            .when(cl('team_abbr').len().eq(2)).then(pl.lit('2TM'))
            .when(cl('team_abbr').len().eq(3)).then(pl.lit('3TM'))
            .when(cl('team_abbr').len().eq(4)).then(pl.lit('4TM'))
            .when(cl('team_abbr').len().eq(5)).then(pl.lit('5TM'))
            .when(cl('team_abbr').len().eq(6)).then(pl.lit('6TM'))
            .otherwise(pl.lit('7+TM'))
        )
    )
    .filter(cl('IP')>0)
    .with_columns(
        Average  = pl.sum_horizontal(list(war_convert))/(len(war_convert)),
        StdDev   = pl.concat_list(list(war_convert)).list.std(),
        ERA      = 9*cl('ER')/cl('IP'),
        RA9      = 9*cl('RA')/cl('IP'),
        BsR9     = 9*cl('BsR')/cl('IP'),
        xBsR9    = 9*cl('xBsR')/cl('IP'),
        dipsBsR9 = 9*cl('dipsBsR')/cl('IP'),
        piBsR9   = 9*cl('piBsR')/cl('IP'),
        stBsR9   = 9*cl('stBsR')/cl('IP'),
        Kpct     = cl('K')/cl('PA'),
    )
    .sort('Average',descending=True)
)

def wcorr(x,y,w):
    μx = (w*x).sum()/w.sum()
    μy = (w*y).sum()/w.sum()
    vx = (w*(x-μx)**2).sum()/w.sum()
    vy = (w*(y-μy)**2).sum()/w.sum()
    cov = (w*(x-μx)*(y-μy)).sum()/w.sum()
    return cov/np.sqrt(vx*vy)

def future_corr(df, x_cols, target, w_col, n_years = 5, cutoff = 15):
    res = np.zeros((n_years,len(x_cols)))
    for year in range(n_years):
        future_df = df.select(
            *[cl(i).alias(f"{i}_future") for i in x_cols],
            pitcher = 'pitcher',
            year = cl('season')-year,
            w_future = w_col,
            target_future = target
        ).filter(cl('w_future')>cutoff)
        joined = (
            df.select(
                *x_cols,
                'pitcher',
                target = target,
                year = cl('season'),
                w=w_col,
            )
            .filter(cl('w')>cutoff)
            .join(
                future_df,
                on=['pitcher','year'],
            ).with_columns(
                w = 2/(1/cl('w') + 1/cl('w_future'))
            ).filter(cl('w')>0)
        )
        w = joined['w'].to_numpy()
        y = joined['target_future'].to_numpy()
        for i,col in enumerate(x_cols):
            res[year,i] = wcorr(joined[col],y,w)
    return res
        
last_date = (
    pl.scan_parquet(data_dir/'play_by_play')
    .select('game_date')
    .max()
    .collect()
    .item()
).strftime('%m/%d/%Y')

st.set_page_config(layout="wide")

st.markdown(f'''

# Pitcher WAR Spectrum

##### Filling in the gaps between rWAR and fWAR, and Beyond!
If you, like me, are a fan of having different WAR perspectives but wished that there 
were instead something like, oh idk, 8 different perspectives to pick and choose from 
for agenda crafting purposes, then boy do I got the leaderboard for you.

Here are those 8 WARs of your dreams &mdash; 6½ of which are new, each differing in 
what it claims is the responsibility of the pitcher, starting by forcing him to reckon 
with each and every run he allowed in their entirety, and then proceeding to strip away
responsibility step by step to the point that the model doesn't even know about what 
happened on any of the pitches he threw, or where they were located for that matter. 

There's a sortable & filterable leaderboard that has all pitcher seasons from 2021-2024,
if you select the rows then a line plot of their WARs will appear below the table :smile:
There are also a few dropdowns where I explain each of the WAR calculations and justify 
my decisions for them all, including an explanation for why I'm using BaseRuns rather 
than something more familiar like (x)wOBA. After that there are some tables which show 
the correlations between all of these WARs and explain the differences between them 
all :cherry_blossom:

*Last Updated: {last_date}*

''')


css = {
    '.ag-header-group-cell-label.ag-sticky-label': {
        'flex-direction': 'column',
        'margin': 'auto',
        'font-size': '12pt'
    }
}

columnDefs = [
    {
        'field': "name", 
        'headerName': "Name",
        'minWidth': 120, 
        'filter': True, 
        'sortable': False, 
        'pinned': 'left'
    },
    {
        'field': "season",
        'headerName': "Year",
        'minWidth':  70,
        'filter': True,
        'sortable': True,
    },
    {
        'field': "Age",
        'minWidth':  70,
        'filter': True,
        'sortable': True, 
        'suppressHeaderFilterButton': False
    },
    {
        'field': "team",
        'headerName': "Team",
        'minWidth':  70,
        'filter': True,
        'sortable': True,
        'suppressHeaderFilterButton': False
    },
    {
        'headerName': "Runs Allowed",
        'headerTooltip': "Pitcher's runs allowed are used",
        'children': [
            {
                'field': 'RA_WAR',
                'headerName': 'Runs Allowed',
                'minWidth': 130,
                'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1,
                'headerTooltip': "Pitcher is responsible for all runs allowed",
                'tooltipValueGetter': JsCode(
                    """function(){return "Pitcher is responsible for all runs allowed"}"""
                )
            },
            {
                'field': 'Rally_WAR',
                'headerName': 'Baseball Reference',
                'minWidth': 150,
                'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1,
                'headerTooltip': "Corrected for team's defence by DRS",
                'tooltipValueGetter': JsCode(
                    """function(){return "Corrected for team's defence"}"""
                )
            },
            {
                'field': 'OAA_WAR',
                'headerName': 'OAA',
                'minWidth': 80,
                'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1,
                'headerTooltip': "Corrected using team OAA when pitcher is on the mound",
                'tooltipValueGetter': JsCode(
                    """function(){return "Corrected using team OAA when pitcher is on the mound"}"""
                )
            },
        ]
    },
    {
        'headerName': "Runs Allowed Estimators",
        'headerTooltip': "A model which estimates a pitcher's runs allowed is used",
        'children': [
            {
                'field': 'BsR_WAR',
                'headerName': 'BaseRuns',
                'minWidth': 110,
                'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1,
                'headerTooltip': "Like OAA-WAR, but with the pitcher's BaseRuns run estimate",
                'tooltipValueGetter': JsCode(
                    """function(){return "Like OAA-WAR but with the pitcher's BaseRuns run estimate"}"""
                )
            },
            {
                'field': 'xBsR_WAR',
                'headerName': 'xBaseRuns',
                'minWidth': 110,
                'headerName': 'xBaseRuns',
                'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1,
                'headerTooltip': "Like BaseRuns-WAR, but the xERA-style xBaseRuns is used",
                'tooltipValueGetter': JsCode(
                    """function(){return "Like BaseRuns-WAR, but the xERA-style xBaseRuns is used"}"""
                )
            },
            {
                'field': 'DIPS_WAR',
                'headerName': 'FIP',
                'minWidth': 80,
                'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1,
                'headerTooltip': "rWAR-style but with FIP",
                'tooltipValueGetter': JsCode(
                    """function(){return "rWAR-style but with FIP"}"""
                )
            },
        ]
    },
    {
        'headerName': "Pitch Modelling",
        'headerTooltip': "A model which estimates a pitcher's runs allowed is used",
        'children': [
            {
                'field': 'Pitch_WAR',
                'headerName': 'Pitching+',
                'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1,
                'headerTooltip': "Like BaseRuns-WAR, but uses Pitching+ style model for xBaseRuns",
                'tooltipValueGetter': JsCode(
                    """function(){return "Like BaseRuns-WAR, but uses Pitching+ style model for xBaseRuns"}"""
                )
            },
            {
                'field': 'Stuff_WAR',
                'headerName': 'Stuff+',
                'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1,
                'headerTooltip': "Like BaseRuns-WAR, but uses Stuff+ style model for xBaseRuns",
                'tooltipValueGetter': JsCode(
                    """function(){return "Like BaseRuns-WAR, but uses Stuff+ style model for xBaseRuns"}"""
                )
            },
        ]
    },
    {
        'headerName': "Statistics",
        'headerTooltip': "Average and standard deviation of these WARs",
        'children': [
            {
                'field': 'Average',
                'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1
            },
            {
                'field': 'StdDev',
                'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1
            } 
        ]
    },
]

gridOptions =  {
    'defaultColDef': {
        'flex': 1,
        'minWidth': 120,
        'filterable': True,
		'groupable': False,
        'editable': False, 
        'wrapText': True,
        'autoHeight': True, 
        'suppressMovable': True,
        'suppressMenu': False
    },
    'columnDefs': columnDefs,
    'initialState': {'rowSelection': [0,1]},
    'tooltipShowDelay': 800, 
    'tooltipMouseTrack': True,
    'rowSelection': 'multiple', 
    'rowMultiSelectWithClick': True, 
    'suppressRowDeselection': False, 
    'suppressRowClickSelection': False, 
    'groupSelectsChildren': False, 
    'groupSelectsFiltered': True
}

st.markdown('''
#### WAR Leaderboard
You can filter columns on mobile by holding down the column header, or on desktop by 
clicking the menu button when you hover over it :blush:  
This lets you, for example, search for specific pitchers, or seasons within a specific 
WAR range.
''')

team_list = wars.select('team').unique().sort('team')['team'].to_numpy().tolist()

left_col,right_col = st.columns(2)
with left_col.expander('Included Years') :
    years_select = st.multiselect(
        "Included years",
        [2021,2022,2023,2024,2025],
        [2025]
    )
with right_col.expander('Included Teams') :
    teams_select = st.multiselect(
        "Included Teams", 
        wars.select('team').unique().sort('team')['team'].to_numpy().tolist(), 
        team_list,
        label_visibility='collapsed'
    )

return_value = AgGrid(
    wars.filter(
        cl('season').is_in(years_select) &
        cl('team').is_in(teams_select)
    ).to_pandas(),
    gridOptions=gridOptions,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    allow_unsafe_jscode=True,
    #fit_columns_on_grid_load=True,
    height=700,
    theme="streamlit",
    #key=None,
    key="stable_grid",
    custom_css=css
)

st.markdown('''#### Selected Players WAR''')
if return_value.selected_rows is None:
    st.write('''Select rows in the table to see a line plot of their WARs''')
else:
    if return_value.selected_rows.shape[0] == 1:
        row = return_value.selected_rows.iloc[0]
        title = row['name'] + ' ' + str(row['season'])
    else:
        title = ''
    f = go.Figure()
    min_war = return_value.selected_rows[list(war_convert)].min().min()
    max_war = return_value.selected_rows[list(war_convert)].max().max()
    for ind,row in return_value.selected_rows.iterrows():
        label = row['name'] + ' ' + str(row['season'])
        f.add_trace(go.Scatter(x=np.arange(8),y=row[list(war_convert)].values,
                               mode='lines+markers',name=label))
    f.update_layout(xaxis = {'tickmode': 'array', 
                             'tickvals': np.arange(8), 
                             'ticktext': list(war_convert.values())},
                    yaxis_range = [min(0,min_war),max_war+0.1],
                    title = title)
    config = {
        'use_containter_width': False,
        'width': 100,
    }
    st.plotly_chart(f,config=config)

st.markdown('''#### Calculation Details''')
raa_exp = st.expander('RAA Calculations for Each WAR')
with raa_exp:
    ra_tab, rw_tab, oa_tab, bs_tab, fi_tab, xb_tab, pi_tab, st_tab = (
        st.tabs([
            'Runs Allowed WAR',
            'Baseball Reference WAR',
            'OAA WAR',
            'BaseRuns WAR',
            'FIP WAR',
            'xBaseRuns WAR',
            'Pitching+ WAR',
            'Stuff+ WAR',
        ])
    )
    with ra_tab:
        st.markdown(r'''
#### Runs Allowed RAA
No corrections for park factors or defence. Corrections are still applied for relief
pitchers, quality of opponents, and for extra-inning automatic runners.
$$
\mathrm{RAA_{runs} = \left(xRA + RP_{adj} + ExIn_{adj}\right) \newline
        - RA + lg_{adj}}
$$
- $\small\mathrm{xRA}$ is how other pitchers performed against this pitcher's opponents 
  on average
- $\small\mathrm{RP_{adj}}$ is an adjustment for how guys perform better as relief 
 pitchers than starters
- $\small\mathrm{ExIn_{adj}}$ is an adjustment for the automatic runners in extra 
  innings
- $\small\mathrm{lg_{adj}}$ is an adjustment to ensure the league's total RAA is $0$
''',unsafe_allow_html=True)

    with rw_tab:
        st.markdown(r'''
#### Baseball Reference RAA
I explain this one in more detail over in the 'Details of the WAR Calculation' 
dropdown, and obviously there's the
[Baseball Reference pitcher WAR explainer](https://www.baseball-reference.com/about/war_explained_pitch.shtml).  

Corrections are applied for the parks the pitcher pitched in, the quality
of his opponents, his team's defensive runs saved and their positioning,
the starter/reliever discrepancy, and for extra-inning automatic runners.
$$
\mathrm{RAA_{rally} = PF_{pit}\times\left(xRA-R_{def}+RP_{adj}\right)+ExIn_{adj}
                                          - RA + lg_{adj}}
$$
- $\small\mathrm{PF_{pit}}$ is the cumulative park factor for the parks in which 
  this pitcher pitched
- $\small\mathrm{xRA}$ is how other pitchers performed against this pitcher's 
  opponents on average
- $\small\mathrm{R_{def} = \frac{BIP_{pit}}{BIP_{Tm}}\left(DRS_{Tm}+PosR_{Tm}\right)}$ 
  is the correction for the team's defence & positioning
- $\small\mathrm{RP_{adj}}$ is an adjustment for how guys perform better as relief 
  pitchers than starters
- $\small\mathrm{ExIn_{adj}}$ is an adjustment for the automatic runners in extra 
  innings
- $\small\mathrm{lg_{adj}}$ is an adjustment to ensure the league's total RAA is $0$
''')

    with oa_tab:
        st.markdown(r'''
#### Baseball Reference but with OAA Defence RAA
This is the same as Baseball Reference WAR but with the team DRS defensive correction 
replaced with the team's actual Fielding Run Value (FRV) while the pitcher is on the 
mound. The team positioning correction remains, though.

Same as rWAR: corrections are applied for the parks the pitcher pitched in, the quality
of his opponents, his team's defensive runs saved and their positioning,
the starter/reliever discrepancy, and for extra-inning automatic runners.
$$
\mathrm{RAA_{oaa} = PF_{pit}\times\left(xRA-PosR_{Tm}\frac{BIP_{pit}}
{BIP_{Tm}}+RP_{adj}\right)-FRV_{pit}-Framing+ExIn_{adj} - RA + lg_{adj}}
$$
- $\small\mathrm{PF_{pit}}$ is the cumulative park factor for the parks in which this 
  pitcher pitched
- $\small\mathrm{xRA}$ is how other pitchers performed against this pitcher's opponents 
  on average
- $\small\mathrm{PosR_{def} = \frac{BIP_{pit}}{BIP_{Tm}}PosR_{Tm}}$ is the correction 
  for the team's positioning
- $\small\mathrm{RP_{adj}}$ is an adjustment for how guys perform better as relief 
  pitchers than starters
- $\small\mathrm{FRV}$ is the team's fielding run value (OAA) while the pitcher is 
  actually on the mound, pre-park factored by Statcast (I think)
- $\small\mathrm{Framing}$ is an adjustment for the framing attributable to this 
  pitcher's catchers while he was pitching.
- $\small\mathrm{ExIn_{adj}}$ is an adjustment for the automatic runners in extra 
  innings
- $\small\mathrm{RA}$ is the pitcher's runs allowed
- $\small\mathrm{lg_{adj}}$ is an adjustment to ensure the league's total RAA is $0$
''')

    with bs_tab:
        st.markdown(r'''
#### BaseRuns RAA
This is the same as OAA WAR but with the pitcher's actual runs allowed replaced by the 
BaseRuns estimate for this runs allowed. 

I explain the BaseRuns estimator over [here](#baseruns), but the purpose of using it 
here is to control for the pitcher's sequencing luck. 

Same as OAA-WAR: corrections are applied for the parks the pitcher pitched in, the 
quality of his opponents, his team's defensive runs saved and their positioning,
the starter/reliever discrepancy; however, since real runs allowed are not used, 
a correction for the extra-innings automatic runners isn't needed.
$$
\mathrm{RAA_{bsr} = PF_{pit}\times\left(xRA-PosR_{Tm}\frac{BIP_{pit}}{BIP_{Tm}}
+RP_{adj}\right)-FRV_{pit}-Framing-BsR+lg_{adj}}
$$
- $\small\mathrm{PF_{pit}}$ is the cumulative park factor for the parks in which this
  pitcher pitched
- $\small\mathrm{xRA}$ is how other pitchers performed against this pitcher's opponents
  on average
- $\small\mathrm{PosR_{def} = \frac{BIP_{pit}}{BIP_{Tm}}PosR_{Tm}}$ is the correction 
  for the team's positioning
- $\small\mathrm{RP_{adj}}$ is an adjustment for how guys perform better as relief
  pitchers than starters
- $\small\mathrm{FRV}$ is the team's fielding run value (OAA) while the pitcher is 
  actually on the mound, pre-park factored by Statcast (I think)
- $\small\mathrm{Framing}$ is an adjustment for the framing attributable to this 
  pitcher's catchers while he was pitching.
- $\small\mathrm{BsR}$ is the pitcher's BaseRuns run estimate
- $\small\mathrm{lg_{adj}}$ is an adjustment to ensure the league's total RAA is $0$
''')

    with xb_tab:
        st.markdown(r'''
#### xBaseRuns RAA
I explain the xBaseRuns estimator over [here](#xbaseruns), but the purpose of using it
here is to control for the pitcher's sequencing luck **and** luck on batted balls.

A park factor correction is still applied, as is the starter/reliever discrepancy 
correction, but no defensive corrections need to be applied. Real walks and strikeouts
are used in xBaseRuns, so framing runs are still corrected.
$$
\mathrm{RAA_{xbsr} = PF_{pit}\times\left(xRA+RP_{adj}\right)-Framing-xBsR+lg_{adj}}
$$
- $\small\mathrm{PF_{pit}}$ is the cumulative park factor for the parks in which this
  pitcher pitched
- $\small\mathrm{xRA}$ is how other pitchers performed against this pitcher's 
  opponents on average
- $\small\mathrm{RP_{adj}}$ is an adjustment for how guys perform better as relief
  pitchers than starters
- $\small\mathrm{Framing}$ is an adjustment for the framing attributable to this 
  pitcher's catchers while he was pitching.
- $\small\mathrm{xBsR}$ is the pitcher's xBaseRuns run estimate
- $\small\mathrm{lg_{adj}}$ is an adjustment to ensure the league's total RAA is $0$
''')

    with fi_tab:
        st.markdown(r'''
#### "FIP" RAA
_(Updated for 2025!)_  
The name is lying to you, this isn't fWAR, this isn't rWAR with FIP either... 
this is Defensive Independent Pitching Baseruns WAR.

The inputs to this are identical to the inputs to fWAR, but rather than using the 
linear ifFIP run estimate that Fangraphs uses, I use the nonlinear Baseruns estimate.
Additionally, this differs from fWAR in the way in which RAA is calculated (this is
done in an rWAR style), how framing is calculated, how park factors are calculated, 
and how the runs-to-win conversion is calculated.

$$
\mathrm{RAA_{dipsbsr} = PF_{pit}\times\left(xRA+RP_{adj}\right)-Framing
-BsR_\mathrm{DIPS}+lg_{adj}}
$$
- $\small\mathrm{PF_{pit}}$ is the cumulative park factor for the parks in which this
  pitcher pitched
- $\small\mathrm{xRA}$ is how other pitchers performed against this pitcher's opponents
  on average
- $\small\mathrm{RP_{adj}}$ is an adjustment for how guys perform better as relief
  pitchers than starters
- $\small\mathrm{Framing}$ is an adjustment for the framing attributable to this 
  pitcher's catchers while he was pitching.
- $\small\mathrm{BsR_\mathrm{DIPS}}$ is the pitcher's [defense independent pitching 
  baseruns estimate](https://vorosmccracken.com/?page_id=14).
- $\small\mathrm{lg_{adj}}$ is an adjustment to ensure the league's total RAA is $0$
''')

    with pi_tab:
        st.markdown(r'''
#### Pitching+ RAA
This uses a Pitching+ style model to estimate the probability of each of the possible 
pitch outcomes for every pitch. These estimated probabilities are used in the xBaseRuns
formula to get an even more expected BaseRuns. This is different from normal
Pitching/Stuff+ models which simply add the run values of each pitch, but I found that
using a BaseRuns approach could better describe same-season runs allowed.

$$
\mathrm{RAA_{piBsR} = \left(\frac{RA_{lg}}{TBF_{lg}}TBF+RP_{adj}\right)-piBsR+lg_{adj}}
$$
- $\small\mathrm{\frac{RA_{lg}}{TBF_{lg}}TBF}$ is the expected number of runs allowed
  in this many batters faced
- $\small\mathrm{RP_{adj}}$ is an adjustment for how guys perform better as relief 
  pitchers than starters
- $\small\mathrm{piBsR}$ is the Pitching+ BaseRuns estimate
- $\small\mathrm{lg_{adj}}$ is an adjustment to ensure the league's total RAA is $0$

I based this Pitching+ model off the 
[Steve Brown](https://x.com/srbrown70)/Baseball Prospectus 
[PitchPro](https://www.baseballprospectus.com/news/article/89245/stuffpro-pitchpro-introduction-new-pitch-metrics-bp/) 
model, with slightly modified inputs, primary/secondary pitch classification, and the
number of sub-models. I did not seriously test any of the changes I made, which were
all out of laziness I assure you, so trust these pitch-modelling results at your own
peril.  

Inputs to the pitching model were:
- Year
- Release position
- Spin axis difference
- Spin efficiency
- Pitcher height
- Extension
- Arm Angle
- The count
- Batter's handedness
- Observed pitch movement
- Initial velocity & acceleration vectors 
- Pitch location 

Pitches were classified into 3 types:
- Fast: Four Seam Fastballs, Sinkers, Hard Cutters
- Slow: Changeups, Splitters, Screwballs
- Bend: Curveballs, Sliders, Sweepers, Knuckle Curves, Slurves, Slow Cutters, 
  Knuckleballs, Ephuses, Forkballs

Pitches were further divided into primary or secondary, based on which pitch the 
pitcher threw the most in each season. 5 models were then made for each classification 
task (swing/take, whiff/foul/bip if swing, etc.), one for:
- Primary fast pitches
- Primary slow pitches (either in the slow or bendy categories above)
- Secondary fast pitches
- Secondary slow pitches
- Secondary bendy pitches

''')

    with st_tab:
        st.markdown(r'''
#### Stuff+ RAA
This is just like the Pitching+ RAA but with a Stuff+ model instead.

$$
\mathrm{RAA_{stBsR} = \left(\frac{RA_{lg}}{TBF_{lg}}TBF+RP_{adj}\right)-stBsR+lg_{adj}}
$$
- $\small\mathrm{\frac{RA_{lg}}{TBF_{lg}}TBF}$ is the expected number of runs allowed
  in this many batters faced
- $\small\mathrm{RP_{adj}}$ is an adjustment for how guys perform better as relief
  pitchers than starters
- $\small\mathrm{stBsR}$ is the Stuff+ BaseRuns estimate
- $\small\mathrm{lg_{adj}}$ is an adjustment to ensure the league's total RAA is $0$

The Stuff model is effectively the same as the Pitching model, with any information
which betray the location of the pitch removed, namely:
- The initial velocity & acceleration vectors are replaced with their respective
  magnitudes
- The pitch location is removed

Other than this, the model is exactly the same as the Pitching model.
''')

baseruns_exp = st.expander('BaseRuns Run Estimator Explanation & Justification')
baseruns_exp.markdown(r'''
#### BaseRuns Run Estimator
The BaseRuns run estimator is a nonlinear run estimator which I have opted to use for
estimating pitching runs allowed for two reasons: first because it allows a consistent
method of estimating runs across all levels of this exercise (in fact its form fits 
naturally fits with how modern classification-based methods, like xwOBA and pitch/stuff
models, work), and second because I found it to be more accurate (in the 
descriptiveness sense) than linear methods across the board. You can read up more about
it 
[here](https://www.tangotiger.net/rc2.html) or [here](https://gosu02.tripod.com/id9.html). 
This approach captures the fact that a double by a pitcher who allows a high OBP is 
worse than a double for a pitcher who does not allow many baserunners. xERA
approximates this behavior by using the square of a pitcher's xwOBA to predict his ERA,
but I figured if I'm doing all this work anyways, why not use the full thing.

##### What it Looks Like & How it Works
Its form is simple,
$$
\mathrm{BaseRuns = Baserunners\times ScoreRate + ForSureRuns}
$$
where $\small\mathrm{Baserunners}$ is the number of baserunners allowed,
$\small\mathrm{ScoreRate}$ is the rate at which baserunners score, and
$\small\mathrm{ForSureRuns}$ effectively just means Home Runs. I've seen some people
include Sac Flys, but I don't like that idea very much so I'm sticking with just
Home Runs. 
This equation as it is written isn't a run estimate, it's exactly the number
of runs allowed. The estimation comes in because the calculation of
$\small\mathrm{ScoreRate}$ from only available statistics is not a trivial task. The 
actual BaseRuns run estimator uses David Smyth's estimate for the
$\small\mathrm{ScoreRate}$, which goes as follows
$$
\mathrm{ScoreRate = \frac{AdvancementFactor}{AdvancementFactor + Outs}}
$$
where the $\small\mathrm{AdvancementFactor}$ is a linear combination of traditional
box score statistics: singles, doubles, triples, homers, walks, stolen bases, etc.
Empirically, this form work really well, so I'll be using it as well.
I opted to fit the weights of this linear combination to individual pitcher seasons 
from 2021-2024 to minimize the weighted RMSE to their runs allowed. Pre-computed
weights exist online, but they are either old and possibly outdated, or fit on team 
seasons rather than pitcher seasons, the transferability of which I'm unsure about.
For all of the implementations I've opted to use the following stats in the
$\small\mathrm{Advancement}$ term:
> Singles, Doubles, Triples, Home Runs, Walks, Sac Flys, GIDPs, Strikeouts,
  Ball-in-Play Outs, Stolen Bases, & Caught Stealings

##### Basic Implementation
The basic implementation looks like this,
$$
\mathrm{BaseRuns} \gets  \begin{cases}
\mathrm{Baserunners}  & \mathrm{H+BB-HR} \\
\mathrm{Advancement}  & \mathbf{b}\cdot\mathrm{[S,D,T,HR,BB,SF,GIDP,SO,BIPOut,SB,CS]} \\
\mathrm{Outs}         & \mathrm{3IP} \\
\mathrm{ForSureRuns}  & \mathrm{HR}
\end{cases}
$$
where $\small\mathbf{b}$ is a vector containing the weights for the corresponding 
events in the following brackets.

##### xBaseRuns: The xERA-style Implementation
This version uses the exact same form as the basic implementation but with real
ball-in-play outcomes replaced by their expected number as determined using an
EV,LA,Sprint Speed based xgboost classifier.
$$
\mathrm{xBaseRuns} \gets  \begin{cases}
\mathrm{Baserunners}   & \mathrm{xS+xD+xT+BB-xHR} \\
\mathrm{Advancement}   & \mathbf{b}\cdot\mathrm{[xS,xD,xT,xHR,BB,xSF,xGIDP,SO,xBIPOut,xSB,xCS]} \\
\mathrm{Outs}          & \mathrm{SO+xBIPOut+xCS+xGIDP} \\
\mathrm{ForSureRuns}   & \mathrm{xHR}
\end{cases}
$$
The pitcher's actual walk and strikeout numbers are still used, but everything else is
replaced with an expected count. The expected number of stolen bases and caught
stealings allowed are from Statcast's pitcher attributed "plus" and "minus" basepath 
advancements, which I believe also include balks and pickoffs rather than just base 
stealing, but I hope this is only a minor error.

##### piBaseRuns & stBaseRuns: The Pitch Modelling Implementation
This version extends the xBaseRuns method to using expected outcomes for all
constituents, no real outcomes are included. The expected numbers of each outcome are
based off a Pitching/Stuff+ model which assigns the probability of every possible 
outcome to each pitch, you can read more about it 
[here](https://www.baseballprospectus.com/news/article/89245/stuffpro-pitchpro-introduction-new-pitch-metrics-bp/)

$$
{pi\mathrm{BaseRuns}\atop st\mathrm{BaseRuns}} \gets  \begin{cases}
\mathrm{Baserunners}   & \mathrm{xS+xD+xT+xBB-xHR} \\
\mathrm{Advancement}   & \mathbf{b}\cdot\mathrm{[xS,xD,xT,xHR,xBB,xSF,xGIDP,xSO,xBIPOut,xSB,xCS]} \\
\mathrm{Outs}          & \mathrm{xSO+xBIPOut+xCS+xGIDP} \\
\mathrm{ForSureRuns}   & \mathrm{xHR}
\end{cases}
$$
The expected number of walks and strikeouts are found from the pitcher's expected 
strikes and balls, which are fed to an absorbing Markov chain model to predict 
strikeouts and walks.

##### Why Am I Using This
The purpose of these WARs is to best describe the pitcher's runs allowed while 
progressively limiting the amount of information given to the models. I've found that
using BaseRuns as the run estimator performs better at describing runs allowed than 
linear methods or even xERA across the board.
This graph shows the correlation of each run estimate to the pitcher's runs allowed 
per 9 over 4 seasons. 
The $\small x$-axis represents years into the future, so $\small x=0$ represents the
correlation between the estimate and the RA9 it is trying to estimate, aka its 
descriptiveness.  Greater values on the $\small x$-axis demonstrate the method's
predictive quality of future runs allowed. xERA and xBaseRuns are highlighted, 
demonstrating that xBaseRuns is similar to xERA in terms of both descriptiveness
and predictiveness (of RA9/ERA respectively), with a slight advantage to xBaseRuns.

''')

x_cols = ['RA9','BsR9','xBsR9','dipsBsR9','piBsR9','stBsR9']
name_dict = {
    'RA9': 'RA9',
    'BsR9': 'BaseRuns9',
    'xBsR9': 'xBaseRuns9',
    'dipsBsR9': 'DIPS BaseRuns9',
    'piBsR9': 'Pitch+ BaseRuns9',
    'stBsR9': 'Stuff+ BaseRuns9',
    'xERA': 'xERA',
}
corrs = future_corr(wars, x_cols, 'RA9', 'PA', cutoff=12)
xera = future_corr(wars, ['xERA'], 'ERA', 'PA', cutoff=12)
corrs = np.c_[corrs,xera]

f = go.Figure()
alphas = [0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 1.0]
for i,stat in enumerate(x_cols+['xERA']):
    f.add_trace(
        go.Scatter(
            x=np.arange(5),
            y=corrs[:,i],
            mode='lines',
            name=name_dict[stat],
            opacity=alphas[i]
        )
    )

f.update_layout(
    xaxis ={'tick0': 0, 'dtick':    1, 'range': [0,3]},
    yaxis ={'tick0': 0, 'dtick': 0.25, 'range': [0,1]},
    title='Correlation to RA9'
)
config = {'width': 100}
baseruns_exp.plotly_chart(f,use_container_width=False,config=config)
baseruns_exp.markdown(r'''
When looking at reliability (self-correlation into the future) xBaseRuns comes out 
slightly above xERA as well.  
''')

corrs = np.zeros((5,7))
for i,x in enumerate(x_cols+['xERA']):
    corrs[:,i] = future_corr(wars, [x], x, 'IP', cutoff=15).squeeze()

lss    = ['dotted','dotted','-','-','-.','dashed','dotted','-']
alphas = [0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 1.]
f = go.Figure()
for i,stat in enumerate(x_cols+['xERA']):
    f.add_trace(
        go.Scatter(
            x=np.arange(4),
            y=corrs[:,i],
            mode='lines',
            name=name_dict[stat],
            opacity=alphas[i]
        )
    )

f.update_layout(xaxis ={'tick0': 0, 'dtick':    1, 'range': [0,3]},
                yaxis ={'tick0': 0, 'dtick': 0.25, 'range': [0,1]},
                title = 'Correlation to Self')
config = {'width': 100}
baseruns_exp.plotly_chart(f,use_container_width=False,config=config)

war_exp = st.expander(
    "Details of the WAR calculation &mdash; if you know how rWAR works you can skip this."
)
war_exp.markdown(r'''
##### If you already know how rWAR works this isn't for you

For the sake of this exercise I think it's good to have a consistent WAR calculation
methodology, and to that end I've opted to base all of the following WARs of the rWAR 
methodology. 

##### 1. Runs Above Average

The rWAR calculation begins with finding the difference between the runs the pitcher
actually allowed and the number of runs the average pitcher would be expected to allow 
had he pitched in the same situations as the real pitcher in question &mdash; this is
his **Runs Above Average**, or **RAA**.
$$
\mathrm{RAA} = \underbrace{\mathrm{PF_{pit}}\times\mathrm{\left(xRA-R_{def}+RP_{adj}\right)+ExIn_{adj}}}_{\text{Avg. Pitcher Runs Allowed}}-\mathrm{RA+lg_{adj}}
$$

This involves first finding the typical performance of all other pitchers against the 
opponents this pitcher faced, $\small\mathrm{xRA}$, removing the contribution of his
team's defence, $\small\mathrm{R_{def}}$. This is then further corrected for the fact 
that guys perform better as relief pitchers than starters, $\small\mathrm{RP_{adj}}$,
and finally a correction for the run expectancy of the extra-inning automatic runner, 
$\small\mathrm{ExIn_{adj}}$. The defensive runs are a combination of the team's 
defensive runs saved and their positioning runs, as determined by Baseball Info
Solutions, prorated to the percent of the team's balls in play which were due to this
pitcher,
$\small\mathrm{R_{def} = \frac{BIP_{pit}}{BIP_{Tm}}\left(DRS_{Tm}+PosR_{Tm}\right)}$. 
Runs which are attributable to the parks in which the pitcher found himself are then
corrected for by scaling this result by the pitcher's personalized park factor 
$\small\mathrm{PF_{pit}}$. The extra innings adjustment appears not park-factor-adjusted 
because it's provided already park-adjusted by Baseball Reference. Altogether, this 
constitutes the amount of runs this pitcher would be expected to allow if he were
completely average. The real pitcher's RAA is then found by subtracting his actual 
runs allowed from these expected runs, and then adding in a final post-facto correction
term to make the league's overall runs above average truly $0$.
Modifications to the calculation of this RAA term will constitute 100% of the changes 
I'll make for any of the other versions of WAR.

##### 2. Wins Above Average

With the pitcher's RAA, we can then find his ***Wins*** **Above Average**, or **WAA**. 
To do this, we will use PythagenPat, a [quite accurate winning percentage estimator](
https://walksaber.blogspot.com/2009/01/runs-per-win-from-pythagenpat.html), 
to find the expected winning percentage of this pitcher if he played on an otherwise 
completely average team. WAA is extracted from this by comparing this percentage to an 
average team's winning percentage, 0.500, and then multiplying by the number of games 
in which this pitcher appeared,
$$
\mathrm{WAA=G\times\left[\left(1+\frac{RA}{RS}^{\left(RS/G+RA/G\right)^{0.285}}\right)^{-1}
-\frac 1 2\right]+lg_{adj}}
$$
where a pitcher's run support, or $\small\mathrm{RS/G}$, is simply their league's
average runs scored per game, and their $\small{\mathrm{RA/G}}$ is determined by
$$
\mathrm{\frac{RA}{G} = \frac{R_{lg}}{G_{lg}} - \frac{RAA}{G}}
$$
This is also corrected with a new $\small\mathrm{lg_{adj}}$ such that the total number
of wins above average sum to $0$.

##### 3. Wins Above Replacement

To find a player's wins above replacement we just need to find the number of wins below 
average a replacement player would be if he were in this pitcher's stead, then add 
those to the real pitcher's wins above average. The runs allowed per out performance of 
a replacement level pitcher is something which is up to personal taste &mdash; where 
you set the replacement level and how you split the total available WAR between 
pitchers and position players. I don't bother with all that, I just use the replacement 
level runs per out as determined by Baseball Reference :angel:
$$
\mathrm{WAR = WAA_{pit} - WAA_{rep} + LI_{adj}}
$$
A final term, $\small\mathrm{LI_{adj}}$, is added to 
[correct for reliever chaining](
http://tangotiger.com/index.php/site/comments/reliever-chaining-and-how-we-view-leverage
). 

''')
framing_exp = st.expander("Explanation of the framing correction model")
framing_exp.markdown(r'''
#### Catcher Framing
Because pitchers also have an effect on catcher framing I opted to go for an approach 
which can control for that and give the pitcher the credit or discredit he deserves for
that effect. For that, I opted to use a framing model in the style of Baseball 
Prospectus's 
[Called Strikes Above Average](https://www.baseballprospectus.com/news/article/25514/moving-beyond-wowy-a-mixed-approach-to-measuring-catcher-framing/).
This is a two stage approach: the first stage captures the probability of a taken pitch
to be called a strike based on its location, movement, the batter's strike zone as
determined by Statcast, and a control for whether the batter belonged to the home team
or not, while the second stage augments these probabilities by incorporating the 
effects of the handedness of the batter and pitcher, the count, and most importantly 
the identity of the pitcher and batter.
A two stage approach is used because the types of models which are capable of
incorporating the spatial information of pitch's location relative to the batter's
personal strike zone are not conducive to also identifying the individual effects of 
every pitcher & catcher, especially when accounting for the structure of the problem.
For the first stage I used a CatBoost model, as I did with my Pitching/Stuff+ models, 
and for the second stage I used a generalized linear mixed effects model which I fit by
maximizing the Laplace approximation of the likelihood.
I didn't feel like getting the home plate umpire's identity for each game so I don't
control for that factor, unlike Baseball Prospectus.   

In my cursory review, the results are very much in-line with Prospectus and Statcast 
framing numbers, though usually less extreme than either of them or between the two,
hardly ever greater. It thinks Austin Hedges is the best framer of the past few years.
Sounds good.
''')

st.markdown('''#### Comparison of the WARs''')

corr_matrix = st.expander("Correlation matrix between each of the WARs.")
corr = wars.select(*[cl(k).alias(v) for k,v in war_convert.items()]).to_pandas().corr()
mask = np.triu(np.ones_like(corr,dtype=bool),k=1)
corr = corr.mask(mask)
corr = np.round(corr,2)
f = px.imshow(corr,text_auto=True)
f.update_layout(title_text="Correlation Matrix",
                title_x=0.5)
corr_matrix.plotly_chart(f)


resp_exp = st.expander("More details for what corrections are applied to each WAR.")
resp_exp.markdown('''
###### The Pitcher's Responsibility

✔️  indicates the pitcher is responsible for this thing, :x: indicates he is not, aka a
correction for it has been applied. :warning: is intermediate in some way.

| WAR Name           | Actual Runs Allowed  | Team Defence/Positioning | Catcher Framing | Ball-in-Play Outcomes                   | Pitch Location | Pitch Shape | 
|--------------------|----------------------|--------------------------|-----------------|----------------------------------------------|----------------|-------------|
| Runs Allowed       | ✔️                    | ✔️                        | ✔️               | ✔️                                            | ✔️              | ✔️           |
| Baseball Reference | ✔️                    | :x:                      | ✔️               | ✔️                                            | ✔️              | ✔️           | 
| OAA Defence        | ✔️                    | :x:                      | :x:             | ✔️                                            | ✔️              | ✔️           | 
| BaseRuns           | :x:                  | :x:                      | :x:             | ✔️                                            | ✔️              | ✔️           | 
| xBaseRuns          | :x:                  | :x:                      | :x:             | :warning: partial credit, quality of contact | ✔️              | ✔️           | 
| DIPS BaseRuns      | :x:                  | :x:                      | :x:             | :warning: home runs, infield flys            | ✔️              | ✔️           | 
| Pitching+          | :x:                  | :x:                      | :x:             | :x:                                          | ✔️              | ✔️           | 
| Stuff+             | :x:                  | :x:                      | :x:             | :x:                                          | :x:            | ✔️           | 

''')

control_exp = st.expander("More details for what corrections are applied to each WAR.")
control_exp.write('''
##### Controls

✔️  indicates I'm controlling for this, :x: indicates I'm not.

| WAR Name           | Real Runs Allowed?              | Opponent Quality | Team Defence              | Team Positioning              | Catcher Framing | Park Factors | Starter/Reliever | Extra-Innings Auto Runners |
|--------------------|---------------------------------|------------------|---------------------------|-------------------------------|-----------------|--------------|------------------|----------------------------|
| Runs Allowed       | ✔️                               | ✔️                | :x:                       | :x:                           | :x:             | :x:          | ✔️                | ✔️                          |
| Baseball Reference | ✔️                               | ✔️                | ✔️  DRS (prorated)         | ✔️  BIS Positioning (prorated) | :x:             | ✔️            | ✔️                | ✔️                          |
| OAA Defence        | ✔️                               | ✔️                | ✔️  OAA while on the mound | ✔️  BIS Positioning (prorated) | ✔️  My model     | ✔️            | ✔️                | ✔️                          |
| BaseRuns           | :x: BaseRuns Estimate           | ✔️                | ✔️  OAA while on the mound | ✔️  BIS Positioning (prorated) | ✔️  My model     | ✔️            | ✔️                | :x:                        |
| xBaseRuns          | :x: xBaseRuns Estimate          | ✔️                | :x:                       | :x:                           | ✔️  My model     | ✔️            | ✔️                | :x:                        |
| DIPS BaseRuns      | :x: DIPS BaseRuns Estimate    | ✔️                | :x:                       | :x:                           | ✔️  My model     | ✔️            | ✔️                | :x:                        |
| Pitching+          | :x: Pitching+ BaseRuns Estimate | :x:              | :x:                       | :x:                           | :x:             | :x:          | ✔️                | :x:                        |
| Stuff+             | :x: Stuff+ BaseRuns Estimate    | :x:              | :x:                       | :x:                           | :x:             | :x:          | ✔️                | :x:                        |
''')

