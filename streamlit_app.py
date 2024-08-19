import streamlit as st, numpy as np, matplotlib.pyplot as plt, pandas as pd, plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode, JsCode


wars = pd.read_pickle('final_wars.pickle')

wars.rename(columns={'name_common': 'Name', 'age': 'Age', 'year_ID':'Year', 'team_ID': 'Team',
                     'ra_war': 'Runs Allowed', 'r_war': 'Baseball Reference',
                     'oaa_war': 'OAA', 'bsr_war': 'BaseRuns', 'xbsr_war': 'xBaseRuns', 'fip_war': 'FIP',
                     'pitch_war': 'Pitching+', 'stuff_war': 'Stuff+'}, inplace=True)

war_names = ['Runs Allowed','Baseball Reference',
             'OAA','BaseRuns','xBaseRuns','FIP',
             'Pitching+','Stuff+']

disp_wars = wars.loc[:,['Name','Year','Age','Team']+war_names]
disp_wars.sort_values('xBaseRuns',ascending=False,ignore_index=True,inplace=True)

st.set_page_config(layout="wide")

st.markdown('''

# Pitcher WAR Spectrum

##### Filling in the gaps between rWAR and fWAR 
If you, like me, are a fan of having different WAR perspectives but wished that there were instead 
something like, oh idk, 8 different perspectives to pick and choose from for agenda crafting purposes, 
then boy do I got the leaderboard for you.

Here are those 8 WARs of your dreams &mdash; 6½ of which are new, each differing in what it claims is the 
responsibility of the pitcher, starting by forcing him to reckon with each and every run he 
allowed in their entirety, and then proceeding to strip away responsibility step by step to the 
point that the model doesn't even know about what happened on any of the pitches he threw, or 
where they were located for that matter. 

There's a sortable & filterable leaderboard that has all pitcher seasons from 2021-2024,
if you select the rows then a line plot of their WARs will appear below the table :smile:
There are also a few dropdowns where I explain each of the WAR calculations and justify my decisions
for them all, including an explanation for why I'm using BaseRuns rather than something more familiar like
(x)wOBA. After that there are some tables which show the correlations between all of these WARs and 
explain the differences between them all :cherry_blossom:

''')


css={'.ag-header-group-cell-label.ag-sticky-label': {'flex-direction': 'column', 'margin': 'auto',
                                                     'font-size': '12pt'}}

columnDefs = [{'field': "Name", 'minWidth': 120, 'filter': True, 'sortable': False, 'pinned': 'left'},
              {'field': "Year", 'minWidth':  70, 'filter': True, 'sortable': False,},
              {'field': "Age",  'minWidth':  70, 'filter': True, 'sortable': True,  'suppressHeaderFilterButton': False},
              {'field': "Team", 'minWidth':  70, 'filter': True, 'sortable': False, 'suppressHeaderFilterButton': False},
              {'headerName': "Runs Allowed",
               'headerTooltip': "Pitcher's runs allowed are used",
               'children': [{'field': 'Runs Allowed',
                             'minWidth': 130,
                             'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1,
                             'headerTooltip': "Pitcher is responsible for all runs allowed",
                             'tooltipValueGetter': JsCode("""function(){return "Pitcher is responsible for all runs allowed"}""")},
                            {'field': 'Baseball Reference',
                             'minWidth': 150,
                             'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1,
                             'headerTooltip': "Corrected for team's defence by DRS",
                             'tooltipValueGetter': JsCode("""function(){return "Corrected for team's defence"}""")},
                            {'field': 'OAA',
                             'minWidth': 80,
                             'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1,
                             'headerTooltip': "Corrected using team OAA when pitcher is on the mound",
                             'tooltipValueGetter': JsCode("""function(){return "Corrected using team OAA when pitcher is on the mound"}""")},
                            ]},
              {'headerName': "Runs Allowed Estimators",
               'headerTooltip': "A model which estimates a pitcher's runs allowed is used",
               'children': [{'field': 'BaseRuns',
                             'minWidth': 110,
                             'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1,
                             'headerTooltip': "Like OAA-WAR, but with the pitcher's BaseRuns run estimate",
                             'tooltipValueGetter': JsCode("""function(){return "Like OAA-WAR but with the pitcher's BaseRuns run estimate"}""")},
                            {'field': 'xBaseRuns',
                             'minWidth': 110,
                             'headerName': 'xBaseRuns',
                             'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1,
                             'headerTooltip': "Like BaseRuns-WAR, but the xERA-style xBaseRuns is used",
                             'tooltipValueGetter': JsCode("""function(){return "Like BaseRuns-WAR, but the xERA-style xBaseRuns is used"}""")},
                            {'field': 'FIP',
                             'minWidth': 80,
                             'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1,
                             'headerTooltip': "rWAR-style but with FIP",
                             'tooltipValueGetter': JsCode("""function(){return "rWAR-style but with FIP"}""")},
                            ]},
              {'headerName': "Pitch Modelling",
               'headerTooltip': "A model which estimates a pitcher's runs allowed is used",
               'children': [{'field': 'Pitching+',
                             'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1,
                             'headerTooltip': "Like BaseRuns-WAR, but uses Pitching+ style model for xBaseRuns",
                             'tooltipValueGetter': JsCode("""function(){return "Like BaseRuns-WAR, but uses Pitching+ style model for xBaseRuns"}""")},
                            {'field': 'Stuff+',
                             'type' : ['numericColumn', 'customNumericFormat'], 'precision': 1,
                             'headerTooltip': "Like BaseRuns-WAR, but uses Stuff+ style model for xBaseRuns",
                             'tooltipValueGetter': JsCode("""function(){return "Like BaseRuns-WAR, but uses Stuff+ style model for xBaseRuns"}""")},
                            ]},
               ]

gridOptions =  {'defaultColDef': {'flex': 1, 'minWidth': 120, 'filterable': True,
								  'groupable': False, 'editable': False, 
                                  'wrapText': True, 'autoHeight': True, 
                                  'suppressMovable': True,
                                  'suppressMenu': False},
				'columnDefs': columnDefs,
                'initialState': {'rowSelection': [0,1]},
				'tooltipShowDelay': 800, 
                'tooltipMouseTrack': True,
                'rowSelection': 'multiple', 
                'rowMultiSelectWithClick': False, 
                'suppressRowDeselection': False, 
                'suppressRowClickSelection': False, 
                'groupSelectsChildren': False, 
                'groupSelectsFiltered': True}

st.markdown('''#### WAR Leaderboard
You can filter columns on mobile by holding down the column header, or on desktop by clicking the menu button when 
you hover over it :blush: This lets you, for example, limit the table to only select pitchers.
''')

left_col,right_col = st.columns(2)
with left_col.expander('Included Years') :
    years_select = st.multiselect("Included years", [2021,2022,2023,2024], [2021,2022,2023,2024])
with right_col.expander('Included Teams') :
    teams_select = st.multiselect("Included Teams", disp_wars.Team.unique(), disp_wars.Team.unique(),
                                    label_visibility='collapsed')

year_filt = disp_wars.Year.isin(years_select)
team_filt = disp_wars.Team.isin(teams_select)

return_value = AgGrid(disp_wars[year_filt & team_filt], 
       gridOptions=gridOptions,
       update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
       allow_unsafe_jscode=True,
       fit_columns_on_grid_load=True,
       height=700,
       theme="streamlit",
       key=None,
       custom_css=css)

st.markdown('''#### Selected Players WAR''')
if return_value.selected_rows is None:
    st.write('''Select rows in the table to see a line plot of their WARs''')
else:
    f = go.Figure()
    min_war = return_value.selected_rows[war_names].min().min()
    max_war = return_value.selected_rows[war_names].max().max()
    for ind,row in return_value.selected_rows.iterrows():
        label = row['Name'] + ' ' + str(row['Year'])
        f.add_trace(go.Scatter(x=np.arange(8),y=row[war_names].values,
                               mode='lines',name=label))
    f.update_layout(xaxis = {'tickmode': 'array', 
                             'tickvals': np.arange(8), 
                             'ticktext': war_names},
                    yaxis_range = [min(0,min_war),max_war+0.1])
    st.plotly_chart(f,use_container_width=False,width=100)


st.markdown('''#### Calculation Details''')
raa_exp = st.expander('RAA Calculations for Each WAR')
with raa_exp:
    ra_tab, rw_tab, oa_tab, bs_tab, xb_tab, fi_tab, pi_tab, st_tab = st.tabs(war_names)
    with ra_tab:
        st.markdown(r'''
#### Runs Allowed RAA
No corrections for park factors or defence. Corrections are still applied for relief
pitchers, quality of opponents, and for extra-inning automatic runners.
$$
\mathrm{RAA_{runs} = \left(xRA + RP_{adj} + ExIn_{adj}\right) \newline
        - RA + lg_{adj}}
$$
- $\small\mathrm{xRA}$ is how other pitchers performed against this pitcher's opponents on average
- $\small\mathrm{RP_{adj}}$ is an adjustment for how guys perform better as relief pitchers than starters
- $\small\mathrm{ExIn_{adj}}$ is an adjustment for the automatic runners in extra innings
- $\small\mathrm{lg_{adj}}$ is an adjustment to ensure the league's total RAA is $0$
''',unsafe_allow_html=True)

    with rw_tab:
        st.markdown(r'''
#### Baseball Reference RAA
I explain this one in more detail over in the 'Details of the WAR Calculation' dropdown, and obviously
there's the [Baseball Reference pitcher WAR explainer](https://www.baseball-reference.com/about/war_explained_pitch.shtml).  

Corrections are applied for the parks the pitcher pitched in, the quality
of his opponents, his team's defensive runs saved and their positioning,
the starter/reliever discrepancy, and for extra-inning automatic runners.
$$
\mathrm{RAA_{rally} = PF_{pit}\times\left(xRA-R_{def}+RP_{adj}\right)+ExIn_{adj} - RA + lg_{adj}}
$$
- $\small\mathrm{PF_{pit}}$ is the cumulative park factor for the parks in which this pitcher pitched
- $\small\mathrm{xRA}$ is how other pitchers performed against this pitcher's opponents on average
- $\small\mathrm{R_{def} = \frac{BIP_{pit}}{BIP_{Tm}}\left(DRS_{Tm}+PosR_{Tm}\right)}$ is the 
correction for the team's defence & positioning
- $\small\mathrm{RP_{adj}}$ is an adjustment for how guys perform better as relief pitchers than starters
- $\small\mathrm{ExIn_{adj}}$ is an adjustment for the automatic runners in extra innings
- $\small\mathrm{lg_{adj}}$ is an adjustment to ensure the league's total RAA is $0$
''')

    with oa_tab:
        st.markdown(r'''
#### Baseball Reference but with OAA Defence RAA
This is the same as Baseball Reference WAR but with the team DRS defensive correction replaced with
the team's actual Fielding Run Value (FRV) while the pitcher is on the mound. The team positioning 
correction remains, though.

Same as rWAR: corrections are applied for the parks the pitcher pitched in, the quality
of his opponents, his team's defensive runs saved and their positioning,
the starter/reliever discrepancy, and for extra-inning automatic runners.
$$
\mathrm{RAA_{oaa} = PF_{pit}\times\left(xRA-PosR_{Tm}\frac{BIP_{pit}}{BIP_{Tm}}+RP_{adj}\right)-FRV_{pit}-Framing+ExIn_{adj} - RA + lg_{adj}}
$$
- $\small\mathrm{PF_{pit}}$ is the cumulative park factor for the parks in which this pitcher pitched
- $\small\mathrm{xRA}$ is how other pitchers performed against this pitcher's opponents on average
- $\small\mathrm{PosR_{def} = \frac{BIP_{pit}}{BIP_{Tm}}PosR_{Tm}}$ is the correction for the team's positioning
- $\small\mathrm{RP_{adj}}$ is an adjustment for how guys perform better as relief pitchers than starters
- $\small\mathrm{FRV}$ is the team's fielding run value (OAA) while the pitcher is actually on the mound,
  pre-park factored by Statcast (I think)
- $\small\mathrm{Framing}$ is an adjustment for the framing attributable to this pitcher's catchers while he was pitching.
- $\small\mathrm{ExIn_{adj}}$ is an adjustment for the automatic runners in extra innings
- $\small\mathrm{RA}$ is the pitcher's runs allowed
- $\small\mathrm{lg_{adj}}$ is an adjustment to ensure the league's total RAA is $0$
''')

    with bs_tab:
        st.markdown(r'''
#### BaseRuns RAA
This is the same as OAA WAR but with the pitcher's actual runs allowed replaced by the BaseRuns
estimate for this runs allowed. 

I explain the BaseRuns estimator over [here](#baseruns), but the purpose of using it here is to 
control for the pitcher's sequencing luck. 

Same as OAA-WAR: corrections are applied for the parks the pitcher pitched in, the quality
of his opponents, his team's defensive runs saved and their positioning,
the starter/reliever discrepancy; however, since real runs allowed are not used, 
a correction for the extra-innings automatic runners isn't needed.
$$
\mathrm{RAA_{bsr} = PF_{pit}\times\left(xRA-PosR_{Tm}\frac{BIP_{pit}}{BIP_{Tm}}+RP_{adj}\right)-FRV_{pit}-Framing-BsR+lg_{adj}}
$$
- $\small\mathrm{PF_{pit}}$ is the cumulative park factor for the parks in which this pitcher pitched
- $\small\mathrm{xRA}$ is how other pitchers performed against this pitcher's opponents on average
- $\small\mathrm{PosR_{def} = \frac{BIP_{pit}}{BIP_{Tm}}PosR_{Tm}}$ is the correction for the team's positioning
- $\small\mathrm{RP_{adj}}$ is an adjustment for how guys perform better as relief pitchers than starters
- $\small\mathrm{FRV}$ is the team's fielding run value (OAA) while the pitcher is actually on the mound, 
  pre-park factored by Statcast (I think)
- $\small\mathrm{Framing}$ is an adjustment for the framing attributable to this pitcher's catchers while he was pitching.
- $\small\mathrm{BsR}$ is the pitcher's BaseRuns run estimate
- $\small\mathrm{lg_{adj}}$ is an adjustment to ensure the league's total RAA is $0$
''')

    with xb_tab:
        st.markdown(r'''
#### xBaseRuns RAA
I explain the xBaseRuns estimator over [here](#xbaseruns), but the purpose of using it here is to 
control for the pitcher's sequencing luck **and** luck on batted balls.

A park factor correction is still applied, as is the starter/reliever discrepancy correction,
but no defensive corrections need to be applied. Real walks and strikeouts are used in xBaseRuns,
so framing runs are still corrected.
$$
\mathrm{RAA_{xbsr} = PF_{pit}\times\left(xRA+RP_{adj}\right)-Framing-xBsR+lg_{adj}}
$$
- $\small\mathrm{PF_{pit}}$ is the cumulative park factor for the parks in which this pitcher pitched
- $\small\mathrm{xRA}$ is how other pitchers performed against this pitcher's opponents on average
- $\small\mathrm{RP_{adj}}$ is an adjustment for how guys perform better as relief pitchers than starters
- $\small\mathrm{Framing}$ is an adjustment for the framing attributable to this pitcher's catchers while he was pitching.
- $\small\mathrm{xBsR}$ is the pitcher's xBaseRuns run estimate
- $\small\mathrm{lg_{adj}}$ is an adjustment to ensure the league's total RAA is $0$
''')

    with fi_tab:
        st.markdown(r'''
#### FIP RAA
This is not fWAR! It's rWAR with FIP!!

This uses the same infield-fly FIP (ifFIP) that Fangraphs uses in their fWAR calculation,
but how the RAA calculation is done is completely different, and while they also include a 
framing correction, my framing model is not the same style as theirs. Further on in the 
WAR calculating process they perform the runs-to-wins conversion differently and allocate
a different amount of the total available WAR to the pitchers than does this.

$$
\mathrm{RAA_{fip} = PF_{pit}\times\left(xRA+RP_{adj}\right)-Framing-ifFIPR+lg_{adj}}
$$
- $\small\mathrm{PF_{pit}}$ is the cumulative park factor for the parks in which this pitcher pitched
- $\small\mathrm{xRA}$ is how other pitchers performed against this pitcher's opponents on average
- $\small\mathrm{RP_{adj}}$ is an adjustment for how guys perform better as relief pitchers than starters
- $\small\mathrm{Framing}$ is an adjustment for the framing attributable to this pitcher's catchers while he was pitching.
- $\small\mathrm{ifFIPR}$ is the pitcher's [infield-fly FIP runs allowed estimate](https://library.fangraphs.com/war/calculating-war-pitchers/)
- $\small\mathrm{lg_{adj}}$ is an adjustment to ensure the league's total RAA is $0$
''')

    with pi_tab:
        st.markdown(r'''
#### Pitching+ RAA
This uses a Pitching+ style model to estimate the probability of each of the possible pitch
outcomes for every pitch. These estimated probabilities are used in the xBaseRuns formula
to get an even more expected BaseRuns. This is different from normal Pitching/Stuff+ models
which simply add the run values of each pitch, but I found that using a BaseRuns approach
could better describe same-season runs allowed.

$$
\mathrm{RAA_{fip} = \left(\frac{RA_{lg}}{TBF_{lg}}TBF+RP_{adj}\right)-piBsR+lg_{adj}}
$$
- $\small\mathrm{\frac{RA_{lg}}{TBF_{lg}}TBF}$ is the expected number of runs allowed in this many batters faced
- $\small\mathrm{RP_{adj}}$ is an adjustment for how guys perform better as relief pitchers than starters
- $\small\mathrm{piBsR}$ is the Pitching+ BaseRuns estimate
- $\small\mathrm{lg_{adj}}$ is an adjustment to ensure the league's total RAA is $0$

I based this Pitching+ model off the [Steve Brown](https://x.com/srbrown70)/Baseball Prospectus 
[PitchPro](https://www.baseballprospectus.com/news/article/89245/stuffpro-pitchpro-introduction-new-pitch-metrics-bp/) 
model, with slightly modified inputs, primary/secondary pitch classification, and the number of sub-models. 
I did not seriously test any of the changes I made, which were all out of laziness I assure you, 
so trust these pitch-modelling results at your own peril.  

Inputs to the pitching model were:
- Year
- Release position
- Spin axis difference
- Spin efficiency
- Pitcher height
- The log of extension (shoutout to [TJStats](https://medium.com/@thomasjamesnestico/modelling-tjstuff-v2-0-f9ee772b4266)])
- The count
- Batter's handedness
- Observed pitch movement
- Initial velocity & acceleration vectors 
- Pitch location 

Pitches were classified into 3 types:
- Fast: Four Seam Fastballs, Sinkers, Hard Cutters
- Slow: Changeups, Splitters, Screwballs
- Bend: Curveballs, Sliders, Sweepers, Knuckle Curves, Slurves, Slow Cutters, Knuckleballs, Ephuses, Forkballs

Pitches were further divided into primary or secondary, based on which pitch the pitcher threw the most in each
season. 5 models were then made for each classification task (swing/take, whiff/foul/bip if swing, etc.), one for:
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
\mathrm{RAA_{fip} = \left(\frac{RA_{lg}}{TBF_{lg}}TBF+RP_{adj}\right)-stBsR+lg_{adj}}
$$
- $\small\mathrm{\frac{RA_{lg}}{TBF_{lg}}TBF}$ is the expected number of runs allowed in this many batters faced
- $\small\mathrm{RP_{adj}}$ is an adjustment for how guys perform better as relief pitchers than starters
- $\small\mathrm{stBsR}$ is the Stuff+ BaseRuns estimate
- $\small\mathrm{lg_{adj}}$ is an adjustment to ensure the league's total RAA is $0$

The Stuff model is effectively the same as the Pitching model, with any information which betray
the location of the pitch removed, namely:
- The initial velocity & acceleration vectors are replaced with their respective magnitudes
- The pitch location is removed

Other than this, the model is exactly the same as the Pitching model.
''')

baseruns_exp = st.expander('BaseRuns Run Estimator Explanation & Justification')
baseruns_exp.markdown(r'''
#### BaseRuns Run Estimator
The BaseRuns run estimator is a nonlinear run estimator which I have opted to use for estimating pitching
runs allowed for two reasons: first because it allows a consistent method of estimating runs across all 
levels of this exercise (in fact its form fits naturally fits with how modern classification-based methods, 
like xwOBA and pitch/stuff models, work), and second because I found it to be more accurate (in the 
descriptiveness sense) than linear methods across the board. You can read up more about it 
[here](https://www.tangotiger.net/rc2.html) or [here](https://gosu02.tripod.com/id9.html). 
This approach captures the fact that a double by a pitcher who allows a high OBP is worse than a double
for a pitcher who does not allow many baserunners. xERA approximates this behavior by using the square
of a pitcher's xwOBA to predict his ERA, but I figured if I'm doing all this work anyways, why not use the
full thing.

##### What it Looks Like & How it Works
Its form is simple,
$$
\mathrm{BaseRuns = Baserunners\times ScoreRate + ForSureRuns}
$$
where $\small\mathrm{Baserunners}$ is the number of baserunners allowed,
$\small\mathrm{ScoreRate}$ is the rate at which baserunners score, and
$\small\mathrm{ForSureRuns}$ effectively just means Home Runs. I've seen some people
include Sac Flys, but I don't like that idea very much so I'm sticking with just Home Runs. 
This equation as it is written isn't a run estimate, it's exactly the number
of runs allowed. The estimation comes in because the calculation of $\small\mathrm{ScoreRate}$ 
from only available statistics is not a trivial task. The actual BaseRuns run estimator
uses David Smyth's estimate for the $\small\mathrm{ScoreRate}$, which goes as follows
$$
\mathrm{ScoreRate = \frac{AdvancementFactor}{AdvancementFactor + Outs}}
$$
where the $\small\mathrm{AdvancementFactor}$ is a linear combination of traditional
box score statistics: singles, doubles, triples, homers, walks, stolen bases, etc.
Empirically, this form work really well, so I'll be using it as well.
I opted to fit the weights of this linear combination to individual pitcher seasons from 2021-2024
to minimize the weighted RMSE to their runs allowed. Pre-computed weights exist online, 
but they are either old and possibly outdated, or fit on team seasons rather than pitcher seasons, 
the transferability of which I'm unsure about.
For all of the implementations I've opted to use the following stats in the $\small\mathrm{Advancement}$ 
term:
> Singles, Doubles, Triples, Home Runs, Walks, Sac Flys, GIDPs, Strikeouts, Ball-in-Play Outs, 
Stolen Bases, & Caught Stealings

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
where $\small\mathbf{b}$ is a vector containing the weights for the corresponding events in the following
brackets.

##### xBaseRuns: The xERA-style Implementation
This version uses the exact same form as the basic implementation but with real ball-in-play outcomes
replaced by their expected number as determined using an EV,LA kNN classifier, just like xwOBA, but 
with no sprint speed correction.
$$
\mathrm{xBaseRuns} \gets  \begin{cases}
\mathrm{Baserunners}   & \mathrm{xS+xD+xT+BB-xHR} \\
\mathrm{Advancement}   & \mathbf{b}\cdot\mathrm{[xS,xD,xT,xHR,BB,xSF,xGIDP,SO,xBIPOut,xSB,xCS]} \\
\mathrm{Outs}          & \mathrm{SO+xBIPOut+xCS+xGIDP} \\
\mathrm{ForSureRuns}   & \mathrm{xHR}
\end{cases}
$$
The pitcher's actual walk and strikeout numbers are still used, but everything else is replaced with
an expected count. The expected number of stolen bases and caught stealings allowed are from Statcast's
pitcher attributed "plus" and "minus" basepath advancements, which I believe also include balks and 
pickoffs rather than just base stealing, but I hope this is only a minor error.

##### piBaseRuns & stBaseRuns: The Pitch Modelling Implementation
This version extends the xBaseRuns method to using expected outcomes for all constituents, no real
outcomes are included. The expected numbers of each outcome are based off a Pitching/Stuff+ model
which assigns the probability of every possible outcome to each pitch, you can read more about it 
[here](https://www.baseballprospectus.com/news/article/89245/stuffpro-pitchpro-introduction-new-pitch-metrics-bp/)

$$
{pi\mathrm{BaseRuns}\atop st\mathrm{BaseRuns}} \gets  \begin{cases}
\mathrm{Baserunners}   & \mathrm{xS+xD+xT+xBB-xHR} \\
\mathrm{Advancement}   & \mathbf{b}\cdot\mathrm{[xS,xD,xT,xHR,xBB,xSF,xGIDP,xSO,xBIPOut,xSB,xCS]} \\
\mathrm{Outs}          & \mathrm{xSO+xBIPOut+xCS+xGIDP} \\
\mathrm{ForSureRuns}   & \mathrm{xHR}
\end{cases}
$$
The expected number of walks and strikeouts are found from the pitcher's expected strikes and balls,
which are fed to a decision tree model to predict strikeouts and walks, along with the pitcher's 
innings pitched per game as I noticed that starters and relievers exhibited different behavior.

##### Why Am I Using This
The purpose of these WARs is to best describe the pitcher's runs allowed while progressively limiting 
the amount of information given to the models. I've found that using BaseRuns as the run estimator
performs better at describing runs allowed than linear methods or even xERA across the board.
This graph shows the correlation of each run estimate to the pitcher's runs allowed per 9 over 4 seasons. 
The $\small x$-axis represents years into the future, so $\small x=0$ represents the correlation between 
the estimate and the RA9 it is trying to estimate, aka its descriptiveness.  Greater values on the
$\small x$-axis demonstrate the method's predictive quality of future runs allowed. xERA and xBaseRuns 
are highlighted, demonstrating that xBaseRuns is capable of better describing current-year production 
than xERA while maintaining the same predictability of future RA9.
''')
xera_comp_df = pd.read_pickle('xera_descriptiveness_df.pickle')
name_dict = {'ra9': 'RA9', 'bsra9': 'BaseRuns9', 
             'xbsra9': 'xBaseRuns9', 'K%':'K%', 'pibsra9': 'Pitching+ BsR9',
             'xERA': 'xERA', 'stbsra9': 'Stuff+ BsR9'}
xera_comp_df.rename(columns=name_dict,inplace=True)
war_columns = xera_comp_df.columns.values[1:].tolist()
xera_comp_df['Years into the Future'] = np.arange(4)
f = go.Figure()
alphas = [0.1, 0.1, 1.0, 1.0, 0.1, 0.1]
for i,stat in enumerate(war_columns):
    f.add_trace(go.Scatter(x=np.arange(4),y=xera_comp_df[stat],
                           mode='lines',name=stat,opacity=alphas[i]))
f.update_layout(xaxis ={'tick0': 0, 'dtick':    1, 'range': [0,3]},
                yaxis ={'tick0': 0, 'dtick': 0.25, 'range': [0,1]},
                title='Correlation to ERA or RA9')

baseruns_exp.plotly_chart(f,use_container_width=False,width=100)
baseruns_exp.markdown(r'''
I observed similar behavior with the pitch modelling approaches, but with more of a trade-off. 
Using pi/stBaseRuns sacrifices some of the reliability and predictiveness of the pitch/stuff
RV models for the sake of better descriptiveness. ''')

pm_descr_df = pd.read_pickle('rv_vs_bsr_descr_df.pickle')
name_dict = {'ra9': 'RA9', 'bsra9': 'BaseRuns9', 
             'xbsra9': 'xBaseRuns9', 'K%':'K%', 'pibsra9': 'Pitching+ BsR9',
             'xERA': 'xERA', 'stbsra9': 'Stuff+ BsR9', 'pirv9': 'Pitching+ RV9',
             'strv9': 'Stuff+ RV9'}
pm_descr_df.rename(columns=name_dict,inplace=True)
war_columns = pm_descr_df.columns.values[1:].tolist()
pm_descr_df['Years into the Future'] = np.arange(4)
lss    = ['dotted','dotted','-','-','-.','dashed','dotted']
alphas = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1]
f = go.Figure()
for i,stat in enumerate(war_columns):
    f.add_trace(go.Scatter(x=np.arange(4),y=pm_descr_df[stat],
                           mode='lines',name=stat,opacity=alphas[i]))
f.update_layout(xaxis ={'tick0': 0, 'dtick':    1, 'range': [0,3]},
                yaxis ={'tick0': 0, 'dtick': 0.25, 'range': [0,1]},
                title = 'Correlation to RA9')
baseruns_exp.plotly_chart(f,use_container_width=False,width=100)

war_exp = st.expander("Details of the WAR calculation &mdash; if you know how rWAR works you can skip this.")
war_exp.markdown(r'''
##### If you already know how rWAR works this isn't for you

For the sake of this exercise I think it's good to have a consistent WAR calculation methodology, and to that 
end I've opted to base all of the following WARs of the rWAR methodology. This includes FIP-WAR, which 
is not fWAR! it's FIP-based rWAR!

##### 1. Runs Above Average

The rWAR calculation begins with finding the difference between the runs the pitcher actually allowed and the number of 
runs the average pitcher would be expected to allow had he pitched in the same situations as the real pitcher 
in question &mdash; this is his **Runs Above Average**, or **RAA**.
$$
\mathrm{RAA} = \underbrace{\mathrm{PF_{pit}}\times\mathrm{\left(xRA-R_{def}+RP_{adj}\right)+ExIn_{adj}}}_{\text{Avg. Pitcher Runs Allowed}}-\mathrm{RA+lg_{adj}}
$$

This involves first finding the typical performance of all other pitchers against the opponents this pitcher faced,
$\small\mathrm{xRA}$, removing the contribution of his team's defence, $\small\mathrm{R_{def}}$. This is then further 
corrected for the fact that guys perform better as relief pitchers than starters, $\small\mathrm{RP_{adj}}$, and finally 
a correction for the run expectancy of the extra-inning automatic runner, $\small\mathrm{ExIn_{adj}}$. The defensive 
runs are a combination of the team's defensive runs saved and their positioning runs, as determined by Baseball Info
Solutions, prorated to the percent of the team's balls in play which were due to this pitcher, 
$\small\mathrm{R_{def} = \frac{BIP_{pit}}{BIP_{Tm}}\left(DRS_{Tm}+PosR_{Tm}\right)}$. Runs which are attributable to 
the parks in which the pitcher found himself are then corrected for by scaling this result by the pitcher's 
personalized park factor $\small\mathrm{PF_{pit}}$. The extra innings adjustment appears not park-factor-adjusted 
because it's provided already park-adjusted by Baseball Reference. Altogether, this constitutes the amount of runs
this pitcher would be expected to allow if he were completely average. The real pitcher's RAA is then found by subtracting 
his actual runs allowed from these expected runs, and then adding in a final post-facto correction term to make the league's 
overall runs above average truly $0$.
Modifications to the calculation of this RAA term will constitute 100% of the changes I'll make for any
of the other versions of WAR.

##### 2. Wins Above Average

With the pitcher's RAA, we can then find his ***Wins*** **Above Average**, or **WAA**. 
To do this, we will use PythagenPat, a [quite accurate winning percentage estimator](https://walksaber.blogspot.com/2009/01/runs-per-win-from-pythagenpat.html), 
to find the expected winning percentage of this pitcher if he played on an otherwise completely average team. WAA is extracted 
from this by comparing this percentage to an average team's winning percentage, 0.500, and then multiplying by the number of 
games in which this pitcher appeared,
$$
\mathrm{WAA=G\times\left[\left(1+\frac{RA}{RS}^{\left(RS/G+RA/G\right)^{0.285}}\right)^{-1}-\frac 1 2\right]+lg_{adj}}
$$
where a pitcher's run support, or $\small\mathrm{RS/G}$, is simply their league's average runs scored per game, and their
$\small{\mathrm{RA/G}}$ is determined by
$$
\mathrm{\frac{RA}{G} = \frac{R_{lg}}{G_{lg}} - \frac{RAA}{G}}
$$
This is also corrected with a new $\small\mathrm{lg_{adj}}$ such that the total number of wins above average
sum to $0$.

##### 3. Wins Above Replacement

To find a player's wins above replacement we just need to find the number of wins below average
a replacement player would be if he were in this pitcher's stead, then add those to the real pitcher's 
wins above average. The runs allowed per out performance of a replacement level pitcher is something which is up
to personal taste &mdash; where you set the replacement level and how you split the total available WAR between 
pitchers and position players. I don't bother with all that, I just use the replacement level runs per out as 
determined by Baseball Reference :angel:
$$
\mathrm{WAR = WAA_{pit} - WAA_{rep} + LI_{adj}}
$$
A final term, $\small\mathrm{LI_{adj}}$, is added to 
[correct for reliever chaining](http://tangotiger.com/index.php/site/comments/reliever-chaining-and-how-we-view-leverage). 

''')
framing_exp = st.expander("Explanation of the framing correction model")
framing_exp.markdown(r'''
#### Catcher Framing
Because pitchers also have an effect on catcher framing I opted to go for an approach which can control for that
and give the pitcher the credit or discredit he deserves for that effect. For that, I opted to use a framing
model in the style of Baseball Prospectus's 
[Called Strikes Above Average](https://www.baseballprospectus.com/news/article/25514/moving-beyond-wowy-a-mixed-approach-to-measuring-catcher-framing/).
This is a two stage approach: the first stage captures the probability of a taken pitch to be called a strike 
based on its location, movement, the batter's strike zone as determined by Statcast, and a control for whether 
the batter belonged to the home team or not, while the second stage augments these probabilities by incorporating
the effects of the handedness of the batter and pitcher, the count, and most importantly the identity of the
pitcher and batter.
A two stage approach is used because the types of models which are capable of incorporating the spatial information
of pitch's location relative to the batter's personal strike zone are not conducive to also identifying the
individual effects of every pitcher & catcher, especially when accounting for the structure of the problem.
For the first stage I used a CatBoost model, as I did with my Pitching/Stuff+ models, and for the second stage
I used a generalized linear mixed effects model which I fit by maximizing the Laplace approximation of the 
likelihood.
I didn't feel like getting the home plate umpire's identity for each game so I don't control
for that factor, unlike Baseball Prospectus.   

In my cursory review, the results are very much in-line with Prospectus and Statcast framing numbers, though 
usually less extreme than either of them or between the two, hardly ever greater. It thinks Austin Hedges 
is the best framer of the past few years. Sounds good.
''')

st.markdown('''#### Comparison of the WARs''')

corr_matrix = st.expander("Correlation matrix between each of the WARs.")
pitcher_years = pd.read_pickle('wars_for_correlation.pickle')
war_columns = ['ra_war', 'r_war', 'oaa_war', 'bsr_war', 'xbsr_war', 'fip_war', 'pitch_war', 'stuff_war']
name_fixer  = {v: war_names[i] for i,v in enumerate(war_columns)}
pitcher_years.rename(columns=name_fixer,inplace=True)
corr = pitcher_years[war_names].corr()
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

✔️  indicates the pitcher is responsible for this thing, :x: indicates he is not, aka a correction for it has been applied. :warning: is intermediate in some way.

| WAR Name           | Actual Runs Allowed  | Team Defence/Positioning | Catcher Framing | Ball-in-Play Outcomes                   | Pitch Location | Pitch Shape | 
|--------------------|----------------------|--------------------------|-----------------|----------------------------------------------|----------------|-------------|
| Runs Allowed       | ✔️                    | ✔️                        | ✔️               | ✔️                                            | ✔️              | ✔️           |
| Baseball Reference | ✔️                    | :x:                      | ✔️               | ✔️                                            | ✔️              | ✔️           | 
| OAA Defence        | ✔️                    | :x:                      | :x:             | ✔️                                            | ✔️              | ✔️           | 
| BaseRuns           | :x:                  | :x:                      | :x:             | ✔️                                            | ✔️              | ✔️           | 
| xBaseRuns          | :x:                  | :x:                      | :x:             | :warning: partial credit, quality of contact | ✔️              | ✔️           | 
| FIP                | :x:                  | :x:                      | :x:             | :warning: home runs, infield flys            | ✔️              | ✔️           | 
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
| FIP                | :x: infield-fly FIP Estimate    | ✔️                | :x:                       | :x:                           | ✔️  My model     | ✔️            | ✔️                | :x:                        |
| Pitching+          | :x: Pitching+ BaseRuns Estimate | :x:              | :x:                       | :x:                           | :x:             | :x:          | ✔️                | :x:                        |
| Stuff+             | :x: Stuff+ BaseRuns Estimate    | :x:              | :x:                       | :x:                           | :x:             | :x:          | ✔️                | :x:                        |
''')

