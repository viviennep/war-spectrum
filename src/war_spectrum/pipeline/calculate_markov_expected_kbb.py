import numpy as np, polars as pl
from war_spectrum.models.xKxBB import markov_pa_outcome_probs
cl = pl.col

# Expression aliases
p_ball = cl('p_ball_pitch').alias('p_ball')
p_called_strike = cl('p_called_strike_pitch').alias('p_called_strike')
p_whiff = cl('p_whiff_pitch').alias('p_whiff')
p_foul = cl('p_foul_pitch').alias('p_foul')
p_hbp = cl('p_hbp_pitch').alias('p_hbp')
p_bip = (
    cl('p_1b_pitch') + cl('p_2b_pitch') + cl('p_3b_pitch') + cl('p_hr_pitch') + 
    cl('p_sf_pitch') + cl('p_gidp_pitch') + cl('p_out_pitch')
).alias('p_bip')

def create_P_from_rows(P_rows):
    n_pitch_outcomes = 6
    P = np.zeros((4,3,n_pitch_outcomes))
    P[*P_rows[:,:2].astype(int).T] = P_rows[:,2:]
    return P

def pitch_outcome_matrix(df,model='stuff'):
    P_rows = (
        df.filter(~cl('balls').eq(4),~cl('strikes').eq(3))
        .group_by('balls','strikes')
        .agg(
            p_ball.mean(), p_called_strike.mean(), p_whiff.mean(), p_foul.mean(), 
            p_hbp.mean(), p_bip.mean()
        )
        .sort('balls','strikes')
    ).to_numpy()
    return create_P_from_rows(P_rows)

def attach_markov_expected_kbb(df,stints,model='stuff',α=50):
    lg_P = pitch_outcome_matrix(df,model=model)
    out_rows = []
    for pitcher_year,split_df in df.group_by('pitcher_year'):
        P = pitch_outcome_matrix(split_df)
        n_pitches = len(split_df)
        P = (n_pitches*P + α*lg_P)/(n_pitches+α)
        p_pa_outcome = markov_pa_outcome_probs(P)
        out_rows.append({
            'pitcher_year': pitcher_year[0],
            f'xK_{model}': p_pa_outcome[0],
            f'xBB_{model}': p_pa_outcome[1],
            f'xHBP_{model}': p_pa_outcome[2],
            f'xBIP_{model}': p_pa_outcome[2],
        })
    xdf = pl.DataFrame(out_rows)
    stints = stints.join(xdf, on='pitcher_year', how='left')
    return stints.with_columns(
        (cl(k)*cl('stint_pa')).alias(k) 
        for k in [f'xK_{model}', f'xBB_{model}', f'xHBP_{model}', f'xBIP_{model}']
    )

