import numpy as np, polars as pl
cl = pl.col

b_terms = {
    'bb'  : 'stint_bb',
    'k'   : 'stint_k',
    '1b'  : 'stint_1b',
    '2b'  : 'stint_2b',
    '3b'  : 'stint_3b',
    'hr'  : 'stint_hr',
    'sf'  : 'stint_sf',
    'gidp': 'stint_gidp',
    'out' : 'stint_out',
    'sb'  : 'n_sb',
    'cs'  : 'n_cs',
    'ra'  : 'runs_allowed',
}

def estimate_baseruns(stints, b_terms=b_terms, cutoff=750, name='baseruns'):
    # RA = baserunners×% who score + homers
    # (RA-HR)/baserunners            = % who score
    # (b·<plays>)/(b·<plays> + outs) = % who score
    # b·<plays> = (% who score)×(b·<plays>) + (% who score)×outs
    # b·<plays> = (% who score)×outs / (1-% who score)
    bs = ['bb', 'k', '1b', '2b', '3b', 'hr', 'sf', 'gidp', 'out', 'sb', 'cs']
    hits = cl(b_terms['1b']) + cl(b_terms['2b']) + cl(b_terms['3b']) + cl(b_terms['hr'])
    baserunners = cl(b_terms['bb']) + hits - cl(b_terms['hr'])
    percent_who_score = (
        pl.when(baserunners.is_not_null() & (baserunners != 0))
          .then((cl(b_terms['ra']).fill_null(0) - cl(b_terms['hr']))/baserunners)
          .otherwise(0.)
          .clip(0.,0.999999)
    )
    outs = (
        cl(b_terms['out']) + cl(b_terms['gidp']) + cl(b_terms['sf']) + 
        cl(b_terms['cs']).fill_null(0)
    )
    scoring_potential = percent_who_score*outs/(1-percent_who_score)
    y = stints.select(scoring_potential.fill_null(0)).to_numpy().ravel()
    X = (
        stints.select(
            cl(b_terms[i]).fill_null(0).alias(f"{np.random.randint(10000)}")
               for i in bs
        )
    ).to_numpy()
    weights = stints.select('stint_pa').to_numpy().squeeze()
    cutoff = 10; mask = y/weights<cutoff
    X_mask = X[mask]; y_mask = y[mask]; weights_mask = weights[mask]
    potentials = (
        np.linalg.pinv(X_mask.T@np.diag(weights_mask)@X_mask)@
        X_mask.T@(weights_mask*y_mask)
    )

    stints = stints.with_columns(B = X@potentials)
    stints = stints.with_columns(
        scoring_potential = pl.when((cl('B')+outs).eq(0))
                              .then(0.)
                              .otherwise(cl('B')/(cl('B')+outs))
    )
    return stints.with_columns(
        (baserunners*cl('scoring_potential')+cl(b_terms['hr'])).alias(name)
    )

