import numpy as np, polars as pl, pathlib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from .glmm import *
cl = pl.col

def load_Xy(df):
    num_features   = ['sz_top','sz_bot','plate_x','plate_z','pfx_x','pfx_z']
    cat_features   = ['balls','strikes','inning_topbot','p_throws','stand','home_team']

    features = num_features + cat_features
    target = ['is_called_strike']

    catchers,pitchers = df.select('catcher_year','pitcher_year').to_numpy().T

    X = df.select(features).to_numpy()
    y = df.select(target).to_numpy().squeeze()
    return X,y,catchers,pitchers,num_features,cat_features

def train_framing_model(
        df,
        model_dir,
        model_name,
        train_base_model=False,
        test_train_split=0.9,
        pitcher_prior_σ = 1.,
        catcher_prior_σ = 1.,
        warm_start=True,
    ):
    X,y,catchers,pitchers,num_features,cat_features = load_Xy(df)
    X_tr, X_va, y_tr, y_va = train_test_split(X,y,train_size=test_train_split)

    features = num_features + cat_features

    base_model = CatBoostClassifier(
        cat_features = list(range(len(num_features),len(features))),
        learning_rate=4e-2,
        iterations=2000,
    )

    base_model_path = model_dir / 'base_model.cbm'
    if train_base_model:
        base_model.fit(X_tr, y_tr, eval_set=(X_va,y_va))
        base_model.save_model(base_model_path)
    else:
        base_model.load_model(base_model_path)

    pred_logit = base_model.predict(X,prediction_type='RawFormulaVal')

    catcher_weights = dict(zip(*np.unique(catchers,return_counts=True)))
    pitcher_weights = dict(zip(*np.unique(pitchers,return_counts=True)))

    if warm_start:
        try:
            framing = load_model(model_dir, model_name=model_name, device='cpu')
            framing.weights_c.update({str(k): float(v) for k, v in catcher_weights.items()})
            framing.weights_p.update({str(k): float(v) for k, v in pitcher_weights.items()})
        except FileNotFoundError:
            framing = CatcherAndPitcherREs(
                prior_σ_p=pitcher_prior_σ,
                prior_σ_c=catcher_prior_σ,
                catcher_weights=catcher_weights,
                pitcher_weights=pitcher_weights,
            )
    else:
        framing = CatcherAndPitcherREs(
            prior_σ_p=pitcher_prior_σ,
            prior_σ_c=catcher_prior_σ,
            catcher_weights=catcher_weights,
            pitcher_weights=pitcher_weights,
        )

    initialize_new_player_params(framing, catchers, pitchers)
    train(framing, pred_logit, y, catchers, pitchers, max_iter=100)
    save_model(framing, model_dir, model_name=model_name)

def predict_called_strike_prob(df, model_dir):
    X,y,catchers,pitchers,num_features,cat_features = load_Xy(df)
    features = num_features + cat_features

    base_model = CatBoostClassifier(
        cat_features = list(range(len(num_features),len(features))),
        learning_rate=4e-2,
        iterations=2000,
    )
    base_model.load_model(model_dir/'base_model.cbm')
    pred_logit = base_model.predict(X, prediction_type='RawFormulaVal')

    framing_model = load_model(model_dir, model_name='framing_random_effects')
    η, re_c, re_p = framing_model(pred_logit, catchers, pitchers)
    η = η.detach().numpy()
    re_c = re_c.detach().numpy()

    sigmoid = lambda x: 1./(1.+np.exp(-x))
    pred_strike = sigmoid(η)
    avg_catcher = sigmoid(η - re_c)
    return pred_strike, avg_catcher


