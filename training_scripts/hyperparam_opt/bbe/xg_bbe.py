import numpy as np, polars as pl, pathlib, joblib, optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

cl = pl.col
features = ['launch_speed','launch_angle','sprint_speed']
targets  = ['is_1b','is_2b','is_3b','is_hr','is_sf','is_gidp','is_out']

def load_Xy(data_path):
    df = pl.read_parquet(data_path)
    X = df.filter('is_tracked').select(features).to_numpy()
    y = df.filter('is_tracked').select(targets).fill_null(False).to_numpy()
    y = y.argmax(1)
    return X,y

def objective(trial, X, y):
    params = {
        'objective'        : 'multi:softprob',
        'num_class'        : len(np.unique(y)),
        'learning_rate'    : trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth'        : trial.suggest_int('max_depth', 3, 10),
        'min_child_weight' : trial.suggest_float('min_child_weight', 1e-2, 10.0, log=True),
        'subsample'        : trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma'            : trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-8, 1e-1, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'n_estimators'     : trial.suggest_int('n_estimators', 300, 2000),
        'device'           : 'cuda',
        'n_jobs'           : -1,
        'eval_metric'      :'mlogloss',
        'early_stopping_rounds':50,
        'verbosity':       1,
        }

    skf = StratifiedKFold(n_splits=3, shuffle=True)
    val_losses = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]
        model = XGBClassifier(**params)
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
        )
        proba = model.predict_proba(Xva)
        val_losses.append(log_loss(yva, proba, labels=np.arange(params['num_class'])))
        trial.report(float(np.mean(val_losses)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(val_losses))

data_dir = pathlib.Path('../data').resolve()
model_dir = pathlib.Path('../models').resolve()

X,y = load_Xy(data_dir/'curated/events')

def _objective(trial):
    return objective(trial, X, y)

backend = JournalFileBackend('./xg-bbe.study.journal')
storage = JournalStorage(backend)
study = optuna.create_study(
    study_name='xg_ss_bbe',
    direction='minimize',
    sampler=optuna.samplers.TPESampler(),
    storage=storage,
    load_if_exists=True,
)
study.optimize(_objective, n_trials=100, show_progress_bar=True, n_jobs=1, gc_after_trial=True)
best = study.best_trial.params


