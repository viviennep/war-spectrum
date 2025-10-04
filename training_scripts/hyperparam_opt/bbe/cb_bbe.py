import numpy as np, polars as pl, pathlib, joblib, optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

cl = pl.col
features = ['launch_speed','launch_angle','sprint_speed','season']
targets  = ['is_1b','is_2b','is_3b','is_hr','is_sf','is_gidp','is_out']

def load_Xy(data_path):
    df = pl.read_parquet(data_path)
    X = df.filter('is_tracked').select(features).to_numpy()
    y = df.filter('is_tracked').select(targets).fill_null(False).to_numpy()
    y = y.argmax(1)
    return X,y

def objective(trial, X, y):
    params = {
        'loss_function'   : 'MultiClass',
        'learning_rate'   : trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'depth'           : trial.suggest_int('depth', 5, 12),
        'l2_leaf_reg'     : trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_strength' : trial.suggest_float('random_strength', 0.0, 2.0),
        'bootstrap_type'  : trial.suggest_categorical('bootstrap_type', ['Bayesian','Bernoulli','MVS']),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'subsample'       : trial.suggest_float('subsample', 0.5, 1.0), 
        'border_count'    : trial.suggest_int('border_bount', 128, 350),
        'iterations'      : trial.suggest_int('iterations', 300, 2000),
        'max_leaves'      : trial.suggest_int('max_leaves', 31, 63),
        'task_type'       : 'GPU',     # switch to 'CPU' if you don't have a GPU
        'devices'         : '0',
        'cat_features'    : [3],
        'allow_writing_files': False,
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True)
    val_losses = []
    num_class = len(np.unique(y))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]
        model = CatBoostClassifier(**params)
        model.fit(
            Xtr, ytr,
            eval_set=(Xva, yva),
            use_best_model=True,
            early_stopping_rounds=50,
            verbose=True,
        )
        proba = model.predict_proba(Xva)
        val_losses.append(log_loss(yva, proba, labels=np.arange(num_class)))
        trial.report(float(np.mean(val_losses)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(val_losses)

data_dir = pathlib.Path('../../../data').resolve()
model_dir = pathlib.Path('../../../models').resolve()

X,y = load_Xy(data_dir/'play_by_play')

def _objective(trial):
    return objective(trial, X, y)

backend = JournalFileBackend('./cb-bbe.study.journal')
storage = JournalStorage(backend)
study = optuna.create_study(
    study_name='cb_ss_bbe',
    direction='minimize',
    sampler=optuna.samplers.TPESampler(),
    storage=storage,
    load_if_exists=True,
)
study.optimize(_objective, n_trials=100, show_progress_bar=True, n_jobs=1, gc_after_trial=True)
best = study.best_trial.params


