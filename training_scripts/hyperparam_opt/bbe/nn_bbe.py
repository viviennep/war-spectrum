import numpy as np, polars as pl, torch, pathlib, joblib
from war_spectrum.models.bbe import *
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import optuna
cl = pl.col

def objective(trial, Xw, y, device='gpu'):
    hidden   = trial.suggest_categorical('hidden', [32, 64, 128, 256])
    depth    = trial.suggest_int('depth', 1, 6)
    dropout  = trial.suggest_float('dropout', 0.0, 0.5)
    lr       = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    wd       = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
    batch_sz = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
    act      = trial.suggest_categorical('activation', ['relu','gelu','silu','tanh'])
    epochs   = trial.suggest_int('epochs', 15, 60)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1234)
    val_losses = []

    for tr_idx, va_idx in skf.split(Xw, y):
        Xtr, ytr = Xw[tr_idx], y[tr_idx]
        Xva, yva = Xw[va_idx], y[va_idx]

        ds_tr = DatasetFromNumpy(Xtr, ytr)
        ds_va = DatasetFromNumpy(Xva, yva)
        dl_tr = DataLoader(ds_tr, batch_size=batch_sz, shuffle=True, num_workers=0, pin_memory=True)
        dl_va = DataLoader(ds_va, batch_size=4096, shuffle=False, num_workers=0, pin_memory=True)

        model = MLPxwOBA(
            in_dim=Xw.shape[1],
            out_dim=7,
            hidden=hidden,
            depth=depth,
            dropout=dropout,
            activation=act
        ).to(device)
        best = train_model(model, dl_tr, dl_va, device, epochs, lr, wd)
        val = evaluate(model, dl_va, device)
        val_losses.append(val['loss'])

        # pruning
        trial.report(np.mean(val_losses), step=len(val_losses))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(val_losses))


data_dir = pathlib.Path('../data').resolve()
model_dir = pathlib.Path('../models').resolve()

X,y = load_Xy(data_dir/'curated/events')

whitener = Whitener().fit(X)
Xw = whitener.transform(X)
joblib.dump({"whitener": whitener}, model_dir/'bbe/whitener.joblib')

def _objective(trial):
    return objective(trial, Xw, y, device='cuda')

backend = JournalFileBackend('./thresh-valid-posadj2.study.journal')
storage = JournalStorage(backend)
study = optuna.create_study(
    study_name='nn_ss_bbe',
    direction='minimize',
    sampler=optuna.samplers.TPESampler(),
    storage=storage,
    load_if_exists=True,
)
study.optimize(_objective, n_trials=100, show_progress_bar=True, n_jobs=1, gc_after_trial=True)
best = study.best_trial.params




