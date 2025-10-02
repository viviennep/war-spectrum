import numpy as np, polars as pl, pathlib, joblib, optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from war_spectrum.models.pitch_quality import PitchQualityModel, impute_arm_angle
from war_spectrum.models.utils import impute_arm_angle
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
cl = pl.col

data_dir = pathlib.Path('../../../data').resolve()
model_dir = pathlib.Path('../../../models').resolve()

df = pl.read_parquet(data_dir / 'curated/events',hive_partitioning=True)

# Add in arm angle for pitches that don't have em
df = impute_arm_angle(df, model_dir / 'pitch_quality/arm_angle_imputer.cbm')

primary_features = [
    'season',
    'p_throws',
    'stand',
    'release_pos_x',
    'release_pos_z',
    'release_extension',
    'spin_axis_diff',
    'release_spin_rate',
    'height',
    'spin_efficiency',
    'pfx_x',
    'pfx_z',
    'release_speed',
    'a_mag',
    'arm_angle',
    'balls',
    'strikes'
]
secondary_features = primary_features + ['diff_release_speed','diff_pfx_x','diff_pfx_z']

pitch_types = [
    'primary_fast','primary_slow','primary_bend',
    'secondary_fast','secondary_slow','secondary_bend',
]

submodel_masks = {
    'is_swing': pl.lit(True), 
    'outcome_given_take': ~cl('is_swing'),
    'outcome_given_swing': cl('is_swing'),
    'outcome_given_bbe': cl('is_bbe'),
}

submodel_targets = {
    'is_swing': 'is_swing', 
    'outcome_given_take': ['is_ball', 'is_called_strike', 'is_hbp'],
    'outcome_given_swing': ['is_whiff', 'is_foul', 'is_bbe'],
    'outcome_given_bbe': [
        'is_1b', 'is_2b', 'is_3b', 'is_hr', 'is_sf', 'is_gidp', 'is_out'
    ]
}

def suggest_params(trial, submodel):
    return {
        'loss_function'   : 'Logloss' if submodel == 'is_swing' else 'MultiClass',
        'learning_rate'   : trial.suggest_float(f'{submodel}.lr', 1e-3, 0.3, log=True),
        'depth'           : trial.suggest_int(f'{submodel}.depth', 4, 10),
        'l2_leaf_reg'     : trial.suggest_float(f'{submodel}.l2', 1e-3, 20.0, log=True),
        'random_strength' : trial.suggest_float(f'{submodel}.rand', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float(f'{submodel}.bagtemp', 0.0, 5.0),
        'iterations'      : trial.suggest_int(f'{submodel}.iters', 300, 1500),
        'od_type'         : 'Iter',
        'od_wait'         : trial.suggest_int(f'{submodel}.od_wait', 20, 200),
        'use_best_model'  : True,
        'task_type'       : 'GPU',
        'cat_features'    : [0,1,2],
    }


def submodel_objective(df, submodel, n_splits=3, fast_rows_per_pitch_type=None):
    targets       = submodel_targets[submodel]
    mask          = submodel_masks[submodel]
    df            = df.filter(mask)
    y             = df.select(targets)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    def objective(trial):
        params = {submodel: suggest_params(trial, submodel)}
        for h in ['is_swing','outcome_given_take','outcome_given_swing','outcome_given_bbe']:
            params.setdefault(h, {
                'loss_function'  : ('Logloss' if h=='is_swing' else 'MultiClass'),
                'task_type'      : 'GPU',
                'use_best_model' : True,
                'verbose'        : True,
                'cat_features'   : [0,1,2]
            })

        fold_losses = []
        for fold, (tr_inds, va_inds) \
                in enumerate(skf.split(np.zeros(len(df)), y), 1):
            model = PitchQualityModel(
                primary_features, secondary_features, pitch_types,
                submodel_masks, submodel_targets, params
            )
            tr_df, va_df = model.make_train_test_split(
                df, 
                train_inds=tr_inds,
                valid_inds=va_inds
            )

            per_pitch_type_losses, per_pitch_type_weights = [], []
            for pitch_type in pitch_types:
                sub = model.fit_submodel(submodel, pitch_type, tr_df, va_df)
                mask = cl('pitch_type_classification').eq(pitch_type) & submodel_masks[submodel]
                if 'primary' in pitch_type:
                    features = model.primary_features
                else:
                    features = model.secondary_features
                targets = submodel_targets[submodel]
                Xva, yva = model.make_eval_set(mask, va_df, features, targets,
                                               multi=isinstance(targets, list))

                if fast_rows_per_pitch_type and (len(yva) > fast_rows_per_pitch_type):
                    sel = np.random.default_rng().choice(
                        len(yva),
                        fast_rows_per_pitch_type,
                        replace=False
                    )
                    Xva, yva = Xva[sel], yva[sel]

                probs = sub.predict_proba(Xva)
                labels = [0,1] if submodel=='is_swing' else list(range(probs.shape[1]))
                per_pitch_type_losses.append(log_loss(yva, probs, labels=labels))
                per_pitch_type_weights.append(len(yva))

            if per_pitch_type_losses:
                fold_losses.append(
                    np.average(
                        per_pitch_type_losses, 
                        weights=np.array(per_pitch_type_weights))
                )
            else:
                fold_losses.append(10.0)

            trial.report(np.mean(fold_losses), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_losses)
    return objective


submodel = 'is_swing'
#{'is_swing.lr': 0.047904358837211704,
#'is_swing.depth': 9,
#'is_swing.l2': 17.755308417978913,
#'is_swing.rand': 0.059891365362389745,
#'is_swing.bagtemp': 1.6980951678626275,
#'is_swing.iters': 908,
#'is_swing.od_wait': 120}

backend = JournalFileBackend(f'./stuff.{submodel}.journal')
storage = JournalStorage(backend)
study = optuna.create_study(
    study_name=f'stuff-{submodel}',
    direction='minimize',
    sampler=optuna.samplers.TPESampler(),
    storage=storage,
    load_if_exists=True,
)
study.optimize(
    submodel_objective(df, submodel, n_splits=3, fast_rows_per_pitch_type=200000), 
    n_trials=100, 
    show_progress_bar=True,
    n_jobs=1,
    gc_after_trial=True
)
best = study.best_trial.params

