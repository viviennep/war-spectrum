import numpy as np, polars as pl, pathlib, joblib, json
from sklearn.model_selection import train_test_split
from war_spectrum.models.pitch_quality import PitchQualityModel, impute_arm_angle
from war_spectrum.models.utils import impute_arm_angle
cl = pl.col

data_dir = pathlib.Path('../data').resolve()
model_dir = pathlib.Path('../models').resolve()

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
    'strikes',
    'plate_x',
    'plate_z',
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

with open('./hyperparam_opt/pitch/pitch_params.json','r') as f:
    opt_submodel_catboost_params = json.load(f)

default_params = {
    'task_type'      : 'GPU',
    'use_best_model' : True,
    'verbose'        : True,
    'cat_features'   : [0,1,2]
}

key_translator = {
    'lr': 'learning_rate',
    'depth': 'depth',
    'l2': 'l2_leaf_reg',
    'rand': 'random_strength',
    'bagtemp': 'bagging_temperature',
    'iters': 'iterations',
    'od_wait': 'od_wait',
}

submodel_catboost_params = {}
for k,v in opt_submodel_catboost_params.items():
    submodel_catboost_params[k] = {}
    submodel_catboost_params[k]['loss_function'] = (
        'Logloss' if k=='is_swing' else 'MultiClass'
    )
    for default_k,default_v in default_params.items():
        submodel_catboost_params[k][default_k] = default_v
    for opt_k,opt_v in v.items():
        submodel_catboost_params[k][key_translator[opt_k]] = opt_v

train_inds, valid_inds = train_test_split(np.arange(len(df)),train_size=0.9)

pitching_model = PitchQualityModel(
    primary_features,
    secondary_features,
    pitch_types,
    submodel_masks,
    submodel_targets,
    submodel_catboost_params,
)
pitching_model.fit(df, train_inds=train_inds, valid_inds=valid_inds)
pitching_model.save(model_dir/'pitch_quality/pitching.joblib')

