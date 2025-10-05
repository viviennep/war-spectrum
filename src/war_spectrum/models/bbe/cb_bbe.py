import numpy as np, polars as pl, pathlib, joblib
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from war_spectrum.models.utils import split_file,assemble_to_tempfile
cl = pl.col

features = ['launch_speed','launch_angle','sprint_speed','season']
targets = ['is_1b','is_2b','is_3b','is_hr','is_sf','is_gidp','is_out']

def load_Xy(data_path):
    df = pl.read_parquet(data_path)
    mask = cl('is_tracked') & pl.any_horizontal(targets)
    X = df.filter(mask).select(features).to_numpy()
    y = df.filter(mask).select(targets).to_numpy()
    y = y.argmax(1)
    return X,y

def train_bbe(data_path, model_save_path, split=True):
    X,y = load_Xy(pathlib.Path(data_path).resolve())
    X_tr, y_tr = X,y

    params = {
        'loss_function': 'MultiClass',
        'learning_rate': 0.05897119217105186,
        'depth': 13,
        'l2_leaf_reg': 8.98599436469612,
        'random_strength': 1.461817753045353,
        'bootstrap_type': 'Bernoulli',
        'border_count': 309,
        'iterations': 1561,
        'subsample': 0.9373346373636807,
        'task_type': 'GPU',
    }

    model = CatBoostClassifier(**params)
    model.fit(
        X_tr, y_tr,
        verbose=True,
    )

    save_path = pathlib.Path(model_save_path).resolve()
    joblib.dump(model, save_path)
    if split:
        split_file(save_path)

def load_model(model_path, split=True):
    if split:
        tmp = assemble_to_tempfile(model_path)
        return joblib.load(tmp)
    else:
        return joblib.load(pathlib.Path(model_path).resolve())


