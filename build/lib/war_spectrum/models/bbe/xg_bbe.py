import numpy as np, polars as pl, pathlib, joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
cl = pl.col

features = ['launch_speed','launch_angle','sprint_speed']
targets = ['is_1b','is_2b','is_3b','is_hr','is_sf','is_gidp','is_out']

def load_Xy(data_path):
    df = pl.read_parquet(data_path)
    X = df.filter('is_tracked').select(features).to_numpy()
    y = df.filter('is_tracked').select(targets).fill_null(False).to_numpy()
    y = y.argmax(1)
    return X,y

def train_bbe(data_path, model_save_path):
    X,y = load_Xy(pathlib.Path(data_path).resolve())
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y,
        test_size=0.10,
        stratify=y
    )

    params = {
        'learning_rate': 0.08888695467636704,
        'max_depth': 9,
        'min_child_weight': 0.023801943335840157,
        'subsample': 0.5828769717794628,
        'colsample_bytree': 0.9245042546885444,
        'gamma': 1.5114932983198943,
        'reg_alpha': 4.569572162730412e-05,
        'reg_lambda': 9.849088309821342,
        'n_estimators': 1718,
        'eval_metric':'mlogloss',
        'early_stopping_rounds':50,
    }
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=len(targets),
        n_jobs=-1,
        **params
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_va, y_va)],
        verbose=False,
    )

    joblib.dump(model, pathlib.Path(model_save_path).resolve())

def load_model(model_path):
    return joblib.load(pathlib.Path(model_path).resolve())


