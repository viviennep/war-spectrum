import numpy as np, polars as pl, pathlib
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
cl = pl.col

def load_Xy(data_path):
    df = pl.read_parquet(data_path)
    nn_df = df.filter(cl('arm_angle').is_not_null(),cl('release_pos_x').is_not_null())
    X = nn_df.select(
        'release_pos_x',
        'release_pos_z',
        'release_extension',
        'height',
        'p_throws'
    ).to_numpy()
    y = nn_df.select('arm_angle').to_numpy().ravel()
    return X,y

def train_arm_angle_imputater(data_path, model_save_dir, test_train_split=0.9):
    X_tr, X_va, y_tr, y_va = train_test_split(X,y,train_size=test_train_split)

    arm_angle_imputer = CatBoostRegressor(
        cat_features=[4],
        iterations=5000,
    ) 
    arm_angle_imputer.fit(
        X_tr,
        y_tr,
        eval_set=(X_va,y_va),
        init_model=model_save_dir/'arm_angle_imputer.cbm'
    )
    arm_angle_imputer.save_model(model_save_dir/'arm_angle_imputer.cbm')

def impute_arm_angle(df, model_path):
    arm_angle_imputer = CatBoostRegressor().load_model(model_path)
    X = df.select(
        'release_pos_x',
        'release_pos_z',
        'release_extension',
        'height',
        'p_throws'
    ).to_numpy()
    pred_aa = arm_angle_imputer.predict(X)
    return df.with_coumns(
        arm_angle = pl.when(cl('arm_angle').is_null()).then(pred_aa).otherwise('arm_angle')
    )


