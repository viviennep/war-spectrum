import numpy as np, polars as pl, pathlib
from catboost import CatBoostRegressor
cl = pl.col

def impute_arm_angle(df, model_dir):
    arm_angle_imputer = CatBoostRegressor().load_model(model_dir)
    X_impute_arm_angle = df.select(
        'release_pos_x',
        'release_pos_z',
        'release_extension',
        'height',
        'p_throws'
    ).to_numpy()
    pred_aa = arm_angle_imputer.predict(X_impute_arm_angle)
    df = df.with_columns(
        arm_angle = pl.when(cl('arm_angle').is_null()).then(pred_aa).otherwise('arm_angle')
    )
    return df


