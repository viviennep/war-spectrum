import numpy as np, polars as pl, pathlib, joblib
from sklearn.model_selection import train_test_split
from war_spectrum.models.framing import load_Xy, train_framing_model 
from war_spectrum.pipeline.apply_event_level_models import apply_framing

cl = pl.col

data_dir = pathlib.Path('../data').resolve()
model_dir = pathlib.Path('../models').resolve()

df = pl.read_parquet(data_dir/'curated/events')
df = df.filter('is_called')

df = apply_framing(df, model_dir/'framing')

#(
#    df.group_by('catcher_year')
#    .agg(0.125*(cl('p_called_strike_catcher')-cl('p_called_strike_avg')).sum())
#    .sort('literal')
#)

