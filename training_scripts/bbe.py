import numpy as np, polars as pl, pathlib
from sklearn.model_selection import train_test_split
from war_spectrum.models.bbe.cb_bbe import *
#from war_spectrum.models.bbe.bbe import *
cl = pl.col

data_dir = pathlib.Path('../data').resolve()
model_dir = pathlib.Path('../models').resolve()

train_bbe(data_dir/'play_by_play', model_dir/'bbe/bbe_classifier.joblib')

