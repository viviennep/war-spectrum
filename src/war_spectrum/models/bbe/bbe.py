import numpy as np, polars as pl, pathlib, joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
cl = pl.col

features = ['launch_speed','launch_angle']
targets = ['is_1b','is_2b','is_3b','is_hr','is_sf','is_gidp','is_out']

class Whitener(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        μ = X.mean(0)
        Xc = X - μ
        Σ = np.cov(Xc.T)
        L = np.linalg.cholesky(np.linalg.inv(Σ))
        self.L = L
        self.μ = μ
        return self
    def transform(self, X):
        Xc = X - self.μ
        return Xc@self.L

def load_Xy(data_path):
    df = pl.read_parquet(data_path)
    X = df.filter('is_tracked').select(features).to_numpy()
    y = df.filter('is_tracked').select(targets).fill_null(False).to_numpy()
    y = y.argmax(1)
    return X,y

def train_bbe(data_path, model_save_path):
    X,y = load_Xy(pathlib.Path(data_path).resolve())
    bbe_classifier = Pipeline([
        ('whiten', Whitener()),
        ('knn', KNeighborsClassifier(n_neighbors=400, n_jobs=-1)),
    ])
    bbe_classifier.fit(X,y)
    joblib.dump(bbe_classifier, pathlib.Path(model_save_path).resolve())

def load_model(model_path):
    return joblib.load(pathlib.Path(model_path).resolve())


