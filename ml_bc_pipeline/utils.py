import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomScaler(TransformerMixin):
    def __init__(self, continuous_idx, dummies_idx):
        self.continuous_idx = continuous_idx
        self.dummies_idx = dummies_idx
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[:, self.continuous_idx])
        return self

    def transform(self, X, y=None, copy=None):
        X_head = self.scaler.transform(X[:, self.continuous_idx])
        return np.concatenate((X_head, X[:, self.dummies_idx]), axis=1)