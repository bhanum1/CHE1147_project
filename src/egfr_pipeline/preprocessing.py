"""Training-fold-only molecular preprocessing."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MolecularPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, normalize=True, threshold=0.9, fingerprint_only=False):
        self.normalize=normalize
        self.threshold=threshold
        self.fingerprint_only=fingerprint_only

    def fit(self,X,y=None):
        X=X.replace([np.inf,-np.inf],np.nan)
        self.input_columns_=list(X.columns)
        self.dropped_={c:'missing' for c in X if X[c].isna().any()}
        cols=[c for c in X if c not in self.dropped_]
        for c in cols:
            if X[c].nunique()<=1: self.dropped_[c]='constant'
        cols=[c for c in cols if c not in self.dropped_]
        if self.fingerprint_only:
            # Tanimoto read-across operates on the native fingerprint. Descriptors
            # are excluded, and correlated bits are kept, because dropping bits
            # changes the similarity the metric reports.
            for c in cols:
                if not c.startswith('morgan_'): self.dropped_[c]='descriptor'
            self.columns_=[c for c in cols if c.startswith('morgan_')]
        else:
            cols=[c for c in cols if not c.startswith('morgan_')]+[c for c in cols if c.startswith('morgan_')]
            corr=np.abs(np.corrcoef(X[cols].to_numpy(dtype=float),rowvar=False))
            keep=[]
            for i,c in enumerate(cols):
                if keep and np.max(corr[i,keep])>self.threshold: self.dropped_[c]='correlated'
                else: keep.append(i)
            self.columns_=[cols[i] for i in keep]
        if not self.columns_: raise ValueError('No usable features remain')
        Z=X[self.columns_].copy()
        self.medians_=Z.median()
        self.continuous_=[c for c in Z if not set(Z[c].unique())<={0,1}]
        self.logged_=[]
        if self.normalize:
            for c in self.continuous_:
                logged=np.sign(Z[c])*np.log1p(np.abs(Z[c]))
                if Z[c].kurt()>7 and logged.kurt()<=7:
                    self.logged_.append(c); Z[c]=logged
        self.means_=Z[self.continuous_].mean()
        self.scales_=Z[self.continuous_].std(ddof=0).replace(0,1)
        return self

    def transform(self,X):
        Z=X[self.columns_].replace([np.inf,-np.inf],np.nan).fillna(self.medians_).copy()
        if self.normalize:
            Z[self.logged_]=np.sign(Z[self.logged_])*np.log1p(np.abs(Z[self.logged_]))
            Z[self.continuous_]=(Z[self.continuous_]-self.means_)/self.scales_
        # Boolean output for the fingerprint-only path, so the Jaccard metric
        # receives bits rather than floats it would convert itself.
        return Z.to_numpy(dtype=bool if self.fingerprint_only else float)

    def get_feature_names_out(self,input_features=None):
        return np.array(self.columns_,dtype=object)
