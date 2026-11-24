"""Molecular preprocessing, fitted on the training fold only.

This is a single scikit-learn transformer rather than a preparation script on
purpose: inside a `Pipeline`, every decision it makes -- which columns to drop,
which to log, what to centre and scale by -- is re-fitted on each training fold
and never sees the validation fold. Doing the same work once over the whole
table, as the earlier version of this project did, leaks fold statistics into
tuning and quietly inflates every score.

The order of operations matters and is: drop unusable columns, drop redundant
ones, then transform what remains.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MolecularPreprocessor(TransformerMixin, BaseEstimator):
    """Filter, optionally log-transform, and optionally z-score molecular features.

    Parameters
    ----------
    normalize:
        Apply the signed-log and z-score steps. Off for trees, which do not need
        them.
    threshold:
        Absolute Pearson correlation above which the later of a pair of columns
        is dropped.
    fingerprint_only:
        Keep the Morgan bits and discard the descriptors, for the Tanimoto
        baseline.

    Attributes set by `fit` all end in an underscore, and `dropped_` records
    *why* each discarded column went, which is what gets written to the run's
    `*_preprocessing.json`.
    """

    def __init__(self, normalize=True, threshold=0.9, fingerprint_only=False):
        self.normalize = normalize
        self.threshold = threshold
        self.fingerprint_only = fingerprint_only

    def fit(self, X, y=None):
        # Infinities come out of a handful of RDKit descriptors (Ipc, mainly).
        # Treating them as missing lets one rule handle both.
        X = X.replace([np.inf, -np.inf], np.nan)
        self.input_columns_ = list(X.columns)

        # A column with any missing value is dropped outright rather than
        # imputed: the median fill in `transform` is a safety net for unseen
        # molecules at prediction time, not a modelling choice.
        self.dropped_ = {c: 'missing' for c in X if X[c].isna().any()}
        cols = [c for c in X if c not in self.dropped_]

        # Constant within this fold carries no information for this fold.
        for c in cols:
            if X[c].nunique() <= 1:
                self.dropped_[c] = 'constant'
        cols = [c for c in cols if c not in self.dropped_]

        if self.fingerprint_only:
            # Tanimoto read-across operates on the native fingerprint. Descriptors
            # are excluded, and correlated bits are kept, because dropping bits
            # changes the similarity the metric reports.
            for c in cols:
                if not c.startswith('morgan_'):
                    self.dropped_[c] = 'descriptor'
            self.columns_ = [c for c in cols if c.startswith('morgan_')]
        else:
            # Descriptors are ordered ahead of the fingerprint bits so that when a
            # descriptor and a bit are redundant, the interpretable one survives.
            cols = ([c for c in cols if not c.startswith('morgan_')]
                    + [c for c in cols if c.startswith('morgan_')])
            corr = np.abs(np.corrcoef(X[cols].to_numpy(dtype=float), rowvar=False))
            # Greedy forward pass: keep a column unless it correlates above the
            # threshold with something already kept. One pass, so the result
            # depends on the column order fixed just above.
            keep = []
            for i, c in enumerate(cols):
                if keep and np.max(corr[i, keep]) > self.threshold:
                    self.dropped_[c] = 'correlated'
                else:
                    keep.append(i)
            self.columns_ = [cols[i] for i in keep]

        if not self.columns_:
            raise ValueError('No usable features remain')

        Z = X[self.columns_].copy()
        self.medians_ = Z.median()
        # Binary columns are left alone throughout: centring a fingerprint bit
        # makes it neither more informative nor more interpretable.
        self.continuous_ = [c for c in Z if not set(Z[c].unique()) <= {0, 1}]

        self.logged_ = []
        if self.normalize:
            # Signed log1p for heavy-tailed descriptors only, and only where it
            # actually helps: excess kurtosis above 7 before, at or below 7 after.
            # Applying it blindly would squash well-behaved descriptors too.
            for c in self.continuous_:
                logged = np.sign(Z[c]) * np.log1p(np.abs(Z[c]))
                if Z[c].kurt() > 7 and logged.kurt() <= 7:
                    self.logged_.append(c)
                    Z[c] = logged

        # Population standard deviation, and zeros replaced by 1 so a column that
        # is constant among the continuous ones cannot divide by zero.
        self.means_ = Z[self.continuous_].mean()
        self.scales_ = Z[self.continuous_].std(ddof=0).replace(0, 1)
        return self

    def transform(self, X):
        # Median fill catches values that are missing at prediction time in
        # columns that were complete during training; the medians are the
        # training fold's, so nothing from the incoming data informs the fill.
        Z = X[self.columns_].replace([np.inf, -np.inf], np.nan).fillna(self.medians_).copy()
        if self.normalize:
            Z[self.logged_] = np.sign(Z[self.logged_]) * np.log1p(np.abs(Z[self.logged_]))
            Z[self.continuous_] = (Z[self.continuous_] - self.means_) / self.scales_
        # Boolean output for the fingerprint-only path, so the Jaccard metric
        # receives bits rather than floats it would convert itself.
        return Z.to_numpy(dtype=bool if self.fingerprint_only else float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_, dtype=object)
