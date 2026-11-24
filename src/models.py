"""Candidate estimators and their search grids.

Four candidates, in increasing order of capacity: a trivial floor, a regularized
linear model, a similarity baseline, and the ensemble expected to win. The point
of the ladder is that each rung answers a question about the one below it -- does
the problem need a model at all, does it need a nonlinear one, and does any of it
beat simply looking up the nearest known analogue.

`l1_ratio` is the current scikit-learn spelling of the penalty (0 is ridge, 1 is
lasso; `penalty` is deprecated from 1.8), so L1 against L2 is selected by
cross-validation like any other hyperparameter rather than reported as two
separate models. liblinear serves both, and it penalizes the intercept, which
`intercept_scaling` offsets.

KNN is the read-across baseline: Jaccard distance is 1 - Tanimoto on binary
fingerprints, which is how medicinal chemists judge molecular similarity, so the
question it answers is whether the learned models beat looking up near analogues.
It is the one candidate fitted on the fingerprint alone.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier


def preprocessing_for(name):
    """Preprocessor settings per candidate.

    Only the linear model needs centred, scaled inputs; trees are invariant to
    monotone rescaling, and the Tanimoto metric needs raw bits (see the
    `fingerprint_only` branch in `MolecularPreprocessor.fit`).
    """
    if name == 'knn':
        return {'normalize': False, 'fingerprint_only': True}
    if name == 'random_forest':
        return {'normalize': False}
    return {'normalize': True}


def candidates():
    """Candidate estimators with their search grids.

    Returns `{name: (estimator, grid)}`. The grids total 51 configurations, so a
    run is 255 fits per split across the five folds.

    `class_weight` is deliberately absent. For logistic regression it moves the
    decision boundary without reordering predictions, so it is close to a
    threshold change in disguise and barely registers on the ROC-AUC that drives
    selection. Leaving it out keeps the models at their unweighted fit and makes
    operating-point choice an explicit, separate decision rather than something
    smuggled in through the grid.
    """
    # Five decades of regularization strength, three points per decade. Wide
    # enough that both ends of the range are visibly bad, which is how you know
    # the optimum is interior rather than at a boundary.
    cs = [.001, .003, .01, .03, .1, .3, 1, 3, 10, 100]

    return {
        'baseline': (
            DummyClassifier(strategy='most_frequent'),
            {}),
        'logistic': (
            LogisticRegression(solver='liblinear', intercept_scaling=100,
                               max_iter=10000, random_state=42),
            {'C': cs, 'l1_ratio': [0, 1]}),
        'knn': (
            # Brute force rather than a tree: Jaccard is not a metric any of the
            # spatial indexes support, and 4000 molecules is small enough that it
            # does not matter.
            KNeighborsClassifier(metric='jaccard', algorithm='brute', n_jobs=1),
            {'n_neighbors': [1, 3, 5, 11, 21, 41], 'weights': ['uniform', 'distance']}),
        'random_forest': (
            # 300 trees is past the point where the out-of-bag error stops moving;
            # it is fixed rather than tuned so the grid spends its budget on the
            # parameters that do change the fit.
            RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=42),
            {'max_depth': [None, 20], 'min_samples_leaf': [1, 2, 5],
             'max_features': ['sqrt', .1, .3]}),
    }
