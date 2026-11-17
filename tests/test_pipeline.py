"""Regression checks for fold isolation and estimator persistence."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from egfr_pipeline.models import candidates
from egfr_pipeline.preprocessing import MolecularPreprocessor

def test_fold_isolation_and_missing():
    train = pd.DataFrame({'morgan_0':[0,1,0,1], 'size':[1.,3.,2.,9.],
                          'constant':[1,1,1,1], 'missing':[1,np.nan,2,3]})
    pre = MolecularPreprocessor().fit(train)
    means = pre.means_.copy()
    valid = train.copy()
    valid['size'] = [np.nan,np.inf,10000,-10000]
    assert np.isfinite(pre.transform(valid)).all()
    pd.testing.assert_series_equal(means, pre.means_)
    assert pre.dropped_['constant'] == 'constant'
    assert pre.dropped_['missing'] == 'missing'
    assert set(pre.transform(train)[:,list(pre.columns_).index('morgan_0')]) == {0,1}

def test_correlated_features_and_persistence(tmp_path):
    frame = pd.DataFrame({'size':[1,3,2,8,5], 'copy':[2,6,4,16,10],
                          'morgan_0':[0,1,0,1,0]})
    pre = MolecularPreprocessor().fit(frame)
    assert pre.dropped_['copy'] == 'correlated'
    path = tmp_path/'pre.joblib'
    joblib.dump(pre,path)
    np.testing.assert_array_equal(pre.transform(frame),joblib.load(path).transform(frame))

def test_fixed_manifest():
    from egfr_pipeline.prepare import DATA
    manifest = pd.read_csv(DATA/'split_manifest.csv')
    assert manifest.chembl_id.is_unique
    assert len(manifest[manifest.split=='train']) == 3985
    assert len(manifest[manifest.split=='test']) == 997
    assert set(manifest.loc[manifest.split=='train','fold']) == set(range(5))

@pytest.mark.parametrize('name',list(candidates()))
def test_all_models_round_trip(name,tmp_path):
    rng=np.random.default_rng(42)
    X=pd.DataFrame(rng.normal(size=(60,8)),columns=[f'desc_{i}' for i in range(8)])
    y=(X.desc_0+X.desc_1>0).astype(int)
    estimator,grid=candidates()[name]
    estimator.set_params(**{key:values[0] for key,values in grid.items()})
    model=Pipeline([('preprocess',MolecularPreprocessor(normalize=name!='random_forest')),
                    ('model',estimator)]).fit(X,y)
    target=tmp_path/'model.joblib'
    joblib.dump(model,target)
    restored=joblib.load(target)
    np.testing.assert_array_equal(model.predict(X),restored.predict(X))
    if hasattr(model,'decision_function'):
        assert np.isfinite(model.decision_function(X)).all()
    else:
        assert np.isfinite(model.predict_proba(X)).all()
