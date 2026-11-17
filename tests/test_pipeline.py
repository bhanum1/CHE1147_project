"""Regression checks for fold isolation and estimator persistence."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from egfr_pipeline.models import candidates, preprocessing_for
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

@pytest.mark.parametrize('module',['prepare','preprocessing','models','train','predict','eda'])
def test_every_module_imports(module):
    """Catches syntax errors in entrypoints no other test imports."""
    __import__(f'egfr_pipeline.{module}')

def test_fingerprint_only_keeps_native_bits():
    frame = pd.DataFrame({'desc_0':[1.,2.,3.,4.], 'morgan_0':[0,1,0,1],
                          'morgan_1':[0,1,0,1], 'morgan_2':[1,1,0,0]})
    pre = MolecularPreprocessor(normalize=False, fingerprint_only=True).fit(frame)
    assert pre.columns_ == ['morgan_0','morgan_1','morgan_2']
    assert pre.dropped_['desc_0'] == 'descriptor'
    assert 'correlated' not in pre.dropped_.values(), 'duplicate bits must survive for Tanimoto'
    assert set(np.unique(pre.transform(frame))) == {0,1}

def test_scaffold_grouping():
    from egfr_pipeline.prepare import scaffolds_of
    cores = scaffolds_of(['c1ccccc1CCN','c1ccccc1CCCO','c1ccncc1C','CCCCO',None])
    assert cores[0] == cores[1], 'same ring system, different side chains'
    assert cores[0] != cores[2], 'benzene and pyridine are different scaffolds'
    assert cores[3].startswith('acyclic:') and cores[4].startswith('acyclic:')
    assert cores[3] != cores[4], 'acyclic molecules do not share one group'

def test_scaffold_split_isolates_scaffolds():
    from egfr_pipeline.prepare import scaffold_manifest
    rings = ['c1ccccc1','c1ccncc1','c1ccc2ccccc2c1','c1cnc2ccccc2c1',
             'c1ccsc1','c1ccoc1','C1CCCCC1','C1CCNCC1']
    data = pd.DataFrame([{'chembl_id':f'M{i}_{k}', 'smiles':ring+'C'*(k+1), 'is_inhibitor':(i+k)%2}
                         for i,ring in enumerate(rings) for k in range(6)])
    manifest = scaffold_manifest(data, test_fraction=0.25, folds=2)
    train = manifest[manifest.split=='train']
    test = manifest[manifest.split=='test']
    assert manifest.scaffold.nunique() == len(rings)
    assert len(test) > 0
    assert not set(train.scaffold) & set(test.scaffold), 'scaffold leaked across the holdout'
    for fold in range(2):
        inside = set(train.loc[train.fold==fold,'scaffold'])
        assert not inside & set(train.loc[train.fold!=fold,'scaffold']), 'scaffold leaked across folds'
    assert set(train.fold) == {0,1} and set(test.fold) == {-1}
    pd.testing.assert_frame_equal(manifest, scaffold_manifest(data, test_fraction=0.25, folds=2),
                                  obj='scaffold split is not reproducible')

@pytest.mark.parametrize('name',list(candidates()))
def test_all_models_round_trip(name,tmp_path):
    rng=np.random.default_rng(42)
    X=pd.concat([pd.DataFrame(rng.normal(size=(60,8)),columns=[f'desc_{i}' for i in range(8)]),
                 pd.DataFrame(rng.integers(0,2,size=(60,12)),columns=[f'morgan_{i}' for i in range(12)])],axis=1)
    y=(X.desc_0+X.desc_1>0).astype(int)
    estimator,grid=candidates()[name]
    estimator.set_params(**{key:values[0] for key,values in grid.items()})
    model=Pipeline([('preprocess',MolecularPreprocessor(**preprocessing_for(name))),
                    ('model',estimator)]).fit(X,y)
    target=tmp_path/'model.joblib'
    joblib.dump(model,target)
    restored=joblib.load(target)
    np.testing.assert_array_equal(model.predict(X),restored.predict(X))
    if hasattr(model,'decision_function'):
        assert np.isfinite(model.decision_function(X)).all()
    else:
        assert np.isfinite(model.predict_proba(X)).all()
