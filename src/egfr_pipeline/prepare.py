"""Prepare raw molecular features and preserve the original split membership."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/processed/egfr_erbB1'
META=['chembl_id','smiles','is_inhibitor']

def scaffolds_of(smiles):
    """Bemis-Murcko scaffold per molecule.

    Acyclic molecules have no Murcko core. They are keyed by their own structure
    rather than sharing one empty-scaffold group, so unrelated acyclic molecules
    are not forced onto the same side of the split.
    """
    cores=[]
    for value in smiles:
        mol=Chem.MolFromSmiles(value) if isinstance(value,str) else None
        core=MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else ''
        cores.append(core or f'acyclic:{value}')
    return cores

def scaffold_manifest(data,test_fraction=0.2,folds=5,seed=42):
    """Split whole scaffold groups, so no test scaffold is ever seen in training.

    Scaffolds are taken largest first into training until it is full; the long
    tail of rare scaffolds becomes the test set. Training folds are grouped by
    scaffold as well, so hyperparameter selection cannot lean on a scaffold that
    its validation fold also contains.

    Equally sized scaffolds are ordered by a seeded shuffle rather than by name.
    Sorting them by SMILES is not chemically neutral: strings beginning 'C(#C'
    sort first, which front-loads the acetylene-bearing EGFR chemotypes into
    whichever side is filled first and skews the class balance between splits.
    """
    frame=pd.DataFrame({'chembl_id':data.chembl_id,'is_inhibitor':data.is_inhibitor,'scaffold':scaffolds_of(data.smiles)})
    limit=round(len(frame)*(1-test_fraction))
    sizes=frame.scaffold.value_counts()
    names=sorted(sizes.index)
    shuffled=dict(zip(names,np.random.default_rng(seed).permutation(len(names))))
    assigned={}; used=0
    for scaffold,size in sorted(sizes.items(),key=lambda item:(-item[1],shuffled[item[0]])):
        if used+size<=limit: assigned[scaffold]='train'; used+=size
        else: assigned[scaffold]='test'
    frame['split']=frame.scaffold.map(assigned); frame['fold']=-1
    train=frame[frame.split=='train']
    splitter=StratifiedGroupKFold(folds,shuffle=True,random_state=seed)
    for fold,(_,valid) in enumerate(splitter.split(train,train.is_inhibitor,groups=train.scaffold)):
        frame.loc[train.index[valid],'fold']=fold
    shared=set(frame.loc[frame.split=='train','scaffold'])&set(frame.loc[frame.split=='test','scaffold'])
    assert not shared,f'{len(shared)} scaffolds appear in both splits'
    return frame[['chembl_id','split','fold','scaffold']]

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-features',action='store_true',help='Rebuild molecular features from the local ChEMBL dictionary')
    parser.add_argument('--scaffold',action='store_true',help='Also write a Bemis-Murcko scaffold split to scaffold_manifest.csv')
    args=parser.parse_args(argv)
    if args.rebuild_features:
        labels=pd.read_csv(ROOT/'data/raw/egfr_erbB1.csv')
        wanted=set(labels.chembl_id)
        matches=[]
        for chunk in pd.read_csv(ROOT/'data/external/chembl_37_chemreps.txt.gz',sep='\t',usecols=['chembl_id','canonical_smiles'],chunksize=100000):
            matches.append(chunk[chunk.chembl_id.isin(wanted)])
        structures=labels.merge(pd.concat(matches),on='chembl_id',validate='one_to_one').rename(columns={'canonical_smiles':'smiles'})
        generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
        records=[]
        for row in structures.itertuples():
            mol=Chem.MolFromSmiles(row.smiles) if isinstance(row.smiles,str) else None
            if mol is None: continue
            features={f'morgan_{i}':int(v) for i,v in enumerate(generator.GetFingerprintAsNumPy(mol))}
            features.update(Descriptors.CalcMolDescriptors(mol))
            records.append({'chembl_id':row.chembl_id,'smiles':row.smiles,**features,'is_inhibitor':row.is_inhibitor})
        rebuilt=pd.DataFrame(records).replace([np.inf,-np.inf],np.nan)
        expected=pd.read_csv(DATA/'features.csv',usecols=['chembl_id'])
        if set(rebuilt.chembl_id)!=set(expected.chembl_id):
            raise ValueError('Rebuilt molecule membership differs from the fixed dataset; original features were not overwritten')
        rebuilt.to_csv(DATA/'features_rebuilt.csv',index=False)
        print('Rebuilt features written separately to features_rebuilt.csv; original features preserved.')
    data=pd.read_csv(DATA/'features.csv')
    rows=[]
    for part in ['train','test']:
        old=pd.read_csv(ROOT/f'legacy/splits/egfr_erbB1/{part}.csv',usecols=['chembl_id','is_inhibitor'])
        merged=old.merge(data,on='chembl_id',validate='one_to_one',suffixes=('_old',''))
        assert len(merged)==len(old)
        assert merged.is_inhibitor_old.equals(merged.is_inhibitor)
        for ident in old.chembl_id: rows.append({'chembl_id':ident,'split':part,'fold':-1})
    manifest=pd.DataFrame(rows)
    assert manifest.chembl_id.is_unique and set(manifest.chembl_id)==set(data.chembl_id)
    train=manifest[manifest.split=='train']
    labels=data.set_index('chembl_id').loc[train.chembl_id,'is_inhibitor']
    for fold,(_,valid) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(train,labels)):
        manifest.loc[train.index[valid],'fold']=fold
    manifest.to_csv(DATA/'split_manifest.csv',index=False)
    print('Prepared fixed split: 3985 training and 997 test molecules.')
    if args.scaffold:
        frame=scaffold_manifest(data)
        frame.to_csv(DATA/'scaffold_manifest.csv',index=False)
        counts=frame.split.value_counts()
        print(f'Prepared scaffold split: {counts.get("train",0)} training and {counts.get("test",0)} test molecules '
              f'over {frame.scaffold.nunique()} scaffolds.')
if __name__=='__main__': main()
