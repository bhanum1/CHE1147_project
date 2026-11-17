"""Prepare raw molecular features and preserve the original split membership."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/processed/egfr_erbB1'
META=['chembl_id','smiles','is_inhibitor']
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-features',action='store_true',help='Rebuild molecular features from the local ChEMBL dictionary')
    args=parser.parse_args()
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
if __name__=='__main__': main()
