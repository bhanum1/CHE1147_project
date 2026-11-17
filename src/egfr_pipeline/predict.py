"""Predict EGFR inhibition from a CSV containing SMILES strings."""
import argparse
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    pipeline = joblib.load(args.model)
    data = pd.read_csv(args.input)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    records = []
    for value in data.smiles:
        mol = Chem.MolFromSmiles(value) if isinstance(value,str) else None
        if mol is None:
            records.append({'prediction':None,'score':None,'status':'Invalid SMILES'})
            continue
        bits = generator.GetFingerprintAsNumPy(mol)
        features = {f'morgan_{i}':int(bit) for i,bit in enumerate(bits)}
        features.update(Descriptors.CalcMolDescriptors(mol))
        frame = pd.DataFrame([features]).reindex(columns=pipeline.named_steps['preprocess'].input_columns_)
        if hasattr(pipeline,'decision_function'):
            score = pipeline.decision_function(frame)[0]; kind = 'decision_score'
        else:
            score = pipeline.predict_proba(frame)[0,1]; kind = 'probability'
        records.append({'prediction':int(pipeline.predict(frame)[0]),'score':float(score),'score_type':kind,'status':'ok'})
    pd.concat([data.reset_index(drop=True),pd.DataFrame(records)],axis=1).to_csv(args.output,index=False)

if __name__ == '__main__':
    main()
