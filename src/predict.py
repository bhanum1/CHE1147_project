"""Score new molecules with a saved pipeline.

The input CSV needs a `smiles` column; everything else in it is carried through
to the output untouched. Features are recomputed here with the same Morgan
settings as the training table, then handed to the pipeline, which applies the
preprocessing it was fitted with.

Molecules RDKit cannot parse are kept as rows with a status rather than dropped,
so the output lines up with the input row for row.

Usage (from the repo root):
    python -m egfr_pipeline.predict \
        --model models/RUN/random_forest.joblib \
        --input molecules.csv --output predictions.csv
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Path to a .joblib pipeline from a run')
    parser.add_argument('--input', required=True, help='CSV with a smiles column')
    parser.add_argument('--output', required=True, help='Where to write the scored CSV')
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    data = pd.read_csv(args.input)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    records = []
    for value in data.smiles:
        mol = Chem.MolFromSmiles(value) if isinstance(value, str) else None
        if mol is None:
            records.append({'prediction': None, 'score': None, 'status': 'Invalid SMILES'})
            continue

        bits = generator.GetFingerprintAsNumPy(mol)
        features = {f'morgan_{i}': int(bit) for i, bit in enumerate(bits)}
        features.update(Descriptors.CalcMolDescriptors(mol))

        # `reindex` to the columns the preprocessor was fitted on: RDKit's
        # descriptor set shifts between versions, so this adds back anything
        # missing as NaN (filled with training medians downstream) and drops
        # anything new, instead of failing on a column mismatch.
        frame = pd.DataFrame([features]).reindex(
            columns=pipeline.named_steps['preprocess'].input_columns_)

        # Same score convention as training: a margin for the linear model, a
        # probability for the rest, labelled so the two are not confused.
        if hasattr(pipeline, 'decision_function'):
            score = pipeline.decision_function(frame)[0]
            kind = 'decision_score'
        else:
            score = pipeline.predict_proba(frame)[0, 1]
            kind = 'probability'

        records.append({'prediction': int(pipeline.predict(frame)[0]),
                        'score': float(score), 'score_type': kind, 'status': 'ok'})

    pd.concat([data.reset_index(drop=True), pd.DataFrame(records)], axis=1).to_csv(
        args.output, index=False)


if __name__ == '__main__':
    main()
