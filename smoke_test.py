from pathlib import Path

import joblib
import pandas as pd

root = Path(__file__).resolve().parent
model = joblib.load(root / 'models' / 'fraud_pipeline.joblib')
features = ['Time', *[f'V{i}' for i in range(1, 29)], 'Amount']
df = pd.read_csv(root / 'data' / 'creditcard.csv', nrows=8)
X = df[features]
probabilities = model.predict_proba(X)[:, 1]
assert len(probabilities) == len(df)
assert ((probabilities >= 0) & (probabilities <= 1)).all()
assert model.predict(X).shape == (len(df),)
print('rows=', len(df))
print('feature_count=', len(features))
print('probability_min=', float(probabilities.min()))
print('probability_max=', float(probabilities.max()))
print('smoke_test=passed')
