import sys
from pathlib import Path

import numpy as np
import joblib
import librosa
import opensmile

if len(sys.argv) < 2:
    sys.exit("usage: python predict.py your_voice.wav")

MODEL_PATH = Path(__file__).resolve().parent / "outputs" / "model.pkl"
if not MODEL_PATH.exists():
    sys.exit(f"no {MODEL_PATH} - run `python main.py train` first")

bundle = joblib.load(MODEL_PATH)
model, features, sr = bundle["model"], bundle["features"], bundle["sample_rate"]

smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                        feature_level=opensmile.FeatureLevel.Functionals)

signal, _ = librosa.load(sys.argv[1], sr=sr)
signal = signal / (np.max(np.abs(signal)) + 1e-9)
signal, _ = librosa.effects.trim(signal, top_db=30)
feats = smile.process_signal(signal, sr)

X = feats[features].to_numpy(float)
prob = float(model.predict_proba(X)[0, 1])
label = "PATHOLOGICAL" if prob >= 0.5 else "HEALTHY"
print(f"P(voice disorder) = {prob:.3f}  ->  {label}")