import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "svd_boost_features.csv"
OUT = HERE / "outputs"
SEED = 42
SAMPLE_RATE = 16000
TEST_FRACTION = 0.2
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff"}
CLASSES = {"healthy": 0, "pathological": 1}


def load_features(path=None):
    path = Path(path) if path else DATA
    if not path.exists():
        sys.exit(f"missing {path} - run `python main.py collect` or restore the feature table")
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    features = [c for c in df.columns if c not in ("speaker", "label")
                and pd.api.types.is_numeric_dtype(df[c])]
    df[features] = df[features].apply(lambda s: s.fillna(s.median()))
    features = [c for c in features if df[c].nunique() > 1]
    return (df[features].to_numpy(float),
            df["label"].astype(int).to_numpy(),
            df["speaker"].astype(str).to_numpy(),
            features)


def extract_egemaps(signal, smile, sr=SAMPLE_RATE):
    import librosa

    signal = signal / (np.max(np.abs(signal)) + 1e-9)
    signal, _ = librosa.effects.trim(signal, top_db=30)
    return signal, smile.process_signal(signal, sr)


def build_model():
    from sklearn.svm import SVC
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(kernel="rbf", C=10, probability=True, class_weight="balanced", random_state=SEED),
    )


def per_speaker(speaker, y, proba):
    agg = (pd.DataFrame({"speaker": speaker, "y": y, "p": proba})
           .groupby("speaker").agg(y=("y", "first"), p=("p", "mean")))
    return agg["y"].to_numpy(), agg["p"].to_numpy()


def score(speaker, y, proba, n_boot=2000):
    from sklearn.metrics import roc_auc_score, accuracy_score

    sy, sp = per_speaker(speaker, y, proba)
    rng, aucs = np.random.default_rng(SEED), []
    for _ in range(n_boot):
        i = rng.integers(0, len(sy), len(sy))
        if len(np.unique(sy[i])) == 2:
            aucs.append(roc_auc_score(sy[i], sp[i]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return {
        "n_recordings": int(len(y)),
        "n_speakers": int(len(sy)),
        "auc_per_record": float(roc_auc_score(y, proba)),
        "auc_per_speaker": float(roc_auc_score(sy, sp)),
        "auc_per_speaker_ci95": [float(lo), float(hi)],
        "accuracy_per_speaker": float(accuracy_score(sy, sp >= 0.5)),
    }, sy, sp


def speaker_of(path, class_dir):
    if path.parent != class_dir:
        return path.parent.name
    stem = path.stem
    for sep in ("-", "_"):
        if sep in stem:
            return stem.split(sep)[0]
    return stem


def collect(audio_dir, out_path):
    import librosa
    import opensmile

    audio_dir = Path(audio_dir)
    if not audio_dir.is_dir():
        sys.exit(f"no such folder: {audio_dir}")
    missing = [c for c in CLASSES if not (audio_dir / c).is_dir()]
    if len(missing) == len(CLASSES):
        sys.exit(f"expected subfolders {sorted(CLASSES)} inside {audio_dir}")

    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    rows, skipped = [], 0
    for class_name, label in CLASSES.items():
        class_dir = audio_dir / class_name
        if not class_dir.is_dir():
            print(f"  no {class_name}/ folder - skipping that class")
            continue
        files = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
        print(f"  {class_name}: {len(files)} file(s)")
        for n, path in enumerate(files, 1):
            try:
                signal, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                print(f"    skipped {path.name}: {e}")
                skipped += 1
                continue
            if signal.size < SAMPLE_RATE // 2:
                print(f"    skipped {path.name}: shorter than 0.5s")
                skipped += 1
                continue
            _, feats = extract_egemaps(signal, smile)
            row = feats.iloc[0].to_dict()
            row["speaker"] = speaker_of(path, class_dir)
            row["label"] = label
            rows.append(row)
            if n % 25 == 0 or n == len(files):
                print(f"    {n}/{len(files)}")

    if not rows:
        sys.exit("no readable audio found - nothing written")

    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c not in ("speaker", "label")] + ["speaker", "label"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(out_path, index=False)

    print(f"\nwrote {out_path}")
    print(f"  {len(df)} recordings, {df['speaker'].nunique()} speakers, "
          f"{len(cols) - 2} features, {skipped} skipped")
    print(f"  labels: {df['label'].value_counts().to_dict()}  (0 = healthy, 1 = pathological)")


def train(data_path=None, test_fraction=TEST_FRACTION):
    import joblib
    from sklearn.model_selection import GroupKFold, cross_val_predict

    OUT.mkdir(exist_ok=True)
    X, y, speaker, features = load_features(data_path)

    speakers = np.unique(speaker)
    n_test = max(1, int(round(len(speakers) * test_fraction)))
    held = set(np.random.default_rng(SEED).choice(speakers, size=n_test, replace=False))
    is_test = np.array([s in held for s in speaker])
    if is_test.all() or not is_test.any():
        sys.exit("test split left one side empty - adjust --test-fraction")

    Xd, yd, gd = X[~is_test], y[~is_test], speaker[~is_test]
    Xt, yt, gt = X[is_test], y[is_test], speaker[is_test]
    print(f"development: {len(yd)} recordings / {len(np.unique(gd))} speakers")
    print(f"held-out test: {len(yt)} recordings / {len(np.unique(gt))} speakers")

    model = build_model()
    print("\nvalidating (5-fold grouped by speaker, a few minutes)...")
    proba_dev = cross_val_predict(model, Xd, yd, cv=GroupKFold(5), groups=gd,
                                  method="predict_proba")[:, 1]
    val, val_sy, val_sp = score(gd, yd, proba_dev)

    print("scoring the held-out test speakers...")
    model.fit(Xd, yd)
    proba_test = model.predict_proba(Xt)[:, 1]
    test, test_sy, test_sp = score(gt, yt, proba_test)

    metrics = {
        "validation": val,
        "test": test,
        "split": {"test_fraction": test_fraction, "seed": SEED,
                  "dev_speakers": int(len(np.unique(gd))),
                  "test_speakers": int(len(np.unique(gt)))},
    }

    model.fit(X, y)
    joblib.dump({"model": model, "features": features, "sample_rate": SAMPLE_RATE},
                OUT / "model.pkl")

    pd.concat([
        pd.DataFrame({"split": "validation", "y_true": val_sy, "prob": val_sp}),
        pd.DataFrame({"split": "test", "y_true": test_sy, "prob": test_sp}),
    ]).to_csv(OUT / "predictions.csv", index=False)
    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2)

    print(json.dumps(metrics, indent=2))
    print("\nsaved: outputs/model.pkl, metrics.json, predictions.csv")


def graphs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

    if not (OUT / "predictions.csv").exists():
        sys.exit("no outputs/predictions.csv - run `python main.py train` first")

    pred = pd.read_csv(OUT / "predictions.csv")
    if "split" not in pred.columns:
        pred["split"] = "validation"

    plt.figure(figsize=(5, 5))
    for name, colour in (("validation", "#1f8a7b"), ("test", "#b4503e")):
        part = pred[pred["split"] == name]
        if part.empty:
            continue
        y, p = part["y_true"].to_numpy(), part["prob"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, lw=2, color=colour,
                 label=f"{name} (AUC {roc_auc_score(y, p):.3f}, n={len(y)})")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC - voice disorder from a vowel (per speaker)")
    plt.legend(loc="lower right", fontsize=9); plt.tight_layout()
    plt.savefig(OUT / "roc_curve.png", dpi=150); plt.close()

    part = pred[pred["split"] == "test"]
    which = "held-out test"
    if part.empty:
        part, which = pred[pred["split"] == "validation"], "validation"
    y, p = part["y_true"].to_numpy(), part["prob"].to_numpy()
    cm = confusion_matrix(y, (p >= 0.5).astype(int))
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["healthy", "pathological"]); plt.yticks([0, 1], ["healthy", "pathological"])
    plt.xlabel("predicted"); plt.ylabel("actual"); plt.title(f"Confusion matrix ({which})")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close()

    X, y_all, _, features = load_features()
    mi = pd.Series(mutual_info_classif(X, y_all, random_state=SEED),
                   index=features).sort_values(ascending=False).head(15)[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(mi)), mi.to_numpy(), color="#1f8a7b")
    plt.yticks(range(len(mi)), mi.index, fontsize=7)
    plt.xlabel("mutual information with diagnosis"); plt.title("Top 15 voice features")
    plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=150); plt.close()

    print("saved: outputs/roc_curve.png, confusion_matrix.png, feature_importance.png")


def serve(host="127.0.0.1", port=5000):
    import joblib
    import librosa
    import opensmile
    import soundfile as sf
    from flask import Flask, jsonify, render_template, request, send_from_directory

    if not (OUT / "model.pkl").exists():
        sys.exit("no outputs/model.pkl - run `python main.py train` first")

    bundle = joblib.load(OUT / "model.pkl")
    model, features, sr = bundle["model"], bundle["features"], bundle["sample_rate"]
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)

    app = Flask(__name__, template_folder=str(HERE / "templates"))

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/outputs/<path:filename>")
    def outputs(filename):
        return send_from_directory(OUT, filename)

    @app.route("/predict", methods=["POST"])
    def predict():
        if "audio" not in request.files:
            return jsonify(error="no audio uploaded"), 400
        try:
            raw = request.files["audio"].read()
            signal, _ = librosa.load(io.BytesIO(raw), sr=sr, mono=True)
        except Exception:
            try:
                signal, sr0 = sf.read(io.BytesIO(raw))
                signal = np.asarray(signal, float)
                if signal.ndim > 1:
                    signal = signal.mean(axis=1)
                signal = librosa.resample(signal, orig_sr=sr0, target_sr=sr)
            except Exception as e:
                return jsonify(error=f"could not read audio: {e}"), 400

        if signal.size < sr // 2:
            return jsonify(error="recording too short - hold the vowel for ~2 seconds"), 400

        signal, feats = extract_egemaps(signal, smile, sr)
        prob = float(model.predict_proba(feats[features].to_numpy(float))[0, 1])

        return jsonify(
            probability=round(prob, 3),
            label="pathological" if prob >= 0.5 else "healthy",
            seconds=round(len(signal) / sr, 1),
        )

    print(f"open  http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="build the feature table from raw audio")
    p_collect.add_argument("--audio-dir", required=True,
                           help="folder holding healthy/ and pathological/ subfolders")
    p_collect.add_argument("--out", default=str(DATA), help=f"output CSV (default: {DATA.name})")

    p_train = sub.add_parser("train", help="train, validate, and score the held-out test split")
    p_train.add_argument("--data", default=None, help="feature CSV (default: data/svd_boost_features.csv)")
    p_train.add_argument("--test-fraction", type=float, default=TEST_FRACTION,
                         help=f"share of speakers held out (default: {TEST_FRACTION})")

    sub.add_parser("graphs", help="draw ROC, confusion matrix, feature importance")
    p_serve = sub.add_parser("serve", help="run the web demo")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=5000)
    sub.add_parser("all", help="train, then graphs")

    args = parser.parse_args()
    if args.command == "collect":
        collect(args.audio_dir, args.out)
    elif args.command == "train":
        train(args.data, args.test_fraction)
    elif args.command == "graphs":
        graphs()
    elif args.command == "serve":
        serve(args.host, args.port)
    elif args.command == "all":
        train()
        graphs()


if __name__ == "__main__":
    main()