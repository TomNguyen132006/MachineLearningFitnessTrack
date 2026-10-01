"""Export real pipeline results to demo/demo_data.json for the HTML demo page.

Reads the pickles produced by the pipeline (run make_dataset.py, remove_outliers.py
and build_features.py first) and recomputes the results the way the pipeline
scripts do, without plotting. Values that cannot be computed are stored as null
with a reason in "missing".

Usage (from any directory):
    python demo/export_demo_data.py
"""

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.special
from scipy.signal import argrelextrema
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

# The pipeline builds feature lists with set(), whose order depends on Python's hash seed.
# Re-run this script with a fixed PYTHONHASHSEED so the column order, and the results, repeat exactly.
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit(subprocess.run([sys.executable, *sys.argv], env={**os.environ, "PYTHONHASHSEED": "0"}).returncode)

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUT_JSON = ROOT / "demo" / "demo_data.json"
OUT_HTML = ROOT / "demo" / "index.html"
HTML_START, HTML_END = "<!-- DEMO_DATA_START -->", "<!-- DEMO_DATA_END -->"

sys.path.insert(0, str(ROOT / "src" / "features"))
sys.path.insert(0, str(ROOT / "src" / "models"))
from DataTransformation import LowPassFilter  # noqa: E402
from LearningAlgorithms import ClassificationAlgorithms  # noqa: E402

# The neural network and random forest in LearningAlgorithms have no random_state, so they
# draw from numpy's global RNG. Seeding it here (export script only) makes runs repeatable.
SEED = 42
np.random.seed(SEED)

SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
MAX_SIGNAL_POINTS = 150
missing = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(name):
    path = INTERIM / name
    if not path.exists():
        missing[name] = f"{path} not found - run the pipeline scripts first"
        return None
    return pd.read_pickle(path)


# --------------------------------------------------------------
# Outlier functions (same logic as src/features/remove_outliers.py)
# --------------------------------------------------------------


def mark_outliers_iqr(dataset, col):
    q1, q3 = dataset[col].quantile(0.25), dataset[col].quantile(0.75)
    iqr = q3 - q1
    return (dataset[col] < q1 - 1.5 * iqr) | (dataset[col] > q3 + 1.5 * iqr)


def mark_outliers_chauvenet(dataset, col, C=2):
    mean, std = dataset[col].mean(), dataset[col].std()
    criterion = 1.0 / (C * len(dataset.index))
    deviation = abs(dataset[col] - mean) / std
    high = deviation / math.sqrt(C)
    prob = 1.0 - 0.5 * (scipy.special.erf(high) - scipy.special.erf(-high))
    return prob < criterion


def mark_outliers_lof(dataset, columns, n=20):
    return LocalOutlierFactor(n_neighbors=n).fit_predict(dataset[columns]) == -1


# --------------------------------------------------------------
# a) Dataset summary  +  b) sample sensor signals
# --------------------------------------------------------------


def dataset_summary(df):
    return {
        "rows": int(len(df)),
        "sampling_rate_hz": 5,
        "participants": sorted(df["participant"].unique().tolist()),
        "labels": df["label"].unique().tolist(),
        "categories": df["category"].unique().tolist(),
        "sets": int(df["set"].nunique()),
        "samples_per_label": {k: int(v) for k, v in df["label"].value_counts().items()},
        "distinct_set_lengths": int(df.groupby("set").size().nunique()),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
    }


def sensor_signals(df):
    signals = {}
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        first_set = int(subset["set"].min())
        s = subset[subset["set"] == first_set]
        step = max(1, math.ceil(len(s) / MAX_SIGNAL_POINTS))
        s = s.iloc[::step]
        signals[label] = {
            "set": first_set,
            "participant": s["participant"].iloc[0],
            "category": s["category"].iloc[0],
            "time_s": [round((t - s.index[0]).total_seconds(), 2) for t in s.index],
            **{c: [round(float(v), 4) for v in s[c]] for c in SENSOR_COLS},
        }
    return signals


# --------------------------------------------------------------
# c) Outlier detection
# --------------------------------------------------------------


def outliers(df, df_removed):
    result = {
        "rows": int(len(df)),
        "iqr": {c: int(mark_outliers_iqr(df, c).sum()) for c in SENSOR_COLS},
        "chauvenet": {c: int(mark_outliers_chauvenet(df, c).sum()) for c in SENSOR_COLS},
        "lof_rows": int(mark_outliers_lof(df, SENSOR_COLS).sum()),
        "lof_note": "LOF uses all six sensor columns together, so it flags rows, not columns.",
    }
    if df_removed is not None:
        # What the pipeline actually removed: Chauvenet applied per exercise label
        result["removed_chauvenet_per_label"] = {
            c: int(df_removed[c].isna().sum()) for c in SENSOR_COLS
        }
    else:
        result["removed_chauvenet_per_label"] = None
    return result


# --------------------------------------------------------------
# d) Model comparison  +  e) best model (same calls as src/models/train_model.py)
# --------------------------------------------------------------


def models(df):
    learner = ClassificationAlgorithms()
    df_train = df.drop(["participant", "category", "set"], axis=1)
    X = df_train.drop("label", axis=1)
    y = df_train["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    square_features = ["acc_r", "gyr_r"]
    pca_features = ["pca_1", "pca_2", "pca_3"]
    time_features = [f for f in df_train.columns if "_temp_" in f]
    freq_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]
    cluster_features = ["cluster"]

    log("Forward feature selection (10 features) - this is the slow part")
    selected_features, _, ordered_scores = learner.forward_selection(10, X_train, y_train)

    feature_sets = {
        "Feature_set_1": list(set(basic_features)),
        "Feature_set_2": list(set(basic_features + square_features + pca_features)),
        "Feature_set_3": list(set(basic_features + time_features)),
        "Feature_set_4": list(set(basic_features + freq_features + cluster_features)),
        "Selected Features": selected_features,
    }

    scores = []
    for name, cols in feature_sets.items():
        log(f"Training models on {name} ({len(cols)} features)")
        tr, te = X_train[cols], X_test[cols]
        preds = {
            "NN": learner.feedforward_neural_network(tr, y_train, te, gridsearch=False)[1],
            "RF": learner.random_forest(tr, y_train, te, gridsearch=True)[1],
            "KNN": learner.k_nearest_neighbor(tr, y_train, te, gridsearch=True)[1],
            "DT": learner.decision_tree(tr, y_train, te, gridsearch=True)[1],
            "NB": learner.naive_bayes(tr, y_train, te)[1],
        }
        for model, pred in preds.items():
            scores.append(
                {"model": model, "feature_set": name, "accuracy": float(accuracy_score(y_test, pred))}
            )

    best = max(scores, key=lambda s: s["accuracy"])
    tied_best = [s for s in scores if s["accuracy"] == best["accuracy"]]

    # train_model.py evaluates a random forest on Feature_set_4 for the confusion matrix
    log("Random forest on Feature_set_4 (confusion matrix)")
    cols = feature_sets["Feature_set_4"]
    _, class_test_y, _, class_test_prob_y = learner.random_forest(
        X_train[cols], y_train, X_test[cols], gridsearch=True
    )
    classes = list(class_test_prob_y.columns)
    cm = confusion_matrix(y_test, class_test_y, labels=classes)

    log("Random forest on Feature_set_4 with participant A held out")
    participant_df = df.drop(["set", "category"], axis=1)
    train_mask = participant_df["participant"] != "A"
    Xp_train = participant_df[train_mask].drop(["label", "participant"], axis=1)
    yp_train = participant_df[train_mask]["label"]
    Xp_test = participant_df[~train_mask].drop(["label", "participant"], axis=1)
    yp_test = participant_df[~train_mask]["label"]
    _, p_pred, _, _ = learner.random_forest(
        Xp_train[cols], yp_train, Xp_test[cols], gridsearch=True
    )

    return {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
        "selected_features": selected_features,
        "forward_selection_scores": [float(s) for s in ordered_scores],
        "scores": scores,
        "best": best,
        "tied_best": tied_best,
        "confusion_matrix": {
            "model": "RF",
            "feature_set": "Feature_set_4",
            "accuracy": float(accuracy_score(y_test, class_test_y)),
            "labels": classes,
            "matrix": cm.tolist(),
            "test_rows": int(len(y_test)),
        },
        "participant_holdout": {
            "model": "RF",
            "feature_set": "Feature_set_4",
            "held_out": "A",
            "accuracy": float(accuracy_score(yp_test, p_pred)),
            "test_rows": int(len(yp_test)),
        },
    }


# --------------------------------------------------------------
# f) Rep counting (same logic as src/features/count_repetitions.py)
# --------------------------------------------------------------


def rep_counting(df):
    df = df[df["label"] != "rest"].copy()
    df["acc_r"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
    df["gyr_r"] = np.sqrt(df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2)

    fs = 1000 / 200
    lowpass = LowPassFilter()

    def count_reps(dataset, cutoff=0.4, order=10, column="acc_r"):
        data = lowpass.low_pass_filter(
            dataset.copy(), col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order
        )
        return len(argrelextrema(data[column + "_lowpass"].values, np.greater)[0])

    rows = []
    for s in df["set"].unique():
        subset = df[df["set"] == s]
        label = subset["label"].iloc[0]
        column, cutoff = "acc_r", 0.4
        if label == "squat":
            cutoff = 0.35
        if label == "row":
            cutoff, column = 0.65, "gyr_x"
        if label == "ohp":
            cutoff = 0.35
        category = subset["category"].iloc[0]
        rows.append(
            {
                "set": int(s),
                "participant": subset["participant"].iloc[0],
                "label": label,
                "category": category,
                "reps": 5 if category == "heavy" else 10,
                "reps_pred": int(count_reps(subset, cutoff=cutoff, column=column)),
            }
        )

    rep_df = pd.DataFrame(rows)
    by_group = (
        rep_df.groupby(["label", "category"])[["reps", "reps_pred"]].mean().round(2).reset_index()
    )
    return {
        "mae": round(float(mean_absolute_error(rep_df["reps"], rep_df["reps_pred"])), 2),
        "exact_match_sets": int((rep_df["reps"] == rep_df["reps_pred"]).sum()),
        "sets": rep_df.to_dict(orient="records"),
        "by_label_category": by_group.to_dict(orient="records"),
        "actual_reps_note": "Actual reps are assumed from the category (heavy = 5, medium = 10), as in count_repetitions.py.",
    }


# --------------------------------------------------------------
# g) Rep-counting replay for the "See it in action" section
# --------------------------------------------------------------


def rep_replay(df, label="bench", column="acc_r", cutoff=0.4, order=10):
    """Filtered signal and peaks for one set, computed as count_repetitions.py does for bench press.

    The set is chosen by a fixed rule: participant A's first heavy set of the exercise.
    """
    sets = df[(df["label"] == label) & (df["participant"] == "A") & (df["category"] == "heavy")]
    s = int(sets["set"].min())
    subset = df[df["set"] == s].copy()
    subset["acc_r"] = np.sqrt(subset["acc_x"] ** 2 + subset["acc_y"] ** 2 + subset["acc_z"] ** 2)
    subset["gyr_r"] = np.sqrt(subset["gyr_x"] ** 2 + subset["gyr_y"] ** 2 + subset["gyr_z"] ** 2)

    fs = 1000 / 200
    data = LowPassFilter().low_pass_filter(
        subset, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order
    )
    filtered = data[column + "_lowpass"].values
    peaks = argrelextrema(filtered, np.greater)[0]
    category = subset["category"].iloc[0]

    return {
        "label": label,
        "set": s,
        "participant": subset["participant"].iloc[0],
        "category": category,
        "column": column,
        "cutoff_hz": cutoff,
        "order": order,
        "sampling_rate_hz": fs,
        "time_s": [round(i / fs, 2) for i in range(len(filtered))],
        "signal": [round(float(v), 5) for v in subset[column]],
        "filtered": [round(float(v), 5) for v in filtered],
        "peaks": [int(i) for i in peaks],
        "reps_counted": int(len(peaks)),
        "reps_actual": 5 if category == "heavy" else 10,
    }


# --------------------------------------------------------------
# Embed the data in index.html so it opens without a server
# --------------------------------------------------------------


def embed_in_html(data):
    if not OUT_HTML.exists():
        print(f"  {OUT_HTML} not found - skipped embedding")
        return
    html = OUT_HTML.read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    block = f'{HTML_START}\n<script id="demo-data" type="application/json">{payload}</script>\n{HTML_END}'
    start, end = html.index(HTML_START), html.index(HTML_END) + len(HTML_END)
    OUT_HTML.write_text(html[:start] + block + html[end:], encoding="utf-8")
    print(f"  Embedded data in {OUT_HTML}")


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------

if __name__ == "__main__":
    started = time.time()
    df1 = load("01_data_processed.pkl")
    df2 = load("02_outlier_removed_chauvenets.pkl")
    df3 = load("03_data_features.pkl")

    data = {
        "meta": {
            "project": "Machine Learning Fitness Tracker",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": SEED,
            "synthetic_data": True,
            "data_note": "Results are computed from synthetic sample data, not real MetaMotion recordings.",
        }
    }

    log("Dataset summary and sensor signals")
    data["summary"] = dataset_summary(df1) if df1 is not None else None
    data["signals"] = sensor_signals(df1) if df1 is not None else None

    log("Outlier detection")
    data["outliers"] = outliers(df1, df2) if df1 is not None else None

    log("Rep counting")
    data["reps"] = rep_counting(df1) if df1 is not None else None

    log("Rep-counting replay (bench press)")
    data["replay"] = rep_replay(df1) if df1 is not None else None
    if data["replay"] and data["reps"]:
        # The replay must agree with the rep-counting results for the same set
        same = [x for x in data["reps"]["sets"] if x["set"] == data["replay"]["set"]]
        assert same and same[0]["reps_pred"] == data["replay"]["reps_counted"], "replay/rep count mismatch"

    data["models"] = models(df3) if df3 is not None else None

    data["missing"] = missing
    OUT_JSON.write_text(json.dumps(data, indent=1), encoding="utf-8")
    embed_in_html(data)

    log(f"Done in {time.time() - started:.0f}s -> {OUT_JSON}")
    if data["summary"]:
        s = data["summary"]
        print(f"  Dataset: {s['rows']} rows, {len(s['participants'])} participants, "
              f"{len(s['labels'])} labels, {s['sets']} sets")
    if data["outliers"]:
        print(f"  Outliers: IQR {sum(data['outliers']['iqr'].values())}, "
              f"Chauvenet {sum(data['outliers']['chauvenet'].values())}, LOF {data['outliers']['lof_rows']} rows")
    if data["models"]:
        b = data["models"]["best"]
        tied = data["models"]["tied_best"]
        print(f"  Best accuracy {b['accuracy']:.4f}, reached by {len(tied)} model/feature-set combination(s): "
              + ", ".join(f"{t['model']} on {t['feature_set']}" for t in tied))
        print(f"  RF Feature_set_4 confusion-matrix accuracy: {data['models']['confusion_matrix']['accuracy']:.4f}")
    if data["reps"]:
        print(f"  Rep counting MAE: {data['reps']['mae']}")
    if data["replay"]:
        rp = data["replay"]
        print(f"  Replay: {rp['label']} set {rp['set']} ({rp['participant']}, {rp['category']}) - "
              f"counted {rp['reps_counted']}, actual {rp['reps_actual']}, peaks at samples {rp['peaks']}")
    if missing:
        print("  Missing:", missing)
