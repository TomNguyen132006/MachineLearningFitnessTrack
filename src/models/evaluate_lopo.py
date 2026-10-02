"""Leave-one-participant-out (LOPO) evaluation of the exercise classifier.

Trains the random forest on Feature_set_4 (the model and features train_model.py uses for
its participant test) once per participant, holding that participant out as the test set,
and reports each accuracy and the mean over all participants. The original paper reports
this mean (85.43%) in Section 6.5.

Run build_features.py first. Usage (from any directory):
    python src/models/evaluate_lopo.py
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Feature_set_4 is built with set(), whose order depends on Python's hash seed.
# Re-run with a fixed PYTHONHASHSEED so the column order, and the results, repeat exactly.
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit(subprocess.run([sys.executable, *sys.argv], env={**os.environ, "PYTHONHASHSEED": "0"}).returncode)

from LearningAlgorithms import ClassificationAlgorithms  # noqa: E402

# Resolve data paths relative to the repo root so the script runs from any directory
ROOT = Path(__file__).resolve().parents[2]

# The random forest in LearningAlgorithms has no random_state, so it draws from numpy's
# global RNG. Seeding it (same seed as demo/export_demo_data.py) makes runs repeatable.
np.random.seed(42)

df = pd.read_pickle(ROOT / "data/interim/03_data_features.pkl")

basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
freq_features = [f for f in df.columns if ("_freq" in f) or ("_pse" in f)]
cluster_features = ["cluster"]
feature_set_4 = list(set(basic_features + freq_features + cluster_features))

learner = ClassificationAlgorithms()
accuracies = {}
for participant in sorted(df["participant"].unique()):
    train = df[df["participant"] != participant]
    test = df[df["participant"] == participant]
    _, pred, _, _ = learner.random_forest(
        train[feature_set_4], train["label"], test[feature_set_4], gridsearch=True
    )
    accuracies[participant] = accuracy_score(test["label"], pred)
    print(f"Participant {participant} held out: {accuracies[participant]:.4f} ({len(test)} test rows)", flush=True)

print(f"LOPO mean accuracy over {len(accuracies)} participants: {np.mean(list(accuracies.values())):.4f}")
