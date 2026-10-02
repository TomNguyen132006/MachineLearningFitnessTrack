# Machine Learning Fitness Tracker

A machine learning pipeline that processes MetaMotion accelerometer and gyroscope data to classify barbell exercises from real workout sessions.

> **Data and credit:** The MetaMotion recordings, the original approach, and the exercise illustration below come from **Dave Ebbelaar**: his [*Full Machine Learning Project: Coding a Fitness Tracker with Python*](https://www.youtube.com/playlist?list=PL-Y17yukoyy0sT2hoSQxn1TdV0J7-MX4K) tutorial and his repo [daveebbelaar/tracking-barbell-exercises](https://github.com/daveebbelaar/tracking-barbell-exercises). The data is not included here; download it from his repo (see [How to Run](#how-to-run)).

![Barbell exercises: bench press, deadlift, overhead press, barbell row, squat](https://raw.githubusercontent.com/daveebbelaar/tracking-barbell-exercises/master/PythonCode/images/barbell_exercises.png)

*Exercise illustration from [Dave Ebbelaar's repo](https://github.com/daveebbelaar/tracking-barbell-exercises/blob/master/PythonCode/images/barbell_exercises.png).*

## Overview

This project converts raw wearable sensor data into a clean, structured dataset for machine learning. The pipeline includes data preprocessing, time-series resampling, visualization, outlier detection, feature engineering, and exercise classification.

The goal of this project is to explore how motion sensor data can be used to recognize workout movement patterns and support data-driven fitness tracking.

## Features

- Processes raw accelerometer and gyroscope CSV files from MetaMotion sensors
- Extracts participant, exercise label, category, and set information from file names
- Converts multi-frequency sensor data into a consistent 5Hz time-series dataset
- Visualizes accelerometer and gyroscope movement patterns across exercises and participants
- Applies outlier detection using IQR, Chauvenet’s criterion, and Local Outlier Factor
- Builds sensor features using low-pass filtering, PCA, temporal abstraction, frequency analysis, and clustering
- Trains and compares multiple classification models for exercise recognition

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Scientific Computing | SciPy |
| Sensor Data | MetaMotion Accelerometer and Gyroscope |

## Project Structure

```text
MachineLearningFitnessTrack/
├── src/
│   ├── data/
│   │   ├── generate_synthetic_data.py   # creates the stand-in dataset
│   │   └── make_dataset.py
│   ├── features/
│   │   ├── remove_outliers.py
│   │   ├── build_features.py
│   │   └── count_repetitions.py
│   ├── models/
│   │   ├── train_model.py
│   │   └── evaluate_lopo.py             # leave-one-participant-out accuracy
│   └── visualization/
│       └── visualize.py
├── demo/
│   ├── export_demo_data.py              # exports pipeline results for the demo page
│   ├── demo_data.json
│   └── index.html                       # single-file demo page
├── requirements.txt
└── README.md
```

## Pipeline

```text
Raw MetaMotion CSV Files
        ↓
Data Cleaning and Preprocessing
        ↓
Resampling to 5Hz Time-Series Data
        ↓
Outlier Detection
        ↓
Feature Engineering
        ↓
Model Training and Evaluation
        ↓
Exercise Classification
```

## Main Workflow

### 1. Data Processing

The project starts by reading raw MetaMotion CSV files from accelerometer and gyroscope sensors. The script extracts useful metadata from file names, including participant, exercise label, workout category, and set number.

The accelerometer and gyroscope streams are then merged and resampled into a consistent 5Hz time-series dataset.

### 2. Visualization

Sensor signals are visualized to compare movement patterns across exercises and participants. This helps identify how different barbell exercises produce different accelerometer and gyroscope patterns.

### 3. Outlier Detection

The project explores several outlier detection methods, including:

- Interquartile Range (IQR)
- Chauvenet’s criterion
- Local Outlier Factor (LOF)

Outlier handling helps reduce noise and improve the quality of the dataset before feature engineering and modeling.

### 4. Feature Engineering

The project creates additional features from the raw sensor signals, including:

- Low-pass filtered sensor signals
- Principal Component Analysis (PCA) features
- Magnitude features from accelerometer and gyroscope axes
- Temporal mean and standard deviation features
- Frequency-domain features using Fourier transformation
- KMeans cluster labels based on sensor movement patterns

### 5. Model Training

Multiple classification models are trained and compared to classify barbell exercises, including:

- Random Forest
- K-Nearest Neighbors
- Decision Tree
- Neural Network
- Naive Bayes

Model performance is evaluated using accuracy scores and confusion matrices.

## How to Run

Tested with Python 3.12. All scripts resolve paths from the project root, so run them from the project root (or anywhere else) in this order.

1. Create a virtual environment and install the pinned packages:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

2. Download the MetaMotion data from Dave Ebbelaar's repo and copy the 187 CSV files in `PythonCode/data/` (not the `form/` subfolder) into `data/raw/MetaMotion/`:

```bash
git clone -c core.longpaths=true https://github.com/daveebbelaar/tracking-barbell-exercises
mkdir -p data/raw/MetaMotion
cp tracking-barbell-exercises/PythonCode/data/*.csv data/raw/MetaMotion/
```

`core.longpaths=true` is needed on Windows because some file names are long. `data/` is in `.gitignore`, so the files stay out of this repo.

Without the real data, `python src/data/generate_synthetic_data.py` creates a synthetic stand-in with the same file names and columns. Results on it do not reflect real-world performance. If you use it, set `"data_source"` to `"synthetic"` in `demo/export_demo_data.py` so the demo page shows a warning.

3. Run the pipeline:

```bash
python src/data/make_dataset.py          # resample to 5 Hz -> data/interim/01_data_processed.pkl
python src/features/remove_outliers.py   # Chauvenet outlier removal -> 02_outlier_removed_chauvenets.pkl
python src/features/build_features.py    # filtering, PCA, temporal/frequency features, clusters -> 03_data_features.pkl
python src/models/train_model.py         # trains and compares the classifiers (slow, see below)
```

4. Optional scripts:

```bash
python src/visualization/visualize.py      # saves sensor plots to reports/figures/
python src/features/count_repetitions.py   # counts reps per set and prints the mean absolute error
python src/models/evaluate_lopo.py         # leave-one-participant-out accuracy (takes a few minutes)
```

`evaluate_lopo.py` trains the random forest on Feature_set_4 five times, each time holding out one participant as the test set, and prints each accuracy and the mean. It tests how well the classifier works on a person it has never seen. On the real data the mean is 94.85% (participants A–E: 99.3%, 87.4%, 87.8%, 100%, 99.7%); the original paper reports 85.43% with its own features and model. Results repeat exactly because the script fixes the random seed and Python's hash seed.

The scripts open plot windows as they run. To run them without windows (for example on a server), set `MPLBACKEND=Agg` first.

`train_model.py` runs forward feature selection and several grid searches, so it takes about 20 minutes.

## Demo

`demo/index.html` is a single-page demo of the results: a replay of a recorded bench press set with live rep counting, then the dataset, sensor signals, outlier detection, model comparison, confusion matrix and rep counting results.

To open it, double-click `demo/index.html`. No server is needed, but the charts load Chart.js from a CDN, so an internet connection is required.

To regenerate its data after running steps 1–3:

```bash
python demo/export_demo_data.py
```

This recomputes every number from the pickles in `data/interim/`, writes `demo/demo_data.json`, and embeds the same data in `demo/index.html`. It retrains the models, so it takes about as long as `train_model.py`. Results are repeatable: the script fixes the random seed and Python's hash seed.

## Key Results

- Converted raw MetaMotion accelerometer and gyroscope files into a structured machine learning dataset
- Standardized 12.5Hz and 25Hz sensor streams into a consistent 5Hz time-series format
- Improved sensor feature quality using filtering, PCA, temporal features, frequency analysis, and clustering
- Compared multiple machine learning models for barbell exercise classification

## Notes

The MetaMotion dataset is not included in this repository. Download it from [Dave Ebbelaar's repo](https://github.com/daveebbelaar/tracking-barbell-exercises) as described in [How to Run](#how-to-run).

## Credits

- **Data:** the MetaMotion accelerometer and gyroscope recordings (5 participants, barbell exercises) were collected by Dave Ebbelaar and are published in [daveebbelaar/tracking-barbell-exercises](https://github.com/daveebbelaar/tracking-barbell-exercises).
- **Original approach:** the pipeline (5 Hz resampling, Chauvenet outlier removal, low-pass filtering, PCA, temporal and frequency features, clustering, model comparison, peak-based rep counting) follows Dave Ebbelaar's [*Full Machine Learning Project: Coding a Fitness Tracker with Python*](https://www.youtube.com/playlist?list=PL-Y17yukoyy0sT2hoSQxn1TdV0J7-MX4K) tutorial and his paper *Exploring the Possibilities of Context Aware Applications for Strength Training*, included in his repo.
- **Exercise illustration:** [`barbell_exercises.png`](https://github.com/daveebbelaar/tracking-barbell-exercises/blob/master/PythonCode/images/barbell_exercises.png) from his repo. The animated bench-press figure on the demo page was drawn for this project.

## Future Improvements

- Add sample figures to the README, such as sensor visualizations and model performance charts
- Refactor exploratory code into reusable functions
