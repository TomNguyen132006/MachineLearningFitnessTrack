# Machine Learning Fitness Tracker

A machine learning pipeline that processes MetaMotion accelerometer and gyroscope data to classify barbell exercises from real workout sessions.

> **Note:** This repo uses synthetic sample data; accuracy and rep-count results do not reflect real-world performance. The real MetaMotion recordings are not included, so `src/data/generate_synthetic_data.py` creates a stand-in dataset with the same file names and columns.

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
│   │   └── train_model.py
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

2. Generate the synthetic data (320 CSV files in `data/raw/MetaMotion/`, identical on every run). To use the real MetaMotion recordings instead, put them in `data/raw/MetaMotion/` and skip this step.

```bash
python src/data/generate_synthetic_data.py
```

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
```

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

The real MetaMotion dataset is not included in this repository because it may contain large files and participant-specific workout data. The included generator creates synthetic sample data with the same structure so the full pipeline can run; results on that data do not reflect real-world performance.

## Future Improvements

- Add sample figures to the README, such as sensor visualizations and model performance charts
- Refactor exploratory code into reusable functions
