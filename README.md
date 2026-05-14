# Machine Learning Fitness Tracker

A machine learning pipeline that processes MetaMotion accelerometer and gyroscope data to classify barbell exercises from real workout sessions.

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
│   │   └── make_dataset.py
│   ├── features/
│   │   ├── remove_outliers.py
│   │   └── build_features.py
│   ├── models/
│   │   └── train_model.py
│   └── visualization/
│       └── visualize.py
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

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy ipython
```

Run the pipeline step by step:

```bash
python src/data/make_dataset.py
python src/features/remove_outliers.py
python src/features/build_features.py
python src/models/train_model.py
```

## Key Results

- Converted raw MetaMotion accelerometer and gyroscope files into a structured machine learning dataset
- Standardized 12.5Hz and 25Hz sensor streams into a consistent 5Hz time-series format
- Improved sensor feature quality using filtering, PCA, temporal features, frequency analysis, and clustering
- Compared multiple machine learning models for barbell exercise classification

## Notes

The raw sensor dataset is not included in this repository because it may contain large files and participant-specific workout data. This repository focuses on the project pipeline, source code, and machine learning workflow.

## Future Improvements

- Add a `requirements.txt` file for easier setup
- Add sample figures to the README, such as sensor visualizations and model performance charts
- Refactor exploratory code into reusable functions
- Add a repetition-counting module if the algorithm is included in the final project version
