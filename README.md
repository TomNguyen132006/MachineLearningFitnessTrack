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
