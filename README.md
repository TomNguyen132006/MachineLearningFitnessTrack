🏋️ Machine Learning Fitness Tracker

A machine learning-based fitness tracking system that transforms raw MetaMotion sensor data into structured insights for exercise classification and performance analysis.

🚀 Project Overview

Machine Learning Fitness Tracker is a data-driven application that processes wearable sensor data (accelerometer and gyroscope) to classify exercises and analyze workout patterns.

The system focuses on building a complete pipeline — from raw data ingestion to preprocessing, feature engineering, and preparation for machine learning modeling.

💡 Business Purpose

Most fitness tracking systems simply record raw activity data without meaningful interpretation. This project addresses that gap by converting motion sensor data into structured, actionable insights that can support:

Exercise recognition
Repetition tracking
Performance analysis
Data-driven fitness decisions
⚙️ Tech Stack
Layer	Technology
Language	Python
Data Processing	Pandas, NumPy
Machine Learning	Scikit-learn
Visualization	Matplotlib, Seaborn
Data Source	MetaMotion Sensor Data
🧠 Architecture Overview
Raw Sensor Data (Accelerometer + Gyroscope)
                ↓
Data Cleaning & Preprocessing
                ↓
Feature Engineering (Labeling, Metadata Extraction)
                ↓
Resampling & Signal Normalization
                ↓
Structured Time-Series Dataset
                ↓
Machine Learning Model (Classification / Prediction)
                ↓
Insights & Performance Analysis
📊 Data Pipeline & Processing

This project processes raw MetaMotion sensor data collected from multiple participants performing different exercises.

Data Characteristics
📡 Accelerometer sampling rate: 12.5 Hz
📡 Gyroscope sampling rate: 25 Hz
📁 Multiple CSV files per exercise and participant
🏷️ Labels extracted directly from filenames (exercise type, category, participant)
Data Processing Steps
Merged accelerometer and gyroscope datasets into a unified structure
Converted timestamps from epoch milliseconds into datetime format
Removed redundant columns (time, elapsed, raw epoch values)
Added structured metadata:
participant
exercise label
category
set number
Resampled data to 200 ms intervals (5 Hz)
Aggregated signals using mean values to reduce noise

👉 Result:
Raw multi-frequency sensor data (12.5–25 Hz) → Clean, consistent 5 Hz time-series dataset

🔄 Transformation

BEFORE
Raw sensor data existed as separate CSV files with inconsistent sampling rates and no structured labeling, making it difficult to analyze or use for machine learning.

AFTER
Built a preprocessing pipeline that merges, cleans, labels, and resamples multi-frequency sensor data into a unified 5 Hz time-series dataset ready for machine learning modeling.

📈 Impact
Built an end-to-end data processing pipeline for multi-source sensor data
Reduced data inconsistency by standardizing sampling rates from 12.5–25 Hz → 5 Hz
Improved signal stability through aggregation and resampling
Automated feature extraction from filenames (participant, exercise type, category)
Enabled downstream machine learning tasks such as exercise classification and repetition detection
🔥 Key Features
📊 Time-series sensor data processing
🔄 Multi-source data merging (accelerometer + gyroscope)
🧠 Feature engineering from raw data and metadata
⚡ Frequency normalization and noise reduction
📈 Data visualization and analysis
🤖 Machine learning-ready dataset generation
📌 Future Improvements
Train and evaluate classification models (accuracy, precision, recall)
Add real-time fitness tracking support
Build a dashboard for visualization and insights
Implement repetition counting using signal peaks
Integrate wearable devices for live data streaming
