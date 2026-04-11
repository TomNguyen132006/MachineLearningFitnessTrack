# 🏋️ Machine Learning Fitness Tracker

A machine learning fitness tracking system that converts raw MetaMotion sensor data into structured insights for exercise classification and performance analysis.

---

## 🚀 Project Overview

This project builds a complete data pipeline to process wearable sensor data (accelerometer and gyroscope) and prepare it for machine learning.

It focuses on transforming raw signals into clean, structured data that can be used to analyze workout patterns and classify exercises.

---

## 💡 Purpose

Most fitness apps only record data without meaningful analysis.

This project turns raw motion data into useful insights to support:
- Exercise recognition
- Repetition tracking
- Performance analysis
- Data-driven fitness decisions

---

## 📊 Results

- Successfully transformed raw MetaMotion sensor data into a structured time-series dataset ready for machine learning  
- Standardized multi-frequency sensor streams (**12.5 Hz and 25 Hz**) into a consistent **5 Hz dataset**, improving data consistency and usability  
- Reduced noise and variability in sensor signals through resampling and aggregation (mean filtering)  
- Automated extraction of key features such as participant, exercise type, and category from raw file data  
- Built a scalable preprocessing pipeline that supports downstream tasks like exercise classification and repetition detection  

---

## 🚀 Key Outcomes

- 📡 Converted unstructured sensor data → clean ML-ready dataset  
- ⚡ Unified inconsistent data streams → stable time-series signals  
- 🧠 Enabled machine learning workflow → classification & pattern detection  
- 🔄 Reduced manual data preparation → automated pipeline  

---

## 🔥 Highlighted Result (Portfolio Style)

> Improved data reliability and consistency by converting multi-source sensor data (12.5–25 Hz) into a unified 5 Hz time-series dataset, enabling efficient machine learning analysis and feature extraction.

## ⚙️ Tech Stack

| Layer | Technology |
|------|------------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Data Source | MetaMotion Sensors |

---

## 🧠 Architecture

```text
Raw Sensor Data
      ↓
Cleaning & Preprocessing
      ↓
Feature Engineering
      ↓
Resampling (5 Hz)
      ↓
ML-Ready Dataset
      ↓
Modeling & Insights
