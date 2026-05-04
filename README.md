# Project_3 - YouTube Content Monetization – Revenue Prediction

## Project Overview

This project builds an end-to-end machine learning pipeline to predict **YouTube ad revenue** based on video performance metrics such as views, engagement, and audience behavior.

It simulates a real-world analytics workflow used in digital content platforms to optimize monetization strategies.

---

## Objectives

* Predict `ad_revenue_usd` using video and engagement metrics
* Identify key factors influencing revenue
* Build a deployable ML model with a user interface

---

## Dataset Features

* Views, Likes, Comments
* Watch Time & Video Length
* Subscriber Count
* Category, Device, Country
* Target: **Ad Revenue (USD)**

---

## Tech Stack

* Python (Pandas, NumPy)
* Machine Learning (Scikit-learn)
* Visualization (Matplotlib)
* Deployment (Streamlit)

---

## Pipeline

### 1. Data Preprocessing

* Handled missing values
* Removed inconsistencies
* Standardized dataset

### 2. EDA

* EDA created using post cleaned data (youtube_ad_revenue_cleaned_afterisnull)
* Revenue vs (Likes, views, comments, subscribers, watch time)
* Correlation Heatmap
* Ad revenue vs (Category, Country, Device)

### 3. Feature Engineering

* Engagement Rate
* Like & Comment Rate
* Watch Time Efficiency
* Subscriber Engagement

### 4. Encoding

* One-hot encoding for categorical variables (text data into numerical format so machine learning models can understand it.)

### 5. Model Building

Tested multiple regression models:

* Linear Regression
* Ridge
* Lasso
* Decision Tree
* Random Forest

Best model selected based on performance metrics

---

## Model Evaluation

* R² Score
* RMSE
* MAE

---

## Key Insights

* Engagement metrics are the strongest drivers of revenue
* Watch time significantly impacts monetization
* Subscriber engagement influences performance
* Content category and device type affect revenue trends

---

## Deployment

Built an interactive web app using Streamlit:

Features:

* User input-based revenue prediction
* Real-time output
* Clean UI

---

## How to Run Locally
use file app.py and run it locally using bash

pip install -r requirements.txt

Command: streamlit run app.py
```

