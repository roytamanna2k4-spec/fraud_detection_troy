Overview

This project focuses on building a machine learning model to detect fraudulent credit card transactions from a highly imbalanced dataset. The goal is to accurately identify fraud cases while minimizing false negatives, ensuring better financial security.

🚀 Features
End-to-end ML pipeline for fraud detection
Handles highly imbalanced data using SMOTE
Data preprocessing (scaling, cleaning, splitting)
Model training and comparison
Evaluation using multiple performance metrics
📊 Dataset
Source: Kaggle Credit Card Dataset
Total transactions: ~284,000
Fraud cases: ~0.17% (highly imbalanced)
⚙️ Tech Stack
Python
Pandas, NumPy
Scikit-learn
Imbalanced-learn (SMOTE)
Matplotlib, Seaborn
🧠 Approach
1. Data Preprocessing
Handled missing values and cleaned dataset
Applied feature scaling using StandardScaler
Performed stratified train-test split
2. Handling Imbalance
Used SMOTE (Synthetic Minority Oversampling Technique)
Improved detection of minority (fraud) class
3. Model Training
Logistic Regression
Random Forest (better performance)
4. Evaluation Metrics
Precision
Recall (focus metric)
F1-score
ROC-AUC
📈 Results
Improved fraud detection using SMOTE
Random Forest outperformed Logistic Regression
Higher recall achieved → reduced false negatives
📌 Key Learnings
Handling imbalanced datasets effectively
Importance of recall in fraud detection
Model evaluation beyond accuracy
Building real-world ML pipelines
