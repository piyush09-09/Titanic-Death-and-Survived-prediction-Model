# Titanic - Machine Learning From Disaster (Ensemble Model)

This repository contains my complete machine learning pipeline for solving the Titanic survival prediction problem from Kaggle. The model uses ensemble learning (Random Forest, SVM, KNN) to improve prediction accuracy.

## Problem Statement

The goal of the project is to build a classifier that predicts whether a passenger survived the Titanic shipwreck based on features such as age, gender, class, etc.

## What I Did

- Data cleaning and preprocessing (handling missing values, encoding, scaling)
- Built and tuned three models: Random Forest, SVM, and KNN
- Evaluated each model using accuracy, precision, recall, and F1-score
- Visualized learning curves to analyze overfitting/underfitting
- Used VotingClassifier (soft voting) for ensembling
- Final prediction and CSV submission for Kaggle

## Models Used

| Model                | Accuracy | Notes                                 |
|---------------------|----------|----------------------------------------|
| Random Forest        | ~79%     | Tuned max_depth=6 to reduce overfitting |
| SVC (RBF kernel)     | ~80%     | Scaled features for better performance |
| K-Nearest Neighbors  | ~79%     | Used StandardScaler and optimal k      |
| Ensembled Model      | ~80%     | Soft Voting Classifier                 |

## Data Preprocessing

- Dropped columns: Cabin (too many missing), Ticket (not useful)
- Imputed missing values:
  - Age: with median
  - Embarked: dropped 2 missing rows
- Categorical encoding: Sex, Embarked, Pclass
- Scaled features using StandardScaler (for SVM and KNN)

## Learning Curve

Visualized learning curves for individual and ensemble models to evaluate:
- Underfitting (low training and validation accuracy)
- Overfitting (high training accuracy, low validation accuracy)

## Files in the Repository

| File             | Description                             |
|------------------|------------------------------------------|
| train.csv        | Provided by Kaggle (training dataset)    |
| test.csv         | Provided by Kaggle (test dataset)        |
| submission.csv   | Final output CSV for Kaggle submission   |
| notebook.ipynb   | Full Jupyter Notebook (EDA to modeling)  |
| README.md        | This file                                |

## How to Run

1. Clone the repository
2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
