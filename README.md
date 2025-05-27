# Titanic Survival Prediction

Predicting the survival of Titanic passengers using machine learning.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Objectives](#project-objectives)
- [Approach](#approach)
- [Features Used](#features-used)
- [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Project Overview

This project uses data science and machine learning to predict which passengers survived the Titanic disaster. It is a classic binary classification problem and a popular introduction to data science concepts. The project demonstrates the full pipeline: data cleaning, preprocessing, feature engineering, model training, and evaluation.

---

## Dataset

- **Source:** [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- **Files Used:**
  - `train.csv` — Training data with survival labels
  - `test.csv` — Test data for prediction
  - `gender_submission.csv` — Sample submission format

**Key Columns:**
- `PassengerId`: Unique identifier for each passenger
- `Survived`: Survival (1 = Yes, 0 = No)
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Sex`: Gender
- `Age`: Age in years
- `SibSp`: # of siblings/spouses aboard
- `Parch`: # of parents/children aboard
- `Fare`: Ticket fare
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---

## Project Objectives

- Predict whether a passenger survived the Titanic disaster.
- Apply data cleaning, preprocessing, and Naive Bayes classification.
- Evaluate model performance using appropriate metrics.

---

## Approach

1. **Data Cleaning:** Handle missing values and drop irrelevant columns.
2. **Feature Engineering:** Encode categorical variables for modeling.
3. **Modeling:** Train a Gaussian Naive Bayes classifier.
4. **Evaluation:** Assess model with accuracy, classification report, and confusion matrix.
5. **Prediction:** Generate predictions for the test set.

---

## Features Used

- Passenger Class (`Pclass`)
- Sex (`Sex`)
- Age (`Age`)
- Siblings/Spouses (`SibSp`)
- Parents/Children (`Parch`)
- Fare (`Fare`)
- Embarked Port (`Embarked`)

---

## Data Cleaning & Preprocessing

- Filled missing `Age` and `Fare` values with median.
- Filled missing `Embarked` values with mode.
- Dropped columns with little predictive value or high missingness: `Name`, `Ticket`, `Cabin`.
- Encoded `Sex` and `Embarked` as numeric.
- Combined train and test data for consistent encoding.

---

## Modeling

- **Algorithm:** Gaussian Naive Bayes (assumes features are independent and normally distributed).
- **Validation:** 80/20 train-validation split on the training data.
- **Metrics:** Accuracy, precision, recall, F1-score, and confusion matrix.

---

## Evaluation

- **Validation Accuracy:** ~75% (typical for Naive Bayes on this dataset)
- **Confusion Matrix:** Shows the number of correct and incorrect predictions for each class.
- **Classification Report:** Details precision, recall, and F1-score.

---

## How to Run

1. Clone the repository or download the code.
2. Upload the Titanic dataset files (`train.csv`, `test.csv`, `gender_submission.csv`) to your working directory or Colab environment.
3. Run the provided Jupyter notebook or Python script.
4. The code will output validation results and generate predictions for the test set.

---

## Results

- The Naive Bayes model achieved a validation accuracy of approximately 75%.
- The model provides a simple, interpretable baseline for Titanic survival prediction.
- Results can be further improved with advanced feature engineering and more complex models.

---

## Future Work

- Explore additional feature engineering (e.g., family size, title extraction).
- Try alternative models (Random Forest, Logistic Regression, XGBoost).
- Perform hyperparameter tuning and cross-validation.
- Visualize feature importance and survival correlations.

---

## Acknowledgments

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- Open-source contributors and the data science community
