# Credit Score Classification

## Project Overview

This project develops a machine learning model to classify individuals into predefined credit score categories based on various financial and demographic features. The goal is to create a reliable tool for financial institutions to assess customer creditworthiness efficiently.

## Table of Contents

- [Dataset](#dataset)
- [Features](#features)
- [Target Variable](#target-variable)
- [Data Preprocessing](#data-preprocessing)
- [Model Development](#model-development)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Dataset

The dataset contains customer financial information with various features that influence credit scoring decisions. The data includes both numerical and categorical variables with some missing values and data quality issues that require preprocessing.

## Features

### Numerical Features
- **Age**: Customer age
- **Annual_Income**: Total yearly income
- **Monthly_Inhand_Salary**: Monthly salary after deductions
- **Num_Bank_Accounts**: Number of bank accounts
- **Num_Credit_Card**: Number of credit cards
- **Interest_Rate**: Applicable interest rate on loans/credit cards
- **Num_of_Loan**: Total number of loans
- **Delay_from_due_date**: Average payment delay in days
- **Num_of_Delayed_Payment**: Number of delayed payments
- **Changed_Credit_Limit**: Changes in credit limits
- **Num_Credit_Inquiries**: Number of credit report inquiries
- **Outstanding_Debt**: Total unpaid debt
- **Credit_Utilization_Ratio**: Percentage of credit used
- **Credit_History_Age**: Age of oldest credit account
- **Total_EMI_per_month**: Total monthly EMI payments
- **Amount_invested_monthly**: Monthly investment amount
- **Monthly_Balance**: Remaining balance after monthly expenses

### Categorical Features
- **Occupation**: Customer occupation
- **Type_of_Loan**: Types of loans (Auto, Personal, Student, etc.)
- **Credit_Mix**: Variety of credit types
- **Payment_of_Min_Amount**: Whether customer pays minimum amount
- **Payment_Behaviour**: General payment behavior patterns

## Target Variable

The target variable categorizes credit scores into three classes:
- **Good**: High credit score, low default risk
- **Standard**: Medium credit score, moderate risk
- **Poor**: Low credit score, high default risk

## Data Preprocessing

### 1. Data Cleaning
- Removed unwanted symbols and characters (underscores, special characters)
- Converted string representations to appropriate data types
- Handled unrealistic values (e.g., negative ages, extreme outliers)

### 2. Missing Value Treatment
- **Categorical Variables**: Random filling based on distribution
- **Numerical Variables**: Median imputation, group-wise filling
- **Type_of_Loan**: Converted string format to list format
- **Credit_History_Age**: Converted to numeric months

### 3. Outlier Handling
- Applied IQR method with 1.5x threshold for most features
- Used 2x threshold for income-related features
- Capped extreme outliers to reduce skewness

### 4. Feature Engineering
- **One-Hot Encoding**: Occupation, Payment_Behaviour
- **Ordinal Encoding**: Credit_Mix (Bad < Standard < Good)
- **Label Encoding**: Credit_Score, Payment_of_Min_Amount
- **Multi-Label Binarization**: Type_of_Loan (multiple loan types per customer)

### 5. Feature Selection
- Removed highly correlated features to reduce multicollinearity:
  - Monthly_Inhand_Salary
  - Num_Credit_Card
  - Delay_from_due_date
  - Outstanding_Debt

## Model Development

### Models Tested
1. **Random Forest Classifier**
2. **Logistic Regression** (with StandardScaler)
3. **XGBoost Classifier**

### Hyperparameter Tuning
Performed RandomizedSearchCV for Random Forest (best performing model):

#### Random Forest Parameters
- n_estimators: [100, 200, 300, 400, 500]
- max_depth: [10, 15, 20, 25, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]
- bootstrap: [True, False]
- criterion: ['gini', 'entropy']

## Results

### Model Performance Comparison

| Model | Accuracy |
|-------|----------|
| **Random Forest** | **76.98%** |
| XGBoost | 73.04% |
| Logistic Regression | 61.47% |

### Best Model: Tuned Random Forest Classifier
- **Test Accuracy**: 76.98%
- **F1 Score (weighted)**: 78.00%

The Random Forest model was selected as the best performer and underwent hyperparameter tuning using RandomizedSearchCV. The tuned model showed strong performance in classifying credit scores across all three categories (Good, Standard, Poor).

**Evaluation Metrics Used:**
- **Accuracy Score**
- **F1 Score (weighted)**
- **Classification Report**
- **Confusion Matrix**
- **Feature Importance Analysis**

Feature importance analysis revealed the most influential factors in credit scoring decisions, providing valuable insights for financial institutions.


### Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import ast
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
```

## Usage

1. **Load the data**:
   ```python
   train = pd.read_csv("train.csv")
   ```

2. **Run the preprocessing pipeline**:
   - Data cleaning
   - Missing value imputation
   - Outlier handling
   - Feature encoding

3. **Train and evaluate models**:
   - Split data into train/test sets
   - Train multiple models
   - Perform hyperparameter tuning
   - Evaluate performance

4. **Analyze results**:
   - Compare model performance
   - Examine feature importance
   - Generate visualizations



## Key Insights

1. **Data Quality**: The original dataset contained significant data quality issues requiring extensive preprocessing
2. **Feature Engineering**: Converting loan types to multi-label format and properly encoding categorical variables was crucial
3. **Outlier Impact**: Extreme outliers heavily skewed distributions, requiring careful handling
4. **Model Performance**: Random Forest emerged as the best performer with 76.98% accuracy and 78.00% F1 score
5. **Feature Importance**: Payment behavior, credit utilization, and income-related features were most predictive of credit scores

## Future Improvements

- Implement ensemble methods combining multiple models
- Explore deep learning approaches
- Add more sophisticated feature engineering
- Implement real-time scoring capabilities
- Add model interpretability tools (SHAP, LIME)

