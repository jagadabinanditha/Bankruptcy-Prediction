# BANKRUPTCY PREDICTION
## Using Decision Trees and Random Forest with Precision–Recall Analysis

### Machine Learning Project

**Degree:** B.Tech CSE  
**Academic Year:** 2026–27

**Team Members:**
- J. Nanditha — 2520030013
- K. Srithika — 2520030361

**Guide:** Mr. Shaik Asif, Assistant Professor

---

# ABSTRACT

Bankruptcy prediction is an important problem in financial analysis because early identification of financially distressed companies can support better decision-making. Traditional financial analysis may require significant time and may not always capture complex relationships among financial indicators.

This project develops a supervised machine learning approach for predicting whether a company is likely to be bankrupt or non-bankrupt using financial indicators. The project uses the Taiwanese Bankruptcy Prediction dataset obtained from the UCI Machine Learning Repository.

Two classification algorithms, Decision Tree and Random Forest, are implemented and compared. Exploratory Data Analysis (EDA) is first performed to understand the dataset, identify missing values and duplicate records, examine class distribution, analyze descriptive statistics, and study relationships between financial indicators and the bankruptcy target.

Since the dataset contains an imbalanced distribution between bankrupt and non-bankrupt companies, evaluation focuses on Precision, Recall, F1-Score, Confusion Matrix, Precision–Recall Curve, and Average Precision rather than relying only on accuracy.

The project also analyzes feature importance to identify financial indicators that contribute to bankruptcy prediction. The overall objective is to develop an interpretable and useful machine learning approach that can support financial decision-making.

---

# 1. INTRODUCTION

Bankruptcy occurs when a company is unable to meet its financial obligations. Predicting bankruptcy in advance can help organizations, investors, financial institutions, and other stakeholders make informed decisions.

Financial datasets contain several indicators related to profitability, liquidity, leverage, and other aspects of a company's financial condition. Machine learning algorithms can analyze these indicators and learn patterns associated with bankruptcy.

In this project, Decision Tree and Random Forest classification algorithms are used to predict bankruptcy.

---

# 2. PROBLEM STATEMENT

Companies experiencing declining profitability, reduced liquidity, and difficulty meeting financial obligations may eventually face bankruptcy.

Traditional financial analysis can be time-consuming and may fail to identify complex relationships among multiple financial indicators.

Therefore, there is a need for a machine learning-based classification approach that can analyze financial indicators and predict whether a company is bankrupt or non-bankrupt.

---

# 3. OBJECTIVES

The main objectives of this project are:

- To predict whether a company is likely to be bankrupt.
- To perform Exploratory Data Analysis on financial indicators.
- To implement a Decision Tree classifier.
- To implement a Random Forest classifier.
- To compare the performance of the two models.
- To analyze the effect of class imbalance.
- To evaluate the models using Precision, Recall, F1-Score, Confusion Matrix, and Precision–Recall analysis.
- To identify important financial features related to bankruptcy.
- To provide an interpretable machine learning approach for bankruptcy prediction.

---

# 4. DATASET

The project uses the **Taiwanese Bankruptcy Prediction** dataset from the **UCI Machine Learning Repository**.

### Dataset Information

| Property | Details |
|---|---|
| Dataset | Taiwanese Bankruptcy Prediction |
| Source | UCI Machine Learning Repository |
| Number of Instances | 6,819 |
| Financial Features | 95 |
| Total Columns | 96 |
| Target Variable | `Bankrupt?` |
| Problem Type | Binary Classification |
| Data Period | 1999–2009 |
| Region | Taiwan |

The target variable is:

- `0` → Non-Bankrupt
- `1` → Bankrupt

The dataset contains financial indicators describing companies.

---

# 5. EXPLORATORY DATA ANALYSIS

Exploratory Data Analysis is performed before model training to understand the structure and characteristics of the dataset.

The EDA includes:

- Dataset dimensions
- First five records
- Column names
- Data types
- Missing-value analysis
- Duplicate-row analysis
- Descriptive statistics
- Target-variable distribution
- Class-distribution visualization
- Constant-feature analysis
- Feature–target correlation analysis
- Correlation heatmap
- Boxplot analysis

EDA also helps identify the class imbalance present in the dataset.

---

# 6. PROJECT WORKFLOW

```text
Dataset Collection
        ↓
Data Loading
        ↓
Exploratory Data Analysis
        ↓
Data Preprocessing
        ↓
Train-Test Split
        ↓
Decision Tree
        ↓
Random Forest
        ↓
Model Evaluation
        ↓
Precision–Recall Analysis
        ↓
Feature Importance
        ↓
Model Comparison
