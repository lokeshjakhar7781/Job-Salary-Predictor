# Job Salary Prediction | Machine Learning

<p align="center">
  <b>End-to-End Machine Learning Pipeline for Job Salary Prediction</b>
</p>

---

## Overview

This project implements an **end-to-end machine learning solution for job salary prediction** using a structured job salary dataset.

The system automatically preprocesses the data, trains a **Random Forest Regression model**, saves the trained model and preprocessing pipeline, and generates salary predictions for unseen data.

---

## Key Features

* Automated data preprocessing
* Missing-value handling
* Numerical feature scaling
* Categorical feature encoding
* Train/test splitting
* Random Forest regression
* Model and pipeline persistence
* Automated inference
* CSV-based prediction output

---

## Machine Learning Workflow

```text
                 ┌──────────────────────────┐
                 │   Job Salary Dataset     │
                 │                          │
                 │ job_salary_prediction_   │
                 │ dataset.csv              │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     Stratified Split     │
                 │      80% Train / 20%     │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    Data Preprocessing    │
                 │                          │
                 │ Numerical → Imputation   │
                 │            → Scaling     │
                 │                          │
                 │ Categorical → Imputation │
                 │            → One-Hot     │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     Random Forest        │
                 │       Regressor          │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Saved Model & Pipeline │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │       input.csv          │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     Salary Prediction    │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │       output.csv         │
                 └──────────────────────────┘
```

---

## Technologies Used

| Technology       | Purpose                      |
| ---------------- | ---------------------------- |
| Random Forest    | Regression model             |
| Pipeline         | Automated preprocessing      |
| Standard Scaler  | Numerical feature scaling    |
| One-Hot Encoding | Categorical feature encoding |
| Joblib           | Model persistence            |
| Pandas           | Data processing              |
| NumPy            | Numerical operations         |
| CSV              | Data input and output        |

---

## Project Structure

```text
Job-Salary-Prediction/
│
├── main(2).py
├── job_salary_prediction_dataset.csv
├── input(2).csv
├── output(2).csv
├── model.pkl
├── pipeline.pkl
└── README.md
```

### File Description

**`main(2).py`**
Contains the complete training and inference workflow.

**`job_salary_prediction_dataset.csv`**
Main dataset used to train the salary prediction model.

**`input(2).csv`**
Input dataset used for generating salary predictions.

**`output(2).csv`**
Contains the generated salary predictions.

**`model.pkl`**
Saved Random Forest regression model.

**`pipeline.pkl`**
Saved preprocessing pipeline.

---

## Preprocessing

### Numerical Features

```text
Missing Values
      ↓
Median Imputation
      ↓
Standard Scaling
```

### Categorical Features

```text
Missing Values
      ↓
Most-Frequent Imputation
      ↓
One-Hot Encoding
```

The numerical and categorical preprocessing pipelines are combined using a **ColumnTransformer** and applied consistently during model training and inference.

---

## Model

The project uses a:

```text
Random Forest Regressor
```

The target variable is:

```text
salary
```

The model is trained using the preprocessed training data and saved using Joblib for future predictions.

---

## Data Splitting

The dataset is divided into **80% training data and 20% test/input data** using a stratified split based on `experience_years`.

A fixed `random_state=42` is used to make the data split reproducible.

---

## Getting Started

### Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib
```

### Run the Project

```bash
python "main(2).py"
```

---

## How It Works

### First Run

If `model.pkl` does not exist:

1. Load the job salary dataset.
2. Create the train/test split.
3. Separate features and target.
4. Identify numerical and categorical features.
5. Build the preprocessing pipeline.
6. Transform the training data.
7. Train the Random Forest Regressor.
8. Save the model and preprocessing pipeline.

### Subsequent Runs

If the model already exists:

1. Load the saved model.
2. Load the preprocessing pipeline.
3. Read the input data.
4. Transform the input features.
5. Generate salary predictions.
6. Save the predictions to the output file.

---

## Prediction Output

The generated output contains:

```text
salary
```

Each row represents the predicted salary corresponding to a record from the input dataset.

---

## Project Goal

The goal of this project is to demonstrate a complete **regression workflow** that transforms structured job-related data into an automated salary prediction system.

```text
Raw Data
   ↓
Train/Test Split
   ↓
Preprocessing
   ↓
Feature Transformation
   ↓
Random Forest Regressor
   ↓
Salary Prediction
```

---

<p align="center">
  <b>From Data to Intelligent Salary Prediction</b>
</p>