# Diabetes Prediction using Logistic Regression

An internship project that builds a custom diabetes dataset, performs exploratory data analysis, trains a Logistic Regression model, and evaluates it using accuracy, confusion matrix, classification report, and ROC-AUC score.

## Project Overview

- Generates a custom synthetic dataset of 650 patient records (not a downloaded dataset)
- Engineers BMI from height and weight, dropping raw values before training
- Trains a Logistic Regression model using Age, Blood Pressure, Glucose, Family History, and BMI
- Creates visualizations for data distribution, correlation, glucose comparison, confusion matrix, and ROC curve
- Predicts diabetes risk for a sample new patient

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure

```text
diabetes-prediction-logistic-regression/
├── main_dp.py
├── requirements.txt
├── generate_dataset.py
├── diabetes_model.py
├── diabetes.csv
├── diabetes.xlsx
├── chart1_outcome_count.png
├── chart2_feature_distributions.png
├── chart3_correlation_heatmap.png
├── chart4_glucose_vs_outcome.png
├── chart5_confusion_matrix.png
├── chart6_roc_curve.png
├── Diabetes_Prediction_Presentation.pptx
└── README.md
```

## How to Run

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full workflow:

```bash
python main_dp.py
```

This will generate the dataset, train the model, save all chart images, and print th
e evaluation metrics and a sample prediction in the terminal.

## Dataset Design

The dataset was custom-built rather than downloaded, using a fixed random seed (42) for reproducibility.

| Column | Description | Range |
|---|---|---|
| Patient_ID | Unique patient identifier | P001 – P650 |
| Age | Patient age in years | 18 – 80 |
| Height_cm | Patient height | 140 – 190 cm |
| Weight_kg | Patient weight | 40 – 120 kg |
| Blood_Pressure | Diastolic blood pressure | 60 – 140 mmHg |
| Glucose | Blood glucose level | 70 – 200 mg/dL |
| Family_History | Family history of diabetes | 0 = No, 1 = Yes |
| Outcome | Target variable | 0 = Not Diabetic, 1 = Diabetic |

The `Outcome` column is generated using a medically-informed risk scoring system (not randomly), based on Glucose, BMI, Age, Blood Pressure, and Family History — so the dataset reflects realistic, learnable patterns.

BMI is calculated from `Height_cm` and `Weight_kg` during feature engineering and used in place of the raw height/weight values for training.

## Output

The project prints model metrics in the terminal and generates the following files:

- `diabetes.csv` — the generated dataset
- `chart1_outcome_count.png` — class balance
- `chart2_feature_distributions.png` — feature histograms
- `chart3_correlation_heatmap.png` — feature correlations
- `chart4_glucose_vs_outcome.png` — glucose comparison by outcome
- `chart5_confusion_matrix.png` — model error breakdown
- `chart6_roc_curve.png` — model quality curve

## Model Performance

Latest verified run:

- **Accuracy:** 78.46%
- **ROC-AUC Score:** 0.8829
- **Confusion Matrix:** TN = 56, TP = 46, FP = 14, FN = 14

Glucose emerged as the strongest predictor (correlation: 0.48), followed by Family History (0.36) and BMI (0.25).

## Presentation

A 10-slide project presentation (`Diabetes_Prediction_Presentation.pptx`) is included, covering the problem statement, dataset design, algorithm explanation, feature importance, model evaluation, and conclusion.

## Note

This project uses a synthetic dataset for learning and demonstration purposes. It is not intended for real medical diagnosis.

## Author

Anushka Shashikant Dhawale — Internship Project, Naviotech Solution Pvt. Ltd.

