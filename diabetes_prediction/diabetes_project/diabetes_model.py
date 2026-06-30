import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

print("✅ Libraries imported successfully!")

df = pd.read_csv("diabetes.csv")
print(f"Dataset shape: {df.shape}")
print("First 5 rows (df.head()):")
print(df.head())
print("Column names:")
print(df.columns)
print("Data types (df.dtypes):")
print(df.dtypes)
print("Basic statistics (df.describe()):")
print(df.describe())
print("Outcome value counts:")
print(df["Outcome"].value_counts())

# BMI feature engineering
# BMI = Weight_kg / (Height_m^2)
df["Height_m"] = df["Height_cm"] / 100.0
df["BMI"] = (df["Weight_kg"] / (df["Height_m"] ** 2)).round(2)
print("Sample Patient_ID, Height_cm, Weight_kg, BMI:")
print(df[["Patient_ID", "Height_cm", "Weight_kg", "BMI"]].head())

df = df.drop(["Patient_ID", "Height_cm", "Weight_kg", "Height_m"], axis=1)
print("Remaining column names:")
print(df.columns)
print("First 5 rows after engineering:")
print(df.head())

print("Checking missing values:")
print(df.isnull().sum())
print(f"BMI min/max: {df['BMI'].min()} / {df['BMI'].max()}")
print(f"Glucose min/max: {df['Glucose'].min()} / {df['Glucose'].max()}")
print(
    f"Blood_Pressure min/max: {df['Blood_Pressure'].min()} / {df['Blood_Pressure'].max()}"
)
print("✅ Data is clean and within expected ranges.")

sns.set(style="whitegrid")

# Chart 1
outcome_counts = df["Outcome"].value_counts().sort_index()
plt.figure(figsize=(6, 4))
colors = ["steelblue", "salmon"]
plt.bar(outcome_counts.index.astype(str), outcome_counts.values, color=colors)
plt.title("Diabetic vs Non-Diabetic Count")
plt.xlabel("Outcome (0 = Not Diabetic, 1 = Diabetic)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("chart1_outcome_count.png")
plt.show()

# Chart 2
plt.figure(figsize=(12, 8))
df.hist(bins=20, color="steelblue")
plt.suptitle("Distribution of All Features")
plt.tight_layout()
plt.savefig("chart2_feature_distributions.png")
plt.show()

# Chart 3
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("chart3_correlation_heatmap.png")
plt.show()

# Chart 4
plt.figure(figsize=(7, 5))
sns.boxplot(x="Outcome", y="Glucose", data=df, palette="Set2")
plt.title("Glucose Levels: Diabetic vs Non-Diabetic")
plt.xlabel("Outcome (0 = Not Diabetic, 1 = Diabetic)")
plt.ylabel("Glucose")
plt.tight_layout()
plt.savefig("chart4_glucose_vs_outcome.png")
plt.show()

print("✅ All 4 EDA charts saved!")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
print("Feature columns:")
print(X.columns)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42, max_iter=200)
model.fit(X_train_scaled, y_train)
print("✅ Model trained successfully!")

coef = model.coef_[0]
importance = np.abs(coef)
feature_importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": importance}
).sort_values("Importance", ascending=False)
print("Feature importance:")
print(feature_importance_df.to_string(index=False))

y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy: {acc * 100:.2f}%")
print(f"ROC-AUC Score: {auc:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Diabetic", "Diabetic"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Diabetic", "Diabetic"],
    yticklabels=["Not Diabetic", "Diabetic"],
)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("chart5_confusion_matrix.png")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="darkorange", label=f"ROC curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Model Quality")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("chart6_roc_curve.png")
plt.show()

print("✅ Model evaluation complete!")

new_patient_age = 47
new_patient_bp = 95
new_patient_glucose = 155
new_patient_family = 1
new_patient_height = 165
new_patient_weight = 82

new_height_m = new_patient_height / 100.0
new_bmi = round(new_patient_weight / (new_height_m**2), 2)

print("\nNew patient details:")
print(f"Age: {new_patient_age}")
print(f"Blood_Pressure: {new_patient_bp}")
print(f"Glucose: {new_patient_glucose}")
print(f"Family_History: {new_patient_family}")
print(f"Height_cm: {new_patient_height}")
print(f"Weight_kg: {new_patient_weight}")
print(f"Calculated BMI: {new_bmi}")

input_array = np.array(
    [
        [
            new_patient_age,
            new_patient_bp,
            new_patient_glucose,
            new_patient_family,
            new_bmi,
        ]
    ]
)
input_scaled = scaler.transform(input_array)

prediction = model.predict(input_scaled)[0]
prob_diabetic = model.predict_proba(input_scaled)[:, 1][0]

if prediction == 1:
    print("⚠️  This patient is likely DIABETIC")
else:
    print("✅  This patient is likely NOT DIABETIC")

print(f"Diabetic probability: {prob_diabetic * 100:.2f}%")
print("TIP: Try changing Glucose to 80 or Family_History to 0 and re-run!")

print("\n" + "=" * 60)
print("Project Summary")
print("=" * 60)
print("Project name: Diabetic Prediction using Logistic Regression")
print(
    "Dataset: Custom-built (650 patients, 7 input columns → 5 model features after BMI engineering)"
)
print("Algorithm: Logistic Regression")
print(f"Model accuracy: {acc * 100:.2f}%")
print(f"ROC-AUC score: {auc:.4f}")

print("Output files generated:")
output_files = [
    "generate_dataset.py",
    "diabetes_model.py",
    "diabetes.csv",
    "chart1_outcome_count.png",
    "chart2_feature_distributions.png",
    "chart3_correlation_heatmap.png",
    "chart4_glucose_vs_outcome.png",
    "chart5_confusion_matrix.png",
    "chart6_roc_curve.png",
]

for f in output_files:
    print(f"- {f}")

print("\n🎉 Project Complete!")
