import numpy as np
import pandas as pd


def main():
    np.random.seed(42)

    num_patients = 650
    patient_ids = [f"P{i:03d}" for i in range(1, num_patients + 1)]

    ages = np.random.randint(18, 81, size=num_patients)
    heights_cm = np.random.randint(140, 191, size=num_patients)
    weights_kg = np.random.randint(40, 121, size=num_patients)
    blood_pressures = np.random.randint(60, 141, size=num_patients)
    glucose_levels = np.random.randint(70, 201, size=num_patients)
    family_history = np.random.randint(0, 2, size=num_patients)

    heights_m = heights_cm / 100.0
    bmi = weights_kg / (heights_m**2)

    risk_score = (
        (glucose_levels > 140) * 3
        + (bmi > 30) * 2
        + (ages > 45) * 1
        + (blood_pressures > 90) * 1
        + family_history * 2
    )

    prob_diabetic = risk_score / 9.0
    noise = np.random.uniform(-0.1, 0.1, size=num_patients)
    prob_diabetic_noisy = np.clip(prob_diabetic + noise, 0, 1)

    outcome = (prob_diabetic_noisy >= 0.5).astype(int)

    df = pd.DataFrame(
        {
            "Patient_ID": patient_ids,
            "Age": ages,
            "Height_cm": heights_cm,
            "Weight_kg": weights_kg,
            "Blood_Pressure": blood_pressures,
            "Glucose": glucose_levels,
            "Family_History": family_history,
            "Outcome": outcome,
        }
    )

    df.to_csv("diabetes.csv", index=False)

    print(f"Dataset shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())
    print("Outcome value counts:")
    print(df["Outcome"].value_counts())
    print("✅ diabetes.csv saved successfully!")


if __name__ == "__main__":
    main()
