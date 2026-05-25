# Synthetic data generation script

"""
Synthetic Cardiac Surgery Risk Calculator Demo
------------------------------------------------

This script generates synthetic EHR-style source tables for a cardiac surgery
mortality risk calculator demo.

IMPORTANT:
- This data is fully synthetic.
- It does not represent real patients.
- It is not intended for clinical use.
"""

from pathlib import Path
import numpy as np
import pandas as pd


RANDOM_SEED = 42
N_PATIENTS = 8000

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "synthetic_raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Convert risk score/logit into probability."""
    return 1 / (1 + np.exp(-x))


def generate_patients(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate patient demographic and baseline comorbidity table."""

    patient_ids = np.arange(100000, 100000 + n)

    age = rng.normal(loc=63, scale=13, size=n).round().astype(int)
    age = np.clip(age, 18, 90)

    sex = rng.choice(["Male", "Female"], size=n, p=[0.58, 0.42])

    race = rng.choice(
        ["White", "Black", "Asian", "Hispanic", "Other"],
        size=n,
        p=[0.48, 0.18, 0.07, 0.22, 0.05],
    )

    diabetes = rng.binomial(1, p=0.28, size=n)
    ckd = rng.binomial(1, p=0.16, size=n)
    heart_failure = rng.binomial(1, p=0.30, size=n)
    copd = rng.binomial(1, p=0.14, size=n)
    prior_cardiac_surgery = rng.binomial(1, p=0.12, size=n)

    bmi = rng.normal(loc=29, scale=6, size=n)
    bmi = np.clip(bmi, 16, 55).round(1)

    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age": age,
            "sex": sex,
            "race": race,
            "diabetes": diabetes,
            "ckd": ckd,
            "heart_failure": heart_failure,
            "copd": copd,
            "prior_cardiac_surgery": prior_cardiac_surgery,
            "bmi": bmi,
        }
    )


def generate_encounters(patients: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate one index surgical encounter per patient."""

    n = len(patients)

    encounter_ids = np.arange(500000, 500000 + n)

    emergency_case = rng.binomial(1, p=0.10, size=n)

    admission_source = rng.choice(
        ["Elective", "Emergency Department", "Transfer", "Clinic"],
        size=n,
        p=[0.58, 0.18, 0.16, 0.08],
    )

    preop_los_days = rng.poisson(lam=2.0, size=n)
    preop_los_days = np.where(emergency_case == 1, preop_los_days + rng.poisson(2, size=n), preop_los_days)
    preop_los_days = np.clip(preop_los_days, 0, 21)

    surgery_year = rng.choice([2021, 2022, 2023, 2024, 2025], size=n)

    return pd.DataFrame(
        {
            "encounter_id": encounter_ids,
            "patient_id": patients["patient_id"].values,
            "surgery_year": surgery_year,
            "admission_source": admission_source,
            "emergency_case": emergency_case,
            "preop_los_days": preop_los_days,
        }
    )


def generate_procedures(encounters: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate cardiac surgery procedure table."""

    n = len(encounters)

    procedure_type = rng.choice(
        ["CABG", "Valve", "Aortic", "Congenital", "Combined", "Other"],
        size=n,
        p=[0.36, 0.26, 0.10, 0.08, 0.14, 0.06],
    )

    # Longer procedure time for more complex surgeries
    base_time = rng.normal(loc=220, scale=45, size=n)

    procedure_time_minutes = base_time.copy()
    procedure_time_minutes += np.where(procedure_type == "CABG", 20, 0)
    procedure_time_minutes += np.where(procedure_type == "Valve", 35, 0)
    procedure_time_minutes += np.where(procedure_type == "Aortic", 70, 0)
    procedure_time_minutes += np.where(procedure_type == "Congenital", 80, 0)
    procedure_time_minutes += np.where(procedure_type == "Combined", 110, 0)

    procedure_time_minutes = np.clip(procedure_time_minutes, 90, 600).round().astype(int)

    return pd.DataFrame(
        {
            "encounter_id": encounters["encounter_id"].values,
            "procedure_type": procedure_type,
            "procedure_time_minutes": procedure_time_minutes,
        }
    )


def generate_labs(patients: pd.DataFrame, encounters: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate preoperative lab/vital features."""

    n = len(patients)

    # Ejection fraction: lower in HF patients
    ef = rng.normal(loc=55, scale=10, size=n)
    ef = ef - patients["heart_failure"].values * rng.normal(loc=12, scale=5, size=n)
    ef = np.clip(ef, 15, 75).round(1)

    # Creatinine: higher in CKD patients
    creatinine = rng.normal(loc=1.0, scale=0.25, size=n)
    creatinine = creatinine + patients["ckd"].values * rng.normal(loc=0.8, scale=0.35, size=n)
    creatinine = np.clip(creatinine, 0.4, 5.5).round(2)

    # Hemoglobin: lower in higher-risk patients
    hemoglobin = rng.normal(loc=13.2, scale=1.4, size=n)
    hemoglobin = hemoglobin - patients["ckd"].values * 0.5
    hemoglobin = np.clip(hemoglobin, 7.0, 17.5).round(1)

    sodium = rng.normal(loc=138, scale=3.5, size=n)
    sodium = np.clip(sodium, 124, 150).round(1)

    return pd.DataFrame(
        {
            "encounter_id": encounters["encounter_id"].values,
            "patient_id": patients["patient_id"].values,
            "ejection_fraction": ef,
            "creatinine": creatinine,
            "hemoglobin": hemoglobin,
            "sodium": sodium,
        }
    )


def generate_outcomes(
    patients: pd.DataFrame,
    encounters: pd.DataFrame,
    procedures: pd.DataFrame,
    labs: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate synthetic 30-day mortality outcome.

    The risk equation intentionally follows clinical intuition:
    higher age, emergency surgery, CKD, heart failure, low EF, high creatinine,
    anemia, longer procedure time, and complex procedures increase mortality.
    """

    df = (
        encounters.merge(patients, on="patient_id", how="left")
        .merge(procedures, on="encounter_id", how="left")
        .merge(labs, on=["encounter_id", "patient_id"], how="left")
    )

    proc = df["procedure_type"]

    procedure_risk = (
        np.where(proc == "CABG", 0.10, 0)
        + np.where(proc == "Valve", 0.18, 0)
        + np.where(proc == "Aortic", 0.45, 0)
        + np.where(proc == "Congenital", 0.55, 0)
        + np.where(proc == "Combined", 0.75, 0)
        + np.where(proc == "Other", 0.20, 0)
    )

    # Logit-style risk score
    risk_score = (
        -5.1
        + 0.035 * (df["age"] - 60)
        + 0.70 * df["emergency_case"]
        + 0.45 * df["ckd"]
        + 0.38 * df["heart_failure"]
        + 0.25 * df["diabetes"]
        + 0.22 * df["copd"]
        + 0.35 * df["prior_cardiac_surgery"]
        + 0.030 * np.maximum(50 - df["ejection_fraction"], 0)
        + 0.45 * np.maximum(df["creatinine"] - 1.2, 0)
        + 0.18 * np.maximum(12 - df["hemoglobin"], 0)
        + 0.002 * np.maximum(df["procedure_time_minutes"] - 240, 0)
        + 0.035 * df["preop_los_days"]
        + procedure_risk
    )

    mortality_probability = sigmoid(risk_score)
    mortality_30d = rng.binomial(1, mortality_probability)

    return pd.DataFrame(
        {
            "encounter_id": df["encounter_id"],
            "patient_id": df["patient_id"],
            "mortality_30d": mortality_30d,
            "synthetic_true_mortality_probability": mortality_probability.round(5),
        }
    )


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    print("Generating synthetic cardiac surgery-style source tables...")

    patients = generate_patients(N_PATIENTS, rng)
    encounters = generate_encounters(patients, rng)
    procedures = generate_procedures(encounters, rng)
    labs = generate_labs(patients, encounters, rng)
    outcomes = generate_outcomes(patients, encounters, procedures, labs, rng)

    patients.to_csv(RAW_DIR / "patients.csv", index=False)
    encounters.to_csv(RAW_DIR / "encounters.csv", index=False)
    procedures.to_csv(RAW_DIR / "procedures.csv", index=False)
    labs.to_csv(RAW_DIR / "labs.csv", index=False)
    outcomes.to_csv(RAW_DIR / "outcomes.csv", index=False)

    print("Done.")
    print(f"Rows generated: {N_PATIENTS}")
    print(f"Output folder: {RAW_DIR}")

    print("\nOutcome distribution:")
    print(outcomes["mortality_30d"].value_counts(normalize=True).rename("rate"))


if __name__ == "__main__":
    main()
