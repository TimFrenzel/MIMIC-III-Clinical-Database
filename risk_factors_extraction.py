#!/usr/bin/env python3
"""
Author: Tim Frenzel
Version: 1.3
Date: 2025-01-23

Description:
This script processes the MIMIC-III Demo database to extract and aggregate risk factor data 
based on the most common diagnoses.

Hypothesis: Different risk factors (diabetes, hypertension, infection history, smoking) impact survival differently for men vs. women across age groups.

The 'mimic_demo.db' SQLite database must be located in the same working directory.
The radar_chart.html is provided in the repo.
"""

import sqlite3
import pandas as pd
import json
import os

def main():
    # --------------------------
    # Set up paths and create output folder if needed
    # --------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, "mimic_demo.db")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------
    # Connect to SQLite
    # --------------------------
    conn = sqlite3.connect(DB_PATH)

    # --------------------------
    # Step 1: Identify the most common diagnoses dynamically
    # --------------------------
    top_diagnoses_query = """
    SELECT d.icd9_code, COUNT(*) as count
    FROM diagnoses_icd d
    GROUP BY d.icd9_code
    ORDER BY count DESC
    LIMIT 20;  -- Start with top 20 to ensure variety before filtering
    """
    top_diagnoses_df = pd.read_sql(top_diagnoses_query, conn)

    # Extract the top diagnosis codes for filtering
    top_diagnoses = top_diagnoses_df["icd9_code"].tolist()

    # --------------------------
    # Step 2: Fetch diagnosis descriptions dynamically
    # --------------------------
    # Build parameter placeholders (e.g., "?, ?, ?, ..." for safety)
    placeholders = ",".join("?" * len(top_diagnoses))
    diagnosis_labels_query = f"""
    SELECT icd9_code, short_title
    FROM d_icd_diagnoses
    WHERE icd9_code IN ({placeholders});
    """
    diagnosis_labels_df = pd.read_sql(diagnosis_labels_query, conn, params=top_diagnoses)
    diagnosis_labels = dict(zip(diagnosis_labels_df["icd9_code"], diagnosis_labels_df["short_title"]))

    # --------------------------
    # Step 3: Fetch patient data with dynamically selected diagnoses
    # --------------------------
    query = f"""
    SELECT 
        p.subject_id,
        p.gender,
        (strftime('%Y', '2200-01-01') - strftime('%Y', p.dob)) AS age,
        a.hospital_expire_flag,  -- 0 = Survived, 1 = Expired
        d.icd9_code
    FROM patients p
    JOIN admissions a ON p.subject_id = a.subject_id
    JOIN diagnoses_icd d ON a.hadm_id = d.hadm_id
    WHERE d.icd9_code IN ({placeholders});
    """
    df = pd.read_sql(query, conn, params=top_diagnoses)
    conn.close()

    # --------------------------
    # Step 4: Convert ICD codes to human-readable labels ("risk factors")
    # --------------------------
    df["risk_factor"] = df["icd9_code"].map(diagnosis_labels)

    # --------------------------
    # Step 5: Define age groups in 20-year spans (from 30 to 110 years)
    # --------------------------
    age_bins = [30, 50, 70, 90, 110]  # Adjusted to capture a "90+" group as 90-110
    age_labels = ["30-50", "50-70", "70-90", "90+"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, include_lowest=True)

    # --------------------------
    # Step 6: Map survival status
    # --------------------------
    df["survival_status"] = df["hospital_expire_flag"].replace({0: "Survived", 1: "Expired"})

    # --------------------------
    # Step 7: Aggregate counts per gender, age group, survival status, and risk factor
    # --------------------------
    aggregated_data = df.groupby(["gender", "age_group", "survival_status", "risk_factor"]) \
                        .size().reset_index(name="count")

    # --------------------------
    # Step 8: Remove Age Groups Where Total Count is Less Than 10
    # --------------------------
    valid_age_groups = aggregated_data.groupby("age_group")["count"].sum()
    valid_age_groups = valid_age_groups[valid_age_groups >= 10].index.tolist()
    aggregated_data = aggregated_data[aggregated_data["age_group"].isin(valid_age_groups)]

    # --------------------------
    # Step 9: Select the 6 Most Common Risk Factors Across the Dataset
    # --------------------------
    top_risk_factors = aggregated_data.groupby("risk_factor")["count"].sum() \
                                      .sort_values(ascending=False).head(6).index.tolist()

    # --------------------------
    # Step 10: Filter the aggregated dataset to keep only the top 6 risk factors
    # --------------------------
    filtered_data = aggregated_data[aggregated_data["risk_factor"].isin(top_risk_factors)]

    # --------------------------
    # Step 11: Convert the aggregated data to JSON format and save
    # --------------------------
    output_path = os.path.join(OUTPUT_DIR, "risk_factors.json")
    with open(output_path, "w") as f:
        json.dump(filtered_data.to_dict(orient="records"), f, indent=4)

    print(f"Updated JSON file saved to: {output_path}")


if __name__ == '__main__':
    main()
