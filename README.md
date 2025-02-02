# MIMIC-III Visualizations and Risk Factors Extraction

This repository contains a collection of advanced data visualizations and data extraction scripts built using the MIMIC-III Demo database. The project demonstrates several analyses on electronic health record (EHR) data, including patient outcomes, diagnosis co-occurrence, lab trends, polypharmacy, and post-surgical infection impacts. Interactive components (using D3.js) are provided via an HTML file and JSON data.

## Project Overview

The project includes two main components:

1. **MIMIC Visualizations (`mimic_visualizations.py`):**  
   Generates five distinct visualizations:
   - **Age vs. ICU LOS by Outcome (Faceted by Gender):**  
     *Key Finding:* Older patients tend to have shorter ICU stays—possibly due to earlier palliative care interventions—while survivors generally show longer stays, with younger patients displaying high variability.
     
   - **Diagnosis Co-Occurrence Network for the Top 10 Diagnoses:**  
     *Key Finding:* Hypertension, Atrial Fibrillation, and Congestive Heart Failure frequently co-occur. Additionally, acute kidney failure is strongly linked to sepsis.
     
   - **Key Lab Trends Before ICU Admission (Survivors vs. Non-Survivors):**  
     *Key Finding:* Non-survivors show sharp declines in WBC/RBC levels and persistently high ferritin/fibrinogen levels, indicating late-stage sepsis and systemic inflammation.
     
   - **Polypharmacy vs. Age Bubble Chart:**  
     *Key Finding:* Older patients, especially in emergency admissions, generally take more medications, which correlates with longer hospital stays.
     
   - **Post-Surgical Infections and Their Impact on ICU LOS:**  
     *Key Finding:* Procedures such as percutaneous abdominal drainage and IV feeding show significantly higher infection-related ICU stays, while some procedures (e.g., coronary artery catheterization) exhibit lower infection risk.

2. **Risk Factors Extraction (`risk_factors_extraction.py`):**  
   Dynamically extracts the top 20 diagnoses, maps them to human-readable labels, aggregates patient data (including age groups and survival status), and outputs the top 6 risk factors in a JSON file.

Additionally, a **Health Risk Radar Chart** is provided as an interactive HTML file (`radar_chart.html`), with a screenshot (`Health Risk Radar Chart.png`) demonstrating its appearance.

## Files in This Repository

- **mimic_visualizations.py:** Python script generating five key visualizations.
- **risk_factors_extraction.py:** Python script for extracting and aggregating risk factor data.
- **risk_factors.json:** Aggregated JSON file with risk factors data.
- **radar_chart.html:** Interactive D3.js-based health risk radar chart.
- **Health Risk Radar Chart.png:** Screenshot of the interactive radar chart.
- **Outputs Folder:** Contains the following output images:
  - age_vs_icu_los_by_gender.png
  - key_lab_trends_with_differences.png
  - polypharmacy_vs_age.png
  - top5_post_surgical_infections_with_counts.png
  - top10_diagnosis_cooccurrence_network.png

## How to Run the Project

1. **Dependencies:**  
   Ensure you have Python 3.x installed. Install required packages using:
   ```bash
   pip install pandas numpy matplotlib seaborn networkx```
    (D3.js is used for the interactive HTML; no Python dependency is needed for it.)
D3.js is used for the interactive HTML; no Python dependency is needed for it.)

2. **Database Setup:**  
    Place the mimic_demo.db SQLite file in the root folder of the repository. (Note: The MIMIC database cannot be shared publicly for privacy reasons. Please request access at www.citiprogram.org.)

3. **Running the Scripts:**
    Execute the visualization script:
   ```bash
    python mimic_visualizations.py```
    and the risk factors extraction script:
   ```bash
    python risk_factors_extraction.py```

4. **Interactive Components:**
    The interactive radar chart is contained in radar_chart.html and uses D3.js. You can view it directly in a web browser or host it via GitHub Pages for a dynamic experience.

## Initial Key Findings
**Age vs. ICU LOS by Outcome:**
Older patients tend to have shorter ICU stays, potentially due to earlier palliative care interventions.
Survivors generally stay longer than non-survivors, with younger patients showing high variability in ICU stay.

**Diagnosis Co-Occurrence Network:**
Hypertension, Atrial Fibrillation, and Congestive Heart Failure are the most frequently co-occurring conditions, indicating common cardiovascular disease clusters.
Acute kidney failure is strongly linked to sepsis, highlighting its role in multi-organ dysfunction.

**Key Lab Trends Before ICU Admission:**
Non-survivors show sharp declines in WBC and RBC levels, suggesting immunosuppression or late-stage sepsis.
Elevated ferritin and fibrinogen levels in non-survivors suggest ongoing systemic inflammation.

**Polypharmacy vs. Age:**
Older patients, especially those in emergency admissions, take more medications—a marker for higher chronic disease burden.
Higher medication counts are correlated with longer hospital stays.

**Post-Surgical Infections:**
Procedures like percutaneous abdominal drainage and IV feeding exhibit significantly higher infection-related ICU stays.
Some procedures, such as coronary artery catheterization, have lower infection risks, suggesting differences in procedural infection susceptibility.

**Health Risk Radar Chart:**
Men exhibit higher risks for conditions such as hypertension and underactive thyroid, whereas women show increased risks for congestive heart failure and acute kidney failure in the 90+ age group.
Type 2 diabetes is a common risk factor across both genders, underscoring the importance of early metabolic control.