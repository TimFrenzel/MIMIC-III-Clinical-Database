#!/usr/bin/env python3
"""
Author: Tim Frenzel
Version: 1.6
Date: 2025-01-23

Description:
This script generates a series of visualizations using the MIMIC-III Demo database.
The visualizations address distinct hypotheses related to EHR data, including:
    
1. Age vs. ICU Length of Stay by Outcome, Faceted by Gender
   - Hypothesis: Older patients may experience longer ICU stays and different outcomes.
   
2. Diagnosis Co-Occurrence Network for the Top 10 Diagnoses
   - Hypothesis: Certain diagnoses frequently co-occur, indicating common comorbidities.
   
3. Key Lab Trends Before ICU Admission: Differences Between Survivors and Non-Survivors
   - Hypothesis: Lab value trends before ICU admission differ between survivors and non-survivors.
   
4. Polypharmacy vs. Age Bubble Chart
   - Hypothesis: The number of medications (polypharmacy) is associated with patient age and may influence hospital stay.
   
5. Post-Surgical Infections and Their Impact on ICU Length of Stay
   - Hypothesis: Some surgical procedures have a higher infection impact on LOS, reflected in both LOS differences and patient counts.
   
Each visualization function queries the SQLite database, processes data, creates the plot, saves it to the 'outputs' folder, and displays the plot.
Other researchers can replicate the code by setting up the MIMIC-III demo database (as an SQLite file named "mimic_demo.db") in the working directory.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from itertools import combinations
from collections import Counter

# --------------------------
# Global Setup: Define paths
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mimic_demo.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set a Seaborn theme (optional)
sns.set_theme(style="whitegrid")


# =============================================================================
# 1. Age vs. ICU LOS by Outcome, Faceted by Gender
# =============================================================================
def visualization_age_vs_icu_los_by_gender():
    """
    Generates a scatter + regression plot showing the relationship between patient age
    and ICU length of stay (LOS), faceted by gender, and colored by hospital outcome.
    Hypothesis: Older patients may experience longer ICU stays and different outcomes.
    """
    # Query the database for admissions, icustays, and patients data.
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        a.admittime,
        a.hospital_expire_flag,
        i.los,
        p.dob,
        p.gender
    FROM admissions a
    JOIN icustays i 
        ON a.hadm_id = i.hadm_id AND a.subject_id = i.subject_id
    JOIN patients p 
        ON a.subject_id = p.subject_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert dates to datetime objects
    df['admittime'] = pd.to_datetime(df['admittime'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df = df.dropna(subset=['admittime', 'dob'])

    # Compute patient age at admission (in years)
    def compute_age(admittime, dob):
        try:
            if admittime >= dob:
                return (admittime - dob).days / 365.25
            else:
                return None
        except Exception:
            return None

    df['age'] = df.apply(lambda row: compute_age(row['admittime'], row['dob']), axis=1)
    df = df.dropna(subset=['age', 'los'])
    df = df[df['age'] <= 120]  # Filter out implausible ages

    # Map outcome: 0 -> 'Survived', 1 -> 'Expired'
    df['outcome'] = df['hospital_expire_flag'].map({0: 'Survived', 1: 'Expired'})

    # Create the visualization using Seaborn's lmplot with facets by gender.
    g = sns.lmplot(
        data=df,
        x='age',
        y='los',
        hue='outcome',
        col='gender',
        markers=['o', 's'],
        palette={"Survived": "blue", "Expired": "red"},
        scatter_kws={'alpha': 0.6},
        ci=None,
        aspect=1.3,
        height=6
    )
    g.set_axis_labels("Age (years)", "ICU Length of Stay (days)")
    g.fig.suptitle("Age vs. ICU LOS by Outcome, Faceted by Gender", fontsize=14)
    g.fig.subplots_adjust(top=0.88)

    # Save and display the plot
    output_file = os.path.join(OUTPUT_DIR, "age_vs_icu_los_by_gender.png")
    g.fig.savefig(output_file, bbox_inches="tight")
    plt.show()
    print(f"[1] Plot saved to {output_file}")


# =============================================================================
# 2. Diagnosis Co-Occurrence Network for Top 10 Diagnoses
# =============================================================================
def visualization_cooccurrence_network():
    """
    Creates a network graph of co-occurring diagnoses.
    Hypothesis: Certain diagnoses frequently co-occur, indicating common comorbid conditions.
    Uses diagnoses_icd and d_icd_diagnoses tables.
    """
    # Connect to the database and query diagnoses data.
    conn = sqlite3.connect(DB_PATH)
    diagnoses_icd_query = "SELECT hadm_id, icd9_code FROM diagnoses_icd"
    diagnoses_df = pd.read_sql_query(diagnoses_icd_query, conn)
    
    d_icd_query = "SELECT icd9_code, short_title FROM d_icd_diagnoses"
    d_icd_df = pd.read_sql_query(d_icd_query, conn)
    conn.close()

    # Merge to get descriptive diagnosis labels.
    merged_df = pd.merge(diagnoses_df, d_icd_df, on="icd9_code", how="left")
    merged_df['diag_label'] = merged_df['short_title'].fillna(merged_df['icd9_code'])

    # Compute co-occurrence frequencies.
    pair_counter = Counter()
    node_counter = Counter()

    for hadm_id, group in merged_df.groupby("hadm_id"):
        diagnoses = sorted(set(group['diag_label']))
        node_counter.update(diagnoses)
        if len(diagnoses) > 1:
            for pair in combinations(diagnoses, 2):
                pair_counter[tuple(sorted(pair))] += 1

    # Identify the top 10 diagnoses by frequency.
    top10 = sorted(node_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_diagnoses = set([diagnosis for diagnosis, count in top10])

    # Filter pairs where both diagnoses are in the top 10.
    filtered_pairs = {pair: count for pair, count in pair_counter.items() 
                      if pair[0] in top10_diagnoses and pair[1] in top10_diagnoses}

    # Build the network graph.
    G = nx.Graph()
    for diagnosis, freq in top10:
        G.add_node(diagnosis, frequency=freq)
    for (diag1, diag2), weight in filtered_pairs.items():
        G.add_edge(diag1, diag2, weight=weight)

    # Visualize using a spring layout.
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    node_sizes = [G.nodes[node]['frequency'] * 50 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightgreen', alpha=0.8)
    
    edges = G.edges(data=True)
    edge_widths = [d['weight'] for (_, _, d) in edges]
    scaled_edge_widths = [w * 0.5 for w in edge_widths]
    nx.draw_networkx_edges(G, pos, width=scaled_edge_widths, alpha=0.7, edge_color='gray')
    
    node_labels = {node: f"{node}\n({G.nodes[node]['frequency']})" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    
    plt.title("Top 10 Most Frequent Diagnoses: Co-Occurrence Network")
    plt.axis('off')
    output_file = os.path.join(OUTPUT_DIR, "top10_diagnosis_cooccurrence_network.png")
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    print(f"[2] Plot saved to {output_file}")


# =============================================================================
# 3. Key Lab Trends Before ICU Admission with Differences
# =============================================================================
def visualization_key_lab_trends_with_differences():
    """
    Creates heatmaps and a trend line chart comparing key lab test trends in the 7 days
    prior to ICU admission between survivors and non-survivors.
    Hypothesis: There are growing differences in lab values between survivors and non-survivors as ICU admission approaches.
    Uses labevents, icustays, d_labitems, and admissions tables.
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        le.subject_id, 
        le.itemid, 
        le.charttime, 
        le.valuenum, 
        icu.intime as icu_admittime,
        di.label as test_name,
        a.hospital_expire_flag
    FROM labevents le
    JOIN icustays icu ON le.subject_id = icu.subject_id
    JOIN d_labitems di ON le.itemid = di.itemid
    JOIN admissions a ON le.subject_id = a.subject_id
    WHERE le.charttime BETWEEN datetime(icu.intime, '-7 days') AND icu.intime
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['charttime'] = pd.to_datetime(df['charttime'])
    df['icu_admittime'] = pd.to_datetime(df['icu_admittime'])
    df['days_before_icu'] = (df['charttime'] - df['icu_admittime']).dt.days

    # Remove tests with too few data points
    min_data_points = 15
    test_counts = df['test_name'].value_counts()
    valid_tests = test_counts[test_counts >= min_data_points].index
    df = df[df['test_name'].isin(valid_tests)]

    # Log-transform lab values (handle zeros appropriately)
    df['log_valuenum'] = np.log1p(df['valuenum'])
    df_survivors = df[df['hospital_expire_flag'] == 0]
    df_non_survivors = df[df['hospital_expire_flag'] == 1]

    heatmap_survivors = df_survivors.groupby(['test_name', 'days_before_icu'])['log_valuenum'].median().unstack()
    heatmap_non_survivors = df_non_survivors.groupby(['test_name', 'days_before_icu'])['log_valuenum'].median().unstack()
    diff_df = abs(heatmap_survivors - heatmap_non_survivors)
    growth_rate = diff_df.diff(axis=1).mean(axis=1)
    top_diff_tests = growth_rate.nlargest(5).index

    heatmap_survivors = heatmap_survivors.loc[top_diff_tests].dropna(axis=1)
    heatmap_non_survivors = heatmap_non_survivors.loc[top_diff_tests].dropna(axis=1)
    diff_df = diff_df.loc[top_diff_tests].dropna(axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [4, 4, 2]})
    sns.heatmap(heatmap_survivors, cmap="magma", annot=True, fmt=".2f", linewidths=0.5, ax=axes[0])
    axes[0].set_title("Survivors: Key Lab Test Trends Before ICU Admission")
    axes[0].set_xlabel("Days Before ICU Admission")
    axes[0].set_ylabel("Lab Test Type")
    
    sns.heatmap(heatmap_non_survivors, cmap="magma", annot=True, fmt=".2f", linewidths=0.5, ax=axes[1])
    axes[1].set_title("Non-Survivors: Key Lab Test Trends Before ICU Admission")
    axes[1].set_xlabel("Days Before ICU Admission")
    axes[1].set_ylabel("Lab Test Type")
    
    for test in diff_df.index:
        axes[2].plot(diff_df.columns, diff_df.loc[test], label=test, marker='o')
    axes[2].set_title("Growing Difference in Lab Trends (Survivors vs. Non-Survivors)")
    axes[2].set_xlabel("Days Before ICU Admission")
    axes[2].set_ylabel("Absolute Difference in Log Lab Values")
    axes[2].legend()
    
    output_path = os.path.join(OUTPUT_DIR, "key_lab_trends_with_differences.png")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()
    print(f"[3] Plot saved to {output_path}")


# =============================================================================
# 4. Polypharmacy vs. Age Bubble Chart
# =============================================================================
def visualization_polypharmacy_vs_age():
    """
    Generates a bubble chart showing the relationship between patient age, the number of medications
    (polypharmacy), and hospital length of stay, with points colored by admission type.
    Hypothesis: Polypharmacy is associated with patient age and may influence hospital stay.
    Uses prescriptions, admissions, patients, and icustays tables.
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        p.subject_id,
        a.admission_type,
        strftime('%Y', p.dob) AS birth_year,
        COUNT(DISTINCT pr.drug) AS num_medications,
        ic.los AS length_of_stay
    FROM prescriptions pr
    JOIN admissions a ON pr.hadm_id = a.hadm_id
    JOIN patients p ON a.subject_id = p.subject_id
    JOIN icustays ic ON a.hadm_id = ic.hadm_id
    WHERE ic.los IS NOT NULL
    GROUP BY p.subject_id, a.admission_type
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['birth_year'] = df['birth_year'].astype(int)
    df['age'] = 2100 - df['birth_year']
    df = df[(df['age'] >= 18) & (df['age'] <= 110)]

    plt.figure(figsize=(12, 7))
    bubble = sns.scatterplot(
        data=df,
        x='age',
        y='num_medications',
        size='length_of_stay',
        hue='admission_type',
        palette='Set2',
        sizes=(20, 1000),
        alpha=0.7,
        edgecolor='black'
    )
    plt.xlabel("Patient Age", fontsize=12)
    plt.ylabel("Number of Medications", fontsize=12)
    plt.title("Polypharmacy and Hospital Stay by Age and Admission Type", fontsize=14, fontweight="bold")
    handles, labels = bubble.get_legend_handles_labels()
    plt.legend(handles[:5], labels[:5], title="Admission Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    output_path = os.path.join(OUTPUT_DIR, "polypharmacy_vs_age.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"[4] Plot saved to {output_path}")


# =============================================================================
# 5. Post-Surgical Infections and LOS Impact
# =============================================================================
def visualization_post_surgical_infections():
    """
    Creates a dual visualization: a box plot (with swarm plot overlay) of ICU length of stay (LOS)
    by procedure type and infection status, and a bar chart of total patient counts.
    Hypothesis: Certain surgical procedures exhibit a significant infection impact on LOS.
    Uses microbiologyevents, procedures_icd, d_icd_procedures, and icustays tables.
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
    WITH infection_status AS (
        SELECT
            m.subject_id,
            m.hadm_id,
            CASE
                WHEN LOWER(m.interpretation) IN ('r', 'resistant', 'positive') THEN 'Infected'
                ELSE 'Not Infected'
            END AS infection_status
        FROM microbiologyevents m
        WHERE m.interpretation IS NOT NULL
    ),
    procedures AS (
        SELECT
            p.subject_id,
            p.hadm_id,
            d.short_title AS procedure_type
        FROM procedures_icd p
        JOIN d_icd_procedures d ON p.icd9_code = d.icd9_code
    ),
    los_data AS (
        SELECT
            i.subject_id,
            i.hadm_id,
            i.los
        FROM icustays i
    )
    SELECT 
        p.procedure_type,
        COALESCE(i.infection_status, 'Not Tested') AS infection_status,
        l.los
    FROM procedures p
    JOIN los_data l ON p.hadm_id = l.hadm_id
    LEFT JOIN infection_status i ON p.hadm_id = i.hadm_id;
    """
    df = pd.read_sql(query, conn)
    conn.close()

    df.dropna(inplace=True)
    procedure_counts = df.groupby(["procedure_type", "infection_status"]).size().unstack(fill_value=0)
    valid_procedures = procedure_counts[(procedure_counts.sum(axis=1) >= 10) & (procedure_counts.min(axis=1) >= 5)].index
    df_filtered = df[df["procedure_type"].isin(valid_procedures)]
    los_99th = df_filtered["los"].quantile(0.99)
    df_filtered = df_filtered[df_filtered["los"] <= los_99th]

    infection_impact = df_filtered.groupby("procedure_type").agg(
        total_cases=("los", "count"),
        avg_los_infected=("los", lambda x: x[df_filtered["infection_status"] == "Infected"].mean()),
        avg_los_noninfected=("los", lambda x: x[df_filtered["infection_status"] == "Not Infected"].mean())
    )
    infection_impact["los_difference"] = infection_impact["avg_los_infected"] - infection_impact["avg_los_noninfected"]
    infection_impact = infection_impact.sort_values(by=["los_difference", "total_cases"], ascending=[False, False])
    top_5_procedures = infection_impact.head(5).index
    df_filtered = df_filtered[df_filtered["procedure_type"].isin(top_5_procedures)]
    procedure_order = list(top_5_procedures)
    color_palette = {"Infected": "red", "Not Infected": "green", "Not Tested": "gray"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    sns.boxplot(
        data=df_filtered,
        x="procedure_type",
        y="los",
        hue="infection_status",
        order=procedure_order,
        palette=color_palette,
        fliersize=2,
        ax=axes[0]
    )
    sns.swarmplot(
        data=df_filtered,
        x="procedure_type",
        y="los",
        hue="infection_status",
        dodge=True,
        alpha=0.5,
        size=3,
        order=procedure_order,
        ax=axes[0]
    )
    axes[0].set_title("Post-Surgical Infection Rates: Top 5 Most Impactful Procedures", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Length of Stay (Days)", fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(title="Infection Status", loc="upper right")
    
    procedure_counts_top5 = procedure_counts.loc[top_5_procedures]
    procedure_counts_top5 = procedure_counts_top5.reindex(procedure_order)
    procedure_counts_top5.plot(
        kind='bar', 
        stacked=True, 
        color=[color_palette.get("Infected", "red"), color_palette.get("Not Infected", "green"), color_palette.get("Not Tested", "gray")],
        ax=axes[1]
    )
    axes[1].set_title("Total Number of Patients per Procedure & Infection Status", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Procedure Type")
    axes[1].set_ylabel("Number of Patients")
    axes[1].legend(title="Infection Status", loc="upper right")
    axes[1].tick_params(axis='x', rotation=45)
    
    output_path = os.path.join(OUTPUT_DIR, "top5_post_surgical_infections_with_counts.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[5] Plot saved to {output_path}")


# =============================================================================
# Main function to call all visualizations
# =============================================================================
def main():
    print("Starting MIMIC-III Visualizations...")
    visualization_age_vs_icu_los_by_gender()
    visualization_cooccurrence_network()
    visualization_key_lab_trends_with_differences()
    visualization_polypharmacy_vs_age()
    visualization_post_surgical_infections()
    print("All visualizations completed.")

    
if __name__ == '__main__':
    main()
