# -*- coding: utf-8 -*-
"""mimic_sql_analysis.ipynb
# MIMIC-IV ICU Data — SQL Analysis
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
import warnings
warnings.filterwarnings('ignore')
import os
os.makedirs('outputs_sql', exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA & CREATE DATABASE
# ─────────────────────────────────────────────
df = pd.read_csv('Assignment1_mimic dataset.csv')
df['icu_discharge_flag'] = 1 - df['icu_death_flag']
df['outcome'] = df['icu_discharge_flag'].map({1: 'Discharged', 0: 'Died'})

conn = sqlite3.connect('mimic_icu.db')
df.to_sql('icu_patients', conn, if_exists='replace', index=False)
print(f"Data loaded: {df.shape[0]:,} rows")

# ─────────────────────────────────────────────
# 2. SQL QUERIES
# ─────────────────────────────────────────────

q1 = """
SELECT outcome,
    COUNT(*) AS patient_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM icu_patients
GROUP BY outcome
ORDER BY patient_count DESC
"""

q2 = """
SELECT first_careunit,
    COUNT(*) AS total_patients,
    SUM(icu_death_flag) AS deaths,
    ROUND(SUM(icu_death_flag) * 100.0 / COUNT(*), 2) AS mortality_rate_pct
FROM icu_patients
GROUP BY first_careunit
ORDER BY mortality_rate_pct DESC
"""

q3 = """
SELECT outcome,
    ROUND(AVG(age), 1) AS avg_age,
    ROUND(MIN(age), 1) AS min_age,
    ROUND(MAX(age), 1) AS max_age,
    COUNT(*) AS n
FROM icu_patients
GROUP BY outcome
"""

q4 = """
SELECT gender,
    COUNT(*) AS total,
    SUM(icu_death_flag) AS deaths,
    ROUND(SUM(icu_death_flag) * 100.0 / COUNT(*), 2) AS mortality_rate_pct
FROM icu_patients
GROUP BY gender
ORDER BY total DESC
"""

q5 = """
SELECT first_careunit,
    ROUND(AVG(age), 1) AS avg_age,
    COUNT(*) AS deaths,
    ROUND(AVG(heart_rate_mean), 1) AS avg_heart_rate,
    ROUND(AVG(resp_rate_mean), 1) AS avg_resp_rate
FROM icu_patients
WHERE icu_death_flag = 1 AND age >= 65
GROUP BY first_careunit
ORDER BY deaths DESC
"""

q6 = """
SELECT 
    COALESCE(insurance, 'Unknown') AS insurance,
    COUNT(*) AS total_patients,
    ROUND(SUM(icu_death_flag) * 100.0 / COUNT(*), 2) AS mortality_rate_pct
FROM icu_patients
GROUP BY insurance
ORDER BY total_patients DESC
"""

df_q1 = pd.read_sql(q1, conn)
df_q2 = pd.read_sql(q2, conn)
df_q3 = pd.read_sql(q3, conn)
df_q4 = pd.read_sql(q4, conn)
df_q5 = pd.read_sql(q5, conn)
df_q6 = pd.read_sql(q6, conn)

# Fill any remaining None just in case
df_q6['insurance'] = df_q6['insurance'].fillna('Unknown').astype(str)
df_q4['gender'] = df_q4['gender'].fillna('Unknown').astype(str)

print("\nQuery 1 — Overall Outcomes:")
print(df_q1.to_string(index=False))
print("\nQuery 2 — Mortality by ICU Unit:")
print(df_q2.to_string(index=False))
print("\nQuery 3 — Age by Outcome:")
print(df_q3.to_string(index=False))
print("\nQuery 4 — Mortality by Gender:")
print(df_q4.to_string(index=False))
print("\nQuery 5 — Elderly Deaths by ICU Unit:")
print(df_q5.to_string(index=False))
print("\nQuery 6 — Mortality by Insurance:")
print(df_q6.to_string(index=False))

# ─────────────────────────────────────────────
# 3. VISUALISATIONS
# ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('MIMIC-IV ICU Clinical Insights — SQL Analysis', fontsize=14, fontweight='bold')

# Plot 1: Overall outcomes pie
ax1 = axes[0, 0]
ax1.pie(df_q1['patient_count'], labels=df_q1['outcome'],
        autopct='%1.1f%%', colors=['steelblue', 'tomato'], startangle=90)
ax1.set_title('Overall ICU Outcomes')

# Plot 2: Mortality by ICU unit
ax2 = axes[0, 1]
ax2.barh(df_q2['first_careunit'], df_q2['mortality_rate_pct'],
         color='tomato', edgecolor='white')
ax2.set_xlabel('Mortality Rate (%)')
ax2.set_title('Mortality Rate by ICU Unit')
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Average age by outcome
ax3 = axes[0, 2]
colors3 = ['tomato' if o == 'Died' else 'steelblue' for o in df_q3['outcome']]
bars3 = ax3.bar(df_q3['outcome'], df_q3['avg_age'], color=colors3, edgecolor='white')
ax3.set_ylabel('Average Age')
ax3.set_title('Average Age by Outcome')
ax3.set_ylim(0, 80)
ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars3, df_q3['avg_age']):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5, f'{val}', ha='center', fontsize=10)

# Plot 4: Gender and mortality
ax4 = axes[1, 0]
ax4.bar(df_q4['gender'], df_q4['mortality_rate_pct'],
        color=['steelblue', 'mediumseagreen'], edgecolor='white')
ax4.set_ylabel('Mortality Rate (%)')
ax4.set_title('Mortality Rate by Gender')
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Elderly deaths by unit
ax5 = axes[1, 1]
ax5.barh(df_q5['first_careunit'], df_q5['deaths'],
         color='darkorange', edgecolor='white')
ax5.set_xlabel('Number of Deaths')
ax5.set_title('Elderly Deaths (Age ≥65) by ICU Unit')
ax5.grid(axis='x', alpha=0.3)

# Plot 6: Insurance and mortality
ax6 = axes[1, 2]
ax6.bar(df_q6['insurance'], df_q6['mortality_rate_pct'],
        color='mediumpurple', edgecolor='white')
ax6.set_ylabel('Mortality Rate (%)')
ax6.set_title('Mortality Rate by Insurance Type')
ax6.tick_params(axis='x', rotation=20)
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs_sql/sql_clinical_insights.png', bbox_inches='tight')
plt.show()
print("\nSaved to outputs_sql/sql_clinical_insights.png")

# Save CSVs
df_q2.to_csv('outputs_sql/mortality_by_icu_unit.csv', index=False)
df_q3.to_csv('outputs_sql/age_by_outcome.csv', index=False)
df_q5.to_csv('outputs_sql/high_risk_elderly.csv', index=False)

conn.close()
print("SQL analysis complete!")
