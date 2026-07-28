import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import mixedlm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import kruskal
import scikit_posthocs as sp
from statsmodels.stats.multitest import multipletests



# Data_frame.csv save structure
# MouseID	Group	Day     Latency
# 001       Group1  1       123
# 001       Group1  2       115
# 002       Group2  1       202
# 002       Group2  2       221
# ...



# ============ 1. Reading ============
file_path = 'X:\SynC_Fig\Fig.2_rotarod\Fig.2e\Rotarod_6g.csv'  # Change to the right Rotarod.csv file path
output_dir = os.path.dirname(file_path)

# Read CSV
rotarod_df = pd.read_csv(file_path)

# Checking the data structure
print(rotarod_df.head())

# Confirm the title of each line
rotarod_df.columns = ['MouseID', 'Group', 'Day', 'Latency']
# 100% normalization, If necessary
rotarod_df_normalized = rotarod_df.copy()

# Calculating the Day1 latency as 100%
day1_means = rotarod_df_normalized[rotarod_df_normalized['Day'] == 1].groupby('Group')['Latency'].mean()

# Function for 100 normalization
def normalize_latency(row):
    group = row['Group']
    day1_mean = day1_means[group]
    return row['Latency'] / day1_mean * 100

rotarod_df_normalized['Latency'] = rotarod_df_normalized.apply(normalize_latency, axis=1)

def analyze_rotarod(df, prefix, output_dir):
    # ----------------- 1. AUC -----------------
    auc_results = []
    for mouse_id, sub_df in df.groupby('MouseID'):
        sub_df = sub_df.sort_values('Day')
        baseline = sub_df['Latency'].iloc[0]  # identification the First trial for each mouse as baseline.
        auc_value = np.trapz(sub_df['Latency']-baseline, sub_df['Day'])
        group = sub_df['Group'].iloc[0]
        auc_results.append({'MouseID': mouse_id, 'Group': group, 'AUC': auc_value})
    auc_df = pd.DataFrame(auc_results)
    auc_table_path = os.path.join(output_dir, f'{prefix}_Rotarod_AUC_Table.csv')
    auc_df.to_csv(auc_table_path, index=False)

    # AUC ANOVA
    # group_list = [auc_df.loc[auc_df['Group'] == g, 'AUC'].values for g in auc_df['Group'].unique()]
    # f_val, p_val = f_oneway(*group_list)
    # print(f"{prefix} AUC ANOVA F值: {f_val:.3f}, P值: {p_val:.4f}")

    # Kruskal-Wallis
    group_list = [auc_df.loc[auc_df['Group'] == g, 'AUC'].values for g in auc_df['Group'].unique()]
    h_val, p_val = kruskal(*group_list)
    print(f"{prefix} AUC Kruskal-Wallis H_value: {h_val:.3f}, P_value: {p_val:.4f}")


    # Tukey post hoc
    # tukey = pairwise_tukeyhsd(auc_df['AUC'], auc_df['Group'])
    # tukey_table_path = os.path.join(output_dir, f'{prefix}_Rotarod_AUC_Tukey.csv')
    # pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]).to_csv(tukey_table_path, index=False)

    # Steel-Dwass (Dunn) + Bonferroni
    steel_results = sp.posthoc_dunn(auc_df, val_col='AUC', group_col='Group')#, p_adjust='bonferroni'
    print(f"{prefix} AUC Steel-Dwass result：\n", steel_results)

    # Saved results
    steel_table_path = os.path.join(output_dir, f'{prefix}_Rotarod_AUC_Steel.csv')
    steel_results.to_csv(steel_table_path)
    print(f"Steel-Bonferroni saved: {steel_table_path}")


    # AUC bar_plot
    plt.figure(figsize=(6,5))
    sns.barplot(x='Group', y='AUC', data=auc_df, errorbar='se', capsize=0.2, palette='Set2')
    sns.stripplot(x='Group', y='AUC', data=auc_df, color='black', alpha=0.6)
    plt.title(f'{prefix} Rotarod AUC by Group')
    plt.ylim(-120,320)
    plt.ylabel('AUC')
    plt.xlabel('Group')
    auc_pdf_path = os.path.join(output_dir, f'{prefix}_Rotarod_AUC.pdf')
    plt.tight_layout()
    plt.savefig(auc_pdf_path)
    plt.close()

    # ----------------- 2. LMM -----------------
    df['Day'] = df['Day'].astype(float)
    model = mixedlm("Latency ~ Group * Day", data=df, groups=df["MouseID"])
    result = model.fit()
    print(result.summary())

    # Save the LMM results
    # ================================
    # Multiple Testing Correction
    # ================================
    # raw_p_value
    raw_pvalues = result.pvalues.values
    effect_names = result.pvalues.index

    # Correction method: "holm", "fdr_bh",'bonferroni', 'sidak', 'fdr_by'
    method = "bonferroni"

    # multipletests returned results
    reject, p_adj, _, _ = multipletests(raw_pvalues, alpha=0.05, method=method)

    lmm_table_path = os.path.join(output_dir, '_LMM_stats.csv')
    result_df = pd.DataFrame({
        'Effect': effect_names,
        'Estimate': result.params.values,
        'Raw_P': raw_pvalues,
        f'Adjusted_P({method})': p_adj,
        'Reject_H0': reject  # True means significant
    })
    result_df.to_csv(lmm_table_path, index=False)

    # LMM Curve with errobar
    plt.figure(figsize=(6,5))
    sns.pointplot(data=df, x='Day', y='Latency', hue='Group',
                  errorbar='se', dodge=True, markers='o', capsize=0.2)
    plt.title(f'{prefix} Rotarod Performance Over Time (LMM)')
    plt.ylabel('Latency (seconds)')
    plt.xlabel('Day')
    lmm_pdf_path = os.path.join(output_dir, f'{prefix}_Rotarod_LMM.pdf')
    plt.tight_layout()
    plt.savefig(lmm_pdf_path)
    plt.close()

    # ----------------- 3. 汇总 PDF -----------------
    """final_pdf_path = os.path.join(output_dir, f'{prefix}_Rotarod_Report.pdf')
    with PdfPages(final_pdf_path) as pdf:
        # AUC图
        plt.figure(figsize=(6,5))
        sns.barplot(x='Group', y='AUC', data=auc_df, ci='sd', capsize=0.2, palette='Set2')
        sns.stripplot(x='Group', y='AUC', data=auc_df, color='black', alpha=0.6)
        plt.title(f'{prefix} Rotarod AUC by Group')
        plt.ylabel('AUC')
        plt.xlabel('Group')
        pdf.savefig()
        plt.close()

        # LMM图
        plt.figure(figsize=(6,5))
        sns.pointplot(data=df, x='Day', y='Latency', hue='Group',
                      errorbar='se', dodge=True, markers='o', capsize=0.2)
        plt.title(f'{prefix} Rotarod Performance Over Time (LMM)')
        plt.ylabel('Latency (seconds)')
        plt.xlabel('Day')
        pdf.savefig()
        plt.close()"""

# Main project
analyze_rotarod(rotarod_df, prefix='raw', output_dir=output_dir)

# Day1归一化为100数据分析
# analyze_rotarod(rotarod_df_normalized, prefix='100', output_dir=output_dir)

print(f"PDF file have saved")
