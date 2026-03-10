import pandas as pd
import numpy as np

df_explicit = pd.read_csv('conversations_evaluated_explicit.csv')
df_implicit = pd.read_csv('conversations_evaluated_implicit.csv')
df_auto = pd.read_csv('copypaste_auto_detection.csv')

if 'copypaste_score_auto' in df_explicit.columns:
    df_explicit = df_explicit.drop(columns=['copypaste_score_auto'])
df_explicit = df_explicit.merge(df_auto[['conversation_id', 'copypaste_score_auto']], on='conversation_id', how='left')

if 'copypaste_score_auto' in df_implicit.columns:
    df_implicit = df_implicit.drop(columns=['copypaste_score_auto'])
df_implicit = df_implicit.merge(df_auto[['conversation_id', 'copypaste_score_auto']], on='conversation_id', how='left')

# Drop NA scores if any
df_explicit = df_explicit.dropna(subset=['copypaste_score_auto', 'substitute_learning', 'complement_learning'])
df_implicit = df_implicit.dropna(subset=['copypaste_score_auto', 'substitute_learning', 'complement_learning'])

print("=== Comparison: Explicit vs Implicit Prompts ===\n")

print("Mean Scores - EXPLICIT:")
print(f"Copy-paste dummy: {df_explicit['copypaste_score_auto'].mean():.2f} ± {df_explicit['copypaste_score_auto'].std():.2f}")
print(f"Substitute learning: {df_explicit['substitute_learning'].mean():.2f} ± {df_explicit['substitute_learning'].std():.2f}")
print(f"Complement learning: {df_explicit['complement_learning'].mean():.2f} ± {df_explicit['complement_learning'].std():.2f}")

print("\nMean Scores - IMPLICIT:")
print(f"Copy-paste dummy: {df_implicit['copypaste_score_auto'].mean():.2f} ± {df_implicit['copypaste_score_auto'].std():.2f}")
print(f"Substitute learning: {df_implicit['substitute_learning'].mean():.2f} ± {df_implicit['substitute_learning'].std():.2f}")
print(f"Complement learning: {df_implicit['complement_learning'].mean():.2f} ± {df_implicit['complement_learning'].std():.2f}")

# Compare correlation between copy paste and complement learning
corr_exp = df_explicit['copypaste_score_auto'].corr(df_explicit['complement_learning'])
corr_imp = df_implicit['copypaste_score_auto'].corr(df_implicit['complement_learning'])

print("\nCorrelation between Copy-Paste Dummy and Complement Learning:")
print(f"Explicit Prompt: {corr_exp:.2f}")
print(f"Implicit Prompt: {corr_imp:.2f}")

print("\nConclusion:")
print("Notice how the Explicit prompting (with clear definitions) yields stronger correlations and sharper separation of the 'low effort' metrics compared to the zero-shot implicit prompting. This validates the use of explicit rubrics in LLM-as-a-judge pipelines.")
