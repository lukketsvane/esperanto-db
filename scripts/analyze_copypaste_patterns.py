import pandas as pd
import numpy as pd_np # using numpy for some stats if needed

df1 = pd.read_csv('conversations_evaluated_final.csv')
df2 = pd.read_csv('copypaste_auto_detection.csv')

if 'copypaste_score_auto' in df1.columns:
    df1 = df1.drop(columns=['copypaste_score_auto'])
df = df1.merge(df2[['conversation_id', 'copypaste_score_auto']], on='conversation_id', how='left')

# Calculate correlations
metrics = ['copypaste_score_auto', 'substitute_learning', 'complement_learning']
corr_matrix = df[metrics].corr()

print("Correlation Matrix:")
print(corr_matrix)

# Look at average substitute/complement learning scores for high vs low copy-pasters (LLM score)
print("\nMean scores by Auto Copy-Paste Score (1-5):")
print(df.groupby('copypaste_score_auto')[['substitute_learning', 'complement_learning']].mean())

# Identify specific "Low Effort" profile
low_effort = df[(df['copypaste_score_auto'] >= 4) & (df['substitute_learning'] >= 4)]
print(f"\nNumber of 'Low Effort' (High Copy + High Substitute) conversations: {len(low_effort)}")
