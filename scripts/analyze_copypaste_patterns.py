import pandas as pd
import numpy as pd_np # using numpy for some stats if needed

df = pd.read_csv('conversations_evaluated_final.csv')

# Calculate correlations
metrics = ['copypaste_dummy', 'copypaste_score_auto', 'substitute_learning', 'complement_learning']
corr_matrix = df[metrics].corr()

print("Correlation Matrix:")
print(corr_matrix)

# Look at average substitute/complement learning scores for high vs low copy-pasters (LLM score)
print("\nMean scores by LLM Copy-Paste Dummy (1-5):")
print(df.groupby('copypaste_dummy')[['substitute_learning', 'complement_learning']].mean())

# Look at average substitute/complement learning scores for high vs low copy-pasters (Auto score)
print("\nMean scores by Auto Copy-Paste Score (1-5):")
print(df.groupby('copypaste_score_auto')[['substitute_learning', 'complement_learning']].mean())

# Identify specific "Low Effort" profile
low_effort = df[(df['copypaste_dummy'] >= 4) & (df['substitute_learning'] >= 4)]
print(f"\nNumber of 'Low Effort' (High Copy + High Substitute) conversations: {len(low_effort)}")
