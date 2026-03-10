import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup plot style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid", palette="muted")

# Ensure figures directory exists
Path("figures").mkdir(exist_ok=True)

print("Loading dataset...")
df1 = pd.read_csv('conversations_evaluated_final.csv')
df2 = pd.read_csv('copypaste_auto_detection.csv')
if 'copypaste_score_auto' in df1.columns:
    df1 = df1.drop(columns=['copypaste_score_auto'])
df = df1.merge(df2[['conversation_id', 'copypaste_score_auto']], on='conversation_id', how='left')

# Drop rows where we don't have evaluation data
eval_df = df.dropna(subset=['copypaste_score_auto', 'substitute_learning', 'complement_learning']).copy()

# Ensure numeric types
for col in ['copypaste_score_auto', 'substitute_learning', 'complement_learning']:
    eval_df[col] = pd.to_numeric(eval_df[col], errors='coerce')

eval_df = eval_df.dropna(subset=['copypaste_score_auto', 'substitute_learning', 'complement_learning'])

print(f"Data ready. Generating plots for {len(eval_df)} evaluated conversations...")

# --- PLOT 1: Substitute vs. Complement Learning by Copy-Paste Level ---
plt.figure(figsize=(10, 6))
# Create a categorical variable for cleaner plotting
eval_df['Copy-Paste Level (LLM)'] = eval_df['copypaste_score_auto'].round().astype(int)

# Use jitter to prevent overplotting
sns.stripplot(
    data=eval_df, 
    x='substitute_learning', 
    y='complement_learning', 
    hue='Copy-Paste Level (LLM)',
    palette='coolwarm',
    size=8, 
    alpha=0.7,
    jitter=0.2
)

plt.title('The "Black Box" Effect: Substitute vs. Complement Learning\nColored by Copy-Paste Severity', fontsize=14, pad=15)
plt.xlabel('Substitute Learning (1 = Low Reliance, 5 = High Reliance)', fontsize=12)
plt.ylabel('Complement Learning (1 = Low Augmentation, 5 = High Augmentation)', fontsize=12)
plt.legend(title='Copy-Paste Score (1-5)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/substitute_vs_complement_scatter.png', dpi=300)
plt.close()

# --- PLOT 2: Average Learning Quality by Copy-Paste Score ---
plt.figure(figsize=(9, 6))
# Calculate means
agg_df = eval_df.groupby('Copy-Paste Level (LLM)')[['substitute_learning', 'complement_learning']].mean().reset_index()
agg_melted = agg_df.melt(id_vars='Copy-Paste Level (LLM)', var_name='Metric', value_name='Average Score')
# Rename metrics for display
agg_melted['Metric'] = agg_melted['Metric'].replace({
    'substitute_learning': 'Substitute Learning (Rely on AI)',
    'complement_learning': 'Complement Learning (Augment with AI)'
})

sns.barplot(
    data=agg_melted,
    x='Copy-Paste Level (LLM)',
    y='Average Score',
    hue='Metric',
    palette=['#e74c3c', '#4facfe']
)

plt.title('Divergent AI Use: How Copy-Pasting Kills Complementary Learning', fontsize=14, pad=15)
plt.xlabel('Copy-Paste Severity (1 = Original Prompts, 5 = Verbatim Copying)', fontsize=12)
plt.ylabel('Average Score (1-5 Scale)', fontsize=12)
plt.ylim(0, 5)
plt.legend(title='AI Usage Type', loc='upper right')
plt.tight_layout()
plt.savefig('figures/copypaste_impact_bar.png', dpi=300)
plt.close()

print("Plots successfully saved to 'figures/' directory.")
