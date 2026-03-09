import pandas as pd

print("Loading evaluation data...")
eval_df = pd.read_csv('conversations_evaluated_final.csv')

print("Loading full sample data...")
conv_df = pd.read_csv('data/01_full_sample_with_prompts.csv', on_bad_lines='skip', engine='python')

print("Merging datasets on 'conversation_id'...")
# Merge, bringing all evaluation metrics into the full dataset
# We use left merge on conv_df to keep all original participant data
merged = conv_df.merge(eval_df, on='conversation_id', how='left', suffixes=('', '_eval'))

# Clean up duplicate columns if any (like csn_eval)
cols_to_drop = [c for c in merged.columns if c.endswith('_eval') or c.endswith('_x') or c.endswith('_y')]
merged = merged.drop(columns=cols_to_drop)

output_file = 'data/01_full_sample_with_prompts_EVALUATED.csv'
merged.to_csv(output_file, index=False)

print(f"Merge complete! Saved to {output_file}")
print(f"Total rows: {len(merged)}")
print(f"Rows with evaluations: {merged['copypaste_dummy'].notna().sum()}")
