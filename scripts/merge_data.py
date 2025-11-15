#!/usr/bin/env python3
import pandas as pd

eval_df = pd.read_csv('conversations_evaluated.csv')
conv_df = pd.read_csv('prompt_conversations_final_for_paper.csv')

merged = conv_df.merge(eval_df, on='conversation_id', how='left', suffixes=('', '_eval'))

cols_to_drop = [c for c in merged.columns if c.endswith('_eval') and c != 'error']
merged = merged.drop(columns=cols_to_drop)

merged.to_csv('conversations_final.csv', index=False)

print(f"Merged {len(merged)} conversations")
print(f"Columns: {len(merged.columns)}")
print(f"With evaluations: {merged['cognitive_engagement'].notna().sum()}")
