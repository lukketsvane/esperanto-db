import pandas as pd

auto_df = pd.read_csv('copypaste_auto_detection.csv')
eval_df = pd.read_csv('conversations_evaluated_explicit.csv')

# Merge and compare
merged = eval_df.merge(auto_df, on='conversation_id', how='left')

print("Comparison of Auto vs LLM Copypaste Score:")
# Just look at some where they differ or where scores are high
high_scores = merged[(merged['copypaste_dummy'] > 3) | (merged['copypaste_score_auto'] > 1)]
print(high_scores[['conversation_id', 'copypaste_dummy', 'copypaste_score_auto']].head(20))

# Save the unified result
merged.to_csv('conversations_evaluated_final.csv', index=False)
print(f"Final merged results saved to conversations_evaluated_final.csv")
