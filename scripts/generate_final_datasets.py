import pandas as pd

# Load main participant data
df_main = pd.read_csv('data/01_full_sample_with_prompts.csv', low_memory=False, on_bad_lines='skip')

# Load explicit evaluation
df_eval = pd.read_csv('conversations_evaluated_explicit.csv')

# Load auto scores
df_auto = pd.read_csv('copypaste_auto_detection.csv')

# Load conversation metadata
df_conv = pd.read_csv('data/conversations_final.csv')

# Merge auto scores into eval
df_eval = df_eval.merge(df_auto[['conversation_id', 'copypaste_score_auto']], on='conversation_id', how='left')

# The complex matching logic from analyze_copypaste_vs_testscore.py
has_data = df_main[df_main['StartDate'].notna()]
linked_conv_ids = set(has_data['conversation_id'].dropna())
linked_pids = set(has_data['participant_id'].dropna())

linked_rows = has_data[has_data['conversation_id'].notna()]
session_info = linked_rows.groupby('session').agg(
    date=('startdate', 'first'),
    hour_min=('starthour', 'min'),
    hour_max=('starthour', 'max')
).to_dict('index')

df_conv['csn_num'] = df_conv['csn'].str.extract(r'(\d+)').astype(float)
df_conv['create_dt'] = pd.to_datetime(df_conv['create_time'], unit='s')
df_conv['create_day'] = df_conv['create_dt'].dt.day
df_conv['create_hour_utc'] = df_conv['create_dt'].dt.hour + df_conv['create_dt'].dt.minute / 60
df_conv['create_hour_local'] = df_conv['create_hour_utc'] + 1  # UTC → CET

def assign_session(row):
    day = row['create_day']
    hour = row['create_hour_local']
    best_session, best_diff = None, float('inf')
    for s, info in session_info.items():
        if info['date'] == day:
            mid = (info['hour_min'] + info['hour_max']) / 2
            diff = abs(hour - mid)
            if diff < best_diff:
                best_diff = diff
                best_session = s
    return best_session

df_conv['estimated_session'] = df_conv.apply(assign_session, axis=1)

unlinked_ai = has_data[
    (has_data['treatment'].isin(['assist', 'guided'])) &
    (has_data['conversation_id'].isna()) &
    (has_data['testscore'].notna())
]

new_matches = {}
for idx, row in unlinked_ai.iterrows():
    pc = row['PC']
    session = row['session']
    candidates = df_conv[
        (df_conv['csn_num'] == pc) &
        (df_conv['estimated_session'] == session) &
        (~df_conv['participant_id'].isin(linked_pids)) &
        (~df_conv['conversation_id'].isin(linked_conv_ids))
    ]
    if len(candidates) == 1:
        match = candidates.iloc[0]
        # map main index to conversation id
        new_matches[idx] = match['conversation_id']
        linked_pids.add(match['participant_id'])
        linked_conv_ids.add(match['conversation_id'])

# Update df_main with newly found conversation_ids
for idx, conv_id in new_matches.items():
    df_main.at[idx, 'conversation_id'] = conv_id

# Merge the full evaluation data onto df_main based on conversation_id
# We should avoid duplicating columns that might already exist
eval_cols_to_merge = [c for c in df_eval.columns if c != 'csn']
df_main_eval = df_main.merge(df_eval[eval_cols_to_merge], on='conversation_id', how='left')

# Save it to EVALUATED
df_main_eval.to_csv('data/01_full_sample_with_prompts_EVALUATED.csv', index=False)
print(f"Generated data/01_full_sample_with_prompts_EVALUATED.csv with {len(df_main_eval)} rows")
print(f"Rows with explicit evaluation data: {df_main_eval['substitute_learning'].notna().sum()}")

# ----------------------------------------------------
# Also, let's clean up conversations_evaluated_final.csv
# so it doesn't have csn_x and csn_y

df_eval_final = pd.read_csv('conversations_evaluated_explicit.csv')
df_eval_final = df_eval_final.merge(df_auto[['conversation_id', 'copypaste_score_auto']], on='conversation_id', how='left')
df_eval_final.to_csv('conversations_evaluated_final.csv', index=False)
print(f"Cleaned conversations_evaluated_final.csv (columns: {len(df_eval_final.columns)})")
