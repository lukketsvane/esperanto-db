"""
Copy-Paste Behavior vs. Test Score Analysis
Examines how copy-paste level (1-5) correlates with exam outcomes.

Maximizes participant linkage by matching unlinked survey respondents
to conversations via CSN/PC mapping and session timing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Load all data sources ────────────────────────────────────────────────────
df_eval = pd.read_csv('conversations_evaluated_final.csv')
df_main = pd.read_csv('data/01_full_sample_with_prompts.csv', on_bad_lines='skip')
df_conv = pd.read_csv('data/conversations_final.csv')

print("=" * 60)
print("DATA LINKAGE REPORT")
print("=" * 60)
print(f"Conversations evaluated:       {len(df_eval)}")
print(f"Unique conversation participants: {df_conv['participant_id'].nunique()}")
print(f"Survey respondents (with data): {df_main['StartDate'].notna().sum()}")
print(f"Survey respondents with testscore: {df_main['testscore'].notna().sum()}")
print()

# ── Step 1: Direct merge (existing linkage) ──────────────────────────────────
df_direct = df_eval.merge(
    df_main[['conversation_id', 'testscore', 'treatment', 'gender', 'highgpa']],
    on='conversation_id', how='inner'
).dropna(subset=['copypaste_dummy', 'testscore'])

print(f"Direct merge (existing links): {len(df_direct)} participants")

# ── Step 2: Match unlinked AI survey respondents via CSN + session ───────────
has_data = df_main[df_main['StartDate'].notna()]
linked_conv_ids = set(df_direct['conversation_id'])
linked_pids = set(df_main.loc[df_main['conversation_id'].notna(), 'participant_id'].dropna())

# Build session → (date, time_range) mapping from already-linked rows
linked_rows = has_data[has_data['conversation_id'].notna()]
session_info = linked_rows.groupby('session').agg(
    date=('startdate', 'first'),
    hour_min=('starthour', 'min'),
    hour_max=('starthour', 'max')
).to_dict('index')

# Prepare conversation data with CSN number and local time
df_conv['csn_num'] = df_conv['csn'].str.extract(r'(\d+)').astype(float)
df_conv['create_dt'] = pd.to_datetime(df_conv['create_time'], unit='s')
df_conv['create_day'] = df_conv['create_dt'].dt.day
df_conv['create_hour_utc'] = df_conv['create_dt'].dt.hour + df_conv['create_dt'].dt.minute / 60
df_conv['create_hour_local'] = df_conv['create_hour_utc'] + 1  # UTC → CET

# Assign each conversation an estimated session based on day + time
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

# Find unlinked AI survey respondents (assist/guided without conversation_id)
unlinked_ai = has_data[
    (has_data['treatment'].isin(['assist', 'guided'])) &
    (has_data['conversation_id'].isna()) &
    (has_data['testscore'].notna())
]

# Match by PC (= CSN number) and session
new_matches = []
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
        new_matches.append({
            'conversation_id': match['conversation_id'],
            'testscore': row['testscore'],
            'treatment': row['treatment'],
            'gender': row.get('gender'),
            'highgpa': row.get('highgpa'),
        })
        linked_pids.add(match['participant_id'])
        linked_conv_ids.add(match['conversation_id'])

print(f"New matches via CSN+session:    {len(new_matches)} participants")

# Merge the new matches with evaluation data
if new_matches:
    df_new = pd.DataFrame(new_matches)
    df_new_merged = df_eval.merge(df_new, on='conversation_id', how='inner')
    df_new_merged = df_new_merged.dropna(subset=['copypaste_dummy', 'testscore'])
    df = pd.concat([df_direct, df_new_merged], ignore_index=True)
else:
    df = df_direct

df['copypaste_dummy'] = df['copypaste_dummy'].astype(int)

# ── Step 3: Deduplicate (some conversations may share same survey row) ────────
df = df.drop_duplicates(subset=['conversation_id'], keep='first')

print(f"Final sample:                   {len(df)} participants")
print()
print(f"Total conversation participants: 375")
print(f"Survey respondents with test score: {has_data['testscore'].notna().sum()}")
print(f"  - AI users (assist+guided): {has_data[has_data['treatment'].isin(['assist','guided'])]['testscore'].notna().sum()}")
print(f"  - Control (no conversations): {has_data[(has_data['treatment']=='control') | ((has_data['treatment'].isna()) & (has_data[['control']].sum(axis=1)>0))]['testscore'].notna().sum() if 'control' in has_data.columns else 'N/A'}")
print(f"Linked with copypaste + testscore: {len(df)}")
print(f"Gap: {375 - len(df)} participants had no survey test score")
print()

# ── Summary statistics ────────────────────────────────────────────────────────
summary = df.groupby('copypaste_dummy')['testscore'].agg(['mean', 'std', 'count', 'median'])
summary.columns = ['Mean', 'Std', 'N', 'Median']
summary['SE'] = summary['Std'] / np.sqrt(summary['N'])
print("Mean test score by copy-paste level:")
print(summary.round(2).to_string())
print()

# Overall correlation
r, p = stats.pearsonr(df['copypaste_dummy'], df['testscore'])
rho, p_rho = stats.spearmanr(df['copypaste_dummy'], df['testscore'])
print(f"Pearson r = {r:.3f}, p = {p:.4f}")
print(f"Spearman ρ = {rho:.3f}, p = {p_rho:.4f}")
print()

# ── Labels ────────────────────────────────────────────────────────────────────
cp_labels = {
    1: "1\nOriginal",
    2: "2\nMostly\nOriginal",
    3: "3\nMixed",
    4: "4\nMostly\nCopied",
    5: "5\nVerbatim\nCopy"
}

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor('#FAFAFA')

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35,
                       left=0.08, right=0.96, top=0.88, bottom=0.10)

PALETTE = {1: '#2166AC', 2: '#74ADD1', 3: '#FEE090', 4: '#F46D43', 5: '#D73027'}
LEVELS = [1, 2, 3, 4, 5]

# ─ Panel A: Bar chart ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(
    [cp_labels[l] for l in LEVELS],
    [summary.loc[l, 'Mean'] if l in summary.index else np.nan for l in LEVELS],
    yerr=[summary.loc[l, 'SE'] * 1.96 if l in summary.index else 0 for l in LEVELS],
    color=[PALETTE[l] for l in LEVELS],
    width=0.6, capsize=5, error_kw=dict(elinewidth=1.5, ecolor='#333333'),
    edgecolor='white', linewidth=0.8
)

# Annotate n=
for i, l in enumerate(LEVELS):
    if l in summary.index:
        n = int(summary.loc[l, 'N'])
        mean = summary.loc[l, 'Mean']
        ax1.text(i, mean + summary.loc[l, 'SE'] * 1.96 + 0.25, f"n={n}",
                 ha='center', va='bottom', fontsize=8.5, color='#444444')

ax1.set_ylim(0, 12)
ax1.set_yticks([0, 2, 4, 6, 8, 10, 12])
ax1.set_ylabel('Mean Test Score ± 95% CI', fontsize=10)
ax1.set_title('A  Mean Test Score by Copy-Paste Level', fontsize=11, fontweight='bold', loc='left', pad=8)
ax1.axhline(df['testscore'].mean(), color='#555555', linestyle='--', linewidth=1, alpha=0.7, label=f'Overall mean ({df["testscore"].mean():.1f})')
ax1.legend(fontsize=8.5, framealpha=0.7)
ax1.set_facecolor('#F7F7F7')
ax1.spines[['top', 'right']].set_visible(False)

# ─ Panel B: Box plot ──────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
groups = [df[df['copypaste_dummy'] == l]['testscore'].dropna().values for l in LEVELS]

bp = ax2.boxplot(groups, patch_artist=True, notch=False,
                 medianprops=dict(color='white', linewidth=2),
                 whiskerprops=dict(linewidth=1.2),
                 capprops=dict(linewidth=1.2),
                 flierprops=dict(marker='o', markersize=4, alpha=0.5))

for patch, l in zip(bp['boxes'], LEVELS):
    patch.set_facecolor(PALETTE[l])
    patch.set_alpha(0.85)

ax2.set_xticklabels([cp_labels[l] for l in LEVELS], fontsize=8.5)
ax2.set_ylabel('Test Score', fontsize=10)
ax2.set_title('B  Distribution of Test Scores\n    by Copy-Paste Level', fontsize=11, fontweight='bold', loc='left', pad=8)
ax2.axhline(df['testscore'].mean(), color='#555555', linestyle='--', linewidth=1, alpha=0.7)
ax2.set_facecolor('#F7F7F7')
ax2.spines[['top', 'right']].set_visible(False)

# ─ Panel C: Scatter with jitter ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])

jitter = np.random.RandomState(42).uniform(-0.18, 0.18, len(df))
colors_scatter = [PALETTE[l] for l in df['copypaste_dummy']]
ax3.scatter(df['copypaste_dummy'] + jitter, df['testscore'],
            c=colors_scatter, s=28, alpha=0.55, edgecolors='none')

# Regression line
slope, intercept, *_ = stats.linregress(df['copypaste_dummy'], df['testscore'])
x_line = np.linspace(0.8, 5.2, 100)
ax3.plot(x_line, slope * x_line + intercept, color='#333333',
         linewidth=2, linestyle='-', label=f'Trend (r={r:.2f}, p={p:.3f})')

ax3.set_xlabel('Copy-Paste Level', fontsize=10)
ax3.set_ylabel('Test Score', fontsize=10)
ax3.set_xticks(LEVELS)
ax3.set_xticklabels([str(l) for l in LEVELS])
ax3.set_title('C  Individual Scores + Trend Line', fontsize=11, fontweight='bold', loc='left', pad=8)
ax3.legend(fontsize=8.5, framealpha=0.8)
ax3.set_facecolor('#F7F7F7')
ax3.spines[['top', 'right']].set_visible(False)

# ─ Panel D: Proportion high/low scorers ──────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

score_cutoff_high = df['testscore'].quantile(0.67)  # top third
score_cutoff_low  = df['testscore'].quantile(0.33)  # bottom third

prop_high = []
prop_low  = []
ns_bar    = []
for l in LEVELS:
    sub = df[df['copypaste_dummy'] == l]['testscore'].dropna()
    n = len(sub)
    ns_bar.append(n)
    prop_high.append((sub >= score_cutoff_high).mean() * 100 if n > 0 else 0)
    prop_low.append((sub <= score_cutoff_low).mean() * 100 if n > 0 else 0)

x = np.arange(len(LEVELS))
width = 0.35
bars_high = ax4.bar(x - width/2, prop_high, width, label='Top 33% scorers',
                    color='#2166AC', alpha=0.85, edgecolor='white')
bars_low  = ax4.bar(x + width/2, prop_low,  width, label='Bottom 33% scorers',
                    color='#D73027', alpha=0.85, edgecolor='white')

ax4.set_xticks(x)
ax4.set_xticklabels([cp_labels[l] for l in LEVELS], fontsize=8.5)
ax4.set_ylabel('% of Group', fontsize=10)
ax4.set_ylim(0, 80)
ax4.set_title('D  Share of High vs. Low Scorers\n    by Copy-Paste Level', fontsize=11, fontweight='bold', loc='left', pad=8)
ax4.legend(fontsize=8.5, framealpha=0.8)
ax4.set_facecolor('#F7F7F7')
ax4.spines[['top', 'right']].set_visible(False)

# ── Main title ────────────────────────────────────────────────────────────────
fig.suptitle(
    f'Copy-Paste Behavior and Test Score  |  N = {len(df)}  |  r = {r:.2f}, p = {p:.3f}',
    fontsize=13, fontweight='bold', y=0.97, color='#222222'
)

# ── Save ──────────────────────────────────────────────────────────────────────
out_png = 'figures/copypaste_vs_testscore.png'
out_pdf = 'figures/copypaste_vs_testscore.pdf'
fig.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
fig.savefig(out_pdf, bbox_inches='tight', facecolor='#FAFAFA')
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")

# ── Print clean summary table ─────────────────────────────────────────────────
print("\n── Summary statistics ──────────────────────────────────────────")
print(f"{'Level':<8} {'Label':<22} {'N':>4} {'Mean':>6} {'Std':>5} {'Median':>7}")
print("-" * 55)
label_map = {1:'Original', 2:'Mostly Original', 3:'Mixed', 4:'Mostly Copied', 5:'Verbatim Copy'}
for l in LEVELS:
    if l in summary.index:
        row = summary.loc[l]
        print(f"{l:<8} {label_map[l]:<22} {int(row['N']):>4} {row['Mean']:>6.2f} {row['Std']:>5.2f} {row['Median']:>7.1f}")
print("-" * 55)
print(f"{'Overall':<8} {'':<22} {len(df):>4} {df['testscore'].mean():>6.2f} {df['testscore'].std():>5.2f} {df['testscore'].median():>7.1f}")
