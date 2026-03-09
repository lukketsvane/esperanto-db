import pandas as pd
import json
from pathlib import Path

# Load evaluated data
df = pd.read_csv('conversations_evaluated_final.csv')

# Find conversations with high copy-paste scores
# We want to show cases where auto-score and LLM-score agree (blatant copy-paste)
# and maybe one where they disagree (structural copy-paste)
blatant_cp = df[(df['copypaste_dummy'] >= 4) & (df['copypaste_score_auto'] >= 3)].head(2)
structural_cp = df[(df['copypaste_dummy'] == 5) & (df['copypaste_score_auto'] == 1)].head(2)

base_dir = Path('raw_data/csn_exports')

def get_conv_text(conv_id, csn):
    folder = base_dir / str(csn)
    if not folder.exists():
        return None
    
    conv_file = folder / "conversations.json"
    if not conv_file.exists():
        subdirs = [d for d in folder.iterdir() if d.is_dir()]
        if subdirs: conv_file = subdirs[0] / "conversations.json"
    
    if not conv_file.exists(): return None
    
    with open(conv_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        conversations = data if isinstance(data, list) else [data]
        for conv in conversations:
            if conv.get('id') == str(conv_id):
                msgs = []
                if 'mapping' in conv:
                    nodes_with_time = []
                    for node_id, node in conv['mapping'].items():
                        msg = node.get('message')
                        if msg and msg.get('author', {}).get('role') in ['user', 'assistant']:
                            role = msg['author']['role']
                            content = msg.get('content', {})
                            create_time = msg.get('create_time', 0) or 0
                            if isinstance(content, dict) and 'parts' in content:
                                parts = content['parts']
                                if isinstance(parts, list) and len(parts) > 0:
                                    text = parts[0]
                                    if text and isinstance(text, str) and text.strip():
                                        nodes_with_time.append((create_time, f"{role.upper()}: {text.strip()}"))
                    nodes_with_time.sort(key=lambda x: x[0])
                    msgs = [x[1] for x in nodes_with_time]
                return msgs
    return None

print("=== EXAMPLES OF COPY-PASTING FOR PAPER PRESENTATION ===\n")

print("1. BLATANT VERBATIM COPY-PASTING (Caught by Auto-Detector & LLM)")
print("These students pasted the exact assignment strings without any framing.")
for _, row in blatant_cp.iterrows():
    csn = row.get('csn_x', row.get('csn'))
    print(f"\n[Participant ID: {row['conversation_id']} | CSN: {csn}]")
    msgs = get_conv_text(row['conversation_id'], csn)
    if msgs:
        # Just print the first 4 interactions
        for m in msgs[:4]:
            # Truncate long assistant responses
            if m.startswith("ASSISTANT:") and len(m) > 150:
                print(m[:150] + " ... [TRUNCATED]")
            else:
                print(m)
    else:
        print("Data not found.")

print("\n\n2. STRUCTURAL/PARAPHRASED COPY-PASTING (Caught only by LLM)")
print("These students copied the intent or slightly reworded the assignment, evading strict string matching but caught by the AI Judge.")
for _, row in structural_cp.iterrows():
    csn = row.get('csn_x', row.get('csn'))
    print(f"\n[Participant ID: {row['conversation_id']} | CSN: {csn}]")
    msgs = get_conv_text(row['conversation_id'], csn)
    if msgs:
        for m in msgs[:4]:
            if m.startswith("ASSISTANT:") and len(m) > 150:
                print(m[:150] + " ... [TRUNCATED]")
            else:
                print(m)
    else:
        print("Data not found.")
