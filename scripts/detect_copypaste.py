import json
import pandas as pd
from pathlib import Path
import re

# Known practice questions (from the data exploration)
PRACTICE_QUESTIONS = [
    "Tom does not sleep",
    "Anna does not learn",
    "The dog does not fly",
    "The bird is small",
    "Tom ne dormas",
    "Anna ne lernas",
    "La hundo ne flugas",
    "La birdo estas malgranda",
    "Anna and Tom are not learning Esperanto quickly",
    "The young boy does not sleep, but the big dog sleeps",
    "Anna kaj Tom ne estas lernas Esperanto rapide",
    "La juna knabino ne dormas, sed la granda hundo dormas",
    "The cats are big",
    "La katoj estas grandaj",
    "Men are slow",
    "Viroj estas malrapidaj",
    "I am happy because I do not work today",
    "Mi estas feliĉa ĉar mi ne laboras hodiaŭ",
    "The young girl",
    "The big table",
    "The beautiful woman",
    "The fast dog",
    "The bad teacher",
    "Anna learns well",
    "La juna knabino",
    "La granda tablo",
    "La bela virino",
    "La rapida hundo",
    "La malbona instruisto",
    "Anna lernas bone"
]

def extract_user_messages(conversation_data):
    messages = []
    if 'mapping' in conversation_data:
        for node_id, node in conversation_data['mapping'].items():
            msg = node.get('message')
            if msg and msg.get('author', {}).get('role') == 'user':
                content = msg.get('content', {})
                if isinstance(content, dict) and 'parts' in content:
                    parts = content['parts']
                    if isinstance(parts, list) and len(parts) > 0:
                        text = parts[0]
                        if text and isinstance(text, str):
                            messages.append(text.strip())
    return messages

def calculate_copypaste_score(messages):
    if not messages:
        return 1
    
    matches = 0
    total_len = len(messages)
    
    for msg in messages:
        for q in PRACTICE_QUESTIONS:
            # Check if question is in message (case insensitive, basic fuzzy)
            if q.lower() in msg.lower():
                matches += 1
                break
                
    ratio = matches / total_len if total_len > 0 else 0
    
    # Map ratio to 1-5 scale
    if ratio == 0: return 1
    if ratio < 0.2: return 2
    if ratio < 0.5: return 3
    if ratio < 0.8: return 4
    return 5

def main():
    base_dir = Path('.').resolve()
    csn_base = base_dir / "raw_data" / "csn_exports"
    
    results = []
    
    csn_folders = sorted([d for d in csn_base.iterdir() if d.is_dir() and d.name.startswith("CSN")])
    for csn_folder in csn_folders:
        csn_name = csn_folder.name
        conv_file = csn_folder / "conversations.json"
        if not conv_file.exists():
            subdirs = [d for d in csn_folder.iterdir() if d.is_dir()]
            if subdirs: conv_file = subdirs[0] / "conversations.json"
        
        if conv_file.exists():
            with open(conv_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    conversations = data if isinstance(data, list) else [data]
                    for conv in conversations:
                        conv_id = conv.get('id', 'unknown')
                        user_msgs = extract_user_messages(conv)
                        score = calculate_copypaste_score(user_msgs)
                        results.append({
                            "conversation_id": conv_id,
                            "csn": csn_name,
                            "copypaste_score_auto": score,
                            "n_user_messages": len(user_msgs)
                        })
                except:
                    continue

    df = pd.DataFrame(results)
    df.to_csv('copypaste_auto_detection.csv', index=False)
    print(f"Auto-detection complete. Saved to copypaste_auto_detection.csv")
    print(df['copypaste_score_auto'].value_counts().sort_index())

if __name__ == "__main__":
    main()
