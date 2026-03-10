import json
import pandas as pd
from pathlib import Path
import re

# Known practice questions (from the filtered & renumbered question list, 92 total)
# Each entry is a key phrase from the question used for string-match copy-paste detection.
PRACTICE_QUESTIONS = [
    # Q1 (Q142) - Q2 (Q143): Translation sentences
    "The bird flies",
    "The boy learns Esperanto",
    # Q3-Q6 (Q18-Q22): Fill-in-the-blank
    "Mi _____ Anna",
    "La knabo _____ Esperanton",
    "La birdo _____ rapide",
    "Tom kaj Anna _____ amikoj",
    # Q7-Q10 (Q145-Q148): Translation sentences
    "Tom dormas",
    "Anna lernas",
    "La hundo flugas",
    "La birdo estas granda",
    # Q11-Q14 (Q25-Q28): Fill-in-the-blank with hints
    "Tom _____ bone hieraŭ",
    "Anna _____ rapide",
    "La instruisto _____ bona",
    "La birdo _____ _____ rapide",
    # Q15-Q19 (Q30-Q34): Vocabulary - translate to English
    "La kato",
    "Apartamento",
    "La instruisto",
    "Papero",
    "La virino",
    # Q20-Q23 (Q151-Q154): Vocabulary - translate to Esperanto
    "woman",
    "cat",
    "boy",
    "dog",
    # Q24-Q26 (Q36-Q38): Fill-in-the-blank with English hint
    "La _____ (man) dormas",
    "La _____ (cat) estas malgrandaj",
    "La _____ (dog) ludas rapide",
    # Q27 (Q39): Plural transformation
    "Make these sentences plural",
    # Q28-Q31 (Q156-Q159): Adjective + noun phrases
    "La granda viro",
    "La malbona hundo",
    "La bela kato",
    "La juna instruisto",
    # Q32-Q34 (Q40-Q42): Fill-in adjective
    "La _____ (beautiful) kato estas dormanta",
    "La _____ (quick) birdo ne flugas",
    "La _____ (good) viroj lernas multe",
    # Q35-Q37 (Q43-Q45): Fill-in adverb
    "La birdo ne flugas _____ (beautifully)",
    "La instruisto instruas _____ (well)",
    "Mi lernas Esperanto _____ (quickly)",
    # Q38 (Q47): Combined adjective + adverb
    "La _____ (beautiful) birdo flugas _____ (beautifully)",
    # Q39 (Q49): Error identification
    "La hundo estas bele",
    # Q40 (Q61)
    "La birdo flugas bele",
    # Q41 (Q63)
    "Mi estas granda viro",
    # Q42 (Q67)
    "Ĉar mi fartas bone, mi estas malgranda kato",
    # Q43 (Q69)
    "La katoj estas granda",
    # Q44 (Q71)
    "Ĉu vi estas instruisto",
    # Q45 (Q73)
    "Viroj estas malrapide",
    # Q46 (Q79)
    "Mi estas feliĉajn ĉar mi ne laboras hodiaŭ",
    # Q47 (Q52)
    "Kie estas la hundo",
    # Q48 (Q81)
    "Ĝi flugas rapide",
    # Q49 (Q83)
    "Kiel vi fartas",
    # Q50 (Q84)
    "La viro dancas malrapide",
    # Q51 (Q85)
    "ĉar ili estas lacaj",
    # Q52 (Q54): Reading comprehension passage 1
    "Kiel vi fartas, Tom? Mi pensas ke la Esperanto-instruisto estas malrapida",
    # Q53-Q57 (Q55-Q59): Comprehension questions for passage 1
    "Is the Esperanto teacher a man or a woman",
    "Who is happier with the teacher, Tom or Anna",
    "Does Tom agree that the teacher is slow",
    "Why does Anna think the teacher is slow",
    "What does Anna say at the end about learning Esperanto",
    # Q58 (Q87): Reading comprehension passage 2
    "Saluton, Anna kaj Tom! Ĉu vi dormis bone",
    # Q59-Q63 (Q88-Q92): Comprehension questions for passage 2
    "Did Anna sleep well",
    "Why didn't Tom sleep well",
    "What does Anna ask about Tom's cat",
    "Why can't Tom's cat play outside the room",
    "What does the teacher say about the cat at the end",
    # Q64 (Q60)
    "The young girl",
    # Q65-Q68 (Q93-Q96): Translation phrases
    "The big table",
    "The beautiful woman",
    "The quick dog",
    "The bad teacher",
    # Q69-Q72 (Q97-Q100): Translation sentences
    "Anna learns well",
    "The dog sleeps quickly",
    "The teacher speaks beautifully",
    "The small bird flies youthfully",
    # Q73 (Q101)
    "The old girls were quick",
    # Q74 (Q102)
    "The big cat sleeps on the table",
    # Q75 (Q104)
    "The dog is small, but it runs fast",
    # Q76 (Q105)
    "The man and the woman are happy because the lesson is interesting",
    # Q77 (Q107)
    "The birds fly beautifully, but the cat is slow",
    # Q78 (Q108)
    "Because the teacher is good, we learn Esperanto fast",
    # Q79 (Q109)
    "The teacher is young and speaks beautifully, but Anna and Tom were tired because they learned a lot",
    # Q80 (Q110)
    "The man is happy because he lives in a beautiful apartment",
    # Q81 (Q111)
    "Are you sad",
    # Q82-Q87 (Q127-Q133): Sentence completion / matching
    "La viro estas laca",
    "ĝi estas feliĉa",
    "La knabo ne ludas",
    "La instruisto lernas multe",
    "La virino kaj la viro",
    "ĉar li estas laca",
    # Q88 (Q13): True/false
    "La Turo Eiffel estas granda",
    # Q89 (Q136)
    "La Taj-Mahalo estas en Francio",
    # Q90 (Q137)
    "La kato flugas rapide",
    # Q91 (Q138)
    "La piramidoj de Egiptio estas tre malgrandaj",
    # Q92 (Q140)
    "La hundo estas feliĉa, ĉar ĝi iras al parko",
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
