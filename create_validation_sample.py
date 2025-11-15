#!/usr/bin/env python3
"""
Create Validation Sample for Manual Coding

Phase 1 of Literature-Grounded Assessment Protocol:
- Randomly sample 50 conversations (10% of 397)
- Export for manual coding by 2 independent raters
- Create structured coding sheets based on validated instruments

Purpose: Establish ground truth for LLM-as-judge validation
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)


def load_conversation_from_csn(csn_folder: Path) -> list:
    """Load conversation data from a CSN folder."""
    conv_file = csn_folder / "conversations.json"

    if not conv_file.exists():
        subdirs = [d for d in csn_folder.iterdir() if d.is_dir()]
        if subdirs:
            conv_file = subdirs[0] / "conversations.json"

    if not conv_file.exists():
        return []

    try:
        with open(conv_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    except Exception:
        return []


def extract_user_messages(conversation_data: dict) -> list:
    """Extract user messages from conversation JSON."""
    user_messages = []

    if 'mapping' in conversation_data:
        for node_id, node in conversation_data['mapping'].items():
            if node.get('message') and node['message'].get('author', {}).get('role') == 'user':
                content = node['message'].get('content', {})
                if isinstance(content, dict) and 'parts' in content:
                    parts = content['parts']
                    if isinstance(parts, list) and len(parts) > 0:
                        text = parts[0]
                        if text and text.strip():
                            user_messages.append(text.strip())

    return user_messages


def create_coding_sheet_template():
    """Generate manual coding sheet template based on validated frameworks."""

    template = """# MANUAL CODING SHEET - ESPERANTO LEARNING CONVERSATIONS
## Validated Assessment Framework

**Rater Name**: _______________
**Date**: _______________
**Conversation ID**: _______________

---

## INSTRUCTIONS
Read the entire conversation before rating. Base all ratings on observable evidence only.
Use the 1-5 or 1-7 scales as indicated. Provide brief justification for each rating.

---

## 1. COGNITIVE LOAD (Klepsch et al. 2017)
**Scale: 1 (Not at all) to 7 (Completely)**

**Intrinsic Load** (Task Complexity):
"The Esperanto topics and questions in this conversation were complex."
Rating: [ 1 | 2 | 3 | 4 | 5 | 6 | 7 ]
Evidence: _______________________________________________________________

**Germane Load** (Learning Investment):
"The learner was really concentrating on understanding the material."
Rating: [ 1 | 2 | 3 | 4 | 5 | 6 | 7 ]
Evidence: _______________________________________________________________

---

## 2. METACOGNITIVE AWARENESS (Schraw & Dennison 1994 MAI)
**Scale: 1 (Low) to 5 (High)**

**Planning**: Setting goals, organizing thoughts, pacing learning
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Evidence: _______________________________________________________________

**Monitoring**: Self-assessing understanding, checking comprehension
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Evidence: _______________________________________________________________

**Evaluation**: Reflecting on effectiveness, summarizing learning
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Evidence: _______________________________________________________________

**Overall Metacognitive Awareness**: [ 1 | 2 | 3 | 4 | 5 ]

---

## 3. LINGUISTIC PRODUCTION (ACTFL 2024 FACT)
**Scale: 1 (Low) to 5 (High)**

**Functions & Tasks**: What communication tasks can the learner accomplish in Esperanto?
(1=Word/Phrase, 3=Sentence, 5=Extended discourse)
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Examples: _______________________________________________________________

**Accuracy**: Grammar, vocabulary, syntax correctness
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Evidence: _______________________________________________________________

**Comprehensibility**: How easily understood is the Esperanto produced?
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Evidence: _______________________________________________________________

**Text Type**: Complexity of production
(1=Words, 2=Phrases, 3=Sentences, 4=Connected sentences, 5=Paragraphs)
Rating: [ 1 | 2 | 3 | 4 | 5 ]

**Overall Linguistic Production**: [ 1 | 2 | 3 | 4 | 5 ]

---

## 4. SELF-REGULATED LEARNING (MSLQ - Pintrich et al. 1991)
**Scale: 1 (Low) to 5 (High)**

**Critical Thinking**: Deep processing, connecting ideas, questioning assumptions
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Evidence: _______________________________________________________________

**Self-Regulation**: Managing own learning, persistence, strategic approach
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Evidence: _______________________________________________________________

**Help-Seeking**: Strategic use of AI (5=Strategic tool, 1=Complete dependency)
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Evidence: _______________________________________________________________

**Overall Self-Directedness**: [ 1 | 2 | 3 | 4 | 5 ]

---

## 5. COGNITIVE ENGAGEMENT (HESES - Kahu et al. 2018)
**Scale: 1 (Low) to 5 (High)**

**Academic Engagement**: Active participation, effort in learning
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Evidence: _______________________________________________________________

**Cognitive Engagement**: Deep thinking, making connections
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Evidence: _______________________________________________________________

**Affective Engagement**: Interest, enthusiasm, emotional investment
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Evidence: _______________________________________________________________

**Overall Cognitive Engagement**: [ 1 | 2 | 3 | 4 | 5 ]

---

## 6. CRITICAL THINKING (Liu et al. 2014)
**Scale: 1 (Low) to 5 (High)**

**Analytical Reasoning**: Analyzing information, evaluating evidence
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Evidence: _______________________________________________________________

**Synthetic Reasoning**: Understanding implications, producing arguments
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Evidence: _______________________________________________________________

**Overall Question Sophistication**: [ 1 | 2 | 3 | 4 | 5 ]

---

## 7. ITERATIVE REFINEMENT
**Scale: 1 (Low) to 5 (High)**

**Follow-up Behavior**: Building on previous exchanges, seeking deeper understanding
(1=No follow-ups, 3=Some clarification, 5=Systematic deepening)
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Evidence: _______________________________________________________________

---

## 8. COGNITIVE DEBT INDICATORS (MIT 2025)
**Scale: 1 (Low) to 5 (High)**

**Memory Retention**: Evidence of retaining information across conversation
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Evidence: _______________________________________________________________

**Ownership**: Learner shows ownership of learning vs. outsourcing to AI
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Evidence: _______________________________________________________________

---

## 9. OVERALL LEARNING QUALITY
**Scale: 1 (Very Poor) to 5 (Excellent)**

Holistic assessment of learning quality in this conversation:
Rating: [ 1 | 2 | 3 | 4 | 5 ]
Justification: _______________________________________________________________

---

## 10. ADDITIONAL OBSERVATIONS

**Question Types** (estimate %):
- Translation: ____%
- Grammar: ____%
- Usage: ____%
- Clarification: ____%
- Application: ____%
- Meta-learning: ____%

**Esperanto Usage**: Approximately ____% of messages contain Esperanto

**Notable Patterns**: _______________________________________________________________

---

**Time to Complete**: _______ minutes
**Confidence in Ratings** (1-5): [ 1 | 2 | 3 | 4 | 5 ]
"""

    return template


def main():
    """Generate validation sample and coding materials."""
    print("="*80)
    print("CREATING VALIDATION SAMPLE FOR MANUAL CODING")
    print("Phase 1: Ground Truth Establishment")
    print("="*80)
    print()

    base_dir = Path("/home/user/esperanto-db")

    # Check for CSN folders
    csn_base = base_dir / "raw_data" / "csn_exports"
    if not csn_base.exists():
        csn_base = base_dir

    # Collect all conversations
    all_conversations = []
    csn_folders = sorted([d for d in csn_base.iterdir() if d.is_dir() and d.name.startswith("CSN")])

    print(f"Loading conversations from {csn_base}...")

    for csn_folder in csn_folders:
        csn_name = csn_folder.name
        conversations = load_conversation_from_csn(csn_folder)

        for conv_data in conversations:
            conv_id = conv_data.get('id', f"{csn_name}_unknown")
            user_msgs = extract_user_messages(conv_data)

            if user_msgs:  # Only include conversations with user messages
                all_conversations.append({
                    'conversation_id': conv_id,
                    'csn': csn_name,
                    'n_messages': len(user_msgs),
                    'data': conv_data
                })

    print(f"Found {len(all_conversations)} valid conversations")
    print()

    # Stratified random sampling (ensure diversity)
    # Sample across different message counts
    df = pd.DataFrame([{
        'conversation_id': c['conversation_id'],
        'csn': c['csn'],
        'n_messages': c['n_messages']
    } for c in all_conversations])

    # Create message count bins
    df['msg_bin'] = pd.cut(df['n_messages'], bins=[0, 5, 10, 20, 100],
                           labels=['1-5', '6-10', '11-20', '20+'])

    # Stratified sample: 50 conversations
    sample_size = 50
    sample_df = df.groupby('msg_bin', group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size // 4 + 5), random_state=42)
    ).head(sample_size)

    print(f"Sampled {len(sample_df)} conversations (stratified by message count)")
    print("\nDistribution:")
    print(sample_df['msg_bin'].value_counts().sort_index())
    print()

    # Create validation folder
    validation_dir = base_dir / "validation_sample"
    validation_dir.mkdir(exist_ok=True)

    # Save sample IDs
    sample_df.to_csv(validation_dir / "validation_sample_ids.csv", index=False)
    print(f"✓ Saved sample IDs: validation_sample/validation_sample_ids.csv")

    # Export conversations for coding
    conversations_for_coding = []

    for idx, row in sample_df.iterrows():
        conv_id = row['conversation_id']

        # Find the full conversation data
        conv_full = next(c for c in all_conversations if c['conversation_id'] == conv_id)
        user_msgs = extract_user_messages(conv_full['data'])

        conversations_for_coding.append({
            'conversation_id': conv_id,
            'csn': row['csn'],
            'n_messages': row['n_messages'],
            'messages': user_msgs
        })

    # Save as JSON for easy reading
    with open(validation_dir / "conversations_for_manual_coding.json", 'w', encoding='utf-8') as f:
        json.dump(conversations_for_coding, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved conversations: validation_sample/conversations_for_manual_coding.json")

    # Create coding sheet template
    template = create_coding_sheet_template()
    with open(validation_dir / "manual_coding_sheet_template.md", 'w') as f:
        f.write(template)
    print(f"✓ Saved coding template: validation_sample/manual_coding_sheet_template.md")

    # Create instructions for raters
    instructions = f"""# MANUAL CODING INSTRUCTIONS

## Overview
You will code {len(sample_df)} Esperanto learning conversations using validated assessment frameworks.

## Time Estimate
- Approximately 5-8 minutes per conversation
- Total time: 4-7 hours (can be split across multiple sessions)

## Materials Provided
1. `conversations_for_manual_coding.json` - The conversations to code
2. `manual_coding_sheet_template.md` - Coding rubric based on validated instruments

## Process
1. Read the entire conversation before rating
2. Use the coding sheet template for each conversation
3. Base all ratings on **observable evidence only**
4. Provide brief justification for each rating
5. Save completed coding sheets with format: `rater_[NAME]_conv_[ID].md`

## Quality Checks
- Take breaks every 10-15 conversations
- If unsure about a rating, note your uncertainty
- Do not discuss ratings with the other rater until inter-rater reliability is calculated

## Validated Frameworks Used
All metrics are grounded in peer-reviewed research:

1. **Cognitive Load** - Klepsch et al. (2017), Frontiers in Psychology
2. **Metacognitive Awareness** - Schraw & Dennison (1994), Contemporary Educational Psychology
3. **Linguistic Production** - ACTFL (2024), ACTFL Proficiency Guidelines
4. **Self-Regulated Learning** - Pintrich et al. (1991), MSLQ Manual
5. **Cognitive Engagement** - Kahu et al. (2018), Research in Higher Education
6. **Critical Thinking** - Liu et al. (2014), ETS Research Report
7. **Cognitive Debt** - MIT Media Lab (2025), arXiv:2506.08872

## Contact
If you have questions about the coding process, contact: [Your Email]

## Timeline
- Start Date: _______________
- Target Completion: _______________ (aim for 1-2 weeks)
- Inter-Rater Reliability Check: After both raters complete all 50

---

Thank you for your contribution to this research!
"""

    with open(validation_dir / "RATER_INSTRUCTIONS.md", 'w') as f:
        f.write(instructions)
    print(f"✓ Saved rater instructions: validation_sample/RATER_INSTRUCTIONS.md")

    print()
    print("="*80)
    print("VALIDATION SAMPLE CREATED SUCCESSFULLY")
    print("="*80)
    print()
    print(f"Location: {validation_dir}")
    print()
    print("Files created:")
    print("  1. validation_sample_ids.csv - List of sampled conversation IDs")
    print("  2. conversations_for_manual_coding.json - Full conversation texts")
    print("  3. manual_coding_sheet_template.md - Coding rubric")
    print("  4. RATER_INSTRUCTIONS.md - Instructions for human raters")
    print()
    print("Next steps:")
    print("  1. Recruit 2 independent raters (educational psychology background preferred)")
    print("  2. Each rater codes all 50 conversations using the template")
    print("  3. Calculate inter-rater reliability (Cohen's Kappa or ICC)")
    print("  4. Use as ground truth to validate LLM-as-judge approach")
    print("  5. If r > 0.70 correlation, proceed with full dataset")
    print()


if __name__ == "__main__":
    main()
