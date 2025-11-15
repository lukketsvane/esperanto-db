#!/usr/bin/env python3
"""
Fast Conversation Evaluation Script - Concurrent Processing
Combines all evaluations into single API call per conversation for speed
"""

import json
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Any
import time
from openai import OpenAI
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# OpenAI API configuration
# Set your API key as environment variable: export OPENAI_API_KEY="your-key-here"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")
client = OpenAI(api_key=OPENAI_API_KEY)

# Use GPT-4 Turbo (latest stable model)
MODEL = "gpt-4-turbo-preview"

# Thread-safe counter for progress tracking
progress_lock = threading.Lock()
progress_counter = {"count": 0, "total": 0}


COMPREHENSIVE_EVALUATION_PROMPT = """
You are an expert in educational psychology, behavioral economics, and language learning research.
Analyze this Esperanto learning conversation and provide a comprehensive evaluation across multiple
literature-grounded dimensions.

CONVERSATION USER MESSAGES:
{user_messages}

Evaluate on the following dimensions (1-5 scale for each):

1. **Cognitive Engagement** (MIT "cognitive debt" research)
   - 1=Passive, 2=Surface, 3=Moderate, 4=Deep, 5=Highly active/metacognitive

2. **Query Sophistication** (Behavioral economics - effort allocation)
   - 1=Simple/vague, 2=Basic, 3=Clear/specific, 4=Well-formed, 5=Sophisticated/nuanced

3. **Self-Directedness** (Cognitive offloading research)
   - 1=Complete dependency, 2=Heavy reliance, 3=Balanced, 4=Highly self-directed, 5=Autonomous

4. **Iterative Refinement** (Deep learning indicator)
   - 1=No follow-up, 2=Minimal, 3=Some clarification, 4=Regular refinement, 5=Systematic deepening

5. **Learning Orientation** (Declarative vs Procedural)
   - 1=Purely declarative (facts), 2=Mostly declarative, 3=Balanced, 4=Mostly procedural (processes), 5=Purely procedural

6. **Agency & Ownership** (MIT "essay ownership" research)
   - 1=No ownership, 2=Minimal agency, 3=Moderate, 4=Strong ownership, 5=Complete ownership

7. **Linguistic Production** (Esperanto usage quality)
   - 1=No Esperanto, 2=Minimal, 3=Some usage, 4=Regular usage, 5=Sophisticated

8. **Memory/Retention Indicators** (MIT cognitive debt)
   - 1=No evidence, 2=Minimal retention, 3=Some retention, 4=Good retention, 5=Excellent

9. **Metacognitive Awareness** (Educational psychology)
   - 1=None, 2=Minimal, 3=Some reflection, 4=Regular monitoring, 5=Sophisticated strategies

Also provide:
- **Question Type Distribution** (%): translation, grammar, usage, clarification, application, meta
- **Esperanto Usage %**: Percentage of messages containing Esperanto
- **Mean Query Length**: Average word count of user messages
- **Total User Messages**: Count of user messages

Return ONLY a valid JSON object with this exact structure:
{{
  "cognitive_engagement": <1-5>,
  "query_sophistication": <1-5>,
  "self_directedness": <1-5>,
  "iterative_refinement": <1-5>,
  "learning_orientation": <1-5>,
  "agency_ownership": <1-5>,
  "linguistic_production": <1-5>,
  "memory_retention": <1-5>,
  "metacognitive_awareness": <1-5>,
  "question_types": {{
    "translation": <0-100>,
    "grammar": <0-100>,
    "usage": <0-100>,
    "clarification": <0-100>,
    "application": <0-100>,
    "meta": <0-100>
  }},
  "esperanto_usage_pct": <0-100>,
  "mean_query_length": <number>,
  "total_user_messages": <number>,
  "overall_learning_quality": <1-5>
}}
"""


def extract_user_messages(conversation_data: Dict) -> List[str]:
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


def call_openai_comprehensive_eval(user_messages: List[str], max_retries: int = 3) -> Dict[str, Any]:
    """Call OpenAI API with comprehensive evaluation prompt."""
    user_messages_text = "\n\n".join([f"Message {i+1}: {msg}" for i, msg in enumerate(user_messages)])

    prompt = COMPREHENSIVE_EVALUATION_PROMPT.format(user_messages=user_messages_text)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert in educational psychology, behavioral economics, and language learning research. Provide precise, evidence-based evaluations in valid JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return {"error": str(e)}


def evaluate_single_conversation(args):
    """Evaluate a single conversation (for parallel processing)."""
    conv_data, conv_id, csn_name = args

    try:
        user_messages = extract_user_messages(conv_data)

        if not user_messages:
            return {
                "conversation_id": conv_id,
                "csn": csn_name,
                "n_user_messages": 0,
                "error": "No user messages"
            }

        # Single API call for all evaluations
        eval_results = call_openai_comprehensive_eval(user_messages)
        eval_results["conversation_id"] = conv_id
        eval_results["csn"] = csn_name

        # Update progress
        with progress_lock:
            progress_counter["count"] += 1
            if progress_counter["count"] % 10 == 0:
                pct = (progress_counter["count"] / progress_counter["total"]) * 100
                print(f"Progress: {progress_counter['count']}/{progress_counter['total']} ({pct:.1f}%)")

        return eval_results

    except Exception as e:
        print(f"Error evaluating {conv_id}: {e}")
        return {
            "conversation_id": conv_id,
            "csn": csn_name,
            "error": str(e)
        }


def load_conversation_from_csn(csn_folder: Path) -> List[Dict]:
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
    except Exception as e:
        print(f"Error loading {conv_file}: {e}")
        return []


def flatten_evaluation_results(results: List[Dict]) -> List[Dict]:
    """Flatten nested JSON structures for CSV export."""
    flattened = []

    for result in results:
        flat = {
            "conversation_id": result.get("conversation_id"),
            "csn": result.get("csn"),
            "n_user_messages": result.get("total_user_messages", result.get("n_user_messages", 0)),
            "cognitive_engagement": result.get("cognitive_engagement"),
            "query_sophistication": result.get("query_sophistication"),
            "self_directedness": result.get("self_directedness"),
            "iterative_refinement": result.get("iterative_refinement"),
            "learning_orientation": result.get("learning_orientation"),
            "agency_ownership": result.get("agency_ownership"),
            "linguistic_production": result.get("linguistic_production"),
            "memory_retention": result.get("memory_retention"),
            "metacognitive_awareness": result.get("metacognitive_awareness"),
            "overall_learning_quality": result.get("overall_learning_quality"),
            "esperanto_usage_pct": result.get("esperanto_usage_pct"),
            "mean_query_length": result.get("mean_query_length"),
        }

        # Flatten question types
        qt = result.get("question_types", {})
        if qt:
            flat["qt_translation"] = qt.get("translation", 0)
            flat["qt_grammar"] = qt.get("grammar", 0)
            flat["qt_usage"] = qt.get("usage", 0)
            flat["qt_clarification"] = qt.get("clarification", 0)
            flat["qt_application"] = qt.get("application", 0)
            flat["qt_meta"] = qt.get("meta", 0)

        flat["error"] = result.get("error", "")

        flattened.append(flat)

    return flattened


def main():
    """Main execution function with concurrent processing."""
    print("="*80)
    print("FAST ESPERANTO CONVERSATION EVALUATION")
    print("Using concurrent processing and combined API calls for speed")
    print("="*80)
    print()

    base_dir = Path("/home/user/esperanto-db")

    # Load data
    participants_df = pd.read_csv(base_dir / "prompt_participants_final_for_paper.csv")
    conversations_df = pd.read_csv(base_dir / "prompt_conversations_final_for_paper.csv")

    print(f"Loaded {len(participants_df)} participants")
    print(f"Loaded {len(conversations_df)} conversations")
    print()

    # Collect all conversations to evaluate
    all_conversations = []
    csn_folders = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("CSN")])

    for csn_folder in csn_folders:
        csn_name = csn_folder.name
        conversations = load_conversation_from_csn(csn_folder)

        for conv_data in conversations:
            conv_id = conv_data.get('id', f"{csn_name}_unknown")
            all_conversations.append((conv_data, conv_id, csn_name))

    print(f"Found {len(all_conversations)} conversations to evaluate")
    progress_counter["total"] = len(all_conversations)
    print()

    # Process conversations concurrently
    print("Starting concurrent evaluation...")
    print(f"Using {MODEL}")
    print(f"Max workers: 10 (for API rate limiting)")
    print()

    all_results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(evaluate_single_conversation, conv_args): conv_args
                   for conv_args in all_conversations}

        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)

    # Flatten and save
    print("\n" + "="*80)
    print("Creating consolidated dataset...")

    flattened_results = flatten_evaluation_results(all_results)
    evaluations_df = pd.DataFrame(flattened_results)

    # Merge with conversations data
    merged_df = conversations_df.merge(
        evaluations_df,
        on='conversation_id',
        how='left',
        suffixes=('', '_eval')
    )

    # Merge with participants data
    final_df = merged_df.merge(
        participants_df,
        left_on='participant_id',
        right_on='id',
        how='left',
        suffixes=('', '_participant')
    )

    # Save results
    output_file = base_dir / "conversations_with_evaluations_final.csv"
    final_df.to_csv(output_file, index=False)
    print(f"✓ Saved consolidated dataset: {output_file}")
    print(f"  Rows: {len(final_df)}, Columns: {len(final_df.columns)}")

    # Save just evaluations
    evals_file = base_dir / "evaluation_results_final.csv"
    evaluations_df.to_csv(evals_file, index=False)
    print(f"✓ Saved evaluation results: {evals_file}")

    # Summary statistics
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    score_columns = ["cognitive_engagement", "query_sophistication", "self_directedness",
                     "iterative_refinement", "learning_orientation", "agency_ownership",
                     "linguistic_production", "memory_retention", "metacognitive_awareness",
                     "overall_learning_quality"]

    print("\nMean Scores (1-5 scale):")
    for col in score_columns:
        if col in evaluations_df.columns:
            mean_score = evaluations_df[col].mean()
            std_score = evaluations_df[col].std()
            print(f"  {col:30s}: {mean_score:.2f} ± {std_score:.2f}")

    print(f"\nEsperanto Usage: {evaluations_df['esperanto_usage_pct'].mean():.1f}% average")
    print(f"Mean Query Length: {evaluations_df['mean_query_length'].mean():.1f} words")

    print(f"\nTotal conversations evaluated: {len(evaluations_df)}")
    print(f"Conversations with errors: {len(evaluations_df[evaluations_df['error'] != ''])}")
    print(f"Unique participants: {len(final_df['participant_id'].unique())}")

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
