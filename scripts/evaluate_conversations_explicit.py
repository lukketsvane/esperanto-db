#!/usr/bin/env python3
import json
import pandas as pd
import os
from pathlib import Path
from google import genai
from google.genai import types
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-3.1-flash-lite-preview"

progress_lock = threading.Lock()
progress_counter = {"count": 0, "total": 0}

EVALUATION_PROMPT = """Rate this Esperanto learning conversation (1-5 scale). Return JSON:
{
  "germane_load": <1-5>,
  "metacog_planning": <1-5>,
  "metacog_monitoring": <1-5>,
  "metacog_evaluation": <1-5>,
  "ling_quantity": <1-5>,
  "ling_accuracy": <1-5>,
  "self_regulation": <1-5>,
  "critical_thinking": <1-5>,
  "engagement_cognitive": <1-5>,
  "engagement_affective": <1-5>,
  "question_sophistication": <1-5>,
  "iterative_depth": <1-5>,
  "memory_retention": <1-5>,
  "ownership": <1-5>,
  "copypaste_dummy": <1-5>,
  "substitute_learning": <1-5>,
  "complement_learning": <1-5>
}"""

SYSTEM_PROMPT = """You are an expert evaluator of learning conversations.
Rate the given Esperanto learning conversation from 1 to 5 on the following variables. Return valid JSON.

Variable Definitions:
- germane_load (1-5): Extent to which the user focuses on meaning, deep understanding, and schema construction rather than superficial features.
- metacog_planning (1-5): Extent to which the user plans their learning (e.g., setting goals, asking for a structured lesson).
- metacog_monitoring (1-5): Extent to which the user monitors their own understanding (e.g., "Wait, I didn't get that").
- metacog_evaluation (1-5): Extent to which the user evaluates their learning progress or outcomes.
- ling_quantity (1-5): Amount of Esperanto language produced by the user.
- ling_accuracy (1-5): Accuracy of the Esperanto language produced by the user.
- self_regulation (1-5): User's ability to self-regulate their learning process and emotions.
- critical_thinking (1-5): Extent to which the user demonstrates critical analysis, reasoning, and questioning.
- engagement_cognitive (1-5): User's cognitive focus and deep mental effort on the task.
- engagement_affective (1-5): User's emotional engagement, interest, or motivation in the learning.
- question_sophistication (1-5): Complexity and depth of questions asked by the user (1=basic facts, 5=complex nuances).
- iterative_depth (1-5): Extent to which the user iterates on previous responses and dives deeper into topics.
- memory_retention (1-5): Indications of the user remembering and applying past learning/vocabulary in the conversation.
- ownership (1-5): Extent to which the user takes control and leads the learning process.
- copypaste_dummy (1-5): 1 = User types their own fully original prompts, 5 = User clearly copy-pasted assignment or quiz questions directly into the chat without adding their own thought.
- substitute_learning (1-5): 1-5 scale on whether the AI is used to replace user effort (e.g., "just give me the answer/translation"). Higher score means higher reliance on AI as a substitute for learning.
- complement_learning (1-5): 1-5 scale on whether the AI is used to augment user effort (e.g., "explain why my answer is wrong"). Higher score means better complementary use of AI.
"""

def extract_all_messages(conversation_data):
    messages = []
    if 'mapping' in conversation_data:
        nodes_with_time = []
        for node_id, node in conversation_data['mapping'].items():
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
        # Sort by creation time to reconstruct conversation flow
        nodes_with_time.sort(key=lambda x: x[0])
        messages = [x[1] for x in nodes_with_time]
    return messages

def call_gemini(all_messages):
    user_text = "\n\n".join([f"Message {i+1}: {msg}" for i, msg in enumerate(all_messages)])
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"{SYSTEM_PROMPT}\n\n{user_text}\n\n{EVALUATION_PROMPT}"),
            ],
        ),
    ]
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1
    )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=config,
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

def evaluate_conversation(args):
    conv_data, conv_id, csn_name = args

    try:
        all_messages = extract_all_messages(conv_data)

        if not all_messages:
            return {
                "conversation_id": conv_id,
                "csn": csn_name,
                "n_messages": 0,
                "error": "No messages found"
            }

        # Rate limiting: add a small delay to avoid hitting rate limits
        time.sleep(0.1)
        result = call_gemini(all_messages)

        if "error" in result:
            return {
                "conversation_id": conv_id,
                "csn": csn_name,
                "n_messages": len(all_messages),
                "error": result["error"]
            }

        output = {
            "conversation_id": conv_id,
            "csn": csn_name,
            "n_messages": len(all_messages)
        }
        output.update(result)

        # Composite metrics
        output["cognitive_engagement"] = result.get("engagement_cognitive", 3.0)
        output["metacognitive_awareness"] = np.mean([
            result.get("metacog_planning", 3.0),
            result.get("metacog_monitoring", 3.0),
            result.get("metacog_evaluation", 3.0)
        ])
        output["linguistic_production"] = np.mean([
            result.get("ling_quantity", 3.0),
            result.get("ling_accuracy", 3.0)
        ])
        output["self_directedness"] = result.get("self_regulation", 3.0)
        output["iterative_refinement"] = result.get("iterative_depth", 2.0)
        output["query_sophistication"] = result.get("question_sophistication", 2.5)
        output["agency_ownership"] = result.get("ownership", 3.0)
        output["memory_retention"] = result.get("memory_retention", 2.0)
        output["overall_learning_quality"] = np.mean([
            output["cognitive_engagement"],
            output["metacognitive_awareness"],
            output["self_directedness"],
            output["iterative_refinement"]
        ])

        with progress_lock:
            progress_counter["count"] += 1
            if progress_counter["count"] % 10 == 0:
                pct = (progress_counter["count"] / progress_counter["total"]) * 100
                print(f"{progress_counter['count']}/{progress_counter['total']} ({pct:.1f}%)")

        return output

    except Exception as e:
        return {
            "conversation_id": conv_id,
            "csn": csn_name,
            "error": str(e)
        }

def load_conversation_from_csn(csn_folder):
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
    except:
        return []

def main():
    print("Running EXPLICIT validated evaluation on all conversations")
    print(f"Model: {MODEL}")
    
    variables = [
        "germane_load", "metacog_planning", "metacog_monitoring", "metacog_evaluation", 
        "ling_quantity", "ling_accuracy", "self_regulation", "critical_thinking", 
        "engagement_cognitive", "engagement_affective", "question_sophistication", 
        "iterative_depth", "memory_retention", "ownership", "copypaste_dummy", 
        "substitute_learning", "complement_learning"
    ]
    print(f"Variables to be evaluated explicitly: {', '.join(variables)}")
    print()

    base_dir = Path(__file__).parent.parent.resolve()
    csn_base = base_dir / "raw_data" / "csn_exports"
    if not csn_base.exists():
        csn_base = base_dir

    all_conversations = []
    csn_folders = sorted([d for d in csn_base.iterdir() if d.is_dir() and d.name.startswith("CSN")])

    for csn_folder in csn_folders:
        csn_name = csn_folder.name
        conversations = load_conversation_from_csn(csn_folder)
        for conv_data in conversations:
            conv_id = conv_data.get('id', f"{csn_name}_unknown")
            all_conversations.append((conv_data, conv_id, csn_name))

    progress_counter["total"] = len(all_conversations)
    print(f"Evaluating {len(all_conversations)} conversations\n")

    all_results = []
    # Using smaller pool size to avoid rate limiting for Gemini
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(evaluate_conversation, conv_args): conv_args
                   for conv_args in all_conversations}

        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)

    results_df = pd.DataFrame(all_results)
    output_file = base_dir / "conversations_evaluated_explicit.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\nComplete: {len(results_df)} conversations")
    print(f"Errors: {len(results_df[results_df.get('error', '') != ''])}")
    print(f"Saved: {output_file}")

    composite_metrics = ["cognitive_engagement", "metacognitive_awareness", "linguistic_production",
                        "self_directedness", "iterative_refinement", "memory_retention",
                        "agency_ownership", "query_sophistication", "overall_learning_quality"]

    print("\nMetrics summary:")
    for metric in composite_metrics:
        if metric in results_df.columns:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            print(f"  {metric}: {mean_val:.2f} ± {std_val:.2f}")

if __name__ == "__main__":
    main()
