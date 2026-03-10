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
import argparse

# --- Configure CLI Arguments ---
parser = argparse.ArgumentParser(description="Evaluate Esperanto conversations using LLM as a Judge.")
parser.add_argument(
    '--prompt-type', 
    choices=['implicit', 'explicit'], 
    default='explicit',
    help='Choose the system prompt type: "implicit" (no definitions) or "explicit" (strict rubrics). Defaults to "explicit".'
)
parser.add_argument(
    '--ids',
    nargs='+',
    help='Specific conversation IDs to evaluate (space-separated). If omitted, evaluates all.'
)
args = parser.parse_args()

# --- Initialize Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-3.1-flash-lite-preview"

# --- Load Prompts ---
base_dir = Path(__file__).parent.parent.resolve()
prompt_dir = base_dir / "prompts"

try:
    with open(prompt_dir / f"system_{args.prompt_type}.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
    with open(prompt_dir / "user_schema.txt", "r", encoding="utf-8") as f:
        USER_SCHEMA = f.read()
except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing prompt file: {e}")

progress_lock = threading.Lock()
progress_counter = {"count": 0, "total": 0}

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
                types.Part.from_text(text=f"{SYSTEM_PROMPT}\n\n[CONVERSATION START]\n{user_text}\n[CONVERSATION END]\n\n{USER_SCHEMA}"),
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

def evaluate_conversation(conv_args):
    conv_data, conv_id, csn_name = conv_args

    try:
        all_messages = extract_all_messages(conv_data)

        if not all_messages:
            return {
                "conversation_id": conv_id,
                "csn": csn_name,
                "n_messages": 0,
                "error": "No messages found"
            }

        # Rate limiting mitigation
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
    print(f"Running LLM Evaluation - Mode: {args.prompt_type.upper()}")
    print(f"Model: {MODEL}")

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
            if args.ids and conv_id not in args.ids:
                continue
            all_conversations.append((conv_data, conv_id, csn_name))

    progress_counter["total"] = len(all_conversations)
    if progress_counter["total"] == 0:
        print("No conversations to evaluate. Check IDs or CSN folders.")
        return
    
    print(f"Evaluating {len(all_conversations)} conversations\n")

    all_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(evaluate_conversation, conv_args): conv_args
                   for conv_args in all_conversations}

        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)

    results_df = pd.DataFrame(all_results)
    
    # Save file with dynamic name based on prompt type
    output_file = base_dir / f"conversations_evaluated_{args.prompt_type}.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\nComplete: {len(results_df)} conversations")
    error_count = len(results_df[results_df['error'].notna()]) if 'error' in results_df.columns else 0
    print(f"Errors: {error_count}")
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
