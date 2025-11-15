#!/usr/bin/env python3
import json
import pandas as pd
import os
from pathlib import Path
from openai import OpenAI
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4-turbo-preview"
progress_lock = threading.Lock()
progress_counter = {"count": 0, "total": 0}

EVALUATION_PROMPT = """Analyze this Esperanto learning conversation and provide ratings (1-5 scale):

COGNITIVE_LOAD (Klepsch et al. 2017):
- Germane load: depth of learning investment

METACOGNITION (Schraw & Dennison 1994 MAI):
- Planning: goal-setting, organization
- Monitoring: comprehension checking
- Evaluation: reflection on effectiveness

LINGUISTIC_PRODUCTION (ACTFL 2024):
- Esperanto usage quantity and quality
- Grammatical accuracy

SELF_REGULATION (Pintrich et al. 1991 MSLQ):
- Autonomy vs dependency on AI
- Critical thinking depth

ENGAGEMENT (Kahu et al. 2018 HESES):
- Cognitive engagement level
- Affective interest

CRITICAL_THINKING (Liu et al. 2014):
- Question sophistication
- Analytical reasoning

ITERATIVE_BEHAVIOR:
- Follow-up depth
- Building on exchanges

COGNITIVE_DEBT (MIT 2025):
- Memory retention across conversation
- Learning ownership vs outsourcing

Return JSON with these exact keys (all 1-5):
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
  "ownership": <1-5>
}

Base ratings on observable evidence only."""

def extract_user_messages(conversation_data):
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

def call_openai(user_messages):
    user_text = "\n\n".join([f"Message {i+1}: {msg}" for i, msg in enumerate(user_messages)])

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert educational assessor. Provide precise ratings in valid JSON."},
                {"role": "user", "content": f"CONVERSATION:\n{user_text}\n\n{EVALUATION_PROMPT}"}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def evaluate_conversation(args):
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

        result = call_openai(user_messages)

        if "error" in result:
            return {
                "conversation_id": conv_id,
                "csn": csn_name,
                "n_user_messages": len(user_messages),
                "error": result["error"]
            }

        output = {
            "conversation_id": conv_id,
            "csn": csn_name,
            "n_user_messages": len(user_messages)
        }
        output.update(result)

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
    print("Running validated evaluation on all conversations")
    print(f"Model: {MODEL}")
    print()

    base_dir = Path("/home/user/esperanto-db")
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
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(evaluate_conversation, conv_args): conv_args
                   for conv_args in all_conversations}

        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)

    results_df = pd.DataFrame(all_results)
    output_file = base_dir / "conversations_evaluated.csv"
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
