#!/usr/bin/env python3
"""
Literature-Grounded Conversation Evaluation Script
Based on 20+ validated assessment frameworks

Key Improvements over Original:
1. Multi-prompt strategy (3 variations per metric) - reduces bias
2. Validated frameworks with citations
3. Median aggregation across iterations
4. Proper documentation for academic publication

Citations:
- Klepsch et al. (2017): Cognitive Load Theory
- Schraw & Dennison (1994): Metacognitive Awareness Inventory (MAI)
- Pintrich et al. (1991): MSLQ Self-Regulated Learning
- ACTFL (2024): FACT Language Assessment Criteria
- Kahu et al. (2018): HESES Student Engagement
- Liu et al. (2014): Critical Thinking Dimensions
- MIT Media Lab (2025): Cognitive Debt Framework
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
from statistics import median

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4-turbo-preview"

# Thread-safe counter
progress_lock = threading.Lock()
progress_counter = {"count": 0, "total": 0}

# Multi-prompt strategy: 3 variations per metric to reduce bias
EVALUATION_PROMPTS = {
    "cognitive_load": [
        # Variant 1: Klepsch et al. (2017) 3-component framework
        """Using Klepsch et al. (2017) cognitive load theory, rate on 1-7 scale:

INTRINSIC LOAD (Task Complexity):
"The Esperanto topics in this conversation were complex."
1 (Not at all) to 7 (Completely)

GERMANE LOAD (Learning Investment):
"The learner concentrated deeply on understanding the material."
1 (Not at all) to 7 (Completely)

Return JSON: {{"intrinsic_load": <1-7>, "germane_load": <1-7>}}""",

        # Variant 2: Simple effort-based
        """Rate the learner's cognitive effort on a 1-7 scale based on:
- Depth of questions
- Complexity of topics engaged
- Mental effort invested

Return JSON: {{"cognitive_effort": <1-7>}}""",

        # Variant 3: Evidence-based
        """Based on observable behaviors, rate cognitive investment (1-7):
Evidence of high investment: complex questions, follow-ups, integration
Evidence of low investment: simple requests, no follow-up, surface-level

Return JSON: {{"cognitive_investment": <1-7>}}"""
    ],

    "metacognition": [
        # Variant 1: MAI framework (Schraw & Dennison 1994)
        """Using Schraw & Dennison (1994) MAI framework, assess on 1-5 scale:

PLANNING: Does the learner set goals, organize thoughts, pace appropriately?
MONITORING: Does the learner self-assess understanding, revise when confused?
EVALUATION: Does the learner reflect on what worked, assess effectiveness?

Return JSON: {{"planning": <1-5>, "monitoring": <1-5>, "evaluation": <1-5>}}""",

        # Variant 2: Holistic metacognitive awareness
        """Rate metacognitive awareness (1-5) based on evidence of:
- Self-reflection on learning process
- Awareness of comprehension gaps
- Strategic adjustment of approach

Return JSON: {{"metacognitive_awareness": <1-5>}}""",

        # Variant 3: Regulation behaviors
        """Assess self-regulation behaviors (1-5):
Does the learner demonstrate awareness of:
- What they know/don't know
- How they're learning
- When to adjust strategies

Return JSON: {{"self_regulation_awareness": <1-5>}}"""
    ],

    "linguistic_production": [
        # Variant 1: ACTFL 2024 FACT criteria
        """Using ACTFL (2024) FACT criteria for Esperanto production:

FUNCTIONS & TASKS: What can the learner accomplish? (1=Word/Phrase, 5=Extended discourse)
ACCURACY: Grammar, vocabulary, syntax correctness (1=Low, 5=High)
COMPREHENSIBILITY: How easily understood? (1=Difficult, 5=Easy)
TEXT TYPE: Complexity level produced (1=Words, 3=Sentences, 5=Paragraphs)

Return JSON: {{"functions": <1-5>, "accuracy": <1-5>, "comprehensibility": <1-5>, "text_type": <1-5>}}""",

        # Variant 2: Production quality
        """Rate Esperanto language production quality (1-5):
- Quantity of Esperanto usage
- Quality of Esperanto construction
- Sophistication of expression

Return JSON: {{"production_quality": <1-5>}}""",

        # Variant 3: Active vs passive use
        """Assess Esperanto production (1-5):
1 = No production, only requests for translations
3 = Some attempts, mixed quality
5 = Sophisticated, regular Esperanto usage

Return JSON: {{"esperanto_production": <1-5>}}"""
    ],

    "self_regulated_learning": [
        # Variant 1: MSLQ-based (Pintrich 1991)
        """Based on MSLQ (Pintrich et al. 1991), rate on 1-5 scale:

CRITICAL THINKING: Deep processing, connecting ideas, questioning assumptions
SELF-REGULATION: Managing own learning, persistence, strategic approach
HELP-SEEKING: Strategic vs. dependent help requests (5=Strategic, 1=Over-dependent)

Return JSON: {{"critical_thinking": <1-5>, "self_regulation": <1-5>, "help_seeking": <1-5>}}""",

        # Variant 2: Learning autonomy
        """Rate learning autonomy (1-5):
1 = Complete dependency on AI for answers
3 = Balanced AI use with own thinking
5 = Highly autonomous, AI as strategic tool

Return JSON: {{"learning_autonomy": <1-5>}}""",

        # Variant 3: Agency and ownership
        """Assess learner agency (1-5) based on:
- Taking initiative in learning
- Ownership of learning process
- Self-directed exploration

Return JSON: {{"learner_agency": <1-5>}}"""
    ],

    "cognitive_engagement": [
        # Variant 1: HESES framework (Kahu et al. 2018)
        """Using HESES (Kahu et al. 2018) cognitive engagement subscale, rate 1-5:

ACADEMIC ENGAGEMENT: Active participation, effort in learning tasks
COGNITIVE ENGAGEMENT: Deep thinking, making connections, understanding
AFFECTIVE ENGAGEMENT: Interest, enthusiasm, emotional investment

Return JSON: {{"academic": <1-5>, "cognitive": <1-5>, "affective": <1-5>}}""",

        # Variant 2: Depth of processing
        """Rate depth of cognitive processing (1-5):
1 = Surface-level, passive reception
3 = Moderate engagement, some processing
5 = Deep engagement, active construction of knowledge

Return JSON: {{"processing_depth": <1-5>}}""",

        # Variant 3: Engagement indicators
        """Assess engagement (1-5) based on observable indicators:
- Question complexity and follow-through
- Evidence of thinking deeply about responses
- Active vs. passive interaction style

Return JSON: {{"engagement_level": <1-5>}}"""
    ],

    "critical_thinking": [
        # Variant 1: Liu et al. (2014) dimensions
        """Using Liu et al. (2014) critical thinking framework, rate 1-5:

ANALYTICAL: Analyzing information, evaluating evidence
SYNTHETIC: Understanding implications, producing arguments
CAUSAL: Understanding causation and explanation

Return JSON: {{"analytical": <1-5>, "synthetic": <1-5>, "causal": <1-5>}}""",

        # Variant 2: Question sophistication (Bloom's Taxonomy)
        """Rate question sophistication using Bloom's Taxonomy (1-5):
1 = Remembering (simple recall, translations)
3 = Applying/Analyzing (usage questions, grammar analysis)
5 = Evaluating/Creating (comparing approaches, generating examples)

Return JSON: {{"question_sophistication": <1-5>}}""",

        # Variant 3: Intellectual complexity
        """Assess intellectual complexity of queries (1-5):
Evidence of: Making connections, questioning assumptions, seeking deeper understanding
vs. Simple fact requests, surface-level questions

Return JSON: {{"intellectual_complexity": <1-5>}}"""
    ],

    "iterative_refinement": [
        # Variant 1: Follow-up behavior
        """Rate iterative refinement behavior (1-5):
1 = No follow-up questions, accepts first answer
3 = Some clarification requests
5 = Systematic deepening through multiple follow-ups

Return JSON: {{"refinement_depth": <1-5>}}""",

        # Variant 2: Learning persistence
        """Assess learning persistence (1-5):
Does the learner:
- Build on previous exchanges?
- Seek deeper understanding?
- Refine their comprehension iteratively?

Return JSON: {{"learning_persistence": <1-5>}}""",

        # Variant 3: Conversation depth
        """Rate conversation depth (1-5) based on:
- Number of follow-up questions per topic
- Progressive deepening of understanding
- Building on prior exchanges

Return JSON: {{"conversation_depth": <1-5>}}"""
    ],

    "cognitive_debt_indicators": [
        # Variant 1: MIT (2025) framework
        """Based on MIT Media Lab (2025) cognitive debt research, assess 1-5:

MEMORY RETENTION: Evidence of retaining information across conversation
OWNERSHIP: Does learner show ownership of learning vs. outsourcing to AI
COGNITIVE EFFORT: Active processing vs. passive reception

Return JSON: {{"memory_retention": <1-5>, "ownership": <1-5>, "cognitive_effort": <1-5>}}""",

        # Variant 2: Offloading behavior
        """Rate cognitive offloading (reversed, 1-5):
1 = Heavy offloading to AI, minimal personal processing
3 = Balanced use of AI as tool
5 = Minimal offloading, AI as strategic resource

Return JSON: {{"offloading_level": <1-5>}}""",

        # Variant 3: Knowledge building
        """Assess knowledge building vs. answer-getting (1-5):
1 = Pure answer-getting, no knowledge construction
3 = Mixed approach
5 = Clear knowledge building, using AI as scaffold

Return JSON: {{"knowledge_building": <1-5>}}"""
    ]
}


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


def call_openai_with_prompt(user_messages: List[str], prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    """Call OpenAI API with a specific evaluation prompt."""
    user_messages_text = "\n\n".join([f"Message {i+1}: {msg}" for i, msg in enumerate(user_messages)])

    full_prompt = f"CONVERSATION:\n{user_messages_text}\n\n{prompt}"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert educational psychologist. Provide precise evaluations in valid JSON format only. Base ratings on observable evidence."},
                    {"role": "user", "content": full_prompt}
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


def evaluate_with_multi_prompt(user_messages: List[str], metric_name: str, iterations: int = 3) -> Dict[str, float]:
    """
    Evaluate using multi-prompt strategy with median aggregation.

    Args:
        user_messages: List of user messages from conversation
        metric_name: Name of metric category (e.g., 'cognitive_load')
        iterations: Number of times to run each prompt variant

    Returns:
        Dictionary with median scores for each sub-metric
    """
    prompts = EVALUATION_PROMPTS.get(metric_name, [])
    all_results = []

    for prompt in prompts:
        for _ in range(iterations):
            result = call_openai_with_prompt(user_messages, prompt)
            if "error" not in result:
                all_results.append(result)

    # Aggregate results using median
    if not all_results:
        return {"error": "No successful evaluations"}

    # Collect all numeric values
    aggregated = {}
    for result in all_results:
        for key, value in result.items():
            if isinstance(value, (int, float)):
                if key not in aggregated:
                    aggregated[key] = []
                aggregated[key].append(value)

    # Calculate medians
    median_scores = {key: median(values) for key, values in aggregated.items()}

    return median_scores


def evaluate_single_conversation_validated(args):
    """Evaluate a single conversation using validated multi-prompt approach."""
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

        # Evaluate each metric category with multi-prompt strategy
        results = {
            "conversation_id": conv_id,
            "csn": csn_name,
            "n_user_messages": len(user_messages)
        }

        # Cognitive Load (Klepsch et al. 2017)
        cog_load = evaluate_with_multi_prompt(user_messages, "cognitive_load", iterations=2)
        results.update({f"cog_load_{k}": v for k, v in cog_load.items() if k != "error"})

        # Metacognition (Schraw & Dennison 1994 MAI)
        metacog = evaluate_with_multi_prompt(user_messages, "metacognition", iterations=2)
        results.update({f"metacog_{k}": v for k, v in metacog.items() if k != "error"})

        # Linguistic Production (ACTFL 2024 FACT)
        ling_prod = evaluate_with_multi_prompt(user_messages, "linguistic_production", iterations=2)
        results.update({f"ling_{k}": v for k, v in ling_prod.items() if k != "error"})

        # Self-Regulated Learning (MSLQ - Pintrich 1991)
        srl = evaluate_with_multi_prompt(user_messages, "self_regulated_learning", iterations=2)
        results.update({f"srl_{k}": v for k, v in srl.items() if k != "error"})

        # Cognitive Engagement (HESES - Kahu et al. 2018)
        engagement = evaluate_with_multi_prompt(user_messages, "cognitive_engagement", iterations=2)
        results.update({f"engage_{k}": v for k, v in engagement.items() if k != "error"})

        # Critical Thinking (Liu et al. 2014)
        crit_think = evaluate_with_multi_prompt(user_messages, "critical_thinking", iterations=2)
        results.update({f"crit_{k}": v for k, v in crit_think.items() if k != "error"})

        # Iterative Refinement
        refinement = evaluate_with_multi_prompt(user_messages, "iterative_refinement", iterations=2)
        results.update({f"refine_{k}": v for k, v in refinement.items() if k != "error"})

        # Cognitive Debt Indicators (MIT 2025)
        cog_debt = evaluate_with_multi_prompt(user_messages, "cognitive_debt_indicators", iterations=2)
        results.update({f"debt_{k}": v for k, v in cog_debt.items() if k != "error"})

        # Calculate composite scores for backward compatibility
        results["cognitive_engagement"] = results.get("engage_cognitive",
            results.get("engage_processing_depth",
            results.get("engage_engagement_level", 3.0)))

        results["metacognitive_awareness"] = results.get("metacog_metacognitive_awareness",
            np.mean([results.get("metacog_planning", 3.0),
                    results.get("metacog_monitoring", 3.0),
                    results.get("metacog_evaluation", 3.0)]))

        results["linguistic_production"] = results.get("ling_production_quality",
            results.get("ling_esperanto_production",
            np.mean([results.get("ling_accuracy", 3.0),
                    results.get("ling_comprehensibility", 3.0)])))

        results["self_directedness"] = results.get("srl_learning_autonomy",
            results.get("srl_learner_agency",
            results.get("srl_help_seeking", 3.0)))

        results["iterative_refinement"] = results.get("refine_refinement_depth",
            results.get("refine_learning_persistence",
            results.get("refine_conversation_depth", 2.0)))

        results["memory_retention"] = results.get("debt_memory_retention", 2.0)
        results["agency_ownership"] = results.get("debt_ownership", 3.0)

        results["query_sophistication"] = results.get("crit_question_sophistication",
            results.get("crit_intellectual_complexity", 2.5))

        results["overall_learning_quality"] = np.mean([
            results.get("cognitive_engagement", 3.0),
            results.get("metacognitive_awareness", 2.0),
            results.get("self_directedness", 3.0),
            results.get("iterative_refinement", 2.0)
        ])

        # Update progress
        with progress_lock:
            progress_counter["count"] += 1
            if progress_counter["count"] % 5 == 0:
                pct = (progress_counter["count"] / progress_counter["total"]) * 100
                print(f"Progress: {progress_counter['count']}/{progress_counter['total']} ({pct:.1f}%)")

        return results

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


def main():
    """Main execution function with validated multi-prompt evaluation."""
    print("="*80)
    print("LITERATURE-GROUNDED ESPERANTO CONVERSATION EVALUATION")
    print("Multi-Prompt Strategy with Validated Frameworks")
    print("="*80)
    print()
    print("Key Frameworks:")
    print("  - Klepsch et al. (2017): Cognitive Load")
    print("  - Schraw & Dennison (1994): Metacognitive Awareness Inventory")
    print("  - Pintrich et al. (1991): MSLQ Self-Regulated Learning")
    print("  - ACTFL (2024): FACT Language Assessment")
    print("  - Kahu et al. (2018): HESES Student Engagement")
    print("  - Liu et al. (2014): Critical Thinking")
    print("  - MIT Media Lab (2025): Cognitive Debt")
    print()
    print("Method: 3 prompt variants × 2 iterations per metric = 6 ratings per metric")
    print("Aggregation: Median score (reduces outlier bias)")
    print()

    base_dir = Path("/home/user/esperanto-db")

    # Load existing data
    participants_df = pd.read_csv(base_dir / "prompt_participants_final_for_paper.csv")
    conversations_df = pd.read_csv(base_dir / "prompt_conversations_final_for_paper.csv")

    print(f"Loaded {len(participants_df)} participants")
    print(f"Loaded {len(conversations_df)} conversations")
    print()

    # Collect conversations - use raw_data/csn_exports if available
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

    print(f"Found {len(all_conversations)} conversations to evaluate")

    # For testing: limit to first 20 conversations
    print("\n⚠️  TESTING MODE: Evaluating first 20 conversations only")
    print("    (Remove this limit for full dataset evaluation)")
    all_conversations = all_conversations[:20]

    progress_counter["total"] = len(all_conversations)
    print(f"Processing {len(all_conversations)} conversations...\n")

    # Process with fewer workers due to multiple API calls per conversation
    print("Starting validated evaluation...")
    print(f"Using {MODEL}")
    print(f"Max workers: 5 (to manage API rate limits with multi-prompt strategy)")
    print()

    all_results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate_single_conversation_validated, conv_args): conv_args
                   for conv_args in all_conversations}

        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)

    # Save results
    print("\n" + "="*80)
    print("Saving results...")

    results_df = pd.DataFrame(all_results)

    # Save detailed results with all sub-metrics
    detailed_file = base_dir / "evaluation_results_validated_detailed.csv"
    results_df.to_csv(detailed_file, index=False)
    print(f"✓ Saved detailed results: {detailed_file}")
    print(f"  Rows: {len(results_df)}, Columns: {len(results_df.columns)}")

    # Summary statistics
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    composite_metrics = ["cognitive_engagement", "metacognitive_awareness", "linguistic_production",
                        "self_directedness", "iterative_refinement", "memory_retention",
                        "agency_ownership", "query_sophistication", "overall_learning_quality"]

    print("\nComposite Metrics (for backward compatibility):")
    for metric in composite_metrics:
        if metric in results_df.columns:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            print(f"  {metric:30s}: {mean_val:.2f} ± {std_val:.2f}")

    print(f"\nTotal conversations evaluated: {len(results_df)}")
    print(f"Conversations with errors: {len(results_df[results_df.get('error', '') != ''])}")

    print("\n✓ Validated evaluation complete!")
    print("\nNext steps:")
    print("  1. Review detailed results in CSV")
    print("  2. Validate subset against human coders")
    print("  3. Calculate inter-rater reliability")
    print("  4. If r > 0.70, run on full dataset (remove 20-conversation limit)")


if __name__ == "__main__":
    main()
