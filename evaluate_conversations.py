#!/usr/bin/env python3
"""
Conversation Evaluation Script
Based on literature-grounded frameworks from:
- MIT Media Lab (2025): "Your Brain on ChatGPT" - Cognitive debt theory
- Behavioral Economics: Cognitive offloading, effort allocation
- Language Learning Research: Self-directed learning, metacognitive awareness
- Educational Psychology: Agency, ownership, iterative refinement
"""

import json
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Any
import time
from openai import OpenAI
import numpy as np

# OpenAI API configuration
# Set your API key as environment variable: export OPENAI_API_KEY="your-key-here"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")
client = OpenAI(api_key=OPENAI_API_KEY)

# Use GPT-4 (standard, not outdated model)
MODEL = "gpt-4-turbo-preview"


class ConversationEvaluator:
    """
    Evaluates ChatGPT conversations for language learning based on
    literature-grounded metrics.
    """

    def __init__(self):
        self.evaluation_prompts = self._build_evaluation_prompts()

    def _build_evaluation_prompts(self) -> Dict[str, str]:
        """
        Build evaluation prompts based on literature-grounded frameworks.
        """
        return {
            "cognitive_engagement": """
Analyze the cognitive engagement level in this Esperanto learning conversation.
Based on MIT's "cognitive debt" research, evaluate on a 1-5 scale:
1 = Passive acceptance, minimal thinking
2 = Surface-level engagement
3 = Moderate engagement with some critical thinking
4 = Deep engagement, questioning, analyzing
5 = Highly active, metacognitive, critical thinking

Consider:
- Depth of questions asked
- Critical evaluation of responses
- Evidence of reflection
- Metacognitive awareness (thinking about thinking)

USER MESSAGES: {user_messages}

Return ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>", "evidence": "<specific examples>"}}
""",

            "query_sophistication": """
Evaluate the linguistic and conceptual sophistication of the learner's queries.
Rate on 1-5 scale:
1 = Simple, vague, single-word queries
2 = Basic queries with minimal context
3 = Clear queries with some specificity
4 = Well-formed, specific queries showing understanding
5 = Sophisticated, nuanced queries demonstrating deep knowledge

Consider:
- Linguistic complexity of questions
- Specificity and precision
- Conceptual depth
- Use of Esperanto vocabulary in questions

USER MESSAGES: {user_messages}

Return ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>", "examples": ["<example 1>", "<example 2>"]}}
""",

            "self_directedness": """
Based on cognitive offloading research, assess the learner's self-directedness vs. AI dependency.
Rate on 1-5 scale:
1 = Complete dependency, no independent thinking
2 = Heavy reliance, minimal self-direction
3 = Balanced - uses AI as tool while maintaining autonomy
4 = Highly self-directed with strategic AI use
5 = Autonomous learner using AI for verification/enhancement only

Consider:
- Do they seek answers or understanding?
- Do they attempt before asking?
- Do they evaluate AI responses critically?
- Do they show independent problem-solving?

USER MESSAGES: {user_messages}

Return ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>", "dependency_indicators": ["<indicator 1>", "<indicator 2>"]}}
""",

            "iterative_refinement": """
Assess the learner's iterative refinement behavior - a key indicator of deep learning.
Rate on 1-5 scale:
1 = No follow-up, accepts first answer
2 = Minimal follow-up
3 = Some clarification questions
4 = Regular refinement and deeper exploration
5 = Systematic iterative deepening with multiple perspectives

Consider:
- Follow-up questions
- Requests for clarification
- Progressive deepening of topic
- Revisions based on feedback

USER MESSAGES: {user_messages}

Return ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>", "refinement_patterns": ["<pattern 1>", "<pattern 2>"]}}
""",

            "declarative_vs_procedural": """
Classify the learner's orientation between declarative (what) and procedural (how) learning.
Return a score from 1-5:
1 = Purely declarative (only facts, translations)
2 = Mostly declarative with some procedural
3 = Balanced mix
4 = Mostly procedural (grammar rules, usage patterns)
5 = Purely procedural (processes, strategies, how to use)

Declarative: "What does X mean?", "Translate Y"
Procedural: "How do I form X?", "When do I use Y?", "What's the rule for Z?"

USER MESSAGES: {user_messages}

Return ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>", "orientation": "<declarative|balanced|procedural>", "examples": {{"declarative": ["<example>"], "procedural": ["<example>"]}}}}
""",

            "agency_ownership": """
Based on MIT's research on "essay ownership", assess learner agency and ownership of their learning.
Rate on 1-5 scale:
1 = No ownership, purely receptive
2 = Minimal agency, follows AI lead
3 = Moderate ownership with some initiative
4 = Strong ownership, directs conversation
5 = Complete ownership, AI as tool for self-directed goals

Consider:
- Do they set learning goals?
- Do they personalize examples?
- Do they relate to their context?
- Do they take initiative in conversation direction?

USER MESSAGES: {user_messages}

Return ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>", "ownership_indicators": ["<indicator 1>", "<indicator 2>"]}}
""",

            "linguistic_production": """
Evaluate the quality of Esperanto linguistic production in the learner's messages.
Rate on 1-5 scale:
1 = No Esperanto usage, English only
2 = Minimal Esperanto (single words)
3 = Some Esperanto usage with English support
4 = Regular Esperanto usage with good structure
5 = Sophisticated Esperanto production

Consider:
- Frequency of Esperanto usage
- Grammatical complexity
- Vocabulary range
- Accuracy (if evident)

USER MESSAGES: {user_messages}

Return ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>", "esperanto_usage_percent": <0-100>, "examples": ["<example 1>", "<example 2>"]}}
""",

            "memory_retention_indicators": """
Assess indicators of memory formation and retention based on MIT cognitive debt research.
Rate on 1-5 scale:
1 = No evidence of retention, repeated questions
2 = Minimal retention, frequent re-asking
3 = Some retention, occasional references to prior learning
4 = Good retention, builds on previous knowledge
5 = Excellent retention, integrates and synthesizes across topics

Consider:
- References to previous conversations
- Building on prior knowledge
- Not re-asking same questions
- Making connections between topics

USER MESSAGES: {user_messages}

Return ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>", "retention_evidence": ["<example 1>", "<example 2>"]}}
""",

            "metacognitive_awareness": """
Evaluate metacognitive awareness - thinking about one's own learning process.
Rate on 1-5 scale:
1 = No metacognitive awareness
2 = Minimal awareness of learning process
3 = Some reflection on learning
4 = Regular metacognitive monitoring
5 = Sophisticated metacognitive strategies

Consider:
- Reflection on learning strategies
- Awareness of difficulties
- Self-monitoring of understanding
- Strategic planning of learning

USER MESSAGES: {user_messages}

Return ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>", "metacognitive_indicators": ["<indicator 1>", "<indicator 2>"]}}
""",

            "question_type_distribution": """
Analyze the distribution of question types in the conversation.
Categorize each question and return percentages.

Categories:
- translation: Simple translation requests
- grammar: Grammar rules and explanations
- usage: How/when to use something
- clarification: Follow-up clarification questions
- application: Applying knowledge to new situations
- meta: Questions about learning process itself

USER MESSAGES: {user_messages}

Return ONLY a JSON object:
{{"translation": <0-100>, "grammar": <0-100>, "usage": <0-100>, "clarification": <0-100>, "application": <0-100>, "meta": <0-100>, "total_questions": <number>}}
"""
        }

    def extract_user_messages(self, conversation_data: Dict) -> List[str]:
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

    def call_openai_evaluation(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Call OpenAI API with retry logic."""
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert in educational psychology, behavioral economics, and language learning research. Provide precise, evidence-based evaluations in valid JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent evaluations
                    response_format={"type": "json_object"}
                )

                result = json.loads(response.choices[0].message.content)
                return result

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    print(f"Error after {max_retries} attempts: {e}")
                    return {"error": str(e)}

    def evaluate_conversation(self, conversation_data: Dict, conversation_id: str) -> Dict[str, Any]:
        """
        Evaluate a single conversation across all metrics.
        """
        user_messages = self.extract_user_messages(conversation_data)

        if not user_messages:
            return {
                "conversation_id": conversation_id,
                "n_user_messages": 0,
                "error": "No user messages found"
            }

        user_messages_text = "\n\n".join([f"Message {i+1}: {msg}" for i, msg in enumerate(user_messages)])

        results = {
            "conversation_id": conversation_id,
            "n_user_messages": len(user_messages)
        }

        # Run each evaluation
        for eval_name, prompt_template in self.evaluation_prompts.items():
            print(f"  Evaluating {eval_name}...")
            prompt = prompt_template.format(user_messages=user_messages_text)
            eval_result = self.call_openai_evaluation(prompt)

            # Flatten results with prefix
            for key, value in eval_result.items():
                results[f"{eval_name}_{key}"] = value

            # Rate limiting
            time.sleep(0.5)

        return results


def load_conversation_from_csn(csn_folder: Path) -> List[Dict]:
    """Load conversation data from a CSN folder."""
    conv_file = csn_folder / "conversations.json"

    if not conv_file.exists():
        # Try subdirectory (like CSN1/csn1/)
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
    """Main execution function."""
    print("="*80)
    print("ESPERANTO CONVERSATION EVALUATION")
    print("Literature-Grounded Metrics Based on:")
    print("- MIT Media Lab (2025): Cognitive Debt Research")
    print("- Behavioral Economics: Cognitive Offloading Theory")
    print("- Language Learning Research: Self-Directed Learning")
    print("="*80)
    print()

    # Load participant and conversation data
    base_dir = Path("/home/user/esperanto-db")
    participants_df = pd.read_csv(base_dir / "prompt_participants_final_for_paper.csv")
    conversations_df = pd.read_csv(base_dir / "prompt_conversations_final_for_paper.csv")

    print(f"Loaded {len(participants_df)} participants")
    print(f"Loaded {len(conversations_df)} conversations")
    print()

    # Initialize evaluator
    evaluator = ConversationEvaluator()

    # Store all evaluation results
    all_evaluations = []

    # Check for existing progress
    progress_file = base_dir / "evaluation_progress.json"
    evaluated_ids = set()
    if progress_file.exists():
        print("Found existing progress file, loading...")
        with open(progress_file, 'r') as f:
            all_evaluations = json.load(f)
            evaluated_ids = {e['conversation_id'] for e in all_evaluations}
        print(f"Resuming from {len(all_evaluations)} already evaluated conversations")

    # Process each CSN folder
    csn_folders = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("CSN")])

    total_processed = len(all_evaluations)
    for csn_folder in csn_folders:
        csn_name = csn_folder.name
        print(f"\nProcessing {csn_name}...")

        conversations = load_conversation_from_csn(csn_folder)

        if not conversations:
            print(f"  No conversations found in {csn_name}")
            continue

        print(f"  Found {len(conversations)} conversation(s)")

        for i, conv_data in enumerate(conversations):
            conv_id = conv_data.get('id', f"{csn_name}_{i}")

            # Skip if already evaluated
            if conv_id in evaluated_ids:
                print(f"  Skipping {conv_id} (already evaluated)")
                continue

            print(f"  Evaluating conversation {conv_id}...")

            try:
                eval_results = evaluator.evaluate_conversation(conv_data, conv_id)
                eval_results['csn'] = csn_name
                all_evaluations.append(eval_results)
                total_processed += 1

                # Save progress every 10 conversations
                if total_processed % 10 == 0:
                    with open(progress_file, 'w') as f:
                        json.dump(all_evaluations, f)
                    print(f"  Progress saved ({total_processed} conversations)")

            except Exception as e:
                print(f"  ERROR evaluating {conv_id}: {e}")
                continue

    # Convert to DataFrame
    print("\n" + "="*80)
    print("Creating consolidated evaluation dataset...")
    evaluations_df = pd.DataFrame(all_evaluations)

    # Merge with conversations data
    merged_df = conversations_df.merge(
        evaluations_df,
        left_on='conversation_id',
        right_on='conversation_id',
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
    output_file = base_dir / "conversations_with_evaluations.csv"
    final_df.to_csv(output_file, index=False)
    print(f"Saved consolidated dataset to: {output_file}")

    # Save just evaluations for reference
    evals_only_file = base_dir / "evaluation_results.csv"
    evaluations_df.to_csv(evals_only_file, index=False)
    print(f"Saved evaluation results to: {evals_only_file}")

    # Print summary statistics
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    score_columns = [col for col in evaluations_df.columns if col.endswith('_score')]
    if score_columns:
        print("\nMean Scores:")
        for col in score_columns:
            mean_score = evaluations_df[col].mean()
            print(f"  {col}: {mean_score:.2f}")

    print(f"\nTotal conversations evaluated: {len(evaluations_df)}")
    print(f"Total participants: {len(final_df['participant_id'].unique())}")
    print(f"\nFinal dataset shape: {final_df.shape}")
    print(f"Columns: {len(final_df.columns)}")
    print("\nDone!")


if __name__ == "__main__":
    main()
