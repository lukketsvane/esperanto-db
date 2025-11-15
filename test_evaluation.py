#!/usr/bin/env python3
"""Test evaluation on a small sample first"""

import json
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_conversations import ConversationEvaluator, load_conversation_from_csn

def test_single_conversation():
    """Test evaluation on a single conversation."""
    print("Testing evaluation system on CSN2...")

    base_dir = Path("/home/user/esperanto-db")
    csn2_folder = base_dir / "CSN2"

    # Load conversations
    conversations = load_conversation_from_csn(csn2_folder)

    if not conversations:
        print("No conversations found!")
        return

    print(f"Found {len(conversations)} conversation(s)")

    # Test on first conversation only
    conv_data = conversations[0]
    conv_id = conv_data.get('id', 'test')

    print(f"\nTesting conversation: {conv_id}")

    evaluator = ConversationEvaluator()

    # Extract user messages first to see what we're working with
    user_messages = evaluator.extract_user_messages(conv_data)
    print(f"\nFound {len(user_messages)} user messages:")
    for i, msg in enumerate(user_messages[:5]):  # Show first 5
        print(f"  {i+1}. {msg[:100]}...")

    # Run evaluation
    print("\nRunning evaluations...")
    results = evaluator.evaluate_conversation(conv_data, conv_id)

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    for key, value in results.items():
        if not key.endswith('_examples') and not key.endswith('_indicators') and not key.endswith('_evidence'):
            print(f"{key}: {value}")

    print("\nTest completed successfully!")
    return results

if __name__ == "__main__":
    test_single_conversation()
