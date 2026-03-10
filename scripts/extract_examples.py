#!/usr/bin/env python3
"""Extract anonymized conversation examples showing complement vs substitute learning patterns."""
import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()

# Exemplar conversation IDs (identified from explicit evaluation data)
EXEMPLARS = {
    "substitute": {
        "id": "674da645-4518-8001-9cef-a87cd1b4a1f0",
        "csn": "CSN1",
        "label": "Substitute Learner (Copy-Paste)",
        "scores": "copypaste=4, substitute=4, complement=2",
        "description": "This participant copied quiz instructions verbatim into ChatGPT, treating the AI as a translation engine. No original questions were asked — every prompt was a direct paste from the assignment.",
    },
    "complement": {
        "id": "67517f3b-39dc-8001-b00d-4aba7f609de4",
        "csn": "CSN1",
        "label": "Complement Learner (Active Engagement)",
        "scores": "copypaste=1, substitute=2, complement=4",
        "description": "This participant typed fully original queries, proactively requesting lessons on grammar, vocabulary, and sentence construction. They used AI as a tutor, not a shortcut.",
    },
    "strategic": {
        "id": "6751d69b-4d18-800e-a80a-4b649aac1206",
        "csn": "CSN10",
        "label": "Strategic Copier (Mixed Approach)",
        "scores": "copypaste=4, substitute=3, complement=3",
        "description": "This participant copied some practice questions but also followed up with requests for explanations. A hybrid pattern showing partial engagement alongside copying behavior.",
    },
}

# Max user messages to show per example
MAX_MSGS_SHOWN = 8


def extract_messages(conversation_data, roles=None):
    """Extract messages sorted by creation time."""
    if roles is None:
        roles = ["user", "assistant"]
    messages = []
    if "mapping" in conversation_data:
        for node in conversation_data["mapping"].values():
            msg = node.get("message")
            if not msg:
                continue
            role = msg.get("author", {}).get("role")
            if role not in roles:
                continue
            content = msg.get("content", {})
            ct = msg.get("create_time", 0) or 0
            if isinstance(content, dict) and "parts" in content:
                parts = content["parts"]
                if parts and isinstance(parts[0], str) and parts[0].strip():
                    messages.append((ct, role, parts[0].strip()))
    messages.sort(key=lambda x: x[0])
    return messages


def anonymize(text):
    """Remove participant IDs and personal identifiers."""
    import re
    # Remove ID patterns like "My ID is 05122024_1645_1"
    text = re.sub(r"\b\d{8}_\d{4}_\d+\b", "[REDACTED_ID]", text)
    # Remove any email patterns
    text = re.sub(r"\S+@\S+\.\S+", "[REDACTED_EMAIL]", text)
    return text


def find_conversation(csn_name, conv_id):
    """Find a conversation by CSN folder and conversation ID."""
    csn_dir = BASE_DIR / "raw_data" / "csn_exports" / csn_name
    if not csn_dir.exists():
        return None
    # Check for conversations.json directly in CSN folder
    candidates = [csn_dir / "conversations.json"]
    # Also check subdirectories
    for sub in csn_dir.iterdir():
        if sub.is_dir():
            candidates.append(sub / "conversations.json")
    for conv_file in candidates:
        if conv_file.exists():
            with open(conv_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            convs = data if isinstance(data, list) else [data]
            for c in convs:
                if c.get("id") == conv_id:
                    return c
    return None


def format_example(key, info):
    """Format a single conversation example as markdown."""
    conv = find_conversation(info["csn"], info["id"])
    if not conv:
        return f"### {info['label']}\n\n*Conversation not found.*\n"

    messages = extract_messages(conv)
    lines = []
    lines.append(f"### {info['label']}")
    lines.append(f"**Evaluation Scores**: {info['scores']}")
    lines.append(f"**Total Messages**: {len(messages)}")
    lines.append("")
    lines.append(f"> {info['description']}")
    lines.append("")

    # Show selected user messages (skip ID messages, limit to MAX_MSGS_SHOWN)
    user_msgs = [(t, r, m) for t, r, m in messages if r == "user"]
    shown = 0
    lines.append("**Selected User Prompts:**")
    lines.append("")
    lines.append("```")
    for _, role, text in user_msgs:
        clean = anonymize(text)
        # Skip ID registration messages
        if "my id is" in clean.lower() or "[REDACTED_ID]" in clean:
            continue
        if shown >= MAX_MSGS_SHOWN:
            lines.append(f"  ... ({len(user_msgs) - shown} more messages)")
            break
        lines.append(f"  USER: {clean[:300]}")
        shown += 1
    lines.append("```")
    lines.append("")

    # Show a short dialogue excerpt (first few exchanges after ID)
    lines.append("**Dialogue Excerpt:**")
    lines.append("")
    excerpt_count = 0
    skip_id = True
    for _, role, text in messages:
        clean = anonymize(text)
        if skip_id and role == "user" and ("my id is" in clean.lower() or "[REDACTED_ID]" in clean):
            continue
        skip_id = False
        if excerpt_count >= 6:
            lines.append("> *... conversation continues ...*")
            break
        prefix = "**User**" if role == "user" else "**Assistant**"
        # Truncate long assistant responses
        display = clean[:250] + "..." if len(clean) > 250 else clean
        lines.append(f"> {prefix}: {display}")
        lines.append(">")
        excerpt_count += 1

    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def main():
    output_lines = []
    output_lines.append("# Conversation Examples: Complement vs Substitute Learning")
    output_lines.append("")
    output_lines.append("These are anonymized excerpts from real participant conversations,")
    output_lines.append("selected to illustrate the three main AI usage patterns identified in our analysis.")
    output_lines.append("Scores are from the explicit LLM-as-a-Judge evaluation (1-5 scale).")
    output_lines.append("")

    for key in ["substitute", "complement", "strategic"]:
        info = EXEMPLARS[key]
        output_lines.append(format_example(key, info))

    # Summary comparison table
    output_lines.append("## Pattern Summary")
    output_lines.append("")
    output_lines.append("| Pattern | Copy-Paste | Substitute | Complement | Behavior |")
    output_lines.append("|---------|-----------|-----------|------------|----------|")
    output_lines.append("| Substitute Learner | 4/5 | 4/5 | 2/5 | Pastes quiz instructions, no original thought |")
    output_lines.append("| Complement Learner | 1/5 | 2/5 | 4/5 | Asks original questions, seeks understanding |")
    output_lines.append("| Strategic Copier | 4/5 | 3/5 | 3/5 | Copies some questions but also asks for explanations |")
    output_lines.append("")

    out_path = BASE_DIR / "figures" / "conversation_examples.md"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"Conversation examples saved to {out_path}")


if __name__ == "__main__":
    main()
