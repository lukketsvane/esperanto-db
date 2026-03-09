# Esperanto Learning with ChatGPT: Behavioral Economics Study

## Dataset

397 ChatGPT conversations from 375 participants learning Esperanto, evaluated using validated frameworks from educational psychology and cognitive science literature.

## Key Findings

### Cognitive Debt Patterns (MIT 2025 Framework)

| Indicator | Score | Interpretation |
|-----------|-------|----------------|
| Iterative Refinement | 2.91/5 | Moderate follow-up depth |
| Memory Retention | 3.68/5 | Good knowledge retention |
| Metacognitive Awareness | 3.22/5 | Moderate self-reflection |

Learners show moderate engagement patterns with above-average memory retention and metacognitive awareness.

### Learning Patterns

**Declarative Bias**: 50% of learners favor fact/translation requests over procedural understanding (grammar, usage rules).

**AI Dependency**: 48% show balanced autonomy, 32% high reliance, 20% high self-direction.

**Question Types**: 40-45% translation, 25-30% grammar, <5% application, <2% meta-learning.

**Esperanto Production**: Mean 38% usage in messages - most interaction remains English-based.

### Correlations (r > 0.6)

- Cognitive Engagement ↔ Learning Quality: r=0.72
- Agency/Ownership ↔ Self-Directedness: r=0.68
- Query Sophistication ↔ Cognitive Engagement: r=0.65

### Learner Profiles (K-means clustering)

1. Low Engagers (25%): Minimal metrics across board
2. Moderate Balanced (38%): Average scores, balanced approach
3. High Agency (20%): Strong self-direction and ownership
4. Production Focused (17%): Higher Esperanto usage, less translation

## Data Files

**data/conversations_final.csv**: Complete dataset (397 conversations, 44 columns including evaluations and metadata)
**data/prompt_participants_final_for_paper.csv**: Participant information (375 participants)
**figures/**: Publication-quality visualizations (PNG+PDF)
**analysis/LITERATURE_REVIEW_COMPREHENSIVE.md**: Framework sources (20+ papers)

## Evaluation Metrics (1-5 scale)

Grounded in validated frameworks:
- Cognitive Load: Klepsch et al. 2017
- Metacognition: Schraw & Dennison 1994 (MAI)
- Self-Regulation: Pintrich et al. 1991 (MSLQ)
- Linguistic Production: ACTFL 2024 (FACT)
- Engagement: Kahu et al. 2018 (HESES)
- Critical Thinking: Liu et al. 2014
- Cognitive Debt: MIT Media Lab 2025
- **Copy-Paste & AI Usage**: (New for March 2026 update)

**Results (March 2026 Update)**:
- cognitive_engagement: 2.44 ± 1.00
- metacognitive_awareness: 1.30 ± 0.51
- linguistic_production: 2.63 ± 0.83
- self_directedness: 2.49 ± 0.99
- iterative_refinement: 2.21 ± 1.03
- memory_retention: 2.66 ± 1.23
- query_sophistication: 1.63 ± 0.72
- agency_ownership: 2.27 ± 0.98
- overall_learning_quality: 2.11 ± 0.81

## Methodology & LLM-as-a-Judge Design

### LLM Evaluation
**Model**: `gemini-3.1-flash-lite-preview` (Updated March 2026)
**Context**: Full-text extraction of both **User** and **Assistant** messages to capture instructional flow.

**Prompting Architecture**:
We employ a parameterized single-script approach (`scripts/evaluate_conversations.py`) that reads system instructions from external files (`prompts/`). This allows us to compare two modes of LLM evaluation, adhering to current LLM-as-a-Judge best practices (e.g., MT-Bench / Zheng et al., 2023, which demonstrate that explicit rubrics significantly increase inter-rater reliability compared to zero-shot inference):
1. **Implicit Mode (`prompts/system_implicit.txt`)**: Relies on the LLM's latent, zero-shot understanding of the psychological variables.
2. **Explicit Mode (`prompts/system_explicit.txt`)**: Provides the LLM with strict, theoretically grounded definitions for every metric (e.g., `germane_load`, `metacog_planning`).

### Copy-Paste Detection: Dual Approach
To ensure data integrity regarding "low-effort" AI usage, we employ two distinct methodologies:
1. **Programmatic Matching (`copypaste_score_auto`)**: A Python-based string matching algorithm that checks user messages against a library of known practice questions (e.g., "Tom ne dormas"). This provides a verbatim baseline for copying.
2. **LLM Judgment (`copypaste_dummy`)**: A GPT/Gemini-based qualitative assessment that identifies "structural" copy-pasting—where a user copies the intent, format, or multi-part instructions that might bypass simple string matching.

These two scores are consolidated in `conversations_evaluated_final.csv` to allow for multi-dimensional filtering of participant effort.

All metrics derived from observable behaviors in conversation transcripts using validated frameworks from educational psychology literature.

## Repository Structure

```
data/
  conversations_final.csv          Complete dataset (397 conversations, 44 columns)
  prompt_participants_final.csv    Participant data (375 participants)
figures/                           Publication figures (PNG+PDF)
scripts/
  evaluate_conversations.py        Evaluation script
  analyze_and_visualize.py         Figure generation
  merge_data.py                    Data merging utility
raw_data/csn_exports/              Original CSN1-CSN22 exports
analysis/
  LITERATURE_REVIEW_COMPREHENSIVE.md  Framework sources (20+ papers)
```

## Citations

**MIT Cognitive Debt Study**:
MIT Media Lab (2025). Your Brain on ChatGPT: Accumulation of cognitive debt when using an AI assistant for essay writing task. arXiv:2506.08872

**Validated Frameworks**:
- Klepsch, M., Schmitz, F., & Seufert, T. (2017). Frontiers in Psychology, 8, 1997.
- Schraw, G., & Dennison, R. S. (1994). Contemporary Educational Psychology, 19(4), 460-475.
- Pintrich, P. R., et al. (1991). MSLQ Manual. University of Michigan.
- ACTFL (2024). ACTFL Proficiency Guidelines 2024.
- Kahu, E. R., et al. (2018). Research in Higher Education, 60(2), 134-159.
- Liu, O. L., et al. (2014). ETS Research Report Series.

See analysis/LITERATURE_REVIEW_COMPREHENSIVE.md for complete bibliography.
