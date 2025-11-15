# Esperanto Learning with ChatGPT: Behavioral Economics Study

## Dataset

397 ChatGPT conversations from 375 participants learning Esperanto, evaluated using validated frameworks from educational psychology and cognitive science literature.

## Key Findings

### Cognitive Debt Patterns (MIT 2025 Framework)

| Indicator | Score | Interpretation |
|-----------|-------|----------------|
| Iterative Refinement | 1.86/5 | Low follow-up depth, passive acceptance |
| Memory Retention | 2.09/5 | Minimal knowledge building |
| Metacognitive Awareness | 1.98/5 | Limited self-reflection |

Evidence of cognitive offloading consistent with MIT's "cognitive debt" findings - outsourcing mental effort to AI reduces engagement and knowledge retention.

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

**conversations_evaluated.csv**: Full dataset (397 rows, evaluation metrics)
**prompt_conversations_final_for_paper.csv**: Conversation metadata
**prompt_participants_final_for_paper.csv**: Participant information (375)
**quiz_answers.json**: Quiz responses
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

**Results**:
- cognitive_engagement: 2.64 ± 0.67
- metacognitive_awareness: 1.98 ± 0.50
- linguistic_production: 2.20 ± 0.96
- self_directedness: 2.70 ± 0.71
- iterative_refinement: 1.86 ± 0.70
- memory_retention: 2.09 ± 0.65
- query_sophistication: 2.36 ± 0.66
- agency_ownership: 2.70 ± 0.71
- overall_learning_quality: 2.48 ± 0.67

## Methodology

**Model**: GPT-4 Turbo (gpt-4-turbo-preview), temperature 0.2
**Success Rate**: 98% (389/397 conversations evaluated)
**Processing**: Concurrent evaluation, validated prompts based on peer-reviewed instruments

All metrics derived from observable behaviors in conversation transcripts. No human validation used - direct LLM assessment with literature-grounded prompts.

## Repository Structure

```
raw_data/csn_exports/      CSN1-CSN22 original exports
analysis/                  Literature review, figures
*.csv                      Processed datasets
evaluate_conversations.py  Evaluation script
analyze_and_visualize.py   Figure generation
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
