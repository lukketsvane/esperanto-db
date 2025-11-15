# Esperanto Learning Conversation Evaluations - Summary

## Quick Start

**Main Dataset**: `conversations_with_evaluations_final.csv` (397 conversations, 375 participants, 56 columns)

This dataset contains all conversation data merged with:
- Literature-grounded evaluation metrics
- Participant information with recovered IDs
- Behavioral economics and cognitive psychology measures

## Key Files

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `conversations_with_evaluations_final.csv` | Complete dataset with all evaluations | 397 | 56 |
| `evaluation_results_final.csv` | Just the evaluation metrics | 397 | 18 |
| `participant_level_evaluations.csv` | Aggregated by participant | 375 | 7 |
| `EVALUATION_FRAMEWORK.md` | Complete documentation | - | - |

## Evaluation Metrics (All 1-5 scale)

Based on **MIT Media Lab (2025)** cognitive debt research and **behavioral economics** literature:

1. **cognitive_engagement** (2.64 ± 0.67): Depth of thinking and metacognitive awareness
2. **query_sophistication** (2.36 ± 0.66): Linguistic and conceptual sophistication
3. **self_directedness** (2.70 ± 0.71): Autonomy vs. AI dependency
4. **iterative_refinement** (1.86 ± 0.70): Progressive deepening behavior
5. **learning_orientation** (2.48 ± 0.74): Declarative (facts) vs. Procedural (processes)
6. **agency_ownership** (2.70 ± 0.71): Taking ownership of learning
7. **linguistic_production** (2.20 ± 0.96): Esperanto production quality
8. **memory_retention** (2.09 ± 0.65): Evidence of retention
9. **metacognitive_awareness** (1.98 ± 0.50): Reflection on learning process
10. **overall_learning_quality** (2.48 ± 0.67): Holistic assessment

## Key Findings

### Concerning Patterns
- **Low Iterative Refinement** (1.86/5): Learners accept first answers without deeper exploration
- **Low Metacognitive Awareness** (1.98/5): Minimal reflection on learning strategies
- **Moderate Cognitive Engagement** (2.64/5): Mostly surface-level thinking
- **Low Memory Retention** (2.09/5): Aligns with MIT "cognitive debt" findings

### Positive Indicators
- **Balanced Self-Directedness** (2.70/5): Not over-reliant on AI
- **Moderate Agency** (2.70/5): Some ownership of learning process

### Behavioral Patterns
- **Esperanto Usage**: 38.1% average (only ~1/3 of messages contain Esperanto)
- **Query Length**: 6.4 words average (relatively short)
- **Learning Orientation**: Slightly more declarative (facts) than procedural (processes)

### Question Type Distribution
- **Translation**: Highest percentage (varies by participant)
- **Grammar**: Second most common
- **Application & Meta**: Lowest (concerning for deep learning)

## Data Quality

- **Evaluation Coverage**: 98% (389/397 conversations successfully evaluated)
- **ID Confidence**:
  - High confidence: 242 participants (64.5%)
  - Imputed medium: 20 participants (5.3%)
  - Synthetic low: 100 participants (26.7%)
- **Multiple Conversations**: 15 participants had multiple conversations

## Usage Examples

### Load Main Dataset
```python
import pandas as pd
df = pd.read_csv('conversations_with_evaluations_final.csv')
```

### Analyze by ID Confidence
```python
high_conf = df[df['id_confidence'] == 'high']
print(f"High confidence participants: {len(high_conf)}")
print(f"Mean cognitive engagement: {high_conf['cognitive_engagement'].mean():.2f}")
```

### Participant-Level Analysis
```python
participant_df = pd.read_csv('participant_level_evaluations.csv')
participant_df.describe()
```

### Correlation Analysis
```python
# Correlation between self-directedness and learning quality
correlation = df[['self_directedness', 'overall_learning_quality']].corr()
```

## Methods

- **Model**: GPT-4 Turbo Preview (gpt-4-turbo-preview)
- **Temperature**: 0.3 (for consistency)
- **Processing**: Concurrent (10 workers)
- **API Calls**: 1 comprehensive evaluation per conversation
- **Total Runtime**: ~15 minutes for 397 conversations

## Literature References

### Primary Sources
1. MIT Media Lab (2025). "Your Brain on ChatGPT: Accumulation of Cognitive Debt when Using an AI Assistant for Essay Writing Task." arXiv:2506.08872

2. Behavioral Economics research on cognitive offloading and decision-making with AI tools

3. Language learning research on self-directed learning, metacognitive awareness, and agency

### Key Concepts
- **Cognitive Debt**: Accumulation of reduced cognitive processing from AI reliance
- **Cognitive Offloading**: Using external tools instead of internal processing
- **Metacognitive Awareness**: Thinking about one's own thinking and learning
- **Agency**: Taking ownership and initiative in learning process
- **Declarative vs. Procedural**: Facts/translations vs. processes/rules

## Scripts

- `evaluate_conversations_fast.py`: Main evaluation script (concurrent processing)
- `test_evaluation.py`: Test script for single conversation
- `evaluate_conversations.py`: Original sequential version (deprecated)

## Next Steps for Analysis

1. **Correlation Studies**: Examine relationships between metrics
2. **ID Confidence Impact**: Compare high vs. low confidence participants
3. **Temporal Analysis**: Learning progression over conversation
4. **Clustering**: Identify learner archetypes
5. **Outcome Prediction**: Correlate with quiz performance or retention tests

## Contact

For questions about the evaluation framework or methodology, see `EVALUATION_FRAMEWORK.md` for complete documentation.

---
**Generated**: 2025-11-15
**Dataset Version**: Final (375 participants, 397 conversations)
**Evaluation Completeness**: 98%
