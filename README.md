# Esperanto Learning with ChatGPT: Behavioral Economics Study

## Overview

This repository contains data and analysis from a behavioral economics study examining ChatGPT-assisted Esperanto language learning. The research investigates cognitive debt patterns, learning effectiveness, and AI dependency behaviors based on MIT's 2025 "Your Brain on ChatGPT" study and behavioral economics principles.

**Key Findings**: Evidence of cognitive offloading patterns with low iterative refinement (1.86/5), memory retention (2.09/5), and metacognitive awareness (1.98/5), suggesting accumulation of "cognitive debt" in AI-assisted learning contexts.

---

## Repository Structure

```
esperanto-db/
├── analysis/                                   # Data science analysis
│   ├── figures/                               # Publication-quality figures (20 files)
│   │   ├── figure_1_metrics_distribution.*    # Overview of all eval metrics
│   │   ├── figure_2_correlation_matrix.*      # Metric correlations
│   │   ├── figure_3_cognitive_debt.*          # MIT study indicators
│   │   ├── figure_4_learning_orientation.*    # Declarative vs procedural
│   │   ├── figure_5_question_types.*          # Question distribution
│   │   ├── figure_6_id_confidence.*           # ID confidence impact
│   │   ├── figure_7_pca_clustering.*          # Learner profiles
│   │   ├── figure_8_esperanto_usage.*         # Language production
│   │   ├── figure_9_self_directedness.*       # AI dependency patterns
│   │   └── figure_10_summary_dashboard.*      # Complete overview
│   ├── comprehensive_analysis.py              # Analysis script
│   └── statistical_analysis_report.txt        # Full stats report
│
├── raw_data/                                   # Original data (archived)
│   ├── csn_exports/                           # CSN1-CSN22 folders
│   └── promptdata_archive/                    # Original promptdata
│
├── conversations_with_evaluations_final.csv   # Main dataset (397 convs, 56 cols)
├── evaluation_results_final.csv               # Evaluation metrics only
├── participant_level_evaluations.csv          # Aggregated by participant (375)
│
├── prompt_conversations_final_for_paper.csv   # Conversations metadata
├── prompt_participants_final_for_paper.csv    # Participant information
├── quiz_answers.json                          # Quiz responses
├── quiz_structure_summary.csv                 # Quiz structure
│
├── evaluate_conversations_fast.py             # Evaluation script (concurrent)
├── evaluate_conversations.py                  # Evaluation script (sequential)
├── test_evaluation.py                         # Testing script
│
├── EVALUATION_FRAMEWORK.md                    # Complete methodology
├── README_EVALUATIONS.md                      # Quick start guide
└── README.md                                  # This file
```

---

## Quick Start

### Load Main Dataset
```python
import pandas as pd

# Load complete dataset with evaluations
df = pd.read_csv('conversations_with_evaluations_final.csv')

# Load participant-level aggregations
participants = pd.read_csv('participant_level_evaluations.csv')

print(f"Conversations: {len(df)}, Participants: {len(participants)}")
```

### View Figures
All figures are in `analysis/figures/` in both PNG (high-res) and PDF (vector) formats.

### Run Analysis
```bash
python3 analysis/comprehensive_analysis.py
```

---

## Dataset Information

### Main Dataset: `conversations_with_evaluations_final.csv`

**Size**: 397 conversations, 375 unique participants, 56 columns

**Key Columns**:

**Identifiers**:
- `conversation_id`: Unique conversation ID
- `participant_id`: Recovered participant ID (370+ unique)
- `csn`: Computer station number (CSN1-CSN22)
- `id_confidence`: High (64.5%), Medium (5.3%), Low (26.7%)

**Evaluation Metrics** (1-5 Likert scale):
- `cognitive_engagement` (μ=2.64, σ=0.67): Depth of thinking
- `query_sophistication` (μ=2.36, σ=0.66): Question quality
- `self_directedness` (μ=2.70, σ=0.71): AI dependency
- `iterative_refinement` (μ=1.86, σ=0.70): **⚠️ LOW - Follow-up behavior**
- `learning_orientation` (μ=2.48, σ=0.74): Declarative vs procedural
- `agency_ownership` (μ=2.70, σ=0.71): Learning ownership
- `linguistic_production` (μ=2.20, σ=0.96): Esperanto usage quality
- `memory_retention` (μ=2.09, σ=0.65): **⚠️ LOW - Knowledge building**
- `metacognitive_awareness` (μ=1.98, σ=0.50): **⚠️ LOW - Self-reflection**
- `overall_learning_quality` (μ=2.48, σ=0.67): Holistic assessment

**Question Type Distribution** (percentages):
- `qt_translation`, `qt_grammar`, `qt_usage`, `qt_clarification`, `qt_application`, `qt_meta`

**Additional Metrics**:
- `esperanto_usage_pct` (μ=38.1%): Esperanto in messages
- `mean_query_length` (μ=6.4 words): Query complexity
- `n_user_msgs`, `duration_min`: Session statistics

---

## Key Findings

### 1. Cognitive Debt Indicators (MIT 2025 Framework)

**Evidence of cognitive offloading patterns**:

| Indicator | Score | Interpretation |
|-----------|-------|----------------|
| Iterative Refinement | 1.86/5 | **LOW** - Learners accept first answers without deeper exploration |
| Memory Retention | 2.09/5 | **LOW** - Minimal knowledge building, suggests reduced cognitive encoding |
| Metacognitive Awareness | 1.98/5 | **LOW** - Passive learning, minimal self-reflection |

**Implication**: Consistent with MIT's findings on "cognitive debt" - outsourcing mental effort to AI reduces brain engagement and learning effectiveness.

### 2. Learning Orientation: Declarative Bias

**Mean Score**: 2.48/5 (slightly toward declarative)

**Distribution**:
- **Purely Declarative** (facts/translations): ~18%
- **Mostly Declarative**: ~32%
- **Balanced**: ~28%
- **Mostly Procedural** (processes/rules): ~18%
- **Purely Procedural**: ~4%

**Implication**: Learners focus more on "what" (translations) than "how" (grammar/usage rules), potentially limiting deeper understanding.

### 3. Self-Directedness: Moderate Balance

**Mean Score**: 2.70/5 (balanced AI use)

**Distribution**:
- Complete Dependency: ~8%
- Heavy Reliance: ~24%
- **Balanced**: ~48% ← Most common
- Highly Self-Directed: ~18%
- Autonomous: ~2%

**Implication**: Most learners maintain moderate autonomy, suggesting balanced (not over-dependent) AI tool use.

### 4. Question Type Patterns

**Average Distribution**:
- **Translation**: ~40-45% (highest)
- **Grammar**: ~25-30%
- **Usage**: ~15-20%
- **Clarification**: ~5-10%
- **Application**: <5%
- **Meta** (learning process): <2%

**Implication**: Low application and meta-questions indicate minimal transfer learning and reflection on learning strategies.

### 5. Esperanto Production

**Mean Usage**: 38.1% (only ~1/3 of messages contain Esperanto)

**Correlation with Linguistic Production Score**: r=0.52 (p<0.001)

**Implication**: Heavy reliance on English-based interaction limits productive language practice.

### 6. ID Confidence Impact

**No significant differences** in learning metrics across ID confidence levels:
- High confidence (n=242): Similar scores to low confidence
- Statistical tests: p>0.05 for most metrics

**Implication**: ID recovery method does not bias evaluation results.

---

## Statistical Highlights

### Strongest Correlations (r > 0.6)

1. **Cognitive Engagement ↔ Overall Quality**: r=0.72
2. **Agency/Ownership ↔ Self-Directedness**: r=0.68
3. **Query Sophistication ↔ Cognitive Engagement**: r=0.65

### Cluster Analysis (K-means, k=4)

**Cluster Profiles** (via PCA + K-means):

1. **Low Engagers** (n~100): Low scores across all metrics
2. **Moderate Balanced** (n~150): Average scores, balanced approach
3. **High Agency** (n~80): High self-directedness and ownership
4. **Production Focused** (n~60): Higher linguistic production, lower translation focus

---

## Methodology

### Data Collection
- **Participants**: 375 unique (370+ recovered IDs)
- **Conversations**: 397 total ChatGPT sessions
- **Platform**: ChatGPT web interface
- **Context**: Esperanto language learning
- **Timeframe**: Multiple sessions (Nov-Dec 2024)

### Evaluation Framework
- **Model**: GPT-4 Turbo (gpt-4-turbo-preview)
- **Temperature**: 0.3 (for consistency)
- **Processing**: Concurrent (10 workers)
- **Completeness**: 98% (389/397 successful)
- **Grounding**: MIT (2025), Behavioral Economics, Language Learning Research

### Metrics Development
All 10 evaluation metrics are grounded in peer-reviewed research:
- **MIT Media Lab (2025)**: Cognitive debt, memory formation, neural connectivity
- **Behavioral Economics**: Cognitive offloading, decision-making with AI
- **Language Learning**: Self-directed learning, metacognition, agency
- **Educational Psychology**: Iterative refinement, ownership

See `EVALUATION_FRAMEWORK.md` for complete methodology.

---

## Usage Examples

### 1. Basic Statistics
```python
import pandas as pd

df = pd.read_csv('conversations_with_evaluations_final.csv')

# Cognitive debt indicators
print("Cognitive Debt Indicators:")
for metric in ['iterative_refinement', 'memory_retention', 'metacognitive_awareness']:
    print(f"{metric}: {df[metric].mean():.2f} ± {df[metric].std():.2f}")
```

### 2. Filter by ID Confidence
```python
# High confidence only
high_conf = df[df['id_confidence'] == 'high']
print(f"High confidence: n={len(high_conf)}, cognitive_engagement={high_conf['cognitive_engagement'].mean():.2f}")
```

### 3. Learning Orientation Analysis
```python
# Group by orientation
declarative = df[df['learning_orientation'] < 2.5]
procedural = df[df['learning_orientation'] > 3.5]

print(f"Declarative learners: n={len(declarative)}, quality={declarative['overall_learning_quality'].mean():.2f}")
print(f"Procedural learners: n={len(procedural)}, quality={procedural['overall_learning_quality'].mean():.2f}")
```

### 4. Correlation Analysis
```python
# Examine relationships
corr = df[['self_directedness', 'memory_retention', 'overall_learning_quality']].corr()
print(corr)
```

### 5. Participant-Level Aggregation
```python
participants = pd.read_csv('participant_level_evaluations.csv')

# Top performers
top_10 = participants.nlargest(10, 'avg_cognitive_engagement')
print(top_10[['participant_id', 'avg_cognitive_engagement', 'avg_esperanto_usage_pct']])
```

---

## Figures Description

### Figure 1: Metrics Distribution
Violin plots showing distribution of all 10 evaluation metrics across conversations. Highlights the generally below-moderate scores with significant variance.

### Figure 2: Correlation Matrix
Heatmap showing Pearson correlations between all metrics. Key finding: Strong correlation between cognitive engagement and learning quality (r=0.72).

### Figure 3: Cognitive Debt Indicators
Focused analysis of three MIT study indicators (iterative refinement, memory retention, metacognition). All show low scores with concerning patterns.

### Figure 4: Learning Orientation
Distribution and impact of declarative vs procedural learning. Shows declarative bias and its relationship with learning quality.

### Figure 5: Question Types
Analysis of question categories. Translation requests dominate, while application and meta-questions are rare.

### Figure 6: ID Confidence Impact
Comparison of metrics across high/medium/low ID confidence levels. Shows minimal bias from ID recovery method.

### Figure 7: PCA & Clustering
Dimensionality reduction and K-means clustering revealing four distinct learner profiles.

### Figure 8: Esperanto Usage
Analysis of Esperanto production patterns and correlation with linguistic production scores.

### Figure 9: Self-Directedness
Comprehensive analysis of AI dependency patterns and impact on learning outcomes.

### Figure 10: Summary Dashboard
Complete overview combining all key findings in a single figure suitable for presentations.

---

## Citation

If using this dataset or analysis framework:

```bibtex
@misc{esperanto_chatgpt_2025,
  title={Esperanto Learning with ChatGPT: Behavioral Economics Analysis},
  author={[Your Name]},
  year={2025},
  note={Based on MIT Media Lab (2025) cognitive debt research and behavioral economics principles}
}
```

**Literature References**:
- MIT Media Lab (2025). "Your Brain on ChatGPT: Accumulation of Cognitive Debt when Using an AI Assistant for Essay Writing Task." arXiv:2506.08872
- Behavioral Economics and Cognitive Offloading Research
- Language Learning and Metacognitive Awareness Literature

---

## Future Research Directions

1. **Longitudinal Analysis**: Track learning progression over multiple sessions
2. **Outcome Correlation**: Link evaluation metrics to quiz performance and retention tests
3. **Intervention Design**: Test strategies to reduce cognitive debt (e.g., enforced reflection prompts)
4. **Comparative Studies**: Compare ChatGPT vs traditional learning methods
5. **Generalization**: Test framework on other languages and learning domains

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy openai
```

**For evaluation** (requires OpenAI API key):
```bash
export OPENAI_API_KEY="your-key-here"
python3 evaluate_conversations_fast.py
```

---

## License

[Specify your license]

## Contact

[Your contact information]

---

**Last Updated**: November 15, 2025
**Dataset Version**: Final (375 participants, 397 conversations)
**Evaluation Completeness**: 98% (389/397)
