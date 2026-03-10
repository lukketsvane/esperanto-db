# Esperanto Learning with ChatGPT: Behavioral Economics Study

## Dataset

397 ChatGPT conversations from 375 participants learning Esperanto, evaluated using validated frameworks from educational psychology and cognitive science literature. Participants were assigned to three treatment conditions: **AI-assisted**, **AI-guided**, and **control**. The final sample size linking complete survey responses, evaluated AI metrics, and programmatic copy-paste scores is **345 participants**.

## Key Findings (Explicit Evaluation, N=397)

### The "Black Box" Effect: Complement vs Substitute Learning

Our central finding explores the relationship between copy-paste behavior and AI usage strategies. We utilize an automated, programmatic copy-paste detection score (`copypaste_score_auto`), identifying the presence of predefined quiz materials in the prompt:

| Metric | Mean | Std | Interpretation |
|--------|------|-----|----------------|
| copypaste_score_auto | 1.16 | 0.38 | Low overall verbatim copying (mostly Original prompts) |
| substitute_learning | 3.21 | 1.25 | Above-midpoint AI reliance |
| complement_learning | 2.46 | 1.26 | Below-midpoint AI augmentation |

**Key correlations:**
- Copy-Paste vs Complement Learning: **r = 0.27** (p<0.001)
- Copy-Paste vs Substitute Learning: **r = -0.14** (p<0.01)
- Substitute vs Complement Learning: **r = -0.57** (p<0.001) — inversely coupled

The automated copy-paste score distribution is tightly clustered towards original queries. Substitute learning continues to strongly inversely correlate with complement learning, reinforcing the dichotomy between students who rely on the AI for answers and those who engage with the AI to augment learning. 

*(See `figures/fig1_substitute_vs_complement.png` and `figures/fig2_copypaste_impact_bar.png`)*

### Real Conversation Examples

To illustrate the complement-substitute distinction, we extracted anonymized excerpts from actual participant conversations. Full examples with dialogue are in `figures/conversation_examples.md`.

**Example A — Substitute Learner**:
This participant pastes quiz instructions verbatim, treating AI purely as a translation engine:
```
USER: Write the negative form of these sentences in Esperanto = Tom dormas.
USER: La hundo flugas
USER: La birdo estas granda
```

**Example B — Complement Learner**:
This participant types original queries, proactively requesting lessons on grammar and structure:
```
USER: esperanto quick lesson
USER: sentence construction
USER: pronouns in esperanto
```

### Cognitive & Learning Metrics

| Metric | Mean | Std |
|--------|------|-----|
| cognitive_engagement | 2.44 | 0.99 |
| metacognitive_awareness | 1.29 | 0.51 |
| linguistic_production | 2.62 | 0.83 |
| self_directedness | 2.49 | 0.99 |
| iterative_refinement | 2.20 | 1.03 |
| memory_retention | 2.65 | 1.23 |
| agency_ownership | 2.27 | 0.98 |
| query_sophistication | 1.63 | 0.72 |
| overall_learning_quality | 2.11 | 0.81 |

**Strong correlations (r > 0.6):**
- Cognitive Engagement <-> Overall Learning Quality: **r = 0.96**
- Agency/Ownership <-> Self-Directedness: **r = 0.84**

*(See `figures/fig4_metric_distributions.png`, `figures/fig5_correlation_matrix.png`)*

### Cognitive Debt Indicators (MIT 2025 Framework)

| Indicator | Score | Interpretation |
|-----------|-------|----------------|
| Iterative Refinement | 2.20/5 | Low follow-up depth |
| Memory Retention | 2.65/5 | Moderate knowledge retention |
| Metacognitive Awareness | 1.29/5 | Very low self-reflection |

The low metacognitive awareness (1.29) is particularly concerning — most learners show minimal monitoring of their own understanding.

*(See `figures/fig7_cognitive_debt.png`)*

### Learner Profiles (K-means Clustering)

PCA + K-means identifies four distinct learner types:

| Cluster | Cognitive Engagement | Metacognitive | Linguistic | Self-Direction | Iterative | Memory |
|---------|---------------------|---------------|------------|----------------|-----------|--------|
| Low Engagers | 1.28 | 1.03 | 1.47 | 1.15 | 1.04 | 1.03 |
| Moderate Balanced | 2.05 | 1.05 | 2.81 | 2.41 | 1.79 | 2.38 |
| Active Learners | 2.98 | 1.32 | 2.97 | 3.00 | 2.76 | 3.53 |
| High Agency | 4.08 | 2.20 | 3.35 | 3.79 | 3.87 | 4.16 |

*(See `figures/fig6_learner_clusters.png`)*

### Heterogeneous Effects: Gender, GPA, and Treatment

Subgroup analysis reveals differentiated impacts across demographics:
- **Gender x Treatment**: Kernel density plots show different distributions of exam scores and learning quality by gender within treatment groups.
- **GPA x Treatment**: High-GPA and low-GPA students show different patterns of complement learning across AI-assisted and AI-guided conditions.

*(See `figures/fig8_subgroup_kde.png`, `figures/fig10_heterogeneous_effects.png`)*

### Messages, Exam Outcomes, and Copy-Paste

- Scatter plots show the relationship between prompting volume (`sum_msg`) and practice questions attempted, as well as total exam score (`testscore`). *(See `figures/fig3_messages_vs_learning.png`)*
- Individual copy-paste levels have been mapped against actual post-survey exam outcomes, revealing subtle distributions of higher vs. lower performing participants based on structural verbatim copying. *(See `figures/fig11_copypaste_vs_testscore.png`)*

## Methodology & LLM-as-a-Judge Design

### LLM Evaluation
**Model**: `gemini-3.1-flash-lite-preview` (March 2026)
**Context**: Full-text extraction of both **User** and **Assistant** messages to capture instructional flow.

**Prompting Architecture** (`scripts/evaluate_conversations.py`):
Two evaluation modes, following LLM-as-a-Judge best practices (MT-Bench / Zheng et al., 2023):
1. **Explicit Mode** (`prompts/system_explicit.txt`): Strict, literature-grounded definitions for all 17 metrics.
2. **Implicit Mode** (`prompts/system_implicit.txt`): Zero-shot inference.

Explicit rubrics yield significantly sharper and consistent evaluation. Thus, all final data utilizes explicit evaluation records.

### Copy-Paste Detection:
Programmatic (`scripts/detect_copypaste.py`): String matching against known practice questions to assign a structured `copypaste_score_auto`. 

## Evaluation Metrics (1-5 scale)

Grounded in validated frameworks:
- Cognitive Load: Klepsch et al. 2017
- Metacognition: Schraw & Dennison 1994 (MAI)
- Self-Regulation: Pintrich et al. 1991 (MSLQ)
- Linguistic Production: ACTFL 2024 (FACT)
- Engagement: Kahu et al. 2018 (HESES)
- Critical Thinking: Liu et al. 2014
- Cognitive Debt: MIT Media Lab 2025
- Copy-Paste & AI Usage: Complement/Substitute framework

**17 raw dimensions** aggregated into 9 composite metrics + 3 AI-usage metrics.

## Figures Directory Checklist

| File | Description |
|------|-------------|
| `fig1_substitute_vs_complement.png` | Scatter: Substitute vs Complement Learning colored by Copy-Paste score |
| `fig2_copypaste_impact_bar.png` | Bar chart: Mean Substitute/Complement by Copy-Paste severity |
| `fig3_messages_vs_learning.png` | Scatter: Messages vs Practice Questions and Exam Score |
| `fig4_metric_distributions.png` | 3x3 histograms of composite metrics |
| `fig5_correlation_matrix.png` | Heatmap of all metric correlations |
| `fig6_learner_clusters.png` | PCA + K-means learner profiles |
| `fig7_cognitive_debt.png` | Boxplot of cognitive debt indicators |
| `fig8_subgroup_kde.png` | KDE plots by gender, GPA, treatment |
| `fig9_relationships.png` | 4-panel scatter with regression lines |
| `fig10_heterogeneous_effects.png` | KDE: Gender x Treatment, GPA x Treatment |
| `fig11_copypaste_vs_testscore.png` | Distribution, trendline, and performance share vs. exam score |
| `summary_statistics.csv` | Descriptive statistics for all metrics |

## Data Files Checklist

| File | Description |
|------|-------------|
| `data/conversations_final.csv` | Complete dataset |
| `data/01_full_sample_with_prompts.csv` | Cleaned and restored participant metadata |
| `conversations_evaluated_explicit.csv` | Explicit LLM evaluation |
| `conversations_evaluated_implicit.csv` | Implicit LLM evaluation |
| `copypaste_auto_detection.csv` | Programmatic copy-paste scores |

*(All transient fixes and old PDF outputs have been cleared. Naming conventions are unified.)*