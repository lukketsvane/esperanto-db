# Esperanto Learning with ChatGPT: Behavioral Economics Study

## Dataset

397 ChatGPT conversations from 375 participants learning Esperanto, evaluated using validated frameworks from educational psychology and cognitive science literature. Participants were assigned to three treatment conditions: **AI-assisted**, **AI-guided**, and **control**.

## Key Findings (Explicit Evaluation, N=394)

### The "Black Box" Effect: Complement vs Substitute Learning

Our central finding is a **strong inverse relationship** between copy-paste behavior and complementary AI usage. Participants who copy-pasted quiz instructions verbatim into ChatGPT treated the AI as a "black box" translation engine, bypassing all cognitive effort:

| Metric | Mean | Std | Interpretation |
|--------|------|-----|----------------|
| copypaste_dummy | 2.23 | 1.56 | Moderate overall copy-pasting |
| substitute_learning | 3.20 | 1.25 | Above-midpoint AI reliance |
| complement_learning | 2.47 | 1.26 | Below-midpoint AI augmentation |

**Key correlations:**
- Copy-Paste vs Complement Learning: **r = -0.54** (p<0.001) — strong inverse
- Copy-Paste vs Substitute Learning: **r = 0.22** (p<0.001) — positive
- Substitute vs Complement Learning: **r = -0.57** (p<0.001) — inversely coupled

At the extremes: participants with **copypaste=5** show average Substitute Learning of ~4.1 and Complement Learning of ~1.0. They never ask "why" — only "give me the answer."

*(See `figures/fig1_substitute_vs_complement.png` and `figures/fig2_copypaste_impact_bar.png`)*

### Real Conversation Examples

To illustrate the complement-substitute distinction, we extracted anonymized excerpts from actual participant conversations. Full examples with dialogue are in `figures/conversation_examples.md`.

**Example A — Substitute Learner** (copypaste=4, substitute=4, complement=2):
This participant pastes quiz instructions verbatim, treating AI purely as a translation engine:

```
USER: Write the negative form of these sentences in Esperanto = Tom dormas.
USER: La hundo flugas
USER: La birdo estas granda
USER: Transform these singular nouns into plural. If the noun has an article,
      keep the article in the plural form La virino
```

**Example B — Complement Learner** (copypaste=1, substitute=2, complement=4):
This participant types original queries, proactively requesting lessons on grammar and structure:

```
USER: esperanto quick lesson
USER: sentence construction
USER: pronouns in esperanto
USER: common vocab used in esperanto
USER: how are nouns structured in esperanto
```

**Example C — Strategic Copier** (copypaste=4, substitute=3, complement=3):
This participant copies some practice questions but also engages with the material — a hybrid pattern:

```
USER: Identify the correct Esperanto translation: The bird flies. The boy learns Esperanto.
USER: Fill in the blank with the most appropriate verb. Mi _____ Anna.
USER: Write the negative form of these sentences in Esperanto: Tom dormas. Anna lernas.
USER: Transform these singular nouns into plural [...]
```

### Cognitive & Learning Metrics

| Metric | Mean | Std |
|--------|------|-----|
| cognitive_engagement | 2.44 | 1.00 |
| metacognitive_awareness | 1.30 | 0.51 |
| linguistic_production | 2.63 | 0.83 |
| self_directedness | 2.49 | 0.99 |
| iterative_refinement | 2.21 | 1.03 |
| memory_retention | 2.66 | 1.23 |
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
| Iterative Refinement | 2.21/5 | Low follow-up depth |
| Memory Retention | 2.66/5 | Moderate knowledge retention |
| Metacognitive Awareness | 1.30/5 | Very low self-reflection |

The low metacognitive awareness (1.30) is particularly concerning — most learners show minimal monitoring of their own understanding.

*(See `figures/fig7_cognitive_debt.png`)*

### Learner Profiles (K-means Clustering)

PCA + K-means identifies four distinct learner types:

| Cluster | Cognitive Engagement | Metacognitive | Linguistic | Self-Direction | Iterative | Memory |
|---------|---------------------|---------------|------------|----------------|-----------|--------|
| Low Engagers | 1.28 | 1.03 | 1.48 | 1.15 | 1.04 | 1.03 |
| Moderate Balanced | 2.05 | 1.05 | 2.81 | 2.40 | 1.79 | 2.37 |
| Active Learners | 2.98 | 1.32 | 2.97 | 3.00 | 2.76 | 3.53 |
| High Agency | 4.08 | 2.20 | 3.35 | 3.79 | 3.87 | 4.16 |

*(See `figures/fig6_learner_clusters.png`)*

### Heterogeneous Effects: Gender, GPA, and Treatment

Subgroup analysis reveals differentiated impacts across demographics:

- **Gender x Treatment**: Kernel density plots show different distributions of exam scores and learning quality by gender within treatment groups.
- **GPA x Treatment**: High-GPA and low-GPA students show different patterns of complement learning across AI-assisted and AI-guided conditions.

These findings connect to the broader literature on heterogeneous effects of AI on learning, where high-ability women tend to benefit most while male and low-GPA students may experience negative effects (cf. the "boy crisis" in education).

*(See `figures/fig8_subgroup_kde.png`, `figures/fig10_heterogeneous_effects.png`)*

### Messages and Learning Outcomes

Scatter plots with regression lines show the relationship between prompting volume (sum_msg) and:
- **Practice questions attempted** (nb_practice_questions)
- **Exam score** (testscore)

*(See `figures/fig3_messages_vs_learning.png`)*

## Methodology & LLM-as-a-Judge Design

### LLM Evaluation
**Model**: `gemini-3.1-flash-lite-preview` (March 2026)
**Context**: Full-text extraction of both **User** and **Assistant** messages to capture instructional flow.

**Prompting Architecture** (`scripts/evaluate_conversations.py`):
Two evaluation modes, following LLM-as-a-Judge best practices (MT-Bench / Zheng et al., 2023):
1. **Explicit Mode** (`prompts/system_explicit.txt`): Strict, literature-grounded definitions for all 17 metrics.
2. **Implicit Mode** (`prompts/system_implicit.txt`): Zero-shot inference.

**Explicit vs Implicit Performance:**
- **Explicit**: Copy-Paste Mean=2.23, Substitute Learning Mean=3.20. Strong inverse correlation with Complement Learning (r=-0.54).
- **Implicit**: Copy-Paste Mean=1.31, Substitute Learning Mean=2.13. Underreports low-effort behaviors.

*Conclusion*: Explicit rubrics yield significantly sharper detection of low-effort learning patterns.

### Copy-Paste Detection: Dual Approach
1. **Programmatic** (`scripts/detect_copypaste.py`): String matching against known practice questions.
2. **LLM Judgment** (`copypaste_dummy`): Structural/intent-based copy detection.

### Conversation Example Extraction
Real participant excerpts extracted via `scripts/extract_examples.py` — anonymized, with IDs redacted.

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

## Figures

| Figure | Description |
|--------|-------------|
| `fig1_substitute_vs_complement` | Scatter: Substitute vs Complement Learning colored by Copy-Paste score |
| `fig2_copypaste_impact_bar` | Bar chart: Mean Substitute/Complement by Copy-Paste severity |
| `fig3_messages_vs_learning` | Scatter: Messages vs Practice Questions and Exam Score |
| `fig4_metric_distributions` | 3x3 histograms of composite metrics |
| `fig5_correlation_matrix` | Heatmap of all metric correlations |
| `fig6_learner_clusters` | PCA + K-means learner profiles |
| `fig7_cognitive_debt` | Boxplot of cognitive debt indicators |
| `fig8_subgroup_kde` | KDE plots by gender, GPA, treatment |
| `fig9_relationships` | 4-panel scatter with regression lines |
| `fig10_heterogeneous_effects` | KDE: Gender x Treatment, GPA x Treatment |

## Data Files

| File | Description |
|------|-------------|
| `data/conversations_final.csv` | Complete dataset (397 conversations, 44 columns) |
| `data/01_full_sample_with_prompts.csv` | Participant metadata (604 rows, 592 columns) |
| `conversations_evaluated_explicit.csv` | Explicit LLM evaluation (398 rows) |
| `conversations_evaluated_implicit.csv` | Implicit LLM evaluation (398 rows) |
| `copypaste_auto_detection.csv` | Programmatic copy-paste scores |
| `figures/conversation_examples.md` | Anonymized conversation excerpts |
| `figures/summary_statistics.csv` | Descriptive statistics for all metrics |

## Repository Structure

```
data/
  conversations_final.csv            Complete dataset
  01_full_sample_with_prompts.csv    Participant metadata
figures/                             Publication figures (PNG+PDF)
  conversation_examples.md           Anonymized conversation excerpts
scripts/
  evaluate_conversations.py          LLM evaluation pipeline
  analyze_and_visualize.py           Figure generation (10 figures, explicit eval only)
  extract_examples.py                Conversation example extraction
  detect_copypaste.py                Programmatic copy-paste detection
  compare_prompts.py                 Explicit vs implicit comparison
  merge_data.py                      Data merging utility
raw_data/csn_exports/                Original CSN1-CSN22 exports
prompts/                             LLM evaluation prompts
analysis/
  LITERATURE_REVIEW_COMPREHENSIVE.md Framework sources (20+ papers)
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

See `analysis/LITERATURE_REVIEW_COMPREHENSIVE.md` for complete bibliography.
