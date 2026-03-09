# Activity Log: Esperanto-DB Evaluation Update (March 9, 2026)

## Overview
This log tracks the updates made to the Esperanto learning conversation evaluation pipeline, focusing on model upgrades, more granular metrics, and automated behavior detection.

## 1. Data Context & Branching
- **Branch**: Switched to `review-2026-march`.
- **Primary Data Source**: `data/01_full_sample_with_prompts.csv` (contains original practice questions and participant metadata).
- **Scope**: 397 unique ChatGPT conversations across 375 participants.

## 2. Evaluation Script Enhancements & LLM-as-a-Judge Architecture
- **Full Context Extraction**: Updated `extract_all_messages()` to capture both **USER** and **ASSISTANT** messages. This provides the evaluator with the full conversational flow, which is critical for understanding "complementary" vs "substitutive" learning.
- **Model Upgrade**: Transitioned from `gpt-3.5-turbo` to `gemini-3.1-flash-lite-preview` for high-throughput, high-reasoning evaluation.
- **New Metrics (1-5 Scale)**:
    - `copypaste_dummy`: Frequency of verbatim or near-verbatim copying of task instructions.
    - `substitute_learning`: Using AI to replace cognitive effort (e.g., "translate this for me").
    - `complement_learning`: Using AI to augment effort (e.g., "explain why this grammar is used").
- **Unified Evaluation Script (`scripts/evaluate_conversations.py`)**: Consolidated the evaluation logic into a single parameterized script that can run different prompt types.
- **Externalized Prompts (`prompts/`)**: Following LLM-as-a-Judge best practices (Zheng et al., 2023), we isolated system instructions into external files to test inter-rater reliability based on prompt specificity:
    - `system_implicit.txt`: Tests the LLM's zero-shot, latent understanding of the variables.
    - `system_explicit.txt`: Tests the LLM using strict, theoretically grounded definitions for every variable.

### Dual Prompt Findings (Implicit vs. Explicit)
Running both evaluation modes yielded striking differences, validating the use of explicit rubrics in LLM-as-a-judge pipelines:
- **Explicit Prompting**: Mean Copy-Paste: 2.23 | Mean Substitute Learning: 3.20. (Strong inverse correlation with Complement Learning: **r = -0.54**).
- **Implicit Prompting**: Mean Copy-Paste: 1.31 | Mean Substitute Learning: 2.13. (Failed to accurately identify substitutive behaviors, clustering scores too tightly).

By explicitly defining variables like `copypaste_dummy` and `substitute_learning`, the LLM was able to accurately penalize low-effort interactions, whereas zero-shot (implicit) inference drastically underreported these behaviors.

## 3. Programmatic Behavior Detection
- **Auto-Detection Script**: Created `scripts/detect_copypaste.py`.
    - **Methodology**: String matching against a library of known practice questions (e.g., "Tom ne dormas", "La hundo ne flugas").
    - **Output**: `copypaste_score_auto` (1-5 scale based on the ratio of copied questions).
- **Comparison & Validation**: Created `scripts/compare_copypaste.py` to align LLM "judgment" with programmatic "fact."
    - **Insight**: The LLM (GPT/Gemini) is superior at catching "structural" copying, while the script provides a hard baseline for "verbatim" copying.

## 4. Key Findings: Copy-Pasting vs. Learning Quality

### Dual Methodology Results
- **Auto-Detection (`copypaste_score_auto`)**: Captured verbatim copy-pasting of specific task questions. Found 59 instances of copying.
- **LLM Assessment (`copypaste_dummy`)**: Captured broader "structural" copying. Found 88 conversations with high copy-pasting behavior (Scores 4-5).
- **The Gap**: Verbatim auto-matching often missed participants who paraphrased or slightly modified questions, while the LLM was able to recognize the intent of copying.

### Correlation with Learning Approach
The analysis reveals a strong inverse relationship between copying behavior and cognitive engagement:
1. **The "Low Effort" Profile**: High copy-pasting (`copypaste_dummy` >= 4) is strongly correlated with **Substitutive Learning** (r = 0.22, but significantly lower Complement scores). 
2. **Substitutive Bias**: In high-copy conversations (Score 5), the **Complement Learning score drops to ~1.0**. These participants use AI exclusively as a "black box" to provide answers without seeking explanation or deeper understanding.
3. **The Inverse Correlation**: A significant negative correlation (**r = -0.54**) exists between `copypaste_dummy` and `complement_learning`. This confirms that copy-pasting is a behavioral proxy for low cognitive engagement.

### Participant Segmentation
- **Type A: Active Learners**: Low copy-paste scores, high `complement_learning`. These students type their own queries and ask "why."
- **Type B: Translators/Copy-Pasters**: High copy-paste scores, high `substitute_learning`. These students treat AI as a translation tool to bypass the lesson.
- **Type C: Strategic Copiers**: High `copypaste_score_auto` but moderate `complement_learning`. Some students copy questions verbatim but then follow up with "Can you explain this?" (Strategic use of AI as a tutor).

## 5. Final Outputs
- **`conversations_evaluated_explicit.csv`**: Raw LLM evaluation results using the new metrics.
- **`copypaste_auto_detection.csv`**: Programmatic match scores.
- **`conversations_evaluated_final.csv`**: Consolidated dataset merging all psychological, linguistic, and behavioral metrics (including the new auto-detection scores).

## Key Statistics (N=397)
- **Overall Learning Quality**: 2.11 ± 0.81
- **Cognitive Engagement**: 2.44 ± 1.00
- **Linguistic Production**: 2.63 ± 0.83
- **Self-Directedness**: 2.49 ± 0.99
- **Memory Retention**: 2.66 ± 1.23

## 6. Improved Visualization Pipeline (March 9, 2026 — Update 2)

### Visualization Overhaul
Rewrote `scripts/analyze_and_visualize.py` to generate **10 publication-quality figure sets** (PNG+PDF), using **only explicit evaluation data** (`conversations_evaluated_explicit.csv`). The script also merges with participant metadata (`data/01_full_sample_with_prompts.csv`) to access `sum_msg`, `testscore`, `nb_practice_questions`, `gender`, `gpa`, `highgpa`, and `treatment`.

### New Figures Generated

| Figure | Type | Description |
|--------|------|-------------|
| `fig1_substitute_vs_complement` | Scatter | Substitute vs Complement Learning, colored by Copy-Paste score |
| `fig2_copypaste_impact_bar` | Bar | Mean Substitute/Complement by Copy-Paste severity (1-5), with SE bars |
| `fig3_messages_vs_learning` | Scatter (2-panel) | Messages vs Practice Questions; Messages vs Exam Score |
| `fig4_metric_distributions` | Histogram (3x3) | All 9 composite metric distributions with mean lines |
| `fig5_correlation_matrix` | Heatmap | All 12 metrics (including AI-usage) with triangular mask |
| `fig6_learner_clusters` | PCA Scatter | 4 learner profiles via K-means with labeled centroids |
| `fig7_cognitive_debt` | Boxplot | Iterative Refinement, Memory Retention, Metacognitive Awareness |
| `fig8_subgroup_kde` | KDE (4-panel) | Learning Quality by Gender/GPA; Complement/Substitute by Treatment |
| `fig9_relationships` | Scatter (4-panel) | Key metric pairs with regression lines and r/p annotations |
| `fig10_heterogeneous_effects` | KDE (2-panel) | Exam Score by Gender x Treatment; Complement by GPA x Treatment |

### Conversation Example Extraction
Created `scripts/extract_examples.py` to extract anonymized conversation excerpts from raw ChatGPT JSON exports. Three exemplar conversations identified:
- **Substitute Learner**: Pastes quiz instructions verbatim (cp=4, sub=4, comp=2)
- **Complement Learner**: Types original queries about grammar and structure (cp=1, sub=2, comp=4)
- **Strategic Copier**: Copies some questions but also requests explanations (cp=4, sub=3, comp=3)

Output: `figures/conversation_examples.md`

### New Findings from Merged Data Analysis
- **Merged dataset**: 394 evaluated conversations linked to participant metadata (118 with full demographic data)
- **Learner cluster profiles** (K-means, 4 clusters):
  - Low Engagers: All metrics near 1.0
  - Moderate Balanced: Mid-range across board
  - Active Learners: cognitive_engagement=2.98, memory_retention=3.53
  - High Agency: cognitive_engagement=4.08, iterative_refinement=3.87
- **Subgroup KDE analysis**: Gender, GPA, and treatment condition show differentiated distributions of learning quality and AI usage patterns
- **Heterogeneous effects**: Gender x Treatment and GPA x Treatment interactions visualized

### Updated Key Correlations (Explicit Eval, N=394)
- copypaste_dummy <-> complement_learning: **r = -0.540** (p<0.001)
- copypaste_dummy <-> substitute_learning: **r = 0.217** (p<0.001)
- cognitive_engagement <-> overall_learning_quality: **r = 0.960** (p<0.001)
- agency_ownership <-> self_directedness: **r = 0.841** (p<0.001)
- substitute_learning <-> complement_learning: **r = -0.567** (p<0.001)

### Full Metric Summary (Explicit Eval, N=394)

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| cognitive_engagement | 2.44 | 1.00 | 2.0 |
| metacognitive_awareness | 1.30 | 0.51 | 1.0 |
| linguistic_production | 2.63 | 0.83 | 3.0 |
| self_directedness | 2.49 | 0.99 | 3.0 |
| iterative_refinement | 2.21 | 1.03 | 2.0 |
| memory_retention | 2.66 | 1.23 | 3.0 |
| agency_ownership | 2.27 | 0.98 | 2.0 |
| query_sophistication | 1.63 | 0.72 | 1.5 |
| overall_learning_quality | 2.11 | 0.81 | 2.0 |
| copypaste_dummy | 2.23 | 1.56 | 1.0 |
| substitute_learning | 3.20 | 1.25 | 4.0 |
| complement_learning | 2.47 | 1.26 | 2.0 |

---
*Log generated on 2026-03-09.*
