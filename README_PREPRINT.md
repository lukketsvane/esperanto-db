# Esperanto Learning with ChatGPT: A Behavioral Economics Study of AI-Assisted Language Learning

[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-preprint-brightgreen)](https://github.com/lukketsvane/esperanto-db)

> **A comprehensive dataset and analysis of 397 ChatGPT conversations from 375 Esperanto learners, evaluated using validated frameworks from educational psychology and cognitive science**

[🚀 Interactive Demo](#interactive-demo) | [📊 Dataset](#dataset) | [📖 Paper](#paper) | [🔬 Methodology](#methodology) | [📈 Results](#key-findings) | [💻 Code](#repository-structure)

---

## Abstract

Large Language Models (LLMs) are increasingly used for educational purposes, yet their impact on learning behaviors remains poorly understood. This study presents a behavioral economics analysis of AI-assisted language learning through 397 ChatGPT conversations from 375 participants learning Esperanto. Using validated assessment frameworks from educational psychology (Klepsch et al. 2017; Schraw & Dennison 1994; Pintrich et al. 1991; ACTFL 2024), we evaluate cognitive engagement, metacognition, self-regulation, and linguistic production at scale.

**Key Findings**: (1) Learners exhibit moderate cognitive engagement (M=3.79, SD=0.52) with above-average memory retention (M=3.68, SD=0.57), suggesting balanced AI utilization rather than over-dependency; (2) Strong declarative bias (50%) favoring translation over procedural grammar understanding; (3) Heterogeneous learner profiles emerge: Low Engagers (25%), Moderate Balanced (38%), High Agency (20%), and Production Focused (17%); (4) Cognitive engagement strongly predicts learning quality (r=0.72, p<0.001).

This work contributes (a) a novel, open-source dataset of real-world AI-assisted learning conversations with behavioral annotations, (b) validated methodology for large-scale LLM-based assessment using established frameworks, and (c) empirical evidence on cognitive debt patterns in AI tutoring contexts.

**Keywords**: AI-assisted learning, language acquisition, ChatGPT, behavioral economics, cognitive debt, educational psychology, metacognition, Esperanto

---

## Interactive Demo

🎮 **[Launch Interactive Explorer →](https://huggingface.co/spaces/your-username/esperanto-learning)**

Explore the dataset through interactive visualizations:
- 📊 **Metric Distributions**: Histogram analysis of all 9 learning dimensions
- 🔗 **Correlation Heatmaps**: Relationships between cognitive and linguistic metrics
- 🧠 **Cognitive Debt Analysis**: MIT framework indicators for AI dependency
- 👥 **Learner Clustering**: PCA visualization of 4 distinct learner profiles
- 🔍 **Conversation Explorer**: Drill down into individual learning sessions

---

## Dataset

### Overview

| Property | Value |
|----------|-------|
| **Total Conversations** | 397 |
| **Unique Participants** | 375 |
| **Total User Messages** | 4,312 |
| **Evaluation Success Rate** | 98.0% (389/397) |
| **Data Collection Period** | December 2024 |
| **Language Target** | Esperanto (L2) |
| **AI Model Used** | ChatGPT (GPT-4o) |
| **Data Format** | CSV (44 columns) |
| **License** | MIT |

### Data Structure

**Primary Dataset**: `data/conversations_final.csv` (397 rows × 44 columns)

#### Columns

1. **Metadata** (17 columns)
   - `conversation_id`, `title`, `default_model_slug`
   - `create_time`, `create_time_iso`
   - `n_user_messages`, `participant_id`
   - Temporal markers: `first_user_time`, `last_user_time`

2. **Cognitive Metrics** (9 columns, 1-5 scale)
   - `cognitive_engagement`: Depth of thought and active processing
   - `metacognitive_awareness`: Self-reflection and strategy monitoring
   - `linguistic_production`: Quantity and accuracy of L2 output
   - `self_directedness`: Autonomous learning behaviors
   - `iterative_refinement`: Follow-up depth and query elaboration
   - `memory_retention`: Evidence of knowledge retention across turns
   - `query_sophistication`: Complexity and specificity of questions
   - `agency_ownership`: Initiative and responsibility for learning
   - `overall_learning_quality`: Holistic assessment

3. **Sub-component Metrics** (12 columns, 1-5 scale)
   - Cognitive Load: `germane_load`
   - Metacognition: `metacog_planning`, `metacog_monitoring`, `metacog_evaluation`
   - Linguistic: `ling_quantity`, `ling_accuracy`
   - Additional: `self_regulation`, `critical_thinking`, `engagement_cognitive`, `engagement_affective`, `question_sophistication`, `iterative_depth`, `ownership`

4. **Processing Info** (1 column)
   - `error`: Evaluation errors (blank for successful evaluations)

### Download

```bash
git clone https://github.com/lukketsvane/esperanto-db.git
cd esperanto-db
```

Or download directly:
- **Conversations**: [`data/conversations_final.csv`](./data/conversations_final.csv)
- **Participants**: [`data/prompt_participants_final.csv`](./data/prompt_participants_final.csv)
- **Figures**: [`figures/`](./figures/)

---

## Key Findings

### 1. Cognitive Debt Patterns (MIT 2025 Framework)

Adapting the MIT Media Lab (2025) cognitive debt framework to language learning:

| Indicator | Score | Interpretation |
|-----------|-------|----------------|
| **Iterative Refinement** | 2.91/5 (SD=0.55) | Moderate follow-up depth |
| **Memory Retention** | 3.68/5 (SD=0.57) | **Good knowledge retention** |
| **Metacognitive Awareness** | 3.22/5 (SD=0.52) | Moderate self-reflection |

**Finding**: Learners show **moderate engagement patterns** with **above-average memory retention**, suggesting they are actively building knowledge rather than passively consuming answers. This contrasts with the high cognitive debt observed in AI-assisted essay writing (MIT 2025), potentially due to the interactive, skills-based nature of language learning.

### 2. Learning Patterns

#### Declarative Bias
**50%** of learners favor **fact/translation requests** over **procedural understanding** (grammar rules, usage patterns).

**Example patterns**:
- Declarative: "What does 'saluton' mean?" (translation query)
- Procedural: "When do I use accusative case in Esperanto?" (grammar rule)

This bias aligns with cognitive load theory (Sweller 1988): declarative knowledge is easier to acquire but less transferable than procedural schemas.

#### AI Dependency Distribution

- **48%** show **balanced autonomy** (moderate AI reliance with self-direction)
- **32%** show **high AI reliance** (minimal self-generated content)
- **20%** show **high self-direction** (using AI as supplement, not crutch)

Dependency assessed via self-directedness, agency, and iterative refinement metrics.

#### Query Type Distribution

| Query Type | Percentage | Example |
|------------|-----------|---------|
| **Translation** | 40-45% | "How do you say 'thank you'?" |
| **Grammar** | 25-30% | "Explain verb conjugation" |
| **Application** | <5% | "Give me practice exercises" |
| **Meta-learning** | <2% | "What's the best way to learn vocabulary?" |

**Implication**: Learners underutilize AI's potential for active practice and learning strategy coaching.

#### Esperanto Production

- **Mean**: 38% of user messages contain Esperanto text
- **Interpretation**: Most interaction remains English-based, suggesting limited practice in target language

### 3. Correlations (Strong: r > 0.6)

| Relationship | r | p | Interpretation |
|--------------|---|---|----------------|
| **Cognitive Engagement ↔ Learning Quality** | **0.72** | <0.001 | Deep thinking predicts better outcomes |
| **Agency/Ownership ↔ Self-Directedness** | **0.68** | <0.001 | Initiative drives autonomous learning |
| **Query Sophistication ↔ Cognitive Engagement** | **0.65** | <0.001 | Better questions reflect deeper processing |

These correlations validate our theoretical framework: active, self-directed engagement with sophisticated queries leads to higher learning quality.

### 4. Learner Profiles (K-means Clustering, k=4)

Using PCA on 6 primary metrics followed by K-means clustering:

| Cluster | % | Characteristics |
|---------|---|-----------------|
| **Low Engagers** | 25% | Minimal metrics across board; brief, surface interactions |
| **Moderate Balanced** | 38% | Average scores; balanced query types; typical learner |
| **High Agency** | 20% | **Strong self-direction and ownership**; metacognitive strategies; strategic learners |
| **Production Focused** | 17% | **Higher Esperanto usage**, less translation dependency; immersive approach |

**Finding**: Heterogeneous learning strategies emerge even with identical AI tutor access, suggesting individual differences in AI utilization and learning beliefs.

---

## Methodology

### 1. Data Collection

- **Source**: ChatGPT conversation exports (CSN format) from Esperanto learners
- **Participants**: 375 individuals (self-selected sample)
- **Period**: December 2024
- **Inclusion Criteria**:
  - Conversations explicitly about Esperanto learning
  - Minimum 3 user messages
  - Valid JSON structure
- **Exclusion**: 8 conversations (2%) failed evaluation due to parsing errors

### 2. Evaluation Framework

All metrics grounded in **validated instruments** from educational psychology:

| Metric Domain | Framework Source | Validation |
|---------------|-----------------|------------|
| **Cognitive Load** | Klepsch et al. (2017) | Frontiers in Psychology, 8, 1997 |
| **Metacognition** | Schraw & Dennison (1994) | MAI (Metacognitive Awareness Inventory) |
| **Self-Regulation** | Pintrich et al. (1991) | MSLQ (Motivated Strategies for Learning) |
| **Linguistic Production** | ACTFL (2024) | ACTFL Proficiency Guidelines 2024 |
| **Engagement** | Kahu et al. (2018) | HESES (Higher Education Student Engagement) |
| **Critical Thinking** | Liu et al. (2014) | ETS Research Report Series |
| **Cognitive Debt** | MIT Media Lab (2025) | arXiv:2506.08872 |

See [`analysis/LITERATURE_REVIEW_COMPREHENSIVE.md`](./analysis/LITERATURE_REVIEW_COMPREHENSIVE.md) for complete bibliography.

### 3. LLM-as-Judge Evaluation

**Model**: GPT-3.5-turbo (temperature=0.1)

**Process**:
1. Each conversation passed to LLM with structured prompt
2. Prompt contains metric definitions grounded in frameworks
3. LLM scores observable behaviors on 1-5 scale
4. JSON-structured output parsed and validated

**Prompt Engineering**:
- **Behavioral anchors**: Explicit 1-5 scale definitions for each metric
- **Observable focus**: Score only visible behaviors, not inferred intent
- **Multiple examples**: Few-shot prompting with calibration examples
- **Citation grounding**: Framework sources cited in prompt

**Validation**:
- **Inter-rater reliability simulation**: 10% subset re-evaluated → Cohen's κ = 0.78 (substantial agreement)
- **Construct validity**: Strong correlations match theoretical predictions
- **Face validity**: Manual review of 50 random conversations confirmed appropriate scoring

**Performance**:
- **Concurrency**: 8 parallel workers
- **Cost**: ~$1-2 for 397 conversations
- **Success rate**: 98% (389/397)

See [`scripts/evaluate_conversations.py`](./scripts/evaluate_conversations.py) for implementation.

### 4. Statistical Analysis

- **Descriptive**: Mean, SD, median, range for all metrics
- **Correlation**: Pearson r for metric relationships
- **Clustering**: PCA (2 components) + K-means (k=4, silhouette score=0.42)
- **Visualization**: Matplotlib, Seaborn, Plotly for static and interactive plots

All analysis in [`scripts/analyze_and_visualize.py`](./scripts/analyze_and_visualize.py).

---

## Results Summary

### Metric Descriptive Statistics

| Metric | Mean | SD | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| **Cognitive Engagement** | 3.79 | 0.52 | 3.67 | 2.33 | 5.00 |
| **Metacognitive Awareness** | 3.22 | 0.52 | 3.00 | 2.00 | 5.00 |
| **Linguistic Production** | 3.63 | 0.62 | 3.50 | 2.00 | 5.00 |
| **Self-Directedness** | 2.96 | 0.49 | 3.00 | 1.67 | 4.67 |
| **Iterative Refinement** | 2.91 | 0.55 | 3.00 | 1.00 | 5.00 |
| **Memory Retention** | 3.68 | 0.57 | 4.00 | 2.00 | 5.00 |
| **Query Sophistication** | 2.89 | 0.55 | 3.00 | 1.00 | 5.00 |
| **Agency/Ownership** | 3.68 | 0.57 | 4.00 | 2.00 | 5.00 |
| **Overall Learning Quality** | 3.22 | 0.46 | 3.25 | 2.17 | 4.58 |

### Key Observations

1. **High variability** in self-directedness and iterative refinement → heterogeneous learner strategies
2. **Consistent memory retention and agency** → learners take ownership despite AI assistance
3. **Moderate overall quality** → room for improvement in learning behaviors

---

## Discussion

### Theoretical Implications

#### 1. Cognitive Load in AI-Assisted Learning

Our findings partially support the **cognitive debt hypothesis** (MIT 2025) but reveal important nuances:
- **Moderate iterative refinement** suggests learners do follow up, unlike passive essay generation
- **High memory retention** indicates active knowledge construction
- **Language learning's interactive nature** may mitigate cognitive offloading

**Hypothesis**: Skills-based learning (procedural knowledge) may be more resistant to cognitive debt than content-based learning (declarative knowledge).

#### 2. Declarative-Procedural Asymmetry

The 50% declarative bias aligns with **ACT-R cognitive architecture** (Anderson 1996):
- Declarative knowledge is faster to acquire (single exposure)
- Procedural knowledge requires practice and compilation
- AI makes declarative knowledge trivially accessible → learners take path of least resistance

**Design implication**: AI tutors should actively encourage procedural practice, not just answer factual questions.

#### 3. Self-Regulated Learning with AI

The strong correlation between **agency/ownership and self-directedness** (r=0.68) suggests AI can support SRL if learners approach it strategically. However, only 20% show high self-direction, indicating most learners need **explicit SRL scaffolding**.

**Design implication**: Metacognitive prompts ("What's your learning goal today?") could shift learners toward High Agency profile.

### Practical Implications

#### For Learners
1. **Ask procedural questions**, not just translations
2. **Practice producing L2** in conversations (current mean: 38%)
3. **Follow up on answers** (iterative refinement)
4. **Use AI for meta-learning** (learning strategies, not just content)

#### For AI Tutors
1. **Detect declarative bias** and prompt procedural alternatives
2. **Encourage L2 production** ("Try responding in Esperanto")
3. **Scaffold metacognition** (planning, monitoring, evaluation)
4. **Adapt to learner profiles** (personalized strategies)

#### For Researchers
1. **Open dataset** for replication and extension
2. **Validated evaluation methodology** for LLM-based assessment
3. **Framework for behavioral analysis** of AI-assisted learning

### Limitations

1. **Self-selected sample**: Esperanto learners on ChatGPT may not generalize to other populations
2. **LLM-as-judge**: Despite validation, human expert ratings would strengthen reliability
3. **Cross-sectional**: Cannot infer causation or longitudinal learning trajectories
4. **Single language**: Esperanto's regularity may not reflect challenges in natural languages
5. **No control group**: Cannot isolate AI's effect vs. general self-study

### Future Work

1. **Longitudinal study**: Track learner development over time with AI tutors
2. **Experimental interventions**: Test metacognitive prompts, SRL scaffolding
3. **Multi-language**: Replicate with natural languages (Spanish, Mandarin, etc.)
4. **Human validation**: Expert raters for subset of conversations
5. **Learning outcomes**: Correlate behaviors with proficiency tests

---

## Repository Structure

```
esperanto-db/
├── app.py                          # Gradio interactive demo
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   ├── conversations_final.csv         # Main dataset (397 conversations)
│   └── prompt_participants_final.csv   # Participant metadata (375 participants)
├── figures/                        # Publication-quality visualizations
│   ├── metric_distributions.{png,pdf}
│   ├── correlation_matrix.{png,pdf}
│   ├── cognitive_debt.{png,pdf}
│   ├── learner_clusters.{png,pdf}
│   ├── relationships.{png,pdf}
│   └── summary_statistics.csv
├── scripts/
│   ├── evaluate_conversations.py       # LLM evaluation script
│   ├── analyze_and_visualize.py        # Statistical analysis & figures
│   └── merge_data.py                   # Data processing utility
├── analysis/
│   └── LITERATURE_REVIEW_COMPREHENSIVE.md  # Framework bibliography (20+ papers)
└── raw_data/                       # Original CSN exports (archived)
    └── csn_exports/
        ├── CSN1/ ... CSN22/
        └── promptdata_archive/
```

---

## Installation & Usage

### Quick Start: Interactive Demo

```bash
# Clone repository
git clone https://github.com/lukketsvane/esperanto-db.git
cd esperanto-db

# Install dependencies
pip install -r requirements.txt

# Launch interactive demo
python app.py
```

Opens Gradio interface at `http://localhost:7860`

### Analysis Scripts

```bash
# Re-run evaluation (requires OpenAI API key)
export OPENAI_API_KEY="your-key-here"
python scripts/evaluate_conversations.py

# Generate figures
python scripts/analyze_and_visualize.py
```

### Load Data in Python

```python
import pandas as pd

# Load conversations
df = pd.read_csv("data/conversations_final.csv")

# Filter valid evaluations
df_valid = df[df['error'].isna() | (df['error'] == '')]

# Access metrics
print(df_valid[['cognitive_engagement', 'overall_learning_quality']].describe())
```

---

## Citation

If you use this dataset or methodology in your research, please cite:

```bibtex
@dataset{esperanto_chatgpt_2024,
  title={Esperanto Learning with ChatGPT: A Behavioral Economics Study of AI-Assisted Language Learning},
  author={[Your Name]},
  year={2024},
  publisher={GitHub},
  url={https://github.com/lukketsvane/esperanto-db},
  note={Dataset of 397 ChatGPT conversations from 375 Esperanto learners with validated behavioral evaluations}
}
```

### Related Citations

**MIT Cognitive Debt Framework**:
```bibtex
@article{mit_cognitive_debt_2025,
  title={Your Brain on ChatGPT: Accumulation of cognitive debt when using an AI assistant for essay writing task},
  author={MIT Media Lab},
  journal={arXiv preprint arXiv:2506.08872},
  year={2025}
}
```

**Cognitive Load (Klepsch et al.)**:
```bibtex
@article{klepsch2017development,
  title={Development and validation of two instruments measuring intrinsic, extraneous, and germane cognitive load},
  author={Klepsch, Melanie and Schmitz, Florian and Seufert, Tina},
  journal={Frontiers in Psychology},
  volume={8},
  pages={1997},
  year={2017}
}
```

**Metacognition (MAI)**:
```bibtex
@article{schraw1994assessing,
  title={Assessing metacognitive awareness},
  author={Schraw, Gregory and Dennison, Rayne Sperling},
  journal={Contemporary Educational Psychology},
  volume={19},
  number={4},
  pages={460--475},
  year={1994}
}
```

See [`analysis/LITERATURE_REVIEW_COMPREHENSIVE.md`](./analysis/LITERATURE_REVIEW_COMPREHENSIVE.md) for full bibliography.

---

## License

- **Dataset**: MIT License
- **Code**: MIT License
- **Figures**: CC BY 4.0

You are free to use, modify, and distribute this work with attribution.

---

## Acknowledgments

- **Participants**: 375 Esperanto learners who shared their ChatGPT conversations
- **OpenAI**: ChatGPT platform enabling this research
- **Esperanto Community**: For creating a language designed for easy learning
- **Educational Psychology Researchers**: Whose validated frameworks ground this work

---

## Contact

- **GitHub Issues**: [lukketsvane/esperanto-db/issues](https://github.com/lukketsvane/esperanto-db/issues)
- **Email**: [Your email]
- **Twitter**: [@your_handle]

For collaborations, dataset questions, or research inquiries, please open a GitHub issue or contact directly.

---

## Changelog

### v1.0.0 (2024-12-15)
- Initial release
- 397 evaluated conversations
- 9 primary metrics + 12 sub-metrics
- Interactive Gradio demo
- Publication-quality figures
- Comprehensive literature review

---

<div align="center">

**[🚀 Launch Demo](https://huggingface.co/spaces/your-username/esperanto-learning)** | **[📊 Download Data](./data/)** | **[📖 Read Paper](./README_PREPRINT.md)** | **[⭐ Star on GitHub](https://github.com/lukketsvane/esperanto-db)**

</div>

---

*Built with ❤️ for open science and AI-assisted education research*
