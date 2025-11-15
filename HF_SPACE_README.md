---
title: Esperanto Learning with ChatGPT
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
tags:
  - education
  - language-learning
  - chatgpt
  - dataset
  - behavioral-economics
  - cognitive-science
  - esperanto
  - educational-psychology
  - metacognition
  - ai-assisted-learning
---

# Esperanto Learning with ChatGPT: Interactive Dataset Explorer

> **A comprehensive analysis of 397 ChatGPT conversations from 375 Esperanto learners, evaluated using validated frameworks from educational psychology and cognitive science**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/lukketsvane/esperanto-db)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Preprint-brightgreen)](https://github.com/lukketsvane/esperanto-db/blob/main/README_PREPRINT.md)

## 🎯 What is this?

This interactive platform lets you explore how people learn languages using AI tutors. We analyzed **397 real ChatGPT conversations** from **375 Esperanto learners** and evaluated them using **validated frameworks** from educational psychology research.

### Key Questions We Answer:

1. **Do AI tutors cause "cognitive debt"?** → Moderate engagement with good memory retention
2. **How do learners use ChatGPT?** → 50% declarative bias (translation) vs. procedural (grammar)
3. **Are there different learner types?** → Yes! 4 distinct profiles identified
4. **What predicts learning quality?** → Cognitive engagement (r=0.72)

## 📊 Interactive Features

### 🔍 Explore the Data

- **Metric Distributions**: See how 397 learners scored across 9 learning dimensions
- **Correlation Heatmap**: Discover which behaviors predict better learning
- **Cognitive Debt Analysis**: Understand AI dependency patterns (MIT 2025 framework)
- **Learner Clusters**: Visualize 4 distinct learner profiles using PCA + K-means
- **Conversation Explorer**: Drill down into individual learning sessions

### 🎓 Validated Science

All metrics grounded in peer-reviewed research:
- **Cognitive Load** (Klepsch et al. 2017)
- **Metacognition** (Schraw & Dennison 1994)
- **Self-Regulation** (Pintrich et al. 1991)
- **Engagement** (Kahu et al. 2018)
- **Cognitive Debt** (MIT Media Lab 2025)

## 🚀 Quick Start

1. Click **"Metric Distributions"** to see how learners scored
2. Try **"Learner Profiles"** to see 4 distinct types
3. Use **"Conversation Explorer"** to analyze individual sessions
4. Read **"About"** for methodology and citations

## 📈 Key Findings

### Cognitive Debt Indicators (1-5 scale)

| Metric | Score | Meaning |
|--------|-------|---------|
| **Iterative Refinement** | 2.91 | Moderate follow-up depth |
| **Memory Retention** | 3.68 | **Good knowledge retention** ✓ |
| **Metacognitive Awareness** | 3.22 | Moderate self-reflection |

### Learner Profiles

1. **Low Engagers (25%)**: Brief, surface-level interactions
2. **Moderate Balanced (38%)**: Typical learner, balanced approach
3. **High Agency (20%)**: Strategic, metacognitive, self-directed
4. **Production Focused (17%)**: More Esperanto usage, less translation

### Strong Correlations

- Cognitive Engagement ↔ Learning Quality: **r=0.72** ⭐
- Agency ↔ Self-Direction: **r=0.68**
- Query Sophistication ↔ Engagement: **r=0.65**

## 📚 Dataset

- **397 conversations** from **375 participants**
- **4,312 total messages**
- **44 evaluation columns** (metadata + metrics)
- **98% success rate** in automated evaluation
- **MIT License** - freely available for research

**Download**: [GitHub Repository](https://github.com/lukketsvane/esperanto-db)

## 🎓 Citation

```bibtex
@dataset{esperanto_chatgpt_2024,
  title={Esperanto Learning with ChatGPT: A Behavioral Economics Study},
  author={[Your Name]},
  year={2024},
  publisher={HuggingFace},
  url={https://github.com/lukketsvane/esperanto-db}
}
```

## 🔗 Links

- **📄 Full Paper**: [README_PREPRINT.md](https://github.com/lukketsvane/esperanto-db/blob/main/README_PREPRINT.md)
- **💻 Source Code**: [GitHub](https://github.com/lukketsvane/esperanto-db)
- **📊 Raw Data**: [data/conversations_final.csv](https://github.com/lukketsvane/esperanto-db/tree/main/data)
- **📚 Literature Review**: [20+ validated frameworks](https://github.com/lukketsvane/esperanto-db/blob/main/analysis/LITERATURE_REVIEW_COMPREHENSIVE.md)

## 🛠️ Built With

- **Gradio 4.44** - Interactive web interface
- **Plotly** - Interactive visualizations
- **GPT-3.5-turbo** - LLM-as-judge evaluation (~$1 for 397 conversations)
- **Scikit-learn** - PCA and K-means clustering
- **Pandas** - Data processing

## 📧 Contact

Questions? Feedback? Want to collaborate?

- **GitHub Issues**: [Open an issue](https://github.com/lukketsvane/esperanto-db/issues)
- **Email**: [Your email]

## 📜 License

**MIT License** - Free to use, modify, and distribute with attribution.

---

*Built with ❤️ for open science and AI-assisted education research*

**Explore the app above to discover insights from 397 Esperanto learners! 🌐**
