# Esperanto Learning with ChatGPT: Behavioral Economics Study

[![HuggingFace Demo](https://img.shields.io/badge/🤗%20Demo-Interactive%20Explorer-blue)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **397 ChatGPT conversations from 375 participants learning Esperanto, evaluated using validated frameworks from educational psychology and cognitive science literature**

🎮 **[Launch Interactive Explorer](https://huggingface.co/spaces)** | 📖 **[Read Full Paper](./README_PREPRINT.md)** | 📊 **[Download Dataset](./data/)**

## Quick Links

- **Interactive Visualizations**: Run `python app.py` for Gradio interface
- **Full Preprint**: See [README_PREPRINT.md](./README_PREPRINT.md) for academic paper format
- **Deployment Guide**: See [DEPLOYMENT.md](./DEPLOYMENT.md) for HuggingFace Spaces setup
- **Literature Review**: See [analysis/LITERATURE_REVIEW_COMPREHENSIVE.md](./analysis/LITERATURE_REVIEW_COMPREHENSIVE.md)

## Dataset

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

**Results**:
- cognitive_engagement: 3.79 ± 0.52
- metacognitive_awareness: 3.22 ± 0.52
- linguistic_production: 3.63 ± 0.62
- self_directedness: 2.96 ± 0.49
- iterative_refinement: 2.91 ± 0.55
- memory_retention: 3.68 ± 0.57
- query_sophistication: 2.89 ± 0.55
- agency_ownership: 3.68 ± 0.57
- overall_learning_quality: 3.22 ± 0.46

## Methodology

**Model**: GPT-3.5-turbo, temperature 0.1
**Success Rate**: 98% (389/397 conversations evaluated)
**Processing**: Concurrent evaluation (8 workers), cost-optimized
**Cost**: ~$1-2 total

All metrics derived from observable behaviors in conversation transcripts using validated frameworks from educational psychology literature.

## Interactive Demo

```bash
# Clone repository
git clone https://github.com/lukketsvane/esperanto-db.git
cd esperanto-db

# Install dependencies
pip install -r requirements.txt

# Launch interactive demo
python app.py
```

Opens Gradio interface at `http://localhost:7860` with:
- 📊 Metric distributions across all conversations
- 🔗 Interactive correlation heatmaps
- 🧠 Cognitive debt analysis visualizations
- 👥 Learner profile clustering (PCA + K-means)
- 🔍 Individual conversation explorer

## Repository Structure

```
esperanto-db/
├── app.py                          # Gradio interactive demo
├── requirements.txt                # Python dependencies
├── README.md                       # This file (quick reference)
├── README_PREPRINT.md             # Full academic paper format
├── DEPLOYMENT.md                   # HuggingFace Spaces deployment guide
├── HF_SPACE_README.md             # HuggingFace Space configuration
├── data/
│   ├── conversations_final.csv         # Complete dataset (397 × 44)
│   └── prompt_participants_final.csv   # Participant metadata (375)
├── figures/                        # Publication-quality visualizations
│   ├── metric_distributions.{png,pdf}
│   ├── correlation_matrix.{png,pdf}
│   ├── cognitive_debt.{png,pdf}
│   ├── learner_clusters.{png,pdf}
│   └── relationships.{png,pdf}
├── scripts/
│   ├── evaluate_conversations.py       # LLM evaluation script
│   ├── analyze_and_visualize.py        # Statistical analysis
│   └── merge_data.py                   # Data processing
├── analysis/
│   └── LITERATURE_REVIEW_COMPREHENSIVE.md  # 20+ framework sources
└── raw_data/                       # Original CSN exports (archived)
```

## Citation

If you use this dataset or methodology in your research, please cite:

```bibtex
@dataset{esperanto_chatgpt_2024,
  title={Esperanto Learning with ChatGPT: A Behavioral Economics Study},
  author={[Your Name]},
  year={2024},
  publisher={GitHub},
  url={https://github.com/lukketsvane/esperanto-db}
}
```

## Key References

**MIT Cognitive Debt Study**:
MIT Media Lab (2025). Your Brain on ChatGPT: Accumulation of cognitive debt when using an AI assistant for essay writing task. arXiv:2506.08872

**Validated Frameworks**:
- Klepsch, M., Schmitz, F., & Seufert, T. (2017). Frontiers in Psychology, 8, 1997.
- Schraw, G., & Dennison, R. S. (1994). Contemporary Educational Psychology, 19(4), 460-475.
- Pintrich, P. R., et al. (1991). MSLQ Manual. University of Michigan.
- ACTFL (2024). ACTFL Proficiency Guidelines 2024.
- Kahu, E. R., et al. (2018). Research in Higher Education, 60(2), 134-159.
- Liu, O. L., et al. (2014). ETS Research Report Series.

See [analysis/LITERATURE_REVIEW_COMPREHENSIVE.md](./analysis/LITERATURE_REVIEW_COMPREHENSIVE.md) for complete bibliography.

## License

**MIT License** - Free to use, modify, and distribute with attribution.

---

📖 **For full academic paper**: See [README_PREPRINT.md](./README_PREPRINT.md)
🚀 **Deploy to HuggingFace**: See [DEPLOYMENT.md](./DEPLOYMENT.md)
💻 **Run interactive demo**: `python app.py`

*Built with ❤️ for open science and AI-assisted education research*
