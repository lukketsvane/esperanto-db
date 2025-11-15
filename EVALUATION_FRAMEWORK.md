# Esperanto Learning Conversation Evaluation Framework

## Overview

This evaluation framework analyzes ChatGPT-based Esperanto learning conversations using literature-grounded metrics from educational psychology, behavioral economics, and neuroscience research.

## Literature Foundation

### Primary Research Sources

1. **MIT Media Lab (2025)**: "Your Brain on ChatGPT: Accumulation of Cognitive Debt"
   - Key concept: Cognitive debt from AI dependency
   - Focus: Memory formation, neural connectivity, brain engagement
   - arXiv: 2506.08872

2. **Behavioral Economics**: Cognitive offloading and effort allocation
   - Decision-making patterns in AI-assisted learning
   - Strategic use of external tools vs. internal processing

3. **Language Learning Research**: Self-directed learning and metacognitive awareness
   - Agency and ownership in learning processes
   - Iterative refinement and deep learning indicators

## Evaluation Dimensions

All metrics use a **1-5 Likert scale** unless otherwise specified.

### 1. Cognitive Engagement (cognitive_engagement)
**Scale**: 1-5
**Theory**: MIT cognitive debt research
**Description**: Depth of thinking and metacognitive awareness

- **1**: Passive acceptance, minimal thinking
- **2**: Surface-level engagement
- **3**: Moderate engagement with some critical thinking
- **4**: Deep engagement, questioning, analyzing
- **5**: Highly active, metacognitive, critical thinking

**Indicators**:
- Depth of questions asked
- Critical evaluation of responses
- Evidence of reflection
- Metacognitive awareness (thinking about thinking)

### 2. Query Sophistication (query_sophistication)
**Scale**: 1-5
**Theory**: Behavioral economics - effort allocation
**Description**: Linguistic and conceptual sophistication of learner queries

- **1**: Simple, vague, single-word queries
- **2**: Basic queries with minimal context
- **3**: Clear queries with some specificity
- **4**: Well-formed, specific queries showing understanding
- **5**: Sophisticated, nuanced queries demonstrating deep knowledge

**Indicators**:
- Linguistic complexity of questions
- Specificity and precision
- Conceptual depth
- Use of Esperanto vocabulary in questions

### 3. Self-Directedness (self_directedness)
**Scale**: 1-5
**Theory**: Cognitive offloading research
**Description**: Learner autonomy vs. AI dependency

- **1**: Complete dependency, no independent thinking
- **2**: Heavy reliance, minimal self-direction
- **3**: Balanced - uses AI as tool while maintaining autonomy
- **4**: Highly self-directed with strategic AI use
- **5**: Autonomous learner using AI for verification/enhancement only

**Indicators**:
- Seeking answers vs. understanding
- Evidence of attempting before asking
- Critical evaluation of AI responses
- Independent problem-solving

### 4. Iterative Refinement (iterative_refinement)
**Scale**: 1-5
**Theory**: Deep learning indicators
**Description**: Progressive deepening and clarification behavior

- **1**: No follow-up, accepts first answer
- **2**: Minimal follow-up
- **3**: Some clarification questions
- **4**: Regular refinement and deeper exploration
- **5**: Systematic iterative deepening with multiple perspectives

**Indicators**:
- Follow-up questions
- Requests for clarification
- Progressive deepening of topic
- Revisions based on feedback

### 5. Learning Orientation (learning_orientation)
**Scale**: 1-5 (Declarative ← → Procedural)
**Theory**: Cognitive science - declarative vs. procedural knowledge
**Description**: Focus on facts vs. processes

- **1**: Purely declarative (only facts, translations)
- **2**: Mostly declarative with some procedural
- **3**: Balanced mix
- **4**: Mostly procedural (grammar rules, usage patterns)
- **5**: Purely procedural (processes, strategies, how to use)

**Examples**:
- Declarative: "What does X mean?", "Translate Y"
- Procedural: "How do I form X?", "When do I use Y?", "What's the rule for Z?"

### 6. Agency & Ownership (agency_ownership)
**Scale**: 1-5
**Theory**: MIT "essay ownership" research
**Description**: Taking ownership of learning process

- **1**: No ownership, purely receptive
- **2**: Minimal agency, follows AI lead
- **3**: Moderate ownership with some initiative
- **4**: Strong ownership, directs conversation
- **5**: Complete ownership, AI as tool for self-directed goals

**Indicators**:
- Setting learning goals
- Personalizing examples
- Relating to their context
- Taking initiative in conversation direction

### 7. Linguistic Production (linguistic_production)
**Scale**: 1-5
**Theory**: Language learning research
**Description**: Quality of Esperanto production in learner messages

- **1**: No Esperanto usage, English only
- **2**: Minimal Esperanto (single words)
- **3**: Some Esperanto usage with English support
- **4**: Regular Esperanto usage with good structure
- **5**: Sophisticated Esperanto production

**Indicators**:
- Frequency of Esperanto usage
- Grammatical complexity
- Vocabulary range
- Accuracy (if evident)

### 8. Memory/Retention Indicators (memory_retention)
**Scale**: 1-5
**Theory**: MIT cognitive debt - memory formation
**Description**: Evidence of retention and knowledge building

- **1**: No evidence of retention, repeated questions
- **2**: Minimal retention, frequent re-asking
- **3**: Some retention, occasional references to prior learning
- **4**: Good retention, builds on previous knowledge
- **5**: Excellent retention, integrates and synthesizes across topics

**Indicators**:
- References to previous conversations
- Building on prior knowledge
- Not re-asking same questions
- Making connections between topics

### 9. Metacognitive Awareness (metacognitive_awareness)
**Scale**: 1-5
**Theory**: Educational psychology
**Description**: Thinking about one's own learning process

- **1**: No metacognitive awareness
- **2**: Minimal awareness of learning process
- **3**: Some reflection on learning
- **4**: Regular metacognitive monitoring
- **5**: Sophisticated metacognitive strategies

**Indicators**:
- Reflection on learning strategies
- Awareness of difficulties
- Self-monitoring of understanding
- Strategic planning of learning

### 10. Overall Learning Quality (overall_learning_quality)
**Scale**: 1-5
**Theory**: Holistic assessment
**Description**: Overall effectiveness of AI-assisted learning

综合评估所有维度的整体学习质量。

## Additional Metrics

### Question Type Distribution
**Type**: Percentages (0-100)
**Categories**:
- **qt_translation**: Simple translation requests
- **qt_grammar**: Grammar rules and explanations
- **qt_usage**: How/when to use something
- **qt_clarification**: Follow-up clarification questions
- **qt_application**: Applying knowledge to new situations
- **qt_meta**: Questions about learning process itself

### Esperanto Usage (esperanto_usage_pct)
**Type**: Percentage (0-100)
**Description**: Percentage of user messages containing Esperanto text

### Mean Query Length (mean_query_length)
**Type**: Number (words)
**Description**: Average word count of user messages

## Dataset Structure

### Files Generated

1. **conversations_with_evaluations_final.csv**
   - All 397 conversations with evaluations
   - Merged with participant data
   - 56 columns total

2. **evaluation_results_final.csv**
   - Just the evaluation results
   - For quick analysis

### Key Columns in Final Dataset

**Original Data**:
- `conversation_id`: Unique conversation identifier
- `participant_id`: Unique participant identifier (from recovered IDs)
- `csn`: Computer station number
- `n_user_msgs`: Number of user messages
- `title`: Conversation title
- `create_time`: Conversation creation timestamp
- `duration_min`: Participant session duration

**Evaluation Metrics** (all 1-5 unless noted):
- `cognitive_engagement`
- `query_sophistication`
- `self_directedness`
- `iterative_refinement`
- `learning_orientation`
- `agency_ownership`
- `linguistic_production`
- `memory_retention`
- `metacognitive_awareness`
- `overall_learning_quality`

**Additional Metrics**:
- `esperanto_usage_pct` (0-100)
- `mean_query_length` (words)
- `qt_translation`, `qt_grammar`, `qt_usage`, `qt_clarification`, `qt_application`, `qt_meta` (0-100)

**Participant Information**:
- `id_date`, `id_time`, `id_computer`: Participant identification
- `is_main_session`: Whether this is the main experimental session
- `id_confidence`: Confidence level in ID recovery (high/medium/low/synthetic)

## Summary Statistics

**From 397 conversations, 375 unique participants:**

| Metric | Mean | Std Dev | Interpretation |
|--------|------|---------|----------------|
| Cognitive Engagement | 2.64 | 0.67 | Moderate, trending toward surface-level |
| Query Sophistication | 2.36 | 0.66 | Basic to moderate |
| Self-Directedness | 2.70 | 0.71 | Balanced AI use |
| Iterative Refinement | 1.86 | 0.70 | Minimal follow-up (concerning) |
| Learning Orientation | 2.48 | 0.74 | Slightly more declarative than procedural |
| Agency & Ownership | 2.70 | 0.71 | Moderate ownership |
| Linguistic Production | 2.20 | 0.96 | Minimal Esperanto production |
| Memory Retention | 2.09 | 0.65 | Minimal to some retention |
| Metacognitive Awareness | 1.98 | 0.50 | Low metacognitive awareness |
| Overall Learning Quality | 2.48 | 0.67 | Below moderate |

**Key Findings**:
- **Esperanto Usage**: 38.1% average (only about 1/3 of messages contain Esperanto)
- **Mean Query Length**: 6.4 words (relatively short queries)
- **Low Iterative Refinement** (1.86): Learners tend to accept first answers without deeper exploration
- **Low Metacognitive Awareness** (1.98): Minimal reflection on learning strategies
- **Moderate Cognitive Engagement** (2.64): Some thinking but mostly surface-level

## Implications for Behavioral Economics Research

### Cognitive Debt Indicators
- Low iterative refinement suggests possible cognitive offloading
- Moderate self-directedness indicates balanced but not optimal AI tool use
- Low memory retention aligns with MIT findings on reduced encoding

### Learning Effectiveness
- Below-moderate overall quality (2.48/5) suggests room for improvement
- Low linguistic production indicates possible over-reliance on comprehension vs. production
- Declarative bias (2.48) may limit deeper understanding

### Recommendations for Future Studies
1. Investigate correlation between iterative refinement and learning outcomes
2. Analyze relationship between metacognitive awareness and retention
3. Study impact of learning orientation on long-term proficiency
4. Examine whether self-directedness predicts better outcomes

## Methods

**Model Used**: GPT-4 Turbo Preview (gpt-4-turbo-preview)
**Temperature**: 0.3 (for consistent evaluations)
**Processing**: Concurrent with 10 workers
**API Calls**: 1 comprehensive call per conversation (397 total)
**Error Rate**: 2.0% (8/397 conversations)

## Citation

If using this evaluation framework, please cite:

```
Esperanto Learning Conversation Evaluation Framework (2025)
Based on:
- MIT Media Lab (2025). "Your Brain on ChatGPT: Accumulation of Cognitive Debt"
- Behavioral Economics and Cognitive Offloading Research
- Language Learning and Metacognitive Awareness Literature
```

## Files

- `evaluate_conversations_fast.py`: Main evaluation script
- `EVALUATION_FRAMEWORK.md`: This documentation
- `conversations_with_evaluations_final.csv`: Full dataset with evaluations
- `evaluation_results_final.csv`: Evaluation results only
