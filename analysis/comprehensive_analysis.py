#!/usr/bin/env python3
"""
Comprehensive Data Science Analysis of Esperanto Learning Conversations
For Behavioral Economics Research Paper

Analysis focuses on:
- Cognitive debt patterns (MIT 2025)
- Learning effectiveness metrics
- Declarative vs. Procedural learning
- ID confidence impact
- Temporal patterns
- Correlation analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Paths
BASE_DIR = Path("/home/user/esperanto-db")
FIGURES_DIR = BASE_DIR / "analysis" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv(BASE_DIR / "conversations_with_evaluations_final.csv")
participant_df = pd.read_csv(BASE_DIR / "participant_level_evaluations.csv")

print(f"Loaded {len(df)} conversations from {len(participant_df)} participants")
print()

# Define evaluation metrics
EVAL_METRICS = [
    'cognitive_engagement',
    'query_sophistication',
    'self_directedness',
    'iterative_refinement',
    'learning_orientation',
    'agency_ownership',
    'linguistic_production',
    'memory_retention',
    'metacognitive_awareness',
    'overall_learning_quality'
]

QUESTION_TYPES = [
    'qt_translation',
    'qt_grammar',
    'qt_usage',
    'qt_clarification',
    'qt_application',
    'qt_meta'
]


def create_figure_1_overview():
    """Figure 1: Evaluation Metrics Overview - Distribution and Comparison"""
    print("Creating Figure 1: Evaluation Metrics Overview...")

    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    fig.suptitle('Distribution of Evaluation Metrics Across All Conversations (N=397)',
                 fontsize=14, fontweight='bold')

    for idx, metric in enumerate(EVAL_METRICS):
        ax = axes[idx // 5, idx % 5]
        data = df[metric].dropna()

        # Violin plot with box plot overlay
        parts = ax.violinplot([data], positions=[0], widths=0.7,
                               showmeans=True, showmedians=True)

        # Color the violin plots
        for pc in parts['bodies']:
            pc.set_facecolor('#8ECAE6')
            pc.set_alpha(0.7)

        # Add scatter points
        y_jitter = np.random.normal(0, 0.04, size=len(data))
        ax.scatter(y_jitter, data, alpha=0.3, s=10, color='#023047')

        # Formatting
        ax.set_ylim(0.5, 5.5)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.axhline(3, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_ylabel('Score (1-5)')
        ax.set_title(f"{metric.replace('_', ' ').title()}\nμ={data.mean():.2f}, σ={data.std():.2f}",
                    fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_1_metrics_distribution.png", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure_1_metrics_distribution.pdf", bbox_inches='tight')
    print(f"  Saved: figure_1_metrics_distribution.png/pdf")
    plt.close()


def create_figure_2_correlation():
    """Figure 2: Correlation Matrix of Evaluation Metrics"""
    print("Creating Figure 2: Correlation Matrix...")

    # Calculate correlation matrix
    corr_data = df[EVAL_METRICS].dropna()
    corr_matrix = corr_data.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Create heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax)

    ax.set_title('Correlation Matrix of Evaluation Metrics\n' +
                 'Pearson Correlation Coefficients (N=389)',
                 fontsize=14, fontweight='bold', pad=20)

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_2_correlation_matrix.png", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure_2_correlation_matrix.pdf", bbox_inches='tight')
    print(f"  Saved: figure_2_correlation_matrix.png/pdf")
    plt.close()


def create_figure_3_cognitive_debt():
    """Figure 3: Cognitive Debt Indicators - Key MIT Study Findings"""
    print("Creating Figure 3: Cognitive Debt Indicators...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Cognitive Debt Indicators (MIT "Your Brain on ChatGPT" 2025)',
                 fontsize=14, fontweight='bold')

    # Three key indicators
    indicators = [
        ('iterative_refinement', 'Iterative Refinement\n(Follow-up & Deepening)'),
        ('memory_retention', 'Memory Retention\n(Knowledge Building)'),
        ('metacognitive_awareness', 'Metacognitive Awareness\n(Self-Reflection)')
    ]

    for idx, (metric, title) in enumerate(indicators):
        ax = axes[idx]
        data = df[metric].dropna()

        # Histogram with KDE
        ax.hist(data, bins=np.arange(0.5, 6, 0.5), alpha=0.6,
                color='#E63946', edgecolor='black', density=True)

        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_range = np.linspace(0.5, 5.5, 100)
        ax.plot(x_range, kde(x_range), color='#1D3557', linewidth=2, label='KDE')

        # Mean line
        mean_val = data.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.2f}')

        # Threshold line at 3.0
        ax.axvline(3.0, color='green', linestyle=':', linewidth=1.5,
                  alpha=0.7, label='Moderate (3.0)')

        ax.set_xlabel('Score (1-5)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.set_xlim(0.5, 5.5)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)

        # Add interpretation text
        if mean_val < 2.5:
            interp = "LOW - Concerning"
            color = 'red'
        elif mean_val < 3.5:
            interp = "MODERATE"
            color = 'orange'
        else:
            interp = "HIGH - Good"
            color = 'green'

        ax.text(0.05, 0.95, interp, transform=ax.transAxes,
                fontsize=10, fontweight='bold', color=color,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_3_cognitive_debt_indicators.png", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure_3_cognitive_debt_indicators.pdf", bbox_inches='tight')
    print(f"  Saved: figure_3_cognitive_debt_indicators.png/pdf")
    plt.close()


def create_figure_4_learning_orientation():
    """Figure 4: Declarative vs Procedural Learning Analysis"""
    print("Creating Figure 4: Declarative vs Procedural Learning...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Declarative vs. Procedural Learning Orientation',
                 fontsize=14, fontweight='bold')

    # Left: Distribution
    ax = axes[0]
    data = df['learning_orientation'].dropna()

    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    labels = ['Purely\nDeclarative', 'Mostly\nDeclarative', 'Balanced',
              'Mostly\nProcedural', 'Purely\nProcedural']

    counts, _ = np.histogram(data, bins=bins)
    x_pos = np.arange(len(labels))

    colors = ['#FF6B6B', '#FFA07A', '#FFD93D', '#6BCF7F', '#4ECDC4']
    bars = ax.bar(x_pos, counts, color=colors, edgecolor='black', alpha=0.7)

    # Add count labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(data)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Number of Conversations')
    ax.set_title('Distribution of Learning Orientation')
    ax.grid(axis='y', alpha=0.3)

    # Right: Relationship with learning quality
    ax = axes[1]

    # Group by orientation categories
    df_copy = df.dropna(subset=['learning_orientation', 'overall_learning_quality'])
    df_copy['orientation_category'] = pd.cut(df_copy['learning_orientation'],
                                              bins=bins, labels=labels, include_lowest=True)

    # Box plot
    positions = range(len(labels))
    data_by_cat = [df_copy[df_copy['orientation_category'] == label]['overall_learning_quality'].values
                   for label in labels]

    bp = ax.boxplot(data_by_cat, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True)

    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Overall Learning Quality (1-5)')
    ax.set_title('Learning Quality by Orientation')
    ax.axhline(df['overall_learning_quality'].mean(), color='red',
              linestyle='--', alpha=0.5, label='Overall Mean')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_4_learning_orientation.png", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure_4_learning_orientation.pdf", bbox_inches='tight')
    print(f"  Saved: figure_4_learning_orientation.png/pdf")
    plt.close()


def create_figure_5_question_types():
    """Figure 5: Question Type Distribution Analysis"""
    print("Creating Figure 5: Question Type Distribution...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Question Type Distribution Across Conversations',
                 fontsize=14, fontweight='bold')

    # Calculate mean percentages
    qt_means = df[QUESTION_TYPES].mean()
    qt_stds = df[QUESTION_TYPES].std()

    # Left: Mean distribution
    ax = axes[0]
    labels = ['Translation', 'Grammar', 'Usage', 'Clarification', 'Application', 'Meta']
    colors = ['#E63946', '#F77F00', '#FCBF49', '#06D6A0', '#118AB2', '#073B4C']

    bars = ax.bar(range(len(labels)), qt_means, yerr=qt_stds,
                  color=colors, alpha=0.7, edgecolor='black', capsize=5)

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, qt_means, qt_stds)):
        ax.text(bar.get_x() + bar.get_width()/2., mean + std + 2,
                f'{mean:.1f}%\n±{std:.1f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Percentage of Questions (%)')
    ax.set_title('Mean Question Type Distribution\n(Error bars show standard deviation)')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(qt_means + qt_stds) * 1.3)

    # Right: Stacked area chart showing diversity
    ax = axes[1]

    # Get data for top 50 conversations by total questions
    df_sorted = df.dropna(subset=QUESTION_TYPES).sort_values('n_user_messages', ascending=False).head(50)

    # Create stacked area
    x = range(len(df_sorted))
    y = np.column_stack([df_sorted[qt].values for qt in QUESTION_TYPES])

    ax.stackplot(x, y.T, labels=labels, colors=colors, alpha=0.7)

    ax.set_xlabel('Conversations (sorted by message count)')
    ax.set_ylabel('Question Type Distribution (%)')
    ax.set_title('Question Type Patterns in Top 50 Conversations')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_5_question_types.png", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure_5_question_types.pdf", bbox_inches='tight')
    print(f"  Saved: figure_5_question_types.png/pdf")
    plt.close()


def create_figure_6_id_confidence():
    """Figure 6: Impact of ID Confidence on Evaluation Metrics"""
    print("Creating Figure 6: ID Confidence Impact...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Impact of ID Confidence on Learning Metrics',
                 fontsize=14, fontweight='bold')

    # Filter data
    df_conf = df[df['id_confidence'].isin(['high', 'imputed_medium', 'synthetic_low'])].copy()

    # Key metrics to compare
    key_metrics = ['cognitive_engagement', 'overall_learning_quality',
                   'memory_retention', 'esperanto_usage_pct']

    for idx, metric in enumerate(key_metrics):
        ax = axes[idx // 2, idx % 2]

        # Prepare data
        high = df_conf[df_conf['id_confidence'] == 'high'][metric].dropna()
        medium = df_conf[df_conf['id_confidence'] == 'imputed_medium'][metric].dropna()
        low = df_conf[df_conf['id_confidence'] == 'synthetic_low'][metric].dropna()

        data = [high, medium, low]
        labels = [f'High\n(n={len(high)})', f'Medium\n(n={len(medium)})', f'Low\n(n={len(low)})']
        colors = ['#06D6A0', '#FFD166', '#EF476F']

        # Create violin plot
        parts = ax.violinplot(data, positions=range(3), widths=0.7,
                              showmeans=True, showmedians=True)

        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        # Add box plot overlay
        bp = ax.boxplot(data, positions=range(3), widths=0.3, patch_artist=True,
                       showfliers=False)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)

        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} by ID Confidence')
        ax.grid(axis='y', alpha=0.3)

        # Statistical test (Kruskal-Wallis)
        if len(high) > 0 and len(medium) > 0 and len(low) > 0:
            h_stat, p_val = stats.kruskal(high, medium, low)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            ax.text(0.02, 0.98, f'Kruskal-Wallis p={p_val:.4f} {sig}',
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_6_id_confidence_impact.png", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure_6_id_confidence_impact.pdf", bbox_inches='tight')
    print(f"  Saved: figure_6_id_confidence_impact.png/pdf")
    plt.close()


def create_figure_7_pca_clustering():
    """Figure 7: PCA and Clustering of Learner Profiles"""
    print("Creating Figure 7: PCA and Clustering...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Learner Profile Analysis: Dimensionality Reduction and Clustering',
                 fontsize=14, fontweight='bold')

    # Prepare data
    df_clean = df[EVAL_METRICS].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Left: PCA scatter
    ax = axes[0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters,
                        cmap='viridis', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

    # Plot cluster centers
    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red',
              s=200, alpha=0.8, edgecolors='black', linewidth=2, marker='X')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.set_title('PCA: Learner Profiles in 2D Space\n(K-means clusters shown)')
    ax.grid(alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=plt.cm.viridis(i/3), label=f'Cluster {i+1}')
                      for i in range(4)]
    legend_elements.append(plt.Line2D([0], [0], marker='X', color='w',
                                     markerfacecolor='r', markersize=10,
                                     label='Cluster Centers'))
    ax.legend(handles=legend_elements, loc='best', fontsize=8)

    # Right: Cluster characteristics
    ax = axes[1]

    # Calculate mean metrics for each cluster
    df_clean_copy = df_clean.copy()
    df_clean_copy['cluster'] = clusters

    cluster_means = df_clean_copy.groupby('cluster')[EVAL_METRICS].mean()

    # Heatmap
    sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='RdYlGn',
                center=3, vmin=1, vmax=5, cbar_kws={'label': 'Mean Score'},
                ax=ax, linewidths=0.5)

    ax.set_xlabel('Cluster')
    ax.set_ylabel('Metric')
    ax.set_title('Mean Metric Scores by Cluster')
    ax.set_xticklabels([f'C{i+1}\n(n={sum(clusters==i)})' for i in range(4)])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_7_pca_clustering.png", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure_7_pca_clustering.pdf", bbox_inches='tight')
    print(f"  Saved: figure_7_pca_clustering.png/pdf")
    plt.close()


def create_figure_8_esperanto_usage():
    """Figure 8: Esperanto Usage Analysis"""
    print("Creating Figure 8: Esperanto Usage...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Esperanto Production and Usage Patterns',
                 fontsize=14, fontweight='bold')

    # Left: Distribution
    ax = axes[0]
    data = df['esperanto_usage_pct'].dropna()

    ax.hist(data, bins=20, alpha=0.7, color='#219EBC', edgecolor='black')
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {data.mean():.1f}%')
    ax.axvline(data.median(), color='green', linestyle=':', linewidth=2,
              label=f'Median: {data.median():.1f}%')

    ax.set_xlabel('Esperanto Usage (%)')
    ax.set_ylabel('Number of Conversations')
    ax.set_title('Distribution of Esperanto Usage')
    ax.legend()
    ax.grid(alpha=0.3)

    # Middle: Relationship with linguistic production score
    ax = axes[1]

    x = df['esperanto_usage_pct'].dropna()
    y = df.loc[x.index, 'linguistic_production']

    ax.scatter(x, y, alpha=0.5, s=30, color='#219EBC')

    # Regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x.sort_values(), p(x.sort_values()), "r--", linewidth=2,
           label=f'y={z[0]:.3f}x+{z[1]:.2f}')

    # Correlation
    corr, p_val = stats.pearsonr(x, y)
    ax.text(0.05, 0.95, f'r={corr:.3f}\np={p_val:.4f}',
           transform=ax.transAxes, fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Esperanto Usage (%)')
    ax.set_ylabel('Linguistic Production Score (1-5)')
    ax.set_title('Usage % vs. Production Quality')
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: By learning orientation
    ax = axes[2]

    df_copy = df.dropna(subset=['learning_orientation', 'esperanto_usage_pct'])
    bins = [0.5, 2.5, 3.5, 5.5]
    labels = ['Declarative', 'Balanced', 'Procedural']
    df_copy['orientation_cat'] = pd.cut(df_copy['learning_orientation'],
                                         bins=bins, labels=labels)

    data_by_orient = [df_copy[df_copy['orientation_cat'] == label]['esperanto_usage_pct'].values
                     for label in labels]

    bp = ax.boxplot(data_by_orient, labels=labels, patch_artist=True, showmeans=True)

    colors = ['#FF6B6B', '#FFD93D', '#4ECDC4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Esperanto Usage (%)')
    ax.set_title('Usage by Learning Orientation')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_8_esperanto_usage.png", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure_8_esperanto_usage.pdf", bbox_inches='tight')
    print(f"  Saved: figure_8_esperanto_usage.png/pdf")
    plt.close()


def create_figure_9_self_directedness():
    """Figure 9: Self-Directedness and AI Dependency Analysis"""
    print("Creating Figure 9: Self-Directedness Analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Self-Directedness and AI Dependency Patterns\n' +
                 '(Behavioral Economics: Cognitive Offloading)',
                 fontsize=14, fontweight='bold')

    # Top-left: Self-directedness distribution
    ax = axes[0, 0]
    data = df['self_directedness'].dropna()

    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    labels = ['Complete\nDependency', 'Heavy\nReliance', 'Balanced',
              'Highly\nSelf-Directed', 'Autonomous']

    counts, _ = np.histogram(data, bins=bins)
    x_pos = np.arange(len(labels))
    colors = ['#D62828', '#F77F00', '#FCBF49', '#06D6A0', '#118AB2']

    bars = ax.bar(x_pos, counts, color=colors, edgecolor='black', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(data)*100:.1f}%)',
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Number of Conversations')
    ax.set_title('Distribution of Self-Directedness Levels')
    ax.grid(axis='y', alpha=0.3)

    # Top-right: Self-directedness vs. learning quality
    ax = axes[0, 1]

    x = df['self_directedness'].dropna()
    y = df.loc[x.index, 'overall_learning_quality']

    ax.scatter(x, y, alpha=0.5, s=40, color='#118AB2')

    # Regression
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2)

    corr, p_val = stats.pearsonr(x, y)
    ax.text(0.05, 0.95, f'Pearson r={corr:.3f}\np<{p_val:.4f}',
           transform=ax.transAxes, fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Self-Directedness (1-5)')
    ax.set_ylabel('Overall Learning Quality (1-5)')
    ax.set_title('Self-Directedness vs. Learning Outcomes')
    ax.grid(alpha=0.3)

    # Bottom-left: Self-directedness vs. cognitive engagement
    ax = axes[1, 0]

    x = df['self_directedness'].dropna()
    y = df.loc[x.index, 'cognitive_engagement']

    # 2D histogram
    h = ax.hist2d(x, y, bins=10, cmap='YlOrRd', cmin=1)
    plt.colorbar(h[3], ax=ax, label='Count')

    ax.set_xlabel('Self-Directedness (1-5)')
    ax.set_ylabel('Cognitive Engagement (1-5)')
    ax.set_title('Joint Distribution:\nSelf-Directedness & Cognitive Engagement')

    # Bottom-right: Impact on memory retention
    ax = axes[1, 1]

    # Categorize self-directedness
    df_copy = df.dropna(subset=['self_directedness', 'memory_retention'])
    df_copy['self_dir_cat'] = pd.cut(df_copy['self_directedness'],
                                      bins=[0, 2.5, 3.5, 6],
                                      labels=['Low\n(Dependent)', 'Moderate\n(Balanced)',
                                             'High\n(Autonomous)'])

    data_by_cat = [df_copy[df_copy['self_dir_cat'] == label]['memory_retention'].values
                   for label in ['Low\n(Dependent)', 'Moderate\n(Balanced)', 'High\n(Autonomous)']]

    bp = ax.boxplot(data_by_cat, labels=['Low\n(Dependent)', 'Moderate\n(Balanced)', 'High\n(Autonomous)'],
                   patch_artist=True, showmeans=True)

    for patch, color in zip(bp['boxes'], ['#D62828', '#FCBF49', '#118AB2']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Memory Retention (1-5)')
    ax.set_title('Memory Retention by Self-Directedness Level')
    ax.grid(axis='y', alpha=0.3)

    # Statistical test
    if all(len(d) > 0 for d in data_by_cat):
        h_stat, p_val = stats.kruskal(*data_by_cat)
        ax.text(0.02, 0.98, f'Kruskal-Wallis\np={p_val:.4f}',
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_9_self_directedness.png", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure_9_self_directedness.pdf", bbox_inches='tight')
    print(f"  Saved: figure_9_self_directedness.png/pdf")
    plt.close()


def create_figure_10_summary():
    """Figure 10: Summary Dashboard"""
    print("Creating Figure 10: Summary Dashboard...")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Esperanto Learning with ChatGPT: Summary Dashboard\n' +
                 'Behavioral Economics Analysis (N=397 conversations, 375 participants)',
                 fontsize=15, fontweight='bold')

    # 1. Overall metrics radar chart (top-left, larger)
    ax1 = fig.add_subplot(gs[0:2, 0], projection='polar')

    categories = [m.replace('_', '\n').title() for m in EVAL_METRICS]
    values = df[EVAL_METRICS].mean().values

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values = np.concatenate((values, [values[0]]))
    angles += angles[:1]

    ax1.plot(angles, values, 'o-', linewidth=2, color='#E63946', label='Mean Score')
    ax1.fill(angles, values, alpha=0.25, color='#E63946')
    ax1.plot(angles, [3]*len(angles), '--', linewidth=1, color='gray', alpha=0.5, label='Moderate (3.0)')

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, size=7)
    ax1.set_ylim(0, 5)
    ax1.set_yticks([1, 2, 3, 4, 5])
    ax1.set_title('Overall Metric Scores\n(1-5 scale)', size=10, pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7)
    ax1.grid(True)

    # 2. Key findings text (top-middle)
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.axis('off')

    findings_text = """
    KEY FINDINGS (Behavioral Economics Perspective):

    🔴 COGNITIVE DEBT INDICATORS (MIT 2025):
       • Iterative Refinement: 1.86/5 - LOW (learners accept first answers)
       • Memory Retention: 2.09/5 - LOW (reduced cognitive encoding)
       • Metacognitive Awareness: 1.98/5 - LOW (passive learning pattern)

    🟡 LEARNING PATTERNS:
       • Learning Orientation: 2.48/5 (Declarative bias - facts over processes)
       • Esperanto Usage: 38.1% average (reliance on English translation)
       • Query Length: 6.4 words (short, minimal context queries)

    🟢 POSITIVE INDICATORS:
       • Self-Directedness: 2.70/5 (Balanced - not over-dependent on AI)
       • Cognitive Engagement: 2.64/5 (Moderate surface-level engagement)

    ⚠️  IMPLICATIONS:
       Evidence of cognitive offloading patterns consistent with MIT research.
       Suggests need for intervention to promote deeper, more active learning.
    """

    ax2.text(0.05, 0.95, findings_text, transform=ax2.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # 3. Distribution summary (middle-left)
    ax3 = fig.add_subplot(gs[1, 1])

    # Box plot of all metrics
    data_all = [df[m].dropna() for m in EVAL_METRICS[:5]]
    bp = ax3.boxplot(data_all, labels=[m.replace('_', '\n')[:10]+'...' for m in EVAL_METRICS[:5]],
                    patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('#8ECAE6')
        patch.set_alpha(0.7)

    ax3.axhline(3, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_ylabel('Score (1-5)')
    ax3.set_title('Top 5 Metrics Distribution', size=9)
    ax3.tick_params(axis='x', labelsize=7)
    ax3.grid(axis='y', alpha=0.3)

    # 4. Question types pie chart (middle-right)
    ax4 = fig.add_subplot(gs[1, 2])

    qt_means = df[QUESTION_TYPES].mean()
    labels = ['Translation', 'Grammar', 'Usage', 'Clarify', 'Apply', 'Meta']
    colors_pie = ['#E63946', '#F77F00', '#FCBF49', '#06D6A0', '#118AB2', '#073B4C']

    wedges, texts, autotexts = ax4.pie(qt_means, labels=labels, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(7)
        autotext.set_fontweight('bold')

    for text in texts:
        text.set_fontsize(7)

    ax4.set_title('Question Type\nDistribution', size=9)

    # 5. ID confidence breakdown (bottom-left)
    ax5 = fig.add_subplot(gs[2, 0])

    conf_counts = df['id_confidence'].value_counts()
    colors_conf = ['#06D6A0', '#FFD166', '#EF476F']

    bars = ax5.bar(range(len(conf_counts)), conf_counts.values,
                  color=colors_conf, edgecolor='black', alpha=0.7)

    ax5.set_xticks(range(len(conf_counts)))
    ax5.set_xticklabels([label.replace('_', '\n') for label in conf_counts.index],
                        fontsize=7, rotation=0)
    ax5.set_ylabel('Count')
    ax5.set_title('ID Confidence Breakdown', size=9)
    ax5.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=7)

    # 6. Correlation with learning quality (bottom-middle)
    ax6 = fig.add_subplot(gs[2, 1])

    corr_with_quality = df[[*EVAL_METRICS[:-1], 'overall_learning_quality']].corr()['overall_learning_quality'][:-1]
    corr_sorted = corr_with_quality.sort_values(ascending=False)

    colors_corr = ['green' if x > 0 else 'red' for x in corr_sorted]
    bars = ax6.barh(range(len(corr_sorted)), corr_sorted.values, color=colors_corr, alpha=0.7)

    ax6.set_yticks(range(len(corr_sorted)))
    ax6.set_yticklabels([m.replace('_', ' ')[:15]+'...' if len(m) > 15 else m.replace('_', ' ')
                         for m in corr_sorted.index], fontsize=7)
    ax6.set_xlabel('Correlation with Learning Quality', fontsize=8)
    ax6.set_title('Metric Correlations\nwith Overall Quality', size=9)
    ax6.axvline(0, color='black', linewidth=0.8)
    ax6.grid(axis='x', alpha=0.3)

    # 7. Sample size info (bottom-right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    stats_text = f"""
    DATASET STATISTICS:

    Total Conversations: {len(df)}
    Unique Participants: {len(participant_df)}
    Evaluation Completeness: 98%

    Session Duration:
      Mean: {df['duration_min'].mean():.1f} min
      Median: {df['duration_min'].median():.1f} min

    Messages per Conversation:
      Mean: {df['n_user_msgs'].mean():.1f}
      Median: {df['n_user_msgs'].median():.0f}

    Multiple Conversations: 15 participants

    Main Sessions: {df['is_main_session'].sum()}
    Non-Main: {(~df['is_main_session']).sum()}
    """

    ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes,
            fontsize=7, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

    plt.savefig(FIGURES_DIR / "figure_10_summary_dashboard.png", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure_10_summary_dashboard.pdf", bbox_inches='tight')
    print(f"  Saved: figure_10_summary_dashboard.png/pdf")
    plt.close()


def generate_statistics_report():
    """Generate comprehensive statistics report"""
    print("\nGenerating Statistical Analysis Report...")

    report = []
    report.append("="*80)
    report.append("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
    report.append("Esperanto Learning with ChatGPT: Behavioral Economics Perspective")
    report.append("="*80)
    report.append("")

    # 1. Descriptive Statistics
    report.append("1. DESCRIPTIVE STATISTICS")
    report.append("-" * 40)
    report.append("\nEvaluation Metrics (1-5 scale):")
    for metric in EVAL_METRICS:
        data = df[metric].dropna()
        report.append(f"\n{metric.replace('_', ' ').title()}:")
        report.append(f"  Mean:   {data.mean():.3f}")
        report.append(f"  Median: {data.median():.3f}")
        report.append(f"  SD:     {data.std():.3f}")
        report.append(f"  Min:    {data.min():.3f}")
        report.append(f"  Max:    {data.max():.3f}")
        report.append(f"  N:      {len(data)}")

    # 2. Correlation Analysis
    report.append("\n\n2. CORRELATION ANALYSIS")
    report.append("-" * 40)
    corr_matrix = df[EVAL_METRICS].corr()

    # Find strongest correlations
    report.append("\nStrongest Positive Correlations (r > 0.6):")
    for i in range(len(EVAL_METRICS)):
        for j in range(i+1, len(EVAL_METRICS)):
            r = corr_matrix.iloc[i, j]
            if r > 0.6:
                report.append(f"  {EVAL_METRICS[i]} <-> {EVAL_METRICS[j]}: r={r:.3f}")

    report.append("\nNotable Negative Correlations (r < -0.2):")
    for i in range(len(EVAL_METRICS)):
        for j in range(i+1, len(EVAL_METRICS)):
            r = corr_matrix.iloc[i, j]
            if r < -0.2:
                report.append(f"  {EVAL_METRICS[i]} <-> {EVAL_METRICS[j]}: r={r:.3f}")

    # 3. ID Confidence Impact
    report.append("\n\n3. ID CONFIDENCE IMPACT ANALYSIS")
    report.append("-" * 40)

    for metric in ['cognitive_engagement', 'overall_learning_quality', 'memory_retention']:
        report.append(f"\n{metric.replace('_', ' ').title()}:")

        high = df[df['id_confidence'] == 'high'][metric].dropna()
        medium = df[df['id_confidence'] == 'imputed_medium'][metric].dropna()
        low = df[df['id_confidence'] == 'synthetic_low'][metric].dropna()

        report.append(f"  High (n={len(high)}):   μ={high.mean():.3f}, σ={high.std():.3f}")
        report.append(f"  Medium (n={len(medium)}): μ={medium.mean():.3f}, σ={medium.std():.3f}")
        report.append(f"  Low (n={len(low)}):    μ={low.mean():.3f}, σ={low.std():.3f}")

        if len(high) > 0 and len(medium) > 0 and len(low) > 0:
            h_stat, p_val = stats.kruskal(high, medium, low)
            report.append(f"  Kruskal-Wallis H={h_stat:.3f}, p={p_val:.4f}")

    # 4. Learning Orientation Analysis
    report.append("\n\n4. LEARNING ORIENTATION ANALYSIS")
    report.append("-" * 40)

    orient_data = df['learning_orientation'].dropna()
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    labels = ['Purely Declarative', 'Mostly Declarative', 'Balanced',
              'Mostly Procedural', 'Purely Procedural']

    counts, _ = np.histogram(orient_data, bins=bins)

    report.append("\nDistribution:")
    for label, count in zip(labels, counts):
        pct = count / len(orient_data) * 100
        report.append(f"  {label}: {count} ({pct:.1f}%)")

    # 5. Cognitive Debt Indicators
    report.append("\n\n5. COGNITIVE DEBT INDICATORS (MIT 2025 Framework)")
    report.append("-" * 40)

    indicators = {
        'iterative_refinement': 'Iterative Refinement (Follow-up behavior)',
        'memory_retention': 'Memory Retention (Knowledge building)',
        'metacognitive_awareness': 'Metacognitive Awareness (Self-reflection)'
    }

    for metric, description in indicators.items():
        data = df[metric].dropna()
        below_moderate = (data < 3).sum()
        pct_below = below_moderate / len(data) * 100

        report.append(f"\n{description}:")
        report.append(f"  Mean: {data.mean():.3f} (SD={data.std():.3f})")
        report.append(f"  Below moderate (<3.0): {below_moderate}/{len(data)} ({pct_below:.1f}%)")

        if data.mean() < 2.5:
            report.append(f"  ⚠️  WARNING: Low score indicates cognitive debt pattern")

    # Save report
    report_path = BASE_DIR / "analysis" / "statistical_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"  Saved: statistical_analysis_report.txt")
    print("\n".join(report[:50]))  # Print first 50 lines
    print(f"\n... (Full report saved to {report_path})")


def main():
    """Run all analyses and create figures"""
    print("\n" + "="*80)
    print("COMPREHENSIVE DATA SCIENCE ANALYSIS")
    print("Esperanto Learning Conversations - Behavioral Economics Perspective")
    print("="*80)
    print()

    # Create all figures
    create_figure_1_overview()
    create_figure_2_correlation()
    create_figure_3_cognitive_debt()
    create_figure_4_learning_orientation()
    create_figure_5_question_types()
    create_figure_6_id_confidence()
    create_figure_7_pca_clustering()
    create_figure_8_esperanto_usage()
    create_figure_9_self_directedness()
    create_figure_10_summary()

    # Generate statistics report
    generate_statistics_report()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nGenerated 10 figures + 1 statistical report")
    print(f"Output directory: {FIGURES_DIR}")
    print(f"\nFiles created:")
    for f in sorted(FIGURES_DIR.glob("*")):
        print(f"  - {f.name}")
    print()


if __name__ == "__main__":
    main()
