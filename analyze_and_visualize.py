#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    base_dir = Path("/home/user/esperanto-db")
    df = pd.read_csv(base_dir / "conversations_evaluated.csv")
    df = df[df['error'].isna() | (df['error'] == '')]
    return df

def create_figures(df):
    base_dir = Path("/home/user/esperanto-db")
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    metrics = ["cognitive_engagement", "metacognitive_awareness", "linguistic_production",
               "self_directedness", "iterative_refinement", "memory_retention",
               "agency_ownership", "query_sophistication", "overall_learning_quality"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for idx, metric in enumerate(metrics):
        if metric in df.columns:
            axes[idx].hist(df[metric].dropna(), bins=20, edgecolor='black', alpha=0.7)
            axes[idx].set_title(metric.replace('_', ' ').title())
            axes[idx].set_xlabel('Score')
            axes[idx].set_ylabel('Frequency')
            axes[idx].axvline(df[metric].mean(), color='red', linestyle='--', linewidth=2)
    plt.tight_layout()
    plt.savefig(fig_dir / "metric_distributions.png", dpi=300, bbox_inches='tight')
    plt.savefig(fig_dir / "metric_distributions.pdf", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    corr_metrics = [m for m in metrics if m in df.columns]
    corr_matrix = df[corr_metrics].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    plt.title('Metric Correlations')
    plt.tight_layout()
    plt.savefig(fig_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.savefig(fig_dir / "correlation_matrix.pdf", bbox_inches='tight')
    plt.close()

    debt_metrics = ["iterative_refinement", "memory_retention", "metacognitive_awareness"]
    fig, ax = plt.subplots(figsize=(10, 6))
    df[debt_metrics].boxplot(ax=ax)
    ax.set_ylabel('Score (1-5)')
    ax.set_title('Cognitive Debt Indicators')
    ax.axhline(y=2.5, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "cognitive_debt.png", dpi=300, bbox_inches='tight')
    plt.savefig(fig_dir / "cognitive_debt.pdf", bbox_inches='tight')
    plt.close()

    pca_metrics = [m for m in metrics[:6] if m in df.columns]
    pca_data = df[pca_metrics].dropna()

    if len(pca_data) > 10:
        pca = PCA(n_components=2)
        components = pca.fit_transform(pca_data)

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(components)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(components[:, 0], components[:, 1], c=clusters,
                           cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('Learner Profiles (PCA + K-means)')
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.savefig(fig_dir / "learner_clusters.png", dpi=300, bbox_inches='tight')
        plt.savefig(fig_dir / "learner_clusters.pdf", bbox_inches='tight')
        plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0,0].scatter(df['cognitive_engagement'], df['overall_learning_quality'], alpha=0.5)
    axes[0,0].set_xlabel('Cognitive Engagement')
    axes[0,0].set_ylabel('Overall Learning Quality')
    if len(df) > 3:
        z = np.polyfit(df['cognitive_engagement'].dropna(),
                      df['overall_learning_quality'].dropna(), 1)
        p = np.poly1d(z)
        axes[0,0].plot(df['cognitive_engagement'].dropna().sort_values(),
                      p(df['cognitive_engagement'].dropna().sort_values()),
                      "r--", alpha=0.8)

    axes[0,1].scatter(df['self_directedness'], df['overall_learning_quality'], alpha=0.5)
    axes[0,1].set_xlabel('Self-Directedness')
    axes[0,1].set_ylabel('Overall Learning Quality')

    axes[1,0].scatter(df['linguistic_production'], df['query_sophistication'], alpha=0.5)
    axes[1,0].set_xlabel('Linguistic Production')
    axes[1,0].set_ylabel('Query Sophistication')

    axes[1,1].scatter(df['iterative_refinement'], df['memory_retention'], alpha=0.5)
    axes[1,1].set_xlabel('Iterative Refinement')
    axes[1,1].set_ylabel('Memory Retention')

    plt.tight_layout()
    plt.savefig(fig_dir / "relationships.png", dpi=300, bbox_inches='tight')
    plt.savefig(fig_dir / "relationships.pdf", bbox_inches='tight')
    plt.close()

    summary_stats = []
    for metric in metrics:
        if metric in df.columns:
            summary_stats.append({
                'Metric': metric,
                'Mean': df[metric].mean(),
                'Std': df[metric].std(),
                'Median': df[metric].median(),
                'Min': df[metric].min(),
                'Max': df[metric].max()
            })

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(fig_dir / "summary_statistics.csv", index=False)

    print(f"\nFigures saved to {fig_dir}/")
    print(f"  - metric_distributions (PNG/PDF)")
    print(f"  - correlation_matrix (PNG/PDF)")
    print(f"  - cognitive_debt (PNG/PDF)")
    print(f"  - learner_clusters (PNG/PDF)")
    print(f"  - relationships (PNG/PDF)")
    print(f"  - summary_statistics.csv")

def main():
    print("Loading evaluation data...")
    df = load_data()
    print(f"Loaded {len(df)} valid conversations")

    print("\nGenerating figures...")
    create_figures(df)

    print("\nAnalysis complete")

if __name__ == "__main__":
    main()
