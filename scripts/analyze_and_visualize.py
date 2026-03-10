#!/usr/bin/env python3
"""
Comprehensive analysis and visualization pipeline for the Esperanto-DB project.
Uses ONLY explicit evaluation data. Generates 10 publication-quality figure sets.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).parent.parent.resolve()
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

COMPOSITE_METRICS = [
    "cognitive_engagement", "metacognitive_awareness", "linguistic_production",
    "self_directedness", "iterative_refinement", "memory_retention",
    "agency_ownership", "query_sophistication", "overall_learning_quality",
]

AI_METRICS = ["copypaste_dummy", "substitute_learning", "complement_learning"]

ALL_METRICS = COMPOSITE_METRICS + AI_METRICS


def load_data():
    """Load explicit eval data and merge with participant metadata."""
    eval_df = pd.read_csv(BASE_DIR / "conversations_evaluated_explicit.csv")
    # Filter out errors
    if "error" in eval_df.columns:
        eval_df = eval_df[eval_df["error"].isna() | (eval_df["error"] == "")]

    # Ensure numeric
    for col in ALL_METRICS + ["n_messages"]:
        if col in eval_df.columns:
            eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")

    # Load participant data for merge
    part_df = pd.read_csv(BASE_DIR / "data" / "01_full_sample_with_prompts.csv",
                          low_memory=False, on_bad_lines="skip")
    merge_cols = ["conversation_id"]
    keep_cols = ["conversation_id", "sum_msg", "testscore", "nb_practice_questions",
                 "gender", "gpa", "highgpa", "treatment"]
    available = [c for c in keep_cols if c in part_df.columns]
    part_sub = part_df[available].copy()
    for col in ["sum_msg", "testscore", "nb_practice_questions", "gender", "gpa", "highgpa"]:
        if col in part_sub.columns:
            part_sub[col] = pd.to_numeric(part_sub[col], errors="coerce")

    # Merge
    if "conversation_id" in eval_df.columns and "conversation_id" in part_sub.columns:
        merged = eval_df.merge(part_sub, on="conversation_id", how="left", suffixes=("", "_part"))
    else:
        merged = eval_df

    print(f"Loaded {len(merged)} conversations (explicit eval)")
    print(f"  Merged participant data: {merged['sum_msg'].notna().sum()} with sum_msg")
    return merged


def annotate_regression(ax, x, y, color="red"):
    """Add regression line and r/p annotation to a scatter axis."""
    mask = x.notna() & y.notna()
    xc, yc = x[mask], y[mask]
    if len(xc) < 5:
        return
    slope, intercept, r_val, p_val, _ = stats.linregress(xc, yc)
    x_line = np.linspace(xc.min(), xc.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2, alpha=0.8)
    p_str = f"p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
    ax.annotate(f"r={r_val:.2f}, {p_str}", xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=10, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


def save_fig(name):
    """Save figure as both PNG and PDF."""
    plt.savefig(FIG_DIR / f"{name}.png")
    plt.savefig(FIG_DIR / f"{name}.pdf")
    plt.close()
    print(f"  Saved {name}")


# ── Figure 1: Substitute vs Complement Scatter ──────────────────────────────

def fig1_substitute_vs_complement(df):
    fig, ax = plt.subplots(figsize=(9, 7))
    sub = df.dropna(subset=["substitute_learning", "complement_learning", "copypaste_dummy"])
    scatter = ax.scatter(
        sub["substitute_learning"], sub["complement_learning"],
        c=sub["copypaste_dummy"], cmap="RdYlGn_r", s=60, alpha=0.7,
        edgecolors="grey", linewidth=0.5, vmin=1, vmax=5,
    )
    annotate_regression(ax, sub["substitute_learning"], sub["complement_learning"], color="darkred")
    cbar = plt.colorbar(scatter, ax=ax, label="Copy-Paste Score (1-5)")
    cbar.set_ticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Substitute Learning (1=Low Reliance, 5=High Reliance)")
    ax.set_ylabel("Complement Learning (1=Low Augmentation, 5=High Augmentation)")
    ax.set_title('The "Black Box" Effect: Substitute vs Complement Learning')
    save_fig("fig1_substitute_vs_complement")


# ── Figure 2: Copy-Paste Impact Bar Chart ────────────────────────────────────

def fig2_copypaste_impact_bar(df):
    sub = df.dropna(subset=["copypaste_dummy", "substitute_learning", "complement_learning"]).copy()
    sub["cp_level"] = sub["copypaste_dummy"].round().astype(int).clip(1, 5)

    agg = sub.groupby("cp_level")[["substitute_learning", "complement_learning"]].agg(["mean", "sem"]).reset_index()
    agg.columns = ["cp_level", "sub_mean", "sub_sem", "comp_mean", "comp_sem"]

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(agg))
    w = 0.35
    ax.bar(x - w/2, agg["sub_mean"], w, yerr=agg["sub_sem"], label="Substitute Learning",
           color="#e74c3c", capsize=4, alpha=0.85)
    ax.bar(x + w/2, agg["comp_mean"], w, yerr=agg["comp_sem"], label="Complement Learning",
           color="#4facfe", capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["cp_level"])
    ax.set_xlabel("Copy-Paste Severity (1=Original, 5=Verbatim Copying)")
    ax.set_ylabel("Average Score (1-5)")
    ax.set_ylim(0, 5.5)
    ax.set_title("How Copy-Pasting Relates to AI Usage Strategy")
    ax.legend(loc="upper left")
    save_fig("fig2_copypaste_impact_bar")


# ── Figure 3: Messages vs Learning Outcomes (paper-style) ───────────────────

def fig3_messages_vs_learning(df):
    sub = df.dropna(subset=["sum_msg"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # (a) sum_msg vs nb_practice_questions
    s1 = sub.dropna(subset=["nb_practice_questions"])
    ax1.scatter(s1["sum_msg"], s1["nb_practice_questions"], alpha=0.5, s=40, color="#2c3e50")
    annotate_regression(ax1, s1["sum_msg"], s1["nb_practice_questions"], color="darkred")
    ax1.set_xlabel("Total Messages (sum_msg)")
    ax1.set_ylabel("Practice Questions Attempted")
    ax1.set_title("(a) Prompts and Practice Questions")

    # (b) sum_msg vs testscore
    s2 = sub.dropna(subset=["testscore"])
    ax2.scatter(s2["sum_msg"], s2["testscore"], alpha=0.5, s=40, color="#2c3e50")
    annotate_regression(ax2, s2["sum_msg"], s2["testscore"], color="darkred")
    ax2.set_xlabel("Total Messages (sum_msg)")
    ax2.set_ylabel("Exam Score")
    ax2.set_title("(b) Prompts and Exam Score")

    plt.tight_layout()
    save_fig("fig3_messages_vs_learning")


# ── Figure 4: Metric Distributions ──────────────────────────────────────────

def fig4_metric_distributions(df):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for idx, metric in enumerate(COMPOSITE_METRICS):
        if metric not in df.columns:
            continue
        data = df[metric].dropna()
        axes[idx].hist(data, bins=20, edgecolor="white", alpha=0.8, color="#3498db")
        axes[idx].axvline(data.mean(), color="red", linestyle="--", linewidth=2,
                         label=f"Mean: {data.mean():.2f}")
        axes[idx].set_title(metric.replace("_", " ").title())
        axes[idx].set_xlabel("Score")
        axes[idx].set_ylabel("Frequency")
        axes[idx].legend(fontsize=9)
    plt.tight_layout()
    save_fig("fig4_metric_distributions")


# ── Figure 5: Correlation Matrix ─────────────────────────────────────────────

def fig5_correlation_matrix(df):
    cols = [c for c in ALL_METRICS if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, mask=mask,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                vmin=-1, vmax=1)
    ax.set_title("Metric Correlations (Explicit Evaluation)")
    labels = [c.replace("_", "\n") for c in cols]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, rotation=0, fontsize=9)
    save_fig("fig5_correlation_matrix")


# ── Figure 6: Learner Clusters (PCA + K-means) ──────────────────────────────

def fig6_learner_clusters(df):
    cluster_cols = [c for c in COMPOSITE_METRICS[:6] if c in df.columns]
    pca_data = df[cluster_cols].dropna()
    if len(pca_data) < 20:
        print("  Skipping fig6: insufficient data for clustering")
        return

    pca = PCA(n_components=2)
    components = pca.fit_transform(pca_data)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(components)

    cluster_names = {0: "Low Engagers", 1: "Moderate Balanced", 2: "High Agency", 3: "Production Focused"}

    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(4):
        mask = clusters == i
        ax.scatter(components[mask, 0], components[mask, 1],
                  label=cluster_names.get(i, f"Cluster {i}"),
                  alpha=0.6, s=50, edgecolors="grey", linewidth=0.3)
    # Plot centroids (already in PCA space since KMeans was fit on components)
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=200, c="black",
              edgecolors="white", linewidth=2, zorder=5, label="Centroids")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Learner Profiles (PCA + K-means Clustering)")
    ax.legend(loc="best")
    save_fig("fig6_learner_clusters")

    # Print cluster profiles
    pca_data_c = pca_data.copy()
    pca_data_c["cluster"] = clusters
    print("  Cluster profiles:")
    print(pca_data_c.groupby("cluster")[cluster_cols].mean().round(2).to_string())


# ── Figure 7: Cognitive Debt Boxplot ─────────────────────────────────────────

def fig7_cognitive_debt(df):
    debt_cols = ["iterative_refinement", "memory_retention", "metacognitive_awareness"]
    available = [c for c in debt_cols if c in df.columns]
    fig, ax = plt.subplots(figsize=(8, 6))
    box_data = [df[c].dropna() for c in available]
    bp = ax.boxplot(box_data, patch_artist=True, labels=[c.replace("_", "\n") for c in available])
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(y=2.5, color="grey", linestyle="--", alpha=0.5, label="Midpoint (2.5)")
    ax.set_ylabel("Score (1-5)")
    ax.set_title("Cognitive Debt Indicators")
    ax.legend()
    save_fig("fig7_cognitive_debt")


# ── Figure 8: Subgroup KDE Plots ────────────────────────────────────────────

def fig8_subgroup_kde(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Overall learning quality by gender
    ax = axes[0, 0]
    gender_map = {1: "Male", 2: "Female"}
    for g, label in gender_map.items():
        sub = df[df["gender"] == g]["overall_learning_quality"].dropna()
        if len(sub) > 5:
            sub.plot.kde(ax=ax, label=f"{label} (n={len(sub)})", linewidth=2)
    ax.set_title("(a) Learning Quality by Gender")
    ax.set_xlabel("Overall Learning Quality")
    ax.legend()
    ax.set_xlim(0, 5)

    # (b) Overall learning quality by GPA
    ax = axes[0, 1]
    for g, label in [(0, "Low GPA"), (1, "High GPA")]:
        sub = df[df["highgpa"] == g]["overall_learning_quality"].dropna()
        if len(sub) > 5:
            sub.plot.kde(ax=ax, label=f"{label} (n={len(sub)})", linewidth=2)
    ax.set_title("(b) Learning Quality by GPA Level")
    ax.set_xlabel("Overall Learning Quality")
    ax.legend()
    ax.set_xlim(0, 5)

    # (c) Complement learning by treatment
    ax = axes[1, 0]
    for t in ["assist", "guided", "control"]:
        sub = df[df["treatment"] == t]["complement_learning"].dropna()
        if len(sub) > 5:
            sub.plot.kde(ax=ax, label=f"{t.title()} (n={len(sub)})", linewidth=2)
    ax.set_title("(c) Complement Learning by Treatment")
    ax.set_xlabel("Complement Learning Score")
    ax.legend()
    ax.set_xlim(0, 5)

    # (d) Substitute learning by treatment
    ax = axes[1, 1]
    for t in ["assist", "guided", "control"]:
        sub = df[df["treatment"] == t]["substitute_learning"].dropna()
        if len(sub) > 5:
            sub.plot.kde(ax=ax, label=f"{t.title()} (n={len(sub)})", linewidth=2)
    ax.set_title("(d) Substitute Learning by Treatment")
    ax.set_xlabel("Substitute Learning Score")
    ax.legend()
    ax.set_xlim(0, 5)

    plt.tight_layout()
    save_fig("fig8_subgroup_kde")


# ── Figure 9: Relationship Scatter Panels ────────────────────────────────────

def fig9_relationships(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    pairs = [
        ("cognitive_engagement", "overall_learning_quality", "Cognitive Engagement", "Overall Learning Quality"),
        ("self_directedness", "overall_learning_quality", "Self-Directedness", "Overall Learning Quality"),
        ("copypaste_dummy", "complement_learning", "Copy-Paste Score", "Complement Learning"),
        ("n_messages", "cognitive_engagement", "Number of Messages", "Cognitive Engagement"),
    ]
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.flatten(), pairs):
        if xcol not in df.columns or ycol not in df.columns:
            continue
        sub = df[[xcol, ycol]].dropna()
        ax.scatter(sub[xcol], sub[ycol], alpha=0.4, s=30, color="#2c3e50")
        annotate_regression(ax, sub[xcol], sub[ycol])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    plt.tight_layout()
    save_fig("fig9_relationships")


# ── Figure 10: Heterogeneous Effects ─────────────────────────────────────────

def fig10_heterogeneous_effects(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) testscore by gender × treatment
    ax = axes[0]
    gender_map = {1: "Male", 2: "Female"}
    treatments = ["assist", "guided"]
    colors = {"Male": "#3498db", "Female": "#e74c3c"}
    styles = {"assist": "-", "guided": "--"}
    for g, glabel in gender_map.items():
        for t in treatments:
            sub = df[(df["gender"] == g) & (df["treatment"] == t)]["testscore"].dropna()
            if len(sub) > 5:
                sub.plot.kde(ax=ax, label=f"{glabel} × {t.title()} (n={len(sub)})",
                           linewidth=2, color=colors[glabel],
                           linestyle=styles[t])
    ax.set_title("(a) Exam Score by Gender × Treatment")
    ax.set_xlabel("Exam Score")
    ax.legend(fontsize=9)

    # (b) complement_learning by GPA × treatment
    ax = axes[1]
    gpa_map = {0: "Low GPA", 1: "High GPA"}
    colors_gpa = {"Low GPA": "#e67e22", "High GPA": "#27ae60"}
    for g, glabel in gpa_map.items():
        for t in treatments:
            sub = df[(df["highgpa"] == g) & (df["treatment"] == t)]["complement_learning"].dropna()
            if len(sub) > 5:
                sub.plot.kde(ax=ax, label=f"{glabel} × {t.title()} (n={len(sub)})",
                           linewidth=2, color=colors_gpa[glabel],
                           linestyle=styles[t])
    ax.set_title("(b) Complement Learning by GPA × Treatment")
    ax.set_xlabel("Complement Learning Score")
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig("fig10_heterogeneous_effects")


# ── Summary Statistics ───────────────────────────────────────────────────────

def generate_summary(df):
    """Generate summary statistics CSV."""
    rows = []
    for metric in ALL_METRICS:
        if metric not in df.columns:
            continue
        data = df[metric].dropna()
        rows.append({
            "Metric": metric,
            "N": len(data),
            "Mean": round(data.mean(), 2),
            "Std": round(data.std(), 2),
            "Median": round(data.median(), 2),
            "Min": round(data.min(), 2),
            "Max": round(data.max(), 2),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(FIG_DIR / "summary_statistics.csv", index=False)
    print("\n  Summary statistics:")
    print(summary.to_string(index=False))

    # Print key correlations
    print("\n  Key correlations:")
    corr_pairs = [
        ("copypaste_dummy", "complement_learning"),
        ("copypaste_dummy", "substitute_learning"),
        ("cognitive_engagement", "overall_learning_quality"),
        ("agency_ownership", "self_directedness"),
        ("substitute_learning", "complement_learning"),
    ]
    for c1, c2 in corr_pairs:
        if c1 in df.columns and c2 in df.columns:
            sub = df[[c1, c2]].dropna()
            if len(sub) > 5:
                r, p = stats.pearsonr(sub[c1], sub[c2])
                print(f"    {c1} <-> {c2}: r={r:.3f}, p={p:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Esperanto-DB: Comprehensive Analysis (Explicit Eval Only)")
    print("=" * 60)

    df = load_data()

    print("\nGenerating figures...")
    fig1_substitute_vs_complement(df)
    fig2_copypaste_impact_bar(df)
    fig3_messages_vs_learning(df)
    fig4_metric_distributions(df)
    fig5_correlation_matrix(df)
    fig6_learner_clusters(df)
    fig7_cognitive_debt(df)
    fig8_subgroup_kde(df)
    fig9_relationships(df)
    fig10_heterogeneous_effects(df)

    generate_summary(df)

    print(f"\nAll figures saved to {FIG_DIR}/")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
