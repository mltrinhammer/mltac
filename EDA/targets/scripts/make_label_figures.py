from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(r"C:/Users/anec/OneDrive - Syddansk Universitet/Projects/PinSoRo")
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = PROJECT_ROOT / "figures"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig1_sessions_by_split() -> None:
    summary = pd.read_csv(OUTPUT_DIR / "label_summary_by_session.csv")
    counts = summary[["split", "session_id"]].drop_duplicates().groupby("split").size().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["#3B82F6", "#10B981"])
    ax.set_title("Sessions by Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Session Count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_01_sessions_by_split.png", dpi=160)
    plt.close(fig)


def fig2_agreement_rate() -> None:
    summary = pd.read_csv(OUTPUT_DIR / "label_summary_by_session.csv")
    grouped = (
        summary.groupby(["split", "color", "task"], as_index=False)[["total_rows", "nonblank_rows"]]
        .sum()
    )
    grouped["agreement_rate"] = grouped["nonblank_rows"] / grouped["total_rows"]
    grouped["group"] = grouped["split"] + " | " + grouped["color"] + " | " + grouped["task"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(grouped["group"], grouped["agreement_rate"], color="#2563EB")
    ax.set_title("Agreement Rate by Split/Color/Task")
    ax.set_xlabel("Group")
    ax.set_ylabel("Agreement Rate")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_02_agreement_rate.png", dpi=160)
    plt.close(fig)


def _plot_distribution(task: str, out_name: str) -> None:
    agg = pd.read_csv(OUTPUT_DIR / "label_aggregate_main_counts.csv")
    data = agg[agg["task"] == task].copy()
    data["group"] = data["split"] + " | " + data["color"]
    piv = data.pivot_table(index="group", columns="label", values="count", aggfunc="sum", fill_value=0)
    proportions = piv.div(piv.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = None
    for col in proportions.columns:
        values = proportions[col].values
        ax.bar(proportions.index, values, bottom=bottom, label=col)
        if bottom is None:
            bottom = values
        else:
            bottom = bottom + values
    ax.set_title(f"{task} Distribution by Split/Color")
    ax.set_xlabel("Group")
    ax.set_ylabel("Proportion")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=160)
    plt.close(fig)


def fig3_task_distribution() -> None:
    _plot_distribution("task_engagement", "figure_03_task_distribution.png")


def fig4_social_distribution() -> None:
    _plot_distribution("social_engagement", "figure_04_social_distribution.png")


def fig5_session_heatmap(task: str, out_name: str) -> None:
    by_session = pd.read_csv(OUTPUT_DIR / "label_class_counts_by_session.csv")
    data = by_session[(by_session["task"] == task) & (by_session["color"] == "purple")].copy()
    data["split_session"] = data["split"].astype(str) + "_" + data["session_id"].astype(str)
    piv = data.pivot_table(index="split_session", columns="label", values="count", aggfunc="sum", fill_value=0)
    norm = piv.div(piv.sum(axis=1), axis=0).fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(norm))))
    im = ax.imshow(norm.values, aspect="auto")
    ax.set_title(f"Session-Level Label Mix (purple, {task})")
    ax.set_xlabel("Label")
    ax.set_ylabel("Session")
    ax.set_xticks(range(len(norm.columns)))
    ax.set_xticklabels(norm.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(norm.index)))
    ax.set_yticklabels(norm.index, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Proportion")
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=160)
    plt.close(fig)


def fig6_numeric_coverage() -> None:
    data = pd.read_csv(OUTPUT_DIR / "label_numeric_annotation_files.csv")
    grouped = data.groupby(["split", "color", "task"], as_index=False).size()
    grouped["group"] = grouped["split"] + " | " + grouped["color"] + " | " + grouped["task"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(grouped["group"], grouped["size"], color="#F59E0B")
    ax.set_title("Numbered Annotator File Coverage")
    ax.set_xlabel("Group")
    ax.set_ylabel("File Count")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_06_numbered_annotator_coverage.png", dpi=160)
    plt.close(fig)


def fig7_class_weights() -> None:
    data = pd.read_csv(OUTPUT_DIR / "label_scenario_class_weights.csv")
    subset = data[(data["scenario"] == "drop_blank_and_nan") & (data["color"] == "purple")].copy()
    subset["group"] = subset["split"] + " | " + subset["task"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for group_name, group_df in subset.groupby("group"):
        ax.plot(group_df["label"], group_df["weight"], marker="o", label=group_name)
    ax.set_title("Class Weights (drop_blank_and_nan, purple)")
    ax.set_xlabel("Label")
    ax.set_ylabel("Weight")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_07_class_weights_purple.png", dpi=160)
    plt.close(fig)


def fig8_purple_cc_vs_cr() -> None:
    data = pd.read_csv(OUTPUT_DIR / "purple_cc_vs_cr_comparison.csv")
    subset = data[data["scenario"] == "drop_blank_and_nan"].copy()
    for task in ("task_engagement", "social_engagement"):
        d = subset[subset["task"] == task].copy()
        fig, ax = plt.subplots(figsize=(9, 4))
        x = range(len(d))
        w = 0.4
        ax.bar([i - w / 2 for i in x], d["cc_proportion"], width=w, label="cc")
        ax.bar([i + w / 2 for i in x], d["cr_proportion"], width=w, label="cr")
        # Place labels under the left bar of each pair so text aligns with a single bar.
        ax.set_xticks([i - w / 2 for i in x])
        ax.set_xticklabels(d["label"], rotation=25, ha="right")
        ax.set_title(f"Purple CC vs CR Proportions ({task}, drop_blank_and_nan)")
        ax.set_xlabel("Label")
        ax.set_ylabel("Proportion")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"figure_08_purple_cc_vs_cr_{task}.png", dpi=160)
        plt.close(fig)


def fig9_blank_rate_heatmap() -> None:
    data = pd.read_csv(OUTPUT_DIR / "disagreement_blank_summary_by_session.csv")
    d = data[data["split"].isin(["train-cc", "train-cr"])].copy()
    d["group"] = d["split"] + " | " + d["color"] + " | " + d["task"]
    d["session"] = d["split"] + "_" + d["session_id"].astype(str)
    piv = d.pivot_table(index="session", columns="group", values="blank_rate", aggfunc="mean", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(piv.index))))
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_title("Blank Rate by Session and Group")
    ax.set_xlabel("Group")
    ax.set_ylabel("Session")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Blank Rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_09_blank_rate_heatmap.png", dpi=160)
    plt.close(fig)


def fig10_stream_coverage_matrix() -> None:
    inv = pd.read_csv(OUTPUT_DIR / "stream_inventory_by_session.csv")
    d = inv[inv["split"].isin(["train-cc", "train-cr"])].copy()
    d["session"] = d["split"] + "_" + d["session_id"].astype(str)
    d["feature_entity"] = d["entity"] + "." + d["feature"]
    d["present"] = d["has_binary_stream"].astype(int)
    piv = d.pivot_table(
        index="session",
        columns="feature_entity",
        values="present",
        aggfunc="max",
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(14, max(5, 0.28 * len(piv.index))))
    im = ax.imshow(piv.values, aspect="auto", vmin=0, vmax=1)
    ax.set_title("Stream Coverage Matrix (Train Splits)")
    ax.set_xlabel("Entity.Feature")
    ax.set_ylabel("Session")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=65, ha="right", fontsize=7)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Presence (0/1)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_10_stream_coverage_matrix.png", dpi=160)
    plt.close(fig)


def fig11_nan_rate_heatmap() -> None:
    data = pd.read_csv(OUTPUT_DIR / "nan_summary_by_session.csv")
    d = data[data["split"].isin(["train-cc", "train-cr"])].copy()
    d["group"] = d["split"] + " | " + d["color"] + " | " + d["task"]
    d["session"] = d["split"] + "_" + d["session_id"].astype(str)
    piv = d.pivot_table(index="session", columns="group", values="nan_rate", aggfunc="mean", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(piv.index))))
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_title("NaN Rate by Session and Group")
    ax.set_xlabel("Group")
    ax.set_ylabel("Session")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="NaN Rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_11_nan_rate_heatmap.png", dpi=160)
    plt.close(fig)


def fig12_nan_rate_by_group() -> None:
    data = pd.read_csv(OUTPUT_DIR / "nan_summary_by_session.csv")
    d = data[data["split"].isin(["train-cc", "train-cr"])].copy()
    grouped = (
        d.groupby(["split", "color", "task"], as_index=False)[["total_rows", "nan_rows"]]
        .sum()
    )
    grouped["nan_rate"] = grouped["nan_rows"] / grouped["total_rows"]
    grouped["group"] = grouped["split"] + " | " + grouped["color"] + " | " + grouped["task"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(grouped["group"], grouped["nan_rate"], color="#A855F7")
    ax.set_title("NaN Rate by Split/Color/Task")
    ax.set_xlabel("Group")
    ax.set_ylabel("NaN Rate")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_12_nan_rate_by_group.png", dpi=160)
    plt.close(fig)


def fig_temporal_01_duration_violin() -> None:
    data = pd.read_csv(OUTPUT_DIR / "label_runs_by_session.csv")
    d = data[data["split"].isin(["train-cc", "train-cr"])].copy()
    d = d[d["length_seconds"] <= d["length_seconds"].quantile(0.99)]
    d["group"] = d["split"] + " | " + d["task"] + " | " + d["label"]
    groups = []
    labels = []
    for group_name, grp in d.groupby("group"):
        groups.append(grp["length_seconds"].values)
        labels.append(group_name)
    fig, ax = plt.subplots(figsize=(12, 5))
    if groups:
        ax.violinplot(groups, showmeans=True, showmedians=False)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    ax.set_title("Run Duration Distribution (seconds)")
    ax.set_ylabel("Run Duration (s)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_temporal_01_duration_violin.png", dpi=160)
    plt.close(fig)


def fig_temporal_02_run_count_vs_total_time() -> None:
    data = pd.read_csv(OUTPUT_DIR / "label_run_summary_by_label.csv")
    d = data[data["split"].isin(["train-cc", "train-cr"])].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for split in sorted(d["split"].unique()):
        ds = d[d["split"] == split]
        ax.scatter(ds["n_runs"], ds["total_seconds"], label=split, alpha=0.8)
        for _, r in ds.iterrows():
            ax.text(r["n_runs"], r["total_seconds"], f"{r['task'][:4]}:{r['label']}", fontsize=6)
    ax.set_title("Run Count vs Total Label Time")
    ax.set_xlabel("Number of Runs")
    ax.set_ylabel("Total Seconds")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_temporal_02_run_count_vs_total_time.png", dpi=160)
    plt.close(fig)


def _temporal_transition_heatmap(task: str, out_name: str) -> None:
    data = pd.read_csv(OUTPUT_DIR / "label_transition_probs.csv")
    d = data[(data["task"] == task) & (data["split"].isin(["train-cc", "train-cr"]))].copy()
    labels = sorted(set(d["from_label"]).union(set(d["to_label"])))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, split in enumerate(["train-cc", "train-cr"]):
        ds = d[d["split"] == split]
        piv = ds.pivot_table(index="from_label", columns="to_label", values="prob", aggfunc="sum", fill_value=0.0)
        piv = piv.reindex(index=labels, columns=labels, fill_value=0.0)
        im = axes[i].imshow(piv.values, aspect="auto")
        axes[i].set_title(split)
        axes[i].set_xlabel("To Label")
        axes[i].set_xticks(range(len(labels)))
        axes[i].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axes[i].set_yticks(range(len(labels)))
        axes[i].set_yticklabels(labels, fontsize=7)
        axes[i].set_ylabel("From Label")
    fig.suptitle(f"Transition Probability Heatmap ({task})")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02, label="Probability")
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=160)
    plt.close(fig)


def fig_temporal_03_transition_heatmap_task() -> None:
    _temporal_transition_heatmap("task_engagement", "fig_temporal_03_transition_heatmap_task.png")


def fig_temporal_04_transition_heatmap_social() -> None:
    _temporal_transition_heatmap("social_engagement", "fig_temporal_04_transition_heatmap_social.png")


def fig_temporal_05_blank_gap_context() -> None:
    data = pd.read_csv(OUTPUT_DIR / "label_transition_with_gap_context.csv")
    d = data[data["split"].isin(["train-cc", "train-cr"])].copy()
    d["pair"] = d["left_label"] + " -> " + d["right_label"]
    top = d["pair"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top.index, top.values, color="#EF4444")
    ax.set_title("Most Common Transition Pairs Across Removed Gaps")
    ax.set_xlabel("Label Pair Across Gap")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=55, labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_temporal_05_blank_gap_context.png", dpi=160)
    plt.close(fig)


def fig_temporal_06_persistence_curves() -> None:
    data = pd.read_csv(OUTPUT_DIR / "label_persistence_curve.csv")
    d = data[data["split"].isin(["train-cc", "train-cr"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for i, task in enumerate(["task_engagement", "social_engagement"]):
        ds = d[d["task"] == task]
        for (split, label), grp in ds.groupby(["split", "label"]):
            axes[i].plot(grp["t_seconds"], grp["survival_prob"], marker="o", label=f"{split}:{label}")
        axes[i].set_title(task)
        axes[i].set_xlabel("t (seconds)")
        axes[i].set_ylabel("P(run >= t)")
        axes[i].grid(alpha=0.3)
    axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle("Label Persistence Curves")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_temporal_06_persistence_curves.png", dpi=160)
    plt.close(fig)


def fig_temporal_07_session_heterogeneity() -> None:
    data = pd.read_csv(OUTPUT_DIR / "label_temporal_heterogeneity.csv")
    d = data[data["split"].isin(["train-cc", "train-cr"])].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    for split in ["train-cc", "train-cr"]:
        ds = d[d["split"] == split]
        ax.scatter(ds["transition_rate_per_min"], ds["mean_run_sec"], label=split, alpha=0.8)
    ax.set_title("Session Temporal Heterogeneity")
    ax.set_xlabel("Transition Rate per Minute")
    ax.set_ylabel("Mean Run Duration (s)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_temporal_07_session_heterogeneity.png", dpi=160)
    plt.close(fig)


def fig13_disagreement_blank_vote_rate_heatmap() -> None:
    data = pd.read_csv(OUTPUT_DIR / "annotator_disagreement_session_summary.csv")
    if data.empty:
        return
    d = data.copy()
    d["group"] = d["split"] + " | " + d["color"] + " | " + d["task"]
    d["session"] = d["split"] + "_" + d["session_id"].astype(str)
    piv = d.pivot_table(index="session", columns="group", values="blank_vote_rate", aggfunc="mean", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(piv.index))))
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_title("Blank Frames With Annotator Votes (Rate)")
    ax.set_xlabel("Group")
    ax.set_ylabel("Session")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_13_disagreement_blank_vote_rate_heatmap.png", dpi=160)
    plt.close(fig)


def fig14_disagreement_top_signatures() -> None:
    data = pd.read_csv(OUTPUT_DIR / "annotator_disagreement_signature_counts.csv")
    if data.empty:
        return
    d = data.copy()
    d["group"] = d["split"] + " | " + d["color"] + " | " + d["task"]
    d = d.sort_values("count", ascending=False)
    top = d.head(20).copy()
    top["label"] = top["group"] + " :: " + top["vote_signature"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(top["label"], top["count"], color="#F97316")
    ax.set_title("Top Annotator Disagreement Signatures")
    ax.set_xlabel("Group and Vote Signature")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=70, labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_14_disagreement_top_signatures.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    fig1_sessions_by_split()
    fig2_agreement_rate()
    fig3_task_distribution()
    fig4_social_distribution()
    fig5_session_heatmap("task_engagement", "figure_05_session_heatmap_task.png")
    fig5_session_heatmap("social_engagement", "figure_05_session_heatmap_social.png")
    fig6_numeric_coverage()
    fig7_class_weights()
    fig8_purple_cc_vs_cr()
    fig9_blank_rate_heatmap()
    fig10_stream_coverage_matrix()
    fig11_nan_rate_heatmap()
    fig12_nan_rate_by_group()
    fig_temporal_01_duration_violin()
    fig_temporal_02_run_count_vs_total_time()
    fig_temporal_03_transition_heatmap_task()
    fig_temporal_04_transition_heatmap_social()
    fig_temporal_05_blank_gap_context()
    fig_temporal_06_persistence_curves()
    fig_temporal_07_session_heterogeneity()
    fig13_disagreement_blank_vote_rate_heatmap()
    fig14_disagreement_top_signatures()
    print(f"Wrote figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
