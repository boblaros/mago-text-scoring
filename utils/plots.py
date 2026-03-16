
from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.metrics import confusion_matrix

from .data import slugify
from .metrics import get_top_models_df

CFG: dict = {}
COLORS: list = []
external_eval_datasets: dict = {}



def display_saved_plot(plot_path: Path, title: str | None = None, figsize=(10, 6)) -> bool:
    plot_path = Path(plot_path)
    if not plot_path.exists():
        return False
    try:
        image = plt.imread(plot_path)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image)
        ax.axis('off')
        if title:
            ax.set_title(title)
        plt.tight_layout()
        plt.show()
        return True
    except Exception as exc:
        print(f'Could not display existing plot {plot_path}: {exc}')
        return False


def plot_model_comparison(results: dict, metric: str = 'f1_macro', split: str = 'test'):
    filtered = {
        key.replace(f' | {split}', ''): value[metric]
        for key, value in results.items()
        if split in key and metric in value
    }
    if not filtered:
        print(f"No results for split='{split}', metric='{metric}'.")
        return

    out_path = CFG['output_paths']['plots_comparison'] / f"comparison_{metric}_{split}.png"
    if display_saved_plot(out_path, title=f"Model Comparison ({metric}, {split})"):
        print(f'Displayed existing plot: {out_path}')
        return

    names = list(filtered.keys())
    values = list(filtered.values())
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.55)))
    bars = ax.barh([names[i] for i in order], [values[i] for i in order], color=sns.color_palette('viridis', len(names)))
    for bar, value in zip(bars, [values[i] for i in order]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2, f'{value:.4f}', va='center', fontsize=9)
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(f"Model Comparison — {metric} ({split.upper()} set)")
    ax.set_xlim(0, min(1.05, max(values) * 1.12))
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.show()


def plot_top_model_comparison(results: dict, metric: str = 'f1_macro', split: str = 'test', top_n: int = 10, force_replot: bool = True):
    top_df = get_top_models_df(results, split=split, metric=metric, top_n=top_n)
    if top_df.empty:
        print(f"No results for split='{split}', metric='{metric}'.")
        return top_df

    out_path = CFG['output_paths']['plots_comparison'] / f"comparison_top{top_n}_{metric}_{split}.png"
    if (not force_replot) and display_saved_plot(out_path, title=f"Top-{top_n} Model Comparison ({metric}, {split})"):
        print(f'Displayed existing plot: {out_path}')
        return top_df

    plot_df = top_df.sort_values(metric, ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(plot_df['model'], plot_df[metric], color=sns.color_palette('viridis', len(plot_df)))
    for bar, value in zip(bars, plot_df[metric]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2, f'{value:.4f}', va='center', fontsize=9)
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(f"Top-{top_n} Model Comparison — {metric} ({split.upper()} set)")
    ax.set_xlim(0, min(1.05, max(plot_df[metric].max() * 1.12, 0.1)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.show()
    return top_df


def plot_confusion_matrix_clf(y_true, y_pred, model_name: str, label_encoder=None):
    fname = CFG["output_paths"]["plots_confusion"] / f"cm_{slugify(model_name)}.png"

    y_true = np.asarray([] if y_true is None else y_true)
    y_pred = np.asarray([] if y_pred is None else y_pred)

    if y_true.size == 0 or y_pred.size != y_true.size:
        if display_saved_plot(fname, title=f"Confusion Matrix — {model_name}", figsize=(13, 5)):
            print(f"Displayed existing confusion matrix: {fname}")
            return

        print(
            f"Confusion matrix unavailable for {model_name}: "
            f"expected {len(y_true)} predictions, got {len(y_pred)}."
        )
        return

    if display_saved_plot(fname, title=f"Confusion Matrix — {model_name}", figsize=(13, 5)):
        print(f"Displayed existing confusion matrix: {fname}")
        return

    cm = confusion_matrix(y_true, y_pred)
    labels = label_encoder.classes_ if label_encoder else np.unique(y_true)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, title, fmt in zip(
        axes, [cm, cm_norm], ["Raw Counts", "Normalized"], ["d", ".2f"]
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap=globals().get("CMAP", "Blues"),
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)

    plt.suptitle(f"Confusion Matrix — {model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.show()


def plot_history(history: dict, model_name: str):
    fname = CFG["output_paths"]["plots_history"] / f"history_{slugify(model_name)}.png"
    if display_saved_plot(fname, title=model_name, figsize=(12, 4)):
        print(f"Displayed existing training curve: {fname}")
        return

    if not history or len(history.get("train_loss", [])) == 0:
        print(f"No training history to plot for {model_name}.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train", marker="o", color=COLORS[0])
    axes[0].plot(history["val_loss"], label="Val", marker="o", color=COLORS[2])
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["val_f1_macro"], marker="o", color=COLORS[1])
    axes[1].set_title("Val F1 Macro")
    axes[1].set_xlabel("Epoch")

    plt.suptitle(model_name, fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.show()


def plot_learning_curve_from_metrics(
    metrics,
    figsize=(10, 6),
    title="Learning Curve",
    save_path=None,
    x_log=True,
    force_replot=False,
):
    required_cols = {"model", "split_key", "train_size", "f1_macro"}

    if isinstance(metrics, pd.DataFrame):
        df = metrics.copy()
    else:
        df = pd.DataFrame(metrics)

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for learning curve: {sorted(missing)}")

    df = df.sort_values(["model", "train_size"]).reset_index(drop=True)

    if save_path is not None:
        save_path = Path(save_path)
        if (not force_replot) and display_saved_plot(save_path, title=title, figsize=figsize):
            print(f"Displayed existing learning curve: {save_path}")
            return df

    fig, ax = plt.subplots(figsize=figsize)

    models = df["model"].drop_duplicates().tolist()

    if len(models) <= 5 and "COLORS" in globals() and len(COLORS) >= len(models):
        palette = COLORS[: len(models)]
    else:
        # High-contrast palette so overlapping curves remain distinguishable.
        tableau10 = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        if len(models) <= len(tableau10):
            palette = tableau10[: len(models)]
        else:
            cmap = plt.get_cmap("tab20")
            palette = [cmap(i % cmap.N) for i in range(len(models))]

    color_map = {m: palette[i] for i, m in enumerate(models)}
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    linestyle_cycle = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    style_map = {
        m: {
            "marker": marker_cycle[i % len(marker_cycle)],
            "linestyle": linestyle_cycle[i % len(linestyle_cycle)],
        }
        for i, m in enumerate(models)
    }

    majority_name = "Majority Class"

    for model_name in models:
        g = df[df["model"] == model_name]
        color = color_map.get(model_name)
        is_majority = model_name.strip().lower() == majority_name.lower()

        if len(g) == 1:
            if is_majority:
                ax.scatter(
                    g["train_size"],
                    g["f1_macro"],
                    s=220,
                    marker="o",
                    color=color,
                    edgecolor="black",
                    linewidth=1.8,
                    label=model_name,
                    zorder=6,
                )
            else:
                ax.scatter(
                    g["train_size"],
                    g["f1_macro"],
                    s=95,
                    marker=style_map[model_name]["marker"],
                    color=color,
                    edgecolor="white",
                    linewidth=0.9,
                    label=model_name,
                    zorder=4,
                )
        else:
            ax.plot(
                g["train_size"],
                g["f1_macro"],
                marker=style_map[model_name]["marker"],
                linestyle=style_map[model_name]["linestyle"],
                linewidth=2.4,
                color=color,
                label=model_name,
                zorder=3,
            )

        for _, row in g.iterrows():
            ax.annotate(
                f"{row['split_key']}: {row['f1_macro']:.3f}",
                (row["train_size"], row["f1_macro"]),
                textcoords="offset points",
                xytext=(0, 10 if is_majority else 7),
                ha="center",
                fontsize=9 if is_majority else 8,
                fontweight="bold" if is_majority else "normal",
                color=color,
            )

    if x_log:
        ax.set_xscale("log")
    ax.set_xticks(sorted(df["train_size"].unique().tolist()))
    ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    if majority_name in set(df["model"]):
        y_min = float(df["f1_macro"].min())
        y_max = float(df["f1_macro"].max())
        if y_max > y_min:
            pad = (y_max - y_min) * 0.08
            ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("F1-macro")
    ax.legend(frameon=True)
    plt.xticks(rotation=25)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return df


def plot_cross_dataset_overview(
    summary_df: pd.DataFrame,
    metric: str = "f1_macro",
    force_replot: bool = True,
    include_family_mean: bool | None = None,
    normalize_model_names: bool = True,
):
    required_cols = {"family", "dataset", "model", metric}
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"Missing columns for cross-dataset plot: {sorted(missing)}")

    plot_root = CFG["output_dir"] / "cross_dataset_eval" / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    out_path = plot_root / f"cross_dataset_overview_{metric}.png"
    matrix_path = plot_root / f"cross_dataset_matrix_{metric}.csv"

    if (not force_replot) and display_saved_plot(
        out_path,
        title=f"Cross-Dataset Overview ({metric})",
        figsize=(16, 9),
    ):
        print(f"Displayed existing plot: {out_path}")
        return

    df = summary_df.copy()
    df = df.dropna(subset=["dataset", "model", metric]).copy()
    if df.empty:
        print("Cross-dataset summary is empty after dropping missing values.")
        return

    if normalize_model_names:
        df["model_display"] = (
            df["model"]
            .astype(str)
            .map(lambda value: re.sub(r"\s*\[[^\]]+\]\s*$", "", value).strip())
        )
    else:
        df["model_display"] = df["model"].astype(str)

    known_family_order = ["classical", "deep", "transformer"]
    present_families = [family for family in known_family_order if family in set(df["family"].dropna())]
    family_order = present_families or sorted(df["family"].dropna().astype(str).unique().tolist())
    family_palette = {
        "classical": "#7A7A7A",
        "deep": COLORS[1],
        "transformer": "#D97706",
    }

    df["family"] = pd.Categorical(df["family"], categories=family_order, ordered=True)

    dataset_order = (
        [name for name in sorted(external_eval_datasets.keys()) if name in set(df["dataset"])]
        if "external_eval_datasets" in globals()
        else sorted(df["dataset"].dropna().unique().tolist())
    )
    if not dataset_order:
        dataset_order = sorted(df["dataset"].dropna().unique().tolist())

    model_rank_df = (
        df.groupby(["family", "model_display"], as_index=False)[metric]
        .mean()
        .sort_values(["family", metric, "model_display"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    model_order = model_rank_df["model_display"].tolist()
    model_family_map = dict(model_rank_df[["model_display", "family"]].drop_duplicates().values)

    model_heatmap_df = (
        df.pivot_table(index="model_display", columns="dataset", values=metric, aggfunc="mean")
        .reindex(index=model_order, columns=dataset_order)
    )
    model_heatmap_df = model_heatmap_df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    family_heatmap_df = (
        df.groupby(["family", "dataset"], as_index=False)[metric]
        .mean()
        .pivot(index="family", columns="dataset", values=metric)
        .reindex(index=family_order, columns=dataset_order)
    )
    family_heatmap_df = family_heatmap_df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if model_heatmap_df.empty:
        print("No cross-dataset values are available for plotting.")
        return

    if include_family_mean is None:
        include_family_mean = len(family_heatmap_df.index) > 1 and not family_heatmap_df.empty
    include_family_mean = bool(include_family_mean and not family_heatmap_df.empty)

    model_heatmap_df.to_csv(matrix_path)
    print(f"Cross-dataset matrix saved to: {matrix_path}")

    value_parts = [model_heatmap_df.stack()]
    if include_family_mean:
        value_parts.append(family_heatmap_df.stack())
    all_values = pd.concat(value_parts)
    all_values = all_values.dropna()

    vmax = float(all_values.max()) if len(all_values) else 1.0
    vmax = max(0.35, min(1.0, vmax))

    cmap = sns.blend_palette(
        ["#F7FAFC", "#D9EAF7", COLORS[2], COLORS[1], COLORS[0]],
        as_cmap=True,
    )

    if include_family_mean:
        fig = plt.figure(figsize=(18, max(7, len(model_order) * 0.42)))
        gs = fig.add_gridspec(1, 2, width_ratios=[5.5, 1.8], wspace=0.18)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_side = fig.add_subplot(gs[0, 1])
    else:
        fig, ax_main = plt.subplots(
            figsize=(max(10, len(model_heatmap_df.columns) * 2.4), max(3.8, len(model_order) * 0.72 + 1.2))
        )
        ax_side = None

    sns.heatmap(
        model_heatmap_df,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        linewidths=0.6,
        linecolor="white",
        cbar=True,
        ax=ax_main,
        cbar_kws={"shrink": 0.85, "label": metric.replace("_", " ").title()},
    )

    if include_family_mean:
        sns.heatmap(
            family_heatmap_df,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            linewidths=0.6,
            linecolor="white",
            cbar=False,
            ax=ax_side,
        )

    ax_main.set_title(
        f"Cross-Dataset Evaluation by Model ({metric.replace('_', ' ').title()})",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    ax_main.set_xlabel("External Dataset")
    ax_main.set_ylabel("Model")

    ax_main.set_xticklabels(
        [x.get_text().replace("_", "\n") for x in ax_main.get_xticklabels()],
        rotation=0,
        ha="center",
    )
    if include_family_mean:
        ax_side.set_title(
            "Family Mean",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax_side.set_xlabel("External Dataset")
        ax_side.set_ylabel("Family")
        ax_side.set_xticklabels(
            [x.get_text().replace("_", "\n") for x in ax_side.get_xticklabels()],
            rotation=0,
            ha="center",
        )

    for tick in ax_main.get_yticklabels():
        model_name = tick.get_text()
        family = model_family_map.get(model_name)
        tick.set_color(family_palette.get(family, "#222222"))

    if include_family_mean:
        for tick in ax_side.get_yticklabels():
            family = tick.get_text()
            tick.set_color(family_palette.get(family, "#222222"))
            tick.set_fontweight("bold")

    family_breaks = []
    prev_family = None
    for idx, model_name in enumerate(model_order):
        current_family = model_family_map.get(model_name)
        if prev_family is not None and current_family != prev_family:
            family_breaks.append(idx)
        prev_family = current_family

    for boundary in family_breaks:
        ax_main.hlines(boundary, *ax_main.get_xlim(), colors="#2B2B2B", linewidth=1.5)

    plt.suptitle(
        "Cross-Dataset Generalization Overview",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.show()

    print(f"Cross-dataset overview plot saved to: {out_path}")


def compute_umap_embeddings(
    texts,
    labels,
    sample_n: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute 2D UMAP embeddings from raw texts using TF-IDF features.

    Parameters
    ----------
    texts    : iterable of strings
    labels   : iterable of class labels (same length as texts)
    sample_n : randomly subsample to this many points if the corpus is larger
    seed     : random seed for UMAP and sampling

    Returns
    -------
    pd.DataFrame with columns ["X", "Y", "label"]
    """
    import random as _random
    from sklearn.feature_extraction.text import TfidfVectorizer
    import umap  # type: ignore[import]

    texts = list(texts)
    labels = list(labels)

    if len(texts) > sample_n:
        _random.seed(seed)
        idx = _random.sample(range(len(texts)), sample_n)
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]

    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(texts)

    reducer = umap.UMAP(n_components=2, random_state=seed)
    embedding = reducer.fit_transform(X)

    return pd.DataFrame({"X": embedding[:, 0], "Y": embedding[:, 1], "label": labels})
