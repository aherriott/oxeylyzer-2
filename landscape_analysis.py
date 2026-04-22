#!/usr/bin/env python3
"""
Keyboard layout landscape analysis: PCA, FAMD, UMAP.

Takes the CSV from `df` REPL command and produces:
- PCA on continuous features (interpretable axes)
- FAMD on mixed features (handles categorical finger assignments)
- UMAP for cluster discovery
- Correlations: PC axes vs score and known features
- Interactive HTML plots

Usage:
  python landscape_analysis.py features.csv [--output landscape.html]

Requirements:
  pip install pandas numpy scikit-learn prince umap-learn plotly
"""

import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def load_features(csv_path, exclude_score_components=True):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} layouts × {len(df.columns)} columns")

    # Identify feature columns
    meta_cols = ["layout_name", "score"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Optionally drop score-component features (they're literally parts of the score,
    # so including them creates circular reasoning in PCA/FAMD).
    score_component_cols = ["score_sfb", "score_stretch", "score_scissors",
                           "score_trigram", "score_magic", "score_finger"]
    if exclude_score_components:
        feature_cols = [c for c in feature_cols if c not in score_component_cols]
        print(f"  excluded {len(score_component_cols)} score-component features")

    # Separate categorical (finger assignments) from continuous
    categorical_cols = [c for c in feature_cols if c.endswith("_finger")]
    continuous_cols = [c for c in feature_cols if c not in categorical_cols]

    print(f"  continuous features: {len(continuous_cols)}")
    print(f"  categorical features: {len(categorical_cols)}")

    # Convert continuous to float
    for c in continuous_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with NaN in critical columns
    df = df.dropna(subset=continuous_cols[:10])
    print(f"  after dropping bad rows: {len(df)}")

    return df, continuous_cols, categorical_cols


def run_pca(df, continuous_cols, n_components=3):
    print("\n=== PCA (continuous features) ===")
    X = df[continuous_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X_scaled)

    print(f"Explained variance: {pca.explained_variance_ratio_}")
    print(f"Cumulative: {np.cumsum(pca.explained_variance_ratio_)}")

    # Interpret: which features load most on each component?
    for i in range(n_components):
        loadings = pca.components_[i]
        top = sorted(zip(continuous_cols, loadings), key=lambda x: -abs(x[1]))[:8]
        print(f"\n  PC{i+1} top loadings (explains {pca.explained_variance_ratio_[i]*100:.1f}%):")
        for name, val in top:
            direction = "+" if val > 0 else "-"
            print(f"    {direction}{abs(val):.3f}  {name}")

    return coords, pca


def run_famd(df, continuous_cols, categorical_cols, n_components=3):
    print("\n=== FAMD (mixed continuous + categorical) ===")
    try:
        import prince
    except ImportError:
        print("  prince not installed — skipping FAMD. Install: pip install prince")
        return None, None

    # Build mixed feature matrix
    feature_cols = continuous_cols + categorical_cols
    X = df[feature_cols].copy()

    # Categorical columns must be category dtype
    for c in categorical_cols:
        X[c] = X[c].astype("category")

    famd = prince.FAMD(n_components=n_components, random_state=42)
    famd = famd.fit(X)

    coords = famd.row_coordinates(X).values

    print(f"Explained variance: {famd.eigenvalues_summary}")

    return coords, famd


def run_umap(df, continuous_cols, categorical_cols, n_components=2):
    print("\n=== UMAP (non-linear manifold) ===")
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed — skipping. Install: pip install umap-learn")
        return None

    # Encode categoricals as one-hot, then combine with continuous
    X_cont = df[continuous_cols].values
    X_cont_scaled = StandardScaler().fit_transform(X_cont)

    if categorical_cols:
        X_cat = pd.get_dummies(df[categorical_cols])
        X_combined = np.hstack([X_cont_scaled, X_cat.values])
    else:
        X_combined = X_cont_scaled

    reducer = umap.UMAP(n_components=n_components, random_state=42,
                       n_neighbors=min(30, len(df) - 1), min_dist=0.1)
    coords = reducer.fit_transform(X_combined)

    return coords


def feature_importance_for_score(df, continuous_cols, categorical_cols):
    print("\n=== Random Forest feature importance (predicting score) ===")
    X = df[continuous_cols].copy()
    if categorical_cols:
        X_cat = pd.get_dummies(df[categorical_cols])
        X = pd.concat([X, X_cat], axis=1)

    y = df["score"].values

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X.values, y)

    importance = list(zip(X.columns, rf.feature_importances_))
    importance.sort(key=lambda x: -x[1])

    print("  Top 20 most important features for score:")
    for name, imp in importance[:20]:
        print(f"    {imp:.4f}  {name}")

    return importance


def correlate_pcs_with_features(coords, df, continuous_cols, name_prefix="PC"):
    print(f"\n=== {name_prefix} correlations with key features ===")
    key_features = [
        "score", "sfbs_pct", "stretches", "inroll_pct", "outroll_pct",
        "alternate_pct", "redirect_pct", "onehandin_pct", "onehandout_pct",
        "home_row_pct", "hand_imbalance", "vowels_left_pct", "magic_rule_count",
    ]

    for i in range(coords.shape[1]):
        print(f"\n  {name_prefix}{i+1} correlations:")
        pc = coords[:, i]
        rows = []
        for feat in key_features:
            if feat not in df.columns:
                continue
            try:
                v = df[feat].astype(float).values
                if np.std(v) < 1e-9:
                    continue
                corr = np.corrcoef(pc, v)[0, 1]
                rows.append((feat, corr))
            except (ValueError, TypeError):
                continue
        rows.sort(key=lambda x: -abs(x[1]))
        for feat, corr in rows[:8]:
            print(f"    r={corr:+.3f}  {feat}")


def compute_axis_label(coords, df, axis_idx, continuous_cols, max_feats=3):
    """Generate a human-readable label for an axis by finding the top-correlating features."""
    if coords is None or axis_idx >= coords.shape[1]:
        return f"Axis {axis_idx + 1}"

    ax = coords[:, axis_idx]
    key_features = continuous_cols + ["score"]
    rows = []
    for feat in key_features:
        if feat not in df.columns:
            continue
        try:
            v = df[feat].astype(float).values
            if np.std(v) < 1e-9:
                continue
            corr = np.corrcoef(ax, v)[0, 1]
            if not np.isnan(corr):
                rows.append((feat, corr))
        except (ValueError, TypeError):
            continue

    rows.sort(key=lambda x: -abs(x[1]))
    top = rows[:max_feats]
    if not top:
        return f"Axis {axis_idx + 1}"

    parts = []
    for feat, corr in top:
        direction = "+" if corr > 0 else "−"
        parts.append(f"{direction}{feat} ({corr:+.2f})")
    return f"PC{axis_idx + 1}: " + " / ".join(parts)


def plot_interactive(df, pca_coords, famd_coords, umap_coords, output_html, continuous_cols=None):
    print(f"\n=== Building interactive plot ({output_html}) ===")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed — skipping HTML plot. Install: pip install plotly")
        return

    n_plots = sum(1 for c in [pca_coords, famd_coords, umap_coords] if c is not None)
    titles = []
    plots = []
    if pca_coords is not None:
        titles.append("PCA (continuous features)")
        plots.append(("PCA", pca_coords))
    if famd_coords is not None:
        titles.append("FAMD (mixed features)")
        plots.append(("FAMD", famd_coords))
    if umap_coords is not None:
        titles.append("UMAP (non-linear)")
        plots.append(("UMAP", umap_coords))

    fig = make_subplots(rows=1, cols=len(plots), subplot_titles=titles,
                       horizontal_spacing=0.1)

    # Compute axis labels based on top-correlating features
    axis_labels = {}
    if continuous_cols:
        for plot_name, coords in plots:
            x_label = compute_axis_label(coords, df, 0, continuous_cols)
            y_label = compute_axis_label(coords, df, 1, continuous_cols)
            axis_labels[plot_name] = (x_label, y_label)

    # Color by score (less negative = better)
    scores = df["score"].values
    min_score = scores.min()
    max_score = scores.max()

    # Categorize layouts for markers
    def categorize(name):
        if name.startswith("random-"):
            return "random"
        if name.startswith("greedy-"):
            return "greedy"
        if name.lower() in ["sturdy", "graphite", "canary", "colemak-dh", "noctum",
                            "nrts-oxey", "qwerty", "dvorak"]:
            return "reference"
        if name.startswith("oxey-v"):
            return "oxey"
        if name.startswith("goalseed-"):
            return "goalseed"
        return "other"

    df["category"] = df["layout_name"].apply(categorize)

    category_colors = {
        "random": "lightgray",
        "greedy": "skyblue",
        "reference": "red",
        "oxey": "orange",
        "goalseed": "green",
        "other": "blue",
    }
    category_size = {
        "random": 3,
        "greedy": 4,
        "reference": 12,
        "oxey": 8,
        "goalseed": 6,
        "other": 5,
    }

    for col_i, (plot_name, coords) in enumerate(plots):
        for category in ["random", "greedy", "other", "goalseed", "oxey", "reference"]:
            mask = df["category"] == category
            if mask.sum() == 0:
                continue
            subset_df = df[mask]
            subset_coords = coords[mask.values]
            hover = [f"{row['layout_name']}<br>score: {row['score']:,.0f}<br>"
                    f"sfbs: {row['sfbs_pct']:.2f}%<br>inroll: {row['inroll_pct']:.1f}% "
                    f"outroll: {row['outroll_pct']:.1f}%<br>"
                    f"alt: {row['alternate_pct']:.1f}% redir: {row['redirect_pct']:.1f}%"
                    for _, row in subset_df.iterrows()]

            fig.add_trace(
                go.Scatter(
                    x=subset_coords[:, 0],
                    y=subset_coords[:, 1],
                    mode="markers",
                    name=f"{category} ({mask.sum()})",
                    legendgroup=category,
                    showlegend=(col_i == 0),
                    marker=dict(
                        size=category_size[category],
                        color=category_colors[category],
                        line=dict(width=0.5, color="black") if category == "reference" else None,
                    ),
                    hovertext=hover,
                    hoverinfo="text",
                ),
                row=1, col=col_i + 1,
            )

    fig.update_layout(
        title=f"Layout Landscape Analysis ({len(df)} layouts)",
        height=650,
        width=500 * len(plots),
    )

    # Apply per-subplot axis labels
    for col_i, (plot_name, _coords) in enumerate(plots):
        if plot_name in axis_labels:
            x_label, y_label = axis_labels[plot_name]
            # Plotly subplots use xaxis, xaxis2, xaxis3, etc.
            axis_suffix = "" if col_i == 0 else str(col_i + 1)
            fig.update_xaxes(title_text=x_label, row=1, col=col_i + 1)
            fig.update_yaxes(title_text=y_label, row=1, col=col_i + 1)

    fig.write_html(output_html)
    print(f"  Saved: {output_html}")
    print(f"  Open in browser to explore.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="CSV from `df` REPL command")
    parser.add_argument("--output", default="landscape.html", help="Output HTML path")
    parser.add_argument("--no-umap", action="store_true", help="Skip UMAP (faster)")
    parser.add_argument("--no-famd", action="store_true", help="Skip FAMD")
    args = parser.parse_args()

    df, continuous_cols, categorical_cols = load_features(args.csv_path)

    # PCA
    pca_coords, pca = run_pca(df, continuous_cols)
    correlate_pcs_with_features(pca_coords, df, continuous_cols, "PCA_PC")

    # FAMD
    famd_coords = None
    if not args.no_famd:
        famd_coords, famd = run_famd(df, continuous_cols, categorical_cols)
        if famd_coords is not None:
            correlate_pcs_with_features(famd_coords, df, continuous_cols, "FAMD_PC")

    # UMAP
    umap_coords = None
    if not args.no_umap:
        umap_coords = run_umap(df, continuous_cols, categorical_cols)

    # Feature importance
    feature_importance_for_score(df, continuous_cols, categorical_cols)

    # Interactive plot
    plot_interactive(df, pca_coords, famd_coords, umap_coords, args.output, continuous_cols)

    print("\nDone.")


if __name__ == "__main__":
    main()
