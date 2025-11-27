import numpy as np
import pandas as pd
import random
from sklearn.metrics import make_scorer, f1_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from tqdm import tqdm


def remove_collinear_features(X, threshold=0.85, method="pearson", verbose=True):
    """
    Remove highly correlated features from a DataFrame.

    For each pair of correlated features (above threshold), keeps the one
    that appears first in the DataFrame and drops the other.

    Args:
        X: pandas DataFrame with features
        threshold: correlation threshold (default: 0.85).
                   Features with |correlation| > threshold are considered collinear.
        method: correlation method - "pearson", "spearman", or "kendall" (default: "pearson")
        verbose: print details about removed features (default: True)

    Returns:
        tuple: (X_filtered, dropped_features, correlation_info)
            - X_filtered: DataFrame with collinear features removed
            - dropped_features: list of removed feature names
            - correlation_info: list of tuples (dropped, kept, correlation)
    """
    # Compute correlation matrix
    corr_matrix = X.corr(method=method).abs()

    # Get upper triangle (to avoid checking pairs twice)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find pairs above threshold
    correlation_info = []
    dropped_features = set()

    for col in upper_tri.columns:
        # Find features correlated with this one above threshold
        correlated = upper_tri.index[upper_tri[col] > threshold].tolist()

        for corr_feat in correlated:
            if corr_feat not in dropped_features and col not in dropped_features:
                # Drop the one that comes later (col), keep the earlier one (corr_feat)
                corr_value = corr_matrix.loc[corr_feat, col]
                correlation_info.append((col, corr_feat, corr_value))
                dropped_features.add(col)

    # Filter DataFrame
    kept_features = [col for col in X.columns if col not in dropped_features]
    X_filtered = X[kept_features]

    if verbose:
        print(f"{'='*60}")
        print(f"COLLINEARITY REMOVAL (threshold: {threshold})")
        print(f"{'='*60}")
        print(f"Original features: {len(X.columns)}")
        print(f"Dropped features:  {len(dropped_features)}")
        print(f"Remaining features: {len(kept_features)}")

        if correlation_info:
            print(f"\n{'Dropped':<30} {'Kept':<30} {'Corr'}")
            print(f"{'─'*30} {'─'*30} {'─'*6}")
            for dropped, kept, corr in sorted(correlation_info, key=lambda x: x[2], reverse=True):
                print(f"{dropped:<30} {kept:<30} {corr:.3f}")
        else:
            print("\nNo collinear features found.")

        print(f"{'='*60}")

    return X_filtered, list(dropped_features), correlation_info


def get_correlation_clusters(X, threshold=0.85, method="pearson"):
    """
    Group features into clusters based on correlation.

    Useful for understanding which features are measuring similar things.

    Args:
        X: pandas DataFrame with features
        threshold: correlation threshold (default: 0.85)
        method: correlation method (default: "pearson")

    Returns:
        list of lists: each inner list contains correlated feature names
    """
    corr_matrix = X.corr(method=method).abs()

    features = list(X.columns)
    visited = set()
    clusters = []

    for feat in features:
        if feat in visited:
            continue

        # Find all features correlated with this one
        cluster = [feat]
        visited.add(feat)

        correlated = corr_matrix.index[corr_matrix[feat] > threshold].tolist()
        for corr_feat in correlated:
            if corr_feat != feat and corr_feat not in visited:
                cluster.append(corr_feat)
                visited.add(corr_feat)

        clusters.append(cluster)

    # Sort clusters by size (largest first)
    clusters.sort(key=len, reverse=True)

    print(f"Found {len(clusters)} feature clusters:")
    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            print(f"  Cluster {i+1} ({len(cluster)} features): {cluster}")
        else:
            print(f"  Standalone: {cluster[0]}")

    return clusters

def find_good_features(clf, X_train, y_train, X_val=None, y_val=None, n_repeats=200, threshold_divisor=10):
    """
    Identifies features that perform better than a random noise feature
    using Permutation Importance with standard F1 score.

    Args:
        clf: Unfitted classifier (will be fitted internally)
        X_train: Training features (used for fitting)
        y_train: Training labels
        X_val: Validation features (used for permutation importance).
               If None, falls back to using X_train.
        y_val: Validation labels. If None, falls back to using y_train.
        n_repeats: Number of permutation repeats (default: 200)
        threshold_divisor: Divisor for selection threshold (default: 10).
                          threshold = n_repeats / threshold_divisor
                          - 10 → need >55% wins (recommended)
                          - 5  → need >60% wins (strict)
                          - 20 → need >52.5% wins (lenient)

    Returns:
        tuple: (list_of_kept_features, list_of_tuples_name_and_score)
    """
    # Backwards compatibility: if no validation set provided, use train
    if X_val is None:
        X_val = X_train
        y_val = y_train

    names_and_score = []

    # 1. Reproducibility
    random.seed(10)
    np.random.seed(10)

    # 2. Standard F1 Scorer
    # Assumes y_train is binary (0/1).
    # If multiclass, change to f1_score(average='macro')
    custom_scorer = make_scorer(f1_score)

    # 3. Create Shadow Feature (Random Noise) on BOTH train and val
    # Use .copy() to treat inputs as immutable
    X_train_augmented = X_train.copy()
    X_val_augmented = X_val.copy()

    # Generate random noise (-5 to 5) for every row
    X_train_augmented["random"] = np.random.uniform(-5, 5, size=len(X_train))
    X_val_augmented["random"] = np.random.uniform(-5, 5, size=len(X_val))

    # 4. Fit on TRAIN, run Permutation Importance on VAL
    clf.fit(X_train_augmented, y_train.values.ravel())

    print(f"Running permutation importance ({n_repeats} repeats)...")
    result = permutation_importance(
        clf,
        X_val_augmented,
        y_val,
        n_repeats=n_repeats,
        random_state=7,
        scoring=custom_scorer,
        n_jobs=-1,
    )

    # 5. Compare Features vs Random Noise
    summary = {col: 0 for col in X_val_augmented.columns if col != "random"}
    random_col_idx = X_val_augmented.columns.get_loc("random")

    # Iterate through every repetition (row in result.importances)
    for i in tqdm(range(n_repeats), desc="Scoring Features"):

        # The threshold for this specific iteration is the score of the random column
        random_threshold = result.importances[random_col_idx, i]

        for col_idx, name in enumerate(X_val_augmented.columns):
            if name == "random":
                continue

            importance = result.importances[col_idx, i]

            # SCORING LOGIC:
            # Start with assumption that the feature failed (-1)
            to_add = -1

            # To gain a point (+1), the feature must:
            # A) Be more important than the random noise column
            # B) Have a positive importance (actually helps the model)
            if importance > random_threshold:
                to_add = 1

            summary[name] += to_add

    # 6. Filter Results
    to_keep = []
    # Threshold: Must beat random noise often enough
    # threshold_divisor=10 means need >55% wins, =5 means >60%, =20 means >52.5%
    selection_threshold = n_repeats / threshold_divisor

    # Calculate wins needed: score = wins - losses = 2*wins - n_repeats
    # So wins = (score + n_repeats) / 2
    wins_needed = int((selection_threshold + n_repeats) / 2) + 1
    win_rate_needed = wins_needed / n_repeats * 100

    print(f"\n{'─'*60}")
    print(f"THRESHOLD: score > {selection_threshold:.0f} (need >{wins_needed}/{n_repeats} wins, >{win_rate_needed:.1f}%)")
    print(f"{'─'*60}")
    print(f"{'Feature':<35} {'Score':>7} {'Wins':>6} {'Win%':>7} {'Margin':>8} {'Status'}")
    print(f"{'─'*35} {'─'*7} {'─'*6} {'─'*7} {'─'*8} {'─'*10}")

    # Sort by score descending
    sorted_summary = sorted(summary.items(), key=lambda x: x[1], reverse=True)

    for col, val in sorted_summary:
        names_and_score.append((col, val))
        wins = int((val + n_repeats) / 2)
        win_pct = wins / n_repeats * 100
        margin = val - selection_threshold

        if val > selection_threshold:
            to_keep.append(col)
            status = "KEPT"
        else:
            status = "dropped"

        # Color-code margin (positive = safe, negative = dropped)
        margin_str = f"{margin:+.0f}"

        print(f"{col:<35} {val:>7.0f} {wins:>6} {win_pct:>6.1f}% {margin_str:>8} {status}")

    print(f"{'─'*60}")
    print(f"Done. Kept {len(to_keep)}/{len(summary)} features.")

    return to_keep, names_and_score


def cv_feature_selection(clf, X, y, n_splits=7, n_repeats=200, min_folds=None, threshold_divisor=10, random_state=42):
    """
    Performs feature selection using cross-validation with configurable voting.

    Args:
        clf: Unfitted classifier (will be cloned for each fold)
        X: Full feature DataFrame
        y: Full target Series
        n_splits: Number of CV folds (default: 7)
        n_repeats: Number of permutation repeats per fold (default: 200)
        min_folds: Minimum number of folds a feature must pass to be kept.
                   If None, defaults to n_splits (unanimous voting).
                   Set to lower value for majority voting (e.g., 4 for 4/7 folds).
        threshold_divisor: Divisor for per-fold selection threshold (default: 10).
                          - 10 → need >55% wins (recommended)
                          - 5  → need >60% wins (strict)
                          - 20 → need >52.5% wins (lenient)
        random_state: Random state for reproducibility (default: 42)

    Returns:
        tuple: (selected_features, fold_results)
            - selected_features: list of features that passed min_folds threshold
            - fold_results: dict with detailed results per fold
    """
    if min_folds is None:
        min_folds = n_splits

    if min_folds > n_splits:
        raise ValueError(f"min_folds ({min_folds}) cannot be greater than n_splits ({n_splits})")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results = {}
    all_fold_features = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'='*60}")

        # Split data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        print(f"Train size: {len(X_train_fold)}, Val size: {len(X_val_fold)}")

        # Clone classifier for this fold (fresh unfitted copy)
        clf_fold = clone(clf)

        # Run feature selection
        good_features, scores = find_good_features(
            clf_fold,
            X_train_fold,
            y_train_fold,
            X_val_fold,
            y_val_fold,
            n_repeats=n_repeats,
            threshold_divisor=threshold_divisor
        )

        # Store results
        fold_results[f"fold_{fold_idx + 1}"] = {
            "features": good_features,
            "scores": scores,
            "n_selected": len(good_features)
        }
        all_fold_features.append(set(good_features))

        print(f"Fold {fold_idx + 1} selected {len(good_features)} features")

    # Count how many folds each feature passed
    feature_counts = {}
    for fold_features in all_fold_features:
        for feat in fold_features:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    # Select features that pass min_folds threshold
    selected_features = [feat for feat, count in feature_counts.items() if count >= min_folds]

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Features selected per fold:")
    for fold_name, fold_data in fold_results.items():
        print(f"  {fold_name}: {fold_data['n_selected']} features")

    print(f"\nFeature pass counts:")
    for feat, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
        status = "KEPT" if count >= min_folds else "dropped"
        print(f"  {feat:<35} {count}/{n_splits} folds ({status})")

    print(f"\nSelected features (passed >= {min_folds}/{n_splits} folds): {len(selected_features)}")
    print(f"Features: {selected_features}")

    return selected_features, fold_results


def cv_shap_feature_selection(clf, X, y, n_splits=7, min_folds=None, top_k=None,
                               percentile_threshold=70, random_state=42):
    """
    SHAP-based feature selection using cross-validation with configurable voting.

    For each fold:
    - Train on train_k, compute SHAP values on val_k
    - Rank features by mean absolute SHAP value
    - Select top features (by top_k or percentile)

    A feature is kept if it's selected in at least min_folds folds.

    Args:
        clf: Unfitted classifier (will be cloned for each fold).
             Must be tree-based (RF, XGBoost, LightGBM) for TreeSHAP.
        X: Full feature DataFrame
        y: Full target Series
        n_splits: Number of CV folds (default: 7)
        min_folds: Minimum folds a feature must be selected in (default: n_splits//2 + 1)
        top_k: Keep top K features per fold. If None, uses percentile_threshold.
        percentile_threshold: Keep features above this percentile (default: 70 = top 30%)
        random_state: Random state for reproducibility (default: 42)

    Returns:
        tuple: (selected_features, fold_results)
            - selected_features: list of features that passed min_folds threshold
            - fold_results: dict with detailed SHAP importances per fold
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required. Install with: pip install shap")

    if min_folds is None:
        min_folds = n_splits // 2 + 1  # majority by default

    if min_folds > n_splits:
        raise ValueError(f"min_folds ({min_folds}) cannot be greater than n_splits ({n_splits})")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results = {}
    all_fold_features = []
    all_shap_importances = []

    print(f"{'='*60}")
    print(f"SHAP FEATURE SELECTION ({n_splits}-fold CV)")
    print(f"{'='*60}")
    if top_k:
        print(f"Selection: top {top_k} features per fold")
    else:
        print(f"Selection: top {100-percentile_threshold}% features per fold (above {percentile_threshold}th percentile)")
    print(f"Voting: feature must pass >= {min_folds}/{n_splits} folds")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'─'*60}")
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'─'*60}")

        # Split data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        print(f"Train: {len(X_train_fold)}, Val: {len(X_val_fold)}")

        # Clone and train classifier
        clf_fold = clone(clf)
        clf_fold.fit(X_train_fold, y_train_fold)

        # Compute SHAP values on validation set
        print("Computing SHAP values...")
        explainer = shap.TreeExplainer(clf_fold)
        shap_values = explainer.shap_values(X_val_fold)

        # Handle different SHAP output formats
        # For binary classification: shap_values can be:
        # - list [class_0, class_1] where each is (n_samples, n_features)
        # - array of shape (n_samples, n_features, 2)
        # - array of shape (n_samples, n_features)
        if isinstance(shap_values, list):
            # Use positive class (index 1)
            shap_vals = shap_values[1]
        elif len(shap_values.shape) == 3:
            # Shape is (n_samples, n_features, n_classes) - take positive class
            shap_vals = shap_values[:, :, 1]
        else:
            shap_vals = shap_values

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)

        # Handle case where result is still 2D
        if len(mean_abs_shap.shape) > 1:
            mean_abs_shap = mean_abs_shap[:, 1] if mean_abs_shap.shape[1] == 2 else mean_abs_shap.mean(axis=1)

        shap_importance = pd.Series(
            mean_abs_shap,
            index=X.columns
        ).sort_values(ascending=False)

        all_shap_importances.append(shap_importance)

        # Select features for this fold
        if top_k:
            threshold = shap_importance.iloc[min(top_k, len(shap_importance)) - 1]
        else:
            threshold = shap_importance.quantile(percentile_threshold / 100)

        fold_features = shap_importance[shap_importance >= threshold].index.tolist()

        # Store results
        fold_results[f"fold_{fold_idx + 1}"] = {
            "features": fold_features,
            "shap_importance": shap_importance.to_dict(),
            "threshold": threshold,
            "n_selected": len(fold_features)
        }
        all_fold_features.append(set(fold_features))

        # Print fold results
        print(f"\n{'Feature':<35} {'SHAP':>10} {'Status'}")
        print(f"{'─'*35} {'─'*10} {'─'*10}")
        for feat, importance in shap_importance.items():
            status = "KEPT" if importance >= threshold else "dropped"
            print(f"{feat:<35} {importance:>10.6f} {status}")
        print(f"\nFold {fold_idx + 1}: selected {len(fold_features)} features (threshold: {threshold:.6f})")

    # Aggregate SHAP importance across folds
    avg_shap_importance = pd.DataFrame(all_shap_importances).mean().sort_values(ascending=False)

    # Count how many folds each feature passed
    feature_counts = {}
    for fold_features in all_fold_features:
        for feat in fold_features:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    # Select features that pass min_folds threshold
    selected_features = [feat for feat, count in feature_counts.items() if count >= min_folds]

    # Sort selected features by average SHAP importance
    selected_features = sorted(selected_features, key=lambda x: avg_shap_importance[x], reverse=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Features selected per fold:")
    for fold_name, fold_data in fold_results.items():
        print(f"  {fold_name}: {fold_data['n_selected']} features")

    print(f"\n{'Feature':<35} {'Avg SHAP':>12} {'Folds':>8} {'Status'}")
    print(f"{'─'*35} {'─'*12} {'─'*8} {'─'*10}")
    for feat in avg_shap_importance.index:
        count = feature_counts.get(feat, 0)
        status = "KEPT" if count >= min_folds else "dropped"
        print(f"{feat:<35} {avg_shap_importance[feat]:>12.6f} {count:>5}/{n_splits}   {status}")

    print(f"\n{'='*60}")
    print(f"Selected features (passed >= {min_folds}/{n_splits} folds): {len(selected_features)}")
    print(f"Features: {selected_features}")

    return selected_features, fold_results