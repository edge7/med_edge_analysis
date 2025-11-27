import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.base import clone
from typing import Dict, Any, Optional, Union, List


# =============================================================================
# CUSTOM SCORERS
# =============================================================================

def precision_on_wrong(y_true, y_pred):
    """
    Precision for class 0 (wrong/incorrect predictions).
    When model says "this will be wrong", how often is it actually wrong?

    High precision_on_wrong = model is reliable when it says "don't trust this answer"
    """
    return precision_score(y_true, y_pred, pos_label=0, zero_division=0)


def recall_on_wrong(y_true, y_pred):
    """
    Recall for class 0 (wrong/incorrect predictions).
    Of all actually wrong answers, how many did the model catch?

    High recall_on_wrong = model catches most wrong answers
    """
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


def f1_on_wrong(y_true, y_pred):
    """
    F1 score for class 0 (wrong/incorrect predictions).
    Balance between precision and recall on the "wrong" class.
    """
    return f1_score(y_true, y_pred, pos_label=0, zero_division=0)


def weighted_wrong_precision(y_true, y_pred, wrong_weight=2.0):
    """
    Weighted score that prioritizes precision on 'wrong' class.

    Args:
        wrong_weight: How much more to weight precision_on_wrong vs precision_on_correct
                     Default 2.0 means precision_on_wrong counts 2x
    """
    prec_wrong = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    prec_correct = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    return (wrong_weight * prec_wrong + prec_correct) / (wrong_weight + 1)


# Pre-built scorers for use in GridSearchCV/RandomizedSearchCV
CUSTOM_SCORERS = {
    "precision_wrong": make_scorer(precision_on_wrong),
    "recall_wrong": make_scorer(recall_on_wrong),
    "f1_wrong": make_scorer(f1_on_wrong),
    "weighted_wrong_2x": make_scorer(weighted_wrong_precision, wrong_weight=2.0),
    "weighted_wrong_3x": make_scorer(weighted_wrong_precision, wrong_weight=3.0),
}


def get_scorer(scoring):
    """
    Get a scorer by name. Supports both sklearn built-in and custom scorers.

    Custom scorers:
        - "precision_wrong": precision on class 0 (wrong answers)
        - "recall_wrong": recall on class 0 (wrong answers)
        - "f1_wrong": F1 on class 0 (wrong answers)
        - "weighted_wrong_2x": 2x weight on precision_wrong
        - "weighted_wrong_3x": 3x weight on precision_wrong
    """
    if scoring in CUSTOM_SCORERS:
        return CUSTOM_SCORERS[scoring]
    return scoring  # Return as-is for sklearn built-in scorers


def tune_classifier(
    clf,
    X,
    y,
    param_grid: Dict[str, List[Any]],
    method: str = "grid",
    n_splits: int = 7,
    scoring: str = "f1",
    n_iter: int = 50,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1
):
    """
    Hyperparameter tuning for any sklearn-compatible classifier.

    Args:
        clf: Unfitted classifier (e.g., RandomForestClassifier(), XGBClassifier())
        X: Feature DataFrame (already filtered to selected features)
        y: Target Series
        param_grid: Dictionary of parameters to search.
                    For GridSearchCV: all combinations are tried.
                    For RandomizedSearchCV: can use distributions.
        method: "grid" for GridSearchCV, "random" for RandomizedSearchCV
        n_splits: Number of CV folds (default: 7)
        scoring: Scoring metric (default: "f1"). Options: "f1", "accuracy",
                 "roc_auc", "precision", "recall", etc.
                 Custom scorers: "precision_wrong", "recall_wrong", "f1_wrong",
                 "weighted_wrong_2x", "weighted_wrong_3x"
        n_iter: Number of iterations for RandomizedSearchCV (ignored if method="grid")
        random_state: Random state for reproducibility
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)

    Returns:
        dict: {
            "best_estimator": fitted best model,
            "best_params": best parameter combination,
            "best_score": best CV score,
            "cv_results": full CV results DataFrame,
            "search_object": the GridSearchCV/RandomizedSearchCV object
        }

    Example:
        # RandomForest
        from sklearn.ensemble import RandomForestClassifier
        rf_params = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "class_weight": ["balanced"]
        }
        result = tune_classifier(RandomForestClassifier(), X, y, rf_params)

        # XGBoost
        from xgboost import XGBClassifier
        xgb_params = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "scale_pos_weight": [1, 5, 10]  # for imbalanced data
        }
        result = tune_classifier(XGBClassifier(), X, y, xgb_params)

        # LightGBM
        from lightgbm import LGBMClassifier
        lgbm_params = {
            "n_estimators": [100, 200],
            "max_depth": [5, 7, -1],
            "learning_rate": [0.01, 0.1],
            "class_weight": ["balanced"]
        }
        result = tune_classifier(LGBMClassifier(), X, y, lgbm_params)
    """
    # Setup CV strategy
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Resolve custom scorers
    scorer = get_scorer(scoring)
    scorer_name = scoring  # Keep original name for display

    print(f"{'='*60}")
    print(f"HYPERPARAMETER TUNING")
    print(f"{'='*60}")
    print(f"Classifier: {clf.__class__.__name__}")
    print(f"Method: {method.upper()}SearchCV")
    print(f"CV Folds: {n_splits}")
    print(f"Scoring: {scorer_name}" + (" (custom)" if scoring in CUSTOM_SCORERS else ""))
    print(f"Parameters to search:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    # Calculate search space size
    if method == "grid":
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        print(f"Total combinations: {total_combinations}")
        print(f"Total fits: {total_combinations * n_splits}")

        search = GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            cv=cv,
            scoring=scorer,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
    else:
        print(f"Random iterations: {n_iter}")
        print(f"Total fits: {n_iter * n_splits}")

        search = RandomizedSearchCV(
            estimator=clf,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scorer,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            return_train_score=True
        )

    print(f"\nRunning search...")
    search.fit(X, y)

    # Results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Best {scorer_name} score: {search.best_score_:.4f}")
    print(f"Best parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")

    # Convert cv_results to more usable format
    import pandas as pd
    cv_results_df = pd.DataFrame(search.cv_results_)

    return {
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "cv_results": cv_results_df,
        "search_object": search
    }


# Pre-defined parameter grids for common classifiers
PARAM_GRIDS = {
    "RandomForest": {
    "n_estimators": [200],
    "max_depth": [3, 5, 7],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "criterion": ["gini", "entropy"],
    "class_weight": ["balanced"]
    },

    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "scale_pos_weight": [1, 5, 10]
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 7, -1],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 50, 100],
        "class_weight": ["balanced"]
    },
    "LogisticRegression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "solver": ["saga"],
        "class_weight": ["balanced"],
        "max_iter": [1000]
    },
    "SVM": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
        "class_weight": ["balanced"]
    }
}


def get_default_param_grid(classifier_name: str) -> Dict[str, List[Any]]:
    """
    Get a default parameter grid for common classifiers.

    Args:
        classifier_name: One of "RandomForest", "XGBoost", "LightGBM",
                        "LogisticRegression", "SVM"

    Returns:
        dict: Parameter grid for the classifier
    """
    if classifier_name not in PARAM_GRIDS:
        available = list(PARAM_GRIDS.keys())
        raise ValueError(f"Unknown classifier '{classifier_name}'. Available: {available}")

    return PARAM_GRIDS[classifier_name].copy()
