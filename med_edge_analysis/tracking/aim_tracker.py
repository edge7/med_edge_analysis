"""
Aim Experiment Tracker - Modular utilities for ML experiment tracking.

Usage:
    from med_edge_analysis.tracking.aim_tracker import ExperimentTracker

    tracker = ExperimentTracker(experiment_name="my_experiment")
    tracker.log_hyperparams({"learning_rate": 0.01, "epochs": 100})
    tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})
    tracker.log_per_dataset_results(results_list)
    tracker.log_feature_importance(feature_importance_series)
    tracker.finish()
"""

from typing import Any, Optional
import pandas as pd

try:
    from aim import Run
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    Run = None

# Default Aim server configuration
DEFAULT_AIM_SERVER = "aim://192.168.1.28:53800"


class ExperimentTracker:
    """
    Wrapper class for Aim experiment tracking with ML-specific utilities.

    Provides methods for logging:
    - Hyperparameters
    - Metrics (scalar and per-dataset)
    - Feature importances
    - Feature selection results
    - Cross-validation results
    """

    def __init__(
        self,
        experiment_name: str = "default",
        repo: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        use_default_server: bool = True,
    ):
        """
        Initialize the experiment tracker.

        Args:
            experiment_name: Name of the experiment group
            repo: Path to Aim repository. If None and use_default_server=True,
                  uses the default remote server.
            run_name: Optional name for this specific run
            tags: Optional list of tags for the run
            use_default_server: If True and repo is None, use the default Aim server
        """
        if not AIM_AVAILABLE:
            raise ImportError(
                "Aim is not installed. Install with: pip install aim"
            )

        # Use default server if no repo specified and use_default_server is True
        if repo is None and use_default_server:
            repo = DEFAULT_AIM_SERVER

        self.experiment_name = experiment_name
        self.repo = repo
        self.run = Run(
            experiment=experiment_name,
            repo=repo,
        )

        if run_name:
            self.run.name = run_name

        if tags:
            for tag in tags:
                self.run.add_tag(tag)

    @property
    def hash(self) -> str:
        """Get the unique hash of this run."""
        return self.run.hash

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """
        Log hyperparameters for the experiment.

        Args:
            params: Dictionary of hyperparameter names and values
        """
        self.run["hparams"] = params

    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log a single metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step/epoch number
            context: Optional context dictionary for grouping
        """
        kwargs = {"name": name}
        if step is not None:
            kwargs["step"] = step
        if context is not None:
            kwargs["context"] = context

        self.run.track(value, **kwargs)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/epoch number
            context: Optional context dictionary for grouping
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step=step, context=context)

    def log_per_dataset_results(
        self,
        results: list[dict[str, Any]],
        dataset_key: str = "Dataset",
        metrics_to_log: Optional[list[str]] = None,
    ) -> None:
        """
        Log results for multiple datasets (e.g., from cross-validation).

        Args:
            results: List of dictionaries with dataset results
            dataset_key: Key in the dictionary that identifies the dataset
            metrics_to_log: List of metric keys to log (default: all numeric values)
        """
        for result in results:
            ds_name = result.get(dataset_key, "unknown")
            context = {"dataset": ds_name}

            for key, value in result.items():
                if key == dataset_key:
                    continue

                if metrics_to_log and key not in metrics_to_log:
                    continue

                if isinstance(value, (int, float)):
                    self.log_metric(key, value, context=context)

    def log_feature_importance(
        self,
        importance: pd.Series | dict[str, float],
        importance_type: str = "model",
    ) -> None:
        """
        Log feature importance values.

        Args:
            importance: Series or dict with feature names as index/keys
            importance_type: Type of importance (e.g., "model", "shap", "permutation")
        """
        if isinstance(importance, pd.Series):
            importance = importance.to_dict()

        for feature, value in importance.items():
            self.log_metric(
                f"feature_importance_{importance_type}",
                value,
                context={"feature": feature}
            )

        # Also store as a sorted list
        sorted_features = sorted(importance.items(), key=lambda x: -x[1])
        self.run[f"feature_importance_{importance_type}_ranking"] = [
            {"feature": f, "importance": v} for f, v in sorted_features
        ]

    def log_feature_selection(
        self,
        feature_summary_df: pd.DataFrame,
        n_folds: int,
        feature_col: str = "Feature",
        count_col: str = "Folds",
        category_col: Optional[str] = "Category",
    ) -> None:
        """
        Log feature selection stability across CV folds.

        Args:
            feature_summary_df: DataFrame with feature selection counts
            n_folds: Total number of folds
            feature_col: Column name for feature names
            count_col: Column name for selection counts
            category_col: Optional column name for feature categories
        """
        for _, row in feature_summary_df.iterrows():
            context = {"feature": row[feature_col]}
            if category_col and category_col in row:
                context["category"] = row[category_col]

            self.log_metric("feature_selection_count", row[count_col], context=context)

        # Log robust features (>=80% selection rate)
        threshold_80 = int(n_folds * 0.8)
        robust_features = feature_summary_df[
            feature_summary_df[count_col] >= threshold_80
        ][feature_col].tolist()
        self.run["robust_features"] = robust_features
        self.run["n_robust_features"] = len(robust_features)

        # Log core features (100% selection rate)
        core_features = feature_summary_df[
            feature_summary_df[count_col] == n_folds
        ][feature_col].tolist()
        self.run["core_features"] = core_features
        self.run["n_core_features"] = len(core_features)

    def log_confusion_matrix(
        self,
        tp: int,
        tn: int,
        fp: int,
        fn: int,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log confusion matrix components.

        Args:
            tp: True positives
            tn: True negatives
            fp: False positives
            fn: False negatives
            context: Optional context dictionary
        """
        self.log_metric("true_positives", tp, context=context)
        self.log_metric("true_negatives", tn, context=context)
        self.log_metric("false_positives", fp, context=context)
        self.log_metric("false_negatives", fn, context=context)

    def log_artifact(self, name: str, value: Any) -> None:
        """
        Log an arbitrary artifact (list, dict, etc.).

        Args:
            name: Name of the artifact
            value: Value to store
        """
        self.run[name] = value

    def add_tag(self, tag: str) -> None:
        """Add a tag to the run."""
        self.run.add_tag(tag)

    def finish(self) -> None:
        """Close the run and flush all data."""
        self.run.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures run is closed."""
        self.finish()
        return False


def create_lodo_tracker(
    experiment_name: str,
    model_name: str,
    model_params: dict[str, Any],
    llm_model: str,
    **extra_params,
) -> ExperimentTracker:
    """
    Convenience function to create a tracker for Leave-One-Dataset-Out experiments.

    Args:
        experiment_name: Name of the experiment
        model_name: Name of the ML model (e.g., "RandomForestClassifier")
        model_params: Dictionary of model hyperparameters
        llm_model: Name of the LLM model used for feature extraction
        **extra_params: Additional parameters to log

    Returns:
        Configured ExperimentTracker instance
    """
    tracker = ExperimentTracker(
        experiment_name=experiment_name,
        tags=["lodo", model_name, llm_model]
    )

    hparams = {
        "model": model_name,
        "llm_model": llm_model,
        **model_params,
        **extra_params,
    }
    tracker.log_hyperparams(hparams)

    return tracker
