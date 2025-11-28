"""
Experiment Results Saver - Save and load experiment results locally.

Structure:
    results/
    ├── {model_name}/
    │   ├── {timestamp}/
    │   │   ├── lodo_results.json
    │   │   ├── feature_importance.json
    │   │   └── meta.json

Usage:
    from results.experiment_saver import ExperimentSaver

    saver = ExperimentSaver(model_name="deepseek-ai_DeepSeek-R1-Distill-Qwen-32B")
    saver.save_lodo_results(lodo_results_list)
    saver.save_feature_importance(feature_importance_dict)
    saver.save_meta(params_dict)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import pandas as pd
import numpy as np


RESULTS_DIR = Path(__file__).parent


def _convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


class ExperimentSaver:
    """Save experiment results to local filesystem."""

    def __init__(
        self,
        model_name: str,
        timestamp: Optional[str] = None,
        base_dir: Optional[Path] = None,
    ):
        """
        Initialize experiment saver.

        Args:
            model_name: Name of the LLM model used
            timestamp: Optional timestamp string. If None, uses current time.
            base_dir: Base directory for results. Defaults to results/ folder.
        """
        self.model_name = model_name
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_dir = base_dir or RESULTS_DIR

        # Create directory structure
        self.experiment_dir = self.base_dir / model_name / self.timestamp
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def save_lodo_results(self, results: list[dict[str, Any]]) -> Path:
        """
        Save Leave-One-Dataset-Out results.

        Args:
            results: List of dicts with keys like 'Dataset', 'Baseline',
                     'Filtered_Acc', 'Gain', 'Coverage', etc.

        Returns:
            Path to saved file.
        """
        filepath = self.experiment_dir / "lodo_results.json"
        with open(filepath, "w") as f:
            json.dump(_convert_to_native(results), f, indent=2)
        return filepath

    def save_feature_importance(
        self,
        importance: dict[str, float] | pd.Series,
        importance_type: str = "model",
    ) -> Path:
        """
        Save feature importance values.

        Args:
            importance: Dict or Series with feature names and importance values.
            importance_type: Type of importance (e.g., "model", "shap").

        Returns:
            Path to saved file.
        """
        if isinstance(importance, pd.Series):
            importance = importance.to_dict()

        # Sort by importance descending
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: -x[1])
        )

        filepath = self.experiment_dir / f"feature_importance_{importance_type}.json"
        with open(filepath, "w") as f:
            json.dump(_convert_to_native(sorted_importance), f, indent=2)
        return filepath

    def save_feature_selection(
        self,
        feature_summary: pd.DataFrame | list[dict],
        n_folds: int,
    ) -> Path:
        """
        Save feature selection stability across folds.

        Args:
            feature_summary: DataFrame or list with feature selection counts.
            n_folds: Total number of folds.

        Returns:
            Path to saved file.
        """
        if isinstance(feature_summary, pd.DataFrame):
            feature_summary = feature_summary.to_dict(orient="records")

        data = {
            "n_folds": n_folds,
            "features": feature_summary,
        }

        filepath = self.experiment_dir / "feature_selection.json"
        with open(filepath, "w") as f:
            json.dump(_convert_to_native(data), f, indent=2)
        return filepath

    def save_predictions(
        self,
        predictions: list[dict[str, Any]],
    ) -> Path:
        """
        Save per-sample predictions for calibration analysis.

        Args:
            predictions: List of dicts with keys:
                - sample_id: Unique identifier
                - dataset: Dataset name
                - y_true: Ground truth (0/1)
                - rf_prob: RF classifier probability
                - ans_prob: LLM answer probability

        Returns:
            Path to saved file.
        """
        filepath = self.experiment_dir / "predictions.json"
        with open(filepath, "w") as f:
            json.dump(_convert_to_native(predictions), f, indent=2)
        return filepath

    def save_meta(
        self,
        classifier_name: str,
        classifier_params: dict[str, Any],
        **extra_info,
    ) -> Path:
        """
        Save experiment metadata.

        Args:
            classifier_name: Name of the classifier (e.g., "RandomForestClassifier")
            classifier_params: Classifier hyperparameters
            **extra_info: Additional metadata (e.g., cv_splits, thresholds)

        Returns:
            Path to saved file.
        """
        meta = {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "classifier": classifier_name,
            "classifier_params": classifier_params,
            **extra_info,
        }

        filepath = self.experiment_dir / "meta.json"
        with open(filepath, "w") as f:
            json.dump(_convert_to_native(meta), f, indent=2)
        return filepath

    def get_experiment_path(self) -> Path:
        """Get the path to this experiment's directory."""
        return self.experiment_dir


class ExperimentLoader:
    """Load and aggregate experiment results."""

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize experiment loader.

        Args:
            base_dir: Base directory for results. Defaults to results/ folder.
        """
        self.base_dir = base_dir or RESULTS_DIR

    def list_models(self) -> list[str]:
        """List all models with saved experiments."""
        models = []
        for path in self.base_dir.iterdir():
            if path.is_dir() and not path.name.startswith("_") and path.name != "__pycache__":
                # Check if it has experiment subdirectories
                if any(p.is_dir() for p in path.iterdir()):
                    models.append(path.name)
        return sorted(models)

    def list_experiments(self, model_name: str) -> list[str]:
        """List all experiment timestamps for a model."""
        model_dir = self.base_dir / model_name
        if not model_dir.exists():
            return []
        return sorted([p.name for p in model_dir.iterdir() if p.is_dir()], reverse=True)

    def load_experiment(self, model_name: str, timestamp: str) -> dict[str, Any]:
        """
        Load all data from an experiment.

        Args:
            model_name: Model name
            timestamp: Experiment timestamp

        Returns:
            Dict with keys: 'meta', 'lodo_results', 'feature_importance', 'feature_selection'
        """
        exp_dir = self.base_dir / model_name / timestamp
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment not found: {exp_dir}")

        result = {"model": model_name, "timestamp": timestamp}

        # Load meta
        meta_file = exp_dir / "meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                result["meta"] = json.load(f)

        # Load LODO results
        lodo_file = exp_dir / "lodo_results.json"
        if lodo_file.exists():
            with open(lodo_file) as f:
                result["lodo_results"] = json.load(f)

        # Load feature importance (check for different types)
        for fi_file in exp_dir.glob("feature_importance_*.json"):
            importance_type = fi_file.stem.replace("feature_importance_", "")
            with open(fi_file) as f:
                result[f"feature_importance_{importance_type}"] = json.load(f)

        # Load feature selection
        fs_file = exp_dir / "feature_selection.json"
        if fs_file.exists():
            with open(fs_file) as f:
                result["feature_selection"] = json.load(f)

        # Load predictions (for calibration analysis)
        pred_file = exp_dir / "predictions.json"
        if pred_file.exists():
            with open(pred_file) as f:
                result["predictions"] = json.load(f)

        return result

    def load_latest(self, model_name: str) -> dict[str, Any]:
        """Load the most recent experiment for a model."""
        experiments = self.list_experiments(model_name)
        if not experiments:
            raise FileNotFoundError(f"No experiments found for model: {model_name}")
        return self.load_experiment(model_name, experiments[0])

    def load_all_latest(self) -> list[dict[str, Any]]:
        """Load the latest experiment for each model."""
        results = []
        for model in self.list_models():
            try:
                results.append(self.load_latest(model))
            except FileNotFoundError:
                continue
        return results

    def compare_models(self) -> pd.DataFrame:
        """
        Create a comparison DataFrame of latest results across all models.

        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        all_results = self.load_all_latest()

        rows = []
        for exp in all_results:
            if "lodo_results" not in exp:
                continue

            # Aggregate metrics
            lodo = exp["lodo_results"]
            df = pd.DataFrame(lodo)

            row = {
                "model": exp["model"],
                "timestamp": exp["timestamp"],
                "avg_baseline": df["Baseline"].mean() if "Baseline" in df else None,
                "avg_filtered": df["Filtered_Acc"].mean() if "Filtered_Acc" in df else None,
                "avg_gain": df["Gain"].mean() if "Gain" in df else None,
                "avg_coverage": df["Coverage"].mean() if "Coverage" in df else None,
            }
            rows.append(row)

        return pd.DataFrame(rows)
