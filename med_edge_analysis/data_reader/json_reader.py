from dataclasses import dataclass
from pandas import DataFrame
import pandas as pd
import json
from pathlib import Path
from typing import List, Optional
from loguru import logger

from med_edge_analysis.constants import DATASETS, MODELS, SPLITS, MED_QA_DATASET
from med_edge_analysis.feature_extractor.log_probs_extractor import extract_meta_features
from med_edge_analysis.feature_extractor.fast_text_features import extract_fast_text_features

MED_QA_FOLDER = "/media/edge7/Extreme Pro/med_edge/benchmarks"
MED_AGENT = "/media/edge7/Extreme Pro/med_edge/benchmarks_test"


@dataclass
class Dataset:
    model: str
    dataset: str
    split: str
    pandas_df : DataFrame

    def get_memory_usage_gb(self) -> float:
        """Returns memory usage of the DataFrame in GB"""
        return self.pandas_df.memory_usage(deep=True).sum() / (1024 ** 3)


def load_dataset(
    dataset_name: Optional[str] = None,
    split: Optional[str] = None,
    model: Optional[str] = None,
    columns: Optional[List[str]] = None,
    extract_features: bool = False
) -> List[Dataset]:
    """
    Loads datasets based on optional filters.

    Args:
        dataset_name: Specific dataset to load (e.g., "med_qa", "AfrimedQA"). If None, loads all datasets.
        split: Specific split to load (e.g., "train", "val", "test"). If None, loads all splits.
        model: Specific model to load (e.g., "openai_gpt-oss-120b"). If None, loads all models.
        columns: Optional list of column names to load. If None, loads all columns. Use this to reduce memory usage.
                 NOTE: Ignored when extract_features=True.
        extract_features: If True, extracts metacognitive features on-the-fly during loading instead of
                         keeping raw data. This drastically reduces memory usage. Default: False.

    Returns:
        List of Dataset objects matching the filters.
    """
    datasets = []

    # Determine which datasets, splits, and models to load
    datasets_to_load = [dataset_name] if dataset_name else DATASETS
    splits_to_load = [split] if split else SPLITS
    models_to_load = [model] if model else MODELS

    for ds_name in datasets_to_load:
        for sp in splits_to_load:
            for mdl in models_to_load:
                # Construct filepath based on dataset type
                if ds_name == MED_QA_DATASET:
                    # MED_QA files: {model}_{split}_raw.jsonl
                    filepath = Path(MED_QA_FOLDER) / f"{mdl}_{sp}_raw.jsonl"
                else:
                    # MED_AGENT files: {model}_medagents_{dataset}_{split}_raw.jsonl
                    filepath = Path(MED_AGENT) / f"{mdl}_medagents_{ds_name}_{sp}_raw.jsonl"

                # Load if file exists
                if filepath.exists():
                    df = _read_jsonl(filepath, columns=columns, extract_features=extract_features)
                    datasets.append(Dataset(
                        model=mdl,
                        dataset=ds_name,
                        split=sp,
                        pandas_df=df
                    ))

    return datasets


def _read_jsonl(filepath: Path, chunk_size: int = 10000, columns: Optional[List[str]] = None, extract_features: bool = False) -> DataFrame:
    """
    Reads a JSONL file (one JSON object per line) and returns a pandas DataFrame.

    Args:
        filepath: Path to the JSONL file
        chunk_size: Number of records to process at once before creating DataFrame chunks
        columns: Optional list of column names to load. If None, loads all columns. Ignored if extract_features=True.
        extract_features: If True, extracts metacognitive features on-the-fly instead of keeping raw data.
    """

    # Check for duplications!
    from collections import defaultdict

    by_id = defaultdict(set)
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_id[r["sample_id"]].add(r["ground_truth"])
    inconsistent_ids = {sid for sid, answers in by_id.items() if len(answers) > 1}
    logger.warning(f"Found {len(inconsistent_ids)} inconsistent sample_ids to skip")

    chunks = []
    current_chunk = []
    skipped_count = 0
    total_count = 0

    # Metadata columns to preserve when extracting features
    METADATA_COLUMNS = ['is_correct', 'sample_id', 'answer', 'ground_truth']

    sample_ids = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total_count += 1
            assert record['finish_reason'] == "stop" # Let's be sure all is good here
            if record['sample_id']  in sample_ids:
                continue # This is a repetition, but at least it's consistent
            if record['sample_id'] in inconsistent_ids:
                continue # skip garbage

            sample_ids.add(record['sample_id'])

            if extract_features:
                # Extract features on-the-fly
                try:
                    # extract_meta_features expects the 'logprobs' field
                    logprobs_data = record.get('logprobs')
                    if logprobs_data is None:
                        logger.warning(f"Record {total_count} missing 'logprobs' field, skipping")
                        skipped_count += 1
                        continue

                    # Extract log-prob features
                    features = extract_meta_features(logprobs_data)

                    # Extract fast text features from reasoning content (if available)
                    reasoning_content = record.get('reasoning_content')
                    if reasoning_content:
                        text_features = extract_fast_text_features(reasoning_content)
                        if text_features:
                            features.update(text_features)

                    # Preserve important metadata
                    for meta_col in METADATA_COLUMNS:
                        if meta_col in record:
                            if meta_col == "sample_id":
                                features[meta_col] = f"{record['dataset_name']}_{record.get('dataset_config', '')}_{record[meta_col]}"
                            else:
                                features[meta_col] = record[meta_col]

                    current_chunk.append(features)

                except AssertionError as e:
                    # Reasoning chain too short (< 6 tokens)
                    logger.debug(f"Record {total_count}: {e}")
                    skipped_count += 1
                    continue
                except Exception as e:
                    # Other errors
                    logger.warning(f"Record {total_count} feature extraction failed: {e}")
                    skipped_count += 1
                    continue
            else:
                # Original behavior: load raw data
                # Filter columns if specified
                if columns is not None:
                    record = {k: v for k, v in record.items() if k in columns}
                record['sample_id'] = f"{record['dataset_name']}_{record.get('dataset_config', '')}_{record['sample_id']}"
                current_chunk.append(record)

            # Process in chunks to avoid memory spikes
            if len(current_chunk) >= chunk_size:
                chunks.append(pd.DataFrame(current_chunk))
                current_chunk = []

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(pd.DataFrame(current_chunk))

    # Log statistics if features were extracted
    if extract_features and total_count > 0:
        success_rate = ((total_count - skipped_count) / total_count) * 100
        logger.info(f"Feature extraction: {total_count - skipped_count}/{total_count} records ({success_rate:.1f}% success, {skipped_count} skipped)")

    # Concatenate all chunks into final DataFrame
    # Use copy=False to avoid unnecessary copies
    if chunks:
        df = pd.concat(chunks, ignore_index=True, copy=False)

        # Optimize dtypes to reduce memory usage (only for raw data, features are already optimized)
        if not extract_features:
            df = _optimize_dtypes(df)

        return df
    else:
        return pd.DataFrame()


def get_baseline_stats(
    dataset_name: str,
    model: str,
    split: Optional[str] = None,
) -> Optional[dict]:
    """
    Efficiently get baseline accuracy and row count without loading full data.

    Single pass through the file, only reads 'is_correct' and 'sample_id'.

    Args:
        dataset_name: Dataset name (e.g., "med_qa", "AfrimedQA")
        model: Model name
        split: Split name. If None and dataset is med_qa, combines all splits.

    Returns:
        Dict with 'accuracy', 'n_correct', 'n_total' or None if file not found.
    """
    # Determine which splits to load
    if split is None and dataset_name == MED_QA_DATASET:
        splits_to_load = SPLITS  # train, val, test
    else:
        splits_to_load = [split] if split else ["test"]

    total_correct = 0
    total_count = 0
    sample_ids_seen = set()

    for sp in splits_to_load:
        # Construct filepath
        if dataset_name == MED_QA_DATASET:
            filepath = Path(MED_QA_FOLDER) / f"{model}_{sp}_raw.jsonl"
        else:
            filepath = Path(MED_AGENT) / f"{model}_medagents_{dataset_name}_{sp}_raw.jsonl"

        if not filepath.exists():
            continue

        # First pass: find inconsistent sample_ids
        from collections import defaultdict
        by_id = defaultdict(set)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    by_id[r["sample_id"]].add(r["ground_truth"])
        inconsistent_ids = {sid for sid, answers in by_id.items() if len(answers) > 1}

        # Second pass: count correct/total (only read what we need)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                # Minimal JSON parsing - only extract needed fields
                record = json.loads(line)

                sample_id = record["sample_id"]
                if sample_id in sample_ids_seen or sample_id in inconsistent_ids:
                    continue

                sample_ids_seen.add(sample_id)
                total_count += 1
                if record["is_correct"]:
                    total_correct += 1

    if total_count == 0:
        return None

    return {
        "accuracy": total_correct / total_count,
        "n_correct": total_correct,
        "n_total": total_count,
    }


def _optimize_dtypes(df: DataFrame) -> DataFrame:
    """
    Optimizes DataFrame dtypes to reduce memory usage.

    - Converts low-cardinality string columns to categorical
    - Downcasts numeric columns where possible
    - Converts bool-like int columns to bool

    Args:
        df: Input DataFrame

    Returns:
        Optimized DataFrame
    """
    # Categorical threshold: if unique values < 50% of total rows, use categorical
    categorical_threshold = 0.5

    for col in df.columns:
        col_type = df[col].dtype

        # Convert object/string columns to categorical if low cardinality
        if col_type == 'object':
            try:
                num_unique = df[col].nunique()
                num_total = len(df)

                if num_unique / num_total < categorical_threshold:
                    df[col] = df[col].astype('category')
            except TypeError:
                # Skip columns with unhashable types (e.g., dicts, lists)
                pass

        # Downcast integers
        elif col_type in ['int64', 'int32', 'int16']:
            # Check if it's a boolean disguised as int (only 0 and 1)
            unique_vals = df[col].unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                df[col] = df[col].astype('bool')
            else:
                df[col] = pd.to_numeric(df[col], downcast='integer')

        # Downcast floats
        elif col_type in ['float64', 'float32']:
            df[col] = pd.to_numeric(df[col], downcast='float')

    return df
