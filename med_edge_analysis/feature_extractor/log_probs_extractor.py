from typing import List
import numpy as np
from scipy.stats import linregress

import numpy as np
from typing import Dict, Any, List


def extract_meta_features(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts metacognitive features from the model's log-probabilities to predict
    hallucinations or errors in medical reasoning.

    The extraction focuses on three dimensions:
    1. Overall Confidence (Static stats like mean, min, percentile).
    2. Reasoning Dynamics (Trend, volatility, and 'conceptual dips' via rolling windows).
    3. Answer Certainty (Entropy and probability of the final selected option).

    Args:
        data (Dict): The raw JSON object containing 'content' (tokens) and indices.

    Returns:
        Dict[str, float]: A flat dictionary of features for the meta-learner.
    """

    # --- 1. Pre-processing & Validation ---
    window_size = 5
    log_probs = data["content"]

    # Filter only reasoning tokens (Chain-of-Thought)
    reasoning_tokens = [x for x in log_probs if x.get("token_type") == "reasoning"]

    # HARD ASSERTION: Chain-of-Thought must be substantial to perform time-series analysis.
    # A reasoning chain shorter than 6 tokens is statistically insignificant for this method.
    assert (
        len(reasoning_tokens) >= 6
    ), f"Reasoning too short ({len(reasoning_tokens)} tokens). Need >= 6 for split analysis."

    # Convert log-space probabilities to a numpy array for vector operations
    r_logprobs = np.array([float(t["logprob"]) for t in reasoning_tokens])

    # Initialize feature dictionary
    feats = {}

    # --- NEW: Entropy and Margin features from top_logprobs ---
    # These capture the model's uncertainty over alternative token choices
    entropies = []
    margins = []

    for token in reasoning_tokens:
        tops = token.get("top_logprobs", [])

        assert len(tops) == 20 # We forced this in the ingestion phase

            # Calculate entropy (measure of uncertainty across all alternatives)
        top_logprobs = np.array([float(t["logprob"]) for t in tops])
        probs = np.exp(top_logprobs)
        probs_norm = probs / np.sum(probs)
        H = -np.sum(probs_norm * np.log(probs_norm + 1e-9))
        entropies.append(H)

        # Calculate margin (confidence gap between top-1 and top-2)
        margin = float(tops[0]["logprob"]) - float(tops[1]["logprob"])
        margins.append(margin)

    # Aggregate entropy features
    if entropies:
        feats["reasoning_avg_entropy"] = np.mean(entropies)
        feats["reasoning_max_entropy"] = np.max(entropies)  # Peak confusion
        feats["reasoning_std_entropy"] = np.std(entropies)  # Volatility of uncertainty
        feats["reasoning_min_entropy"] = np.min(entropies)  # Most confident moment
        feats["reasoning_entropy_range"] = np.max(entropies) - np.min(entropies)  # Spread

        # Coefficient of variation: normalized volatility (std/mean)
        # High CV = uncertainty varies a lot relative to average
        feats["reasoning_entropy_cv"] = np.std(entropies) / (np.mean(entropies) + 1e-9)
    else:
        feats["reasoning_avg_entropy"] = 0.0
        feats["reasoning_max_entropy"] = 0.0
        feats["reasoning_std_entropy"] = 0.0
        feats["reasoning_min_entropy"] = 0.0
        feats["reasoning_entropy_range"] = 0.0
        feats["reasoning_entropy_cv"] = 0.0

    # Aggregate margin features
    if margins:
        feats["reasoning_avg_margin"] = np.mean(margins)
        feats["reasoning_min_margin"] = np.min(margins)  # Most uncertain choice
    else:
        raise Exception("This should not happen. Margin should be available")

    # --- NEW: Temporal Evolution of Entropy/Margin (3-part split) ---
    # Track how uncertainty evolves during reasoning: convergence vs divergence
    if entropies:
        entropy_array = np.array(entropies)
        entropy_parts = np.array_split(entropy_array, 3)

        feats["entropy_start"] = np.mean(entropy_parts[0])
        feats["entropy_middle"] = np.mean(entropy_parts[1])
        feats["entropy_end"] = np.mean(entropy_parts[2])

        # Convergence/Divergence trends
        # Negative trend = model converges (uncertainty decreases) → GOOD
        # Positive trend = model diverges (uncertainty increases) → BAD
        feats["entropy_trend"] = feats["entropy_end"] - feats["entropy_start"]
        feats["entropy_delta_middle"] = feats["entropy_middle"] - feats["entropy_start"]
    else:
        raise Exception("This should not happen. Entropies should be available")

    if margins:
        margin_array = np.array(margins)
        margin_parts = np.array_split(margin_array, 3)

        feats["margin_start"] = np.mean(margin_parts[0])
        feats["margin_middle"] = np.mean(margin_parts[1])
        feats["margin_end"] = np.mean(margin_parts[2])

        # Convergence/Divergence trends
        # Positive trend = confidence increases → GOOD
        # Negative trend = confidence decreases → BAD
        feats["margin_trend"] = feats["margin_end"] - feats["margin_start"]
        feats["margin_delta_middle"] = feats["margin_middle"] - feats["margin_start"]
    else:
        feats["margin_start"] = 0.0
        feats["margin_middle"] = 0.0
        feats["margin_end"] = 0.0
        feats["margin_trend"] = 0.0
        feats["margin_delta_middle"] = 0.0

    # --- 2. Global Static Statistics ---
    # Captures the general "mood" of the model across the entire reasoning trace.
    feats["global_avg_lp"] = np.mean(r_logprobs)
    feats["global_min_lp"] = np.min(
        r_logprobs
    )  # The single point of maximum uncertainty
    feats["global_p10_lp"] = np.percentile(
        r_logprobs, 10
    )  # Robust minimum (ignoring outliers)
    feats["global_std_lp"] = np.std(r_logprobs)  # Overall volatility

    # --- 3. Local Dynamics (Rolling Window) ---
    # Identifies "conceptual dips": consecutive tokens with low confidence.
    # Unlike 'min_lp' (which can be a single rare word), this captures weak arguments.
    rolling_avgs = np.convolve(
        r_logprobs, np.ones(window_size) / window_size, mode="valid"
    )
    feats["min_rolling_lp"] = np.min(
        rolling_avgs
    )  # Lowest confidence over a 5-token span

    # --- 4. Structural Dynamics (3-Part Split) ---
    # Splits reasoning into: 1. Setup/Premise -> 2. Conflict/Evaluation -> 3. Conclusion.
    # This captures the "narrative arc" of the solution.
    parts = np.array_split(r_logprobs, 3)
    p1, p2, p3 = parts[0], parts[1], parts[2]

    # Calculate stats for each section to detect drifts
    feats["start_avg_lp"] = np.mean(p1)
    feats["middle_avg_lp"] = np.mean(p2)
    feats["end_avg_lp"] = np.mean(p3)

    # --- 5. Trend Analysis (Deltas) ---
    # Crucial for detecting hallucinations.
    # Positive trend = Convergence (Problem solved).
    # Negative trend = Divergence (Model got confused).

    # "Initial Shock": Confidence drop when moving from premise to complex evaluation
    feats["delta_middle_start"] = feats["middle_avg_lp"] - feats["start_avg_lp"]

    # "Resolution Recovery": Confidence gain when moving to conclusion
    feats["delta_end_middle"] = feats["end_avg_lp"] - feats["middle_avg_lp"]

    # "Overall Trend": Did the model end up more confident than it started?
    feats["trend_overall"] = feats["end_avg_lp"] - feats["start_avg_lp"]

    # --- 6. Answer Metrics (Final Choice) ---
    # Analyzes the specific token selected as the final answer (e.g., "A", "B", "Option 1").
    ans_idx = data["answer_token_index"]
    ans_token = log_probs[ans_idx]

    # Linear Probability of the chosen option (0.0 to 1.0)
    feats["ans_prob"] = np.exp(float(ans_token["logprob"]))
    feats["reasoning_length"] = len(reasoning_tokens)

    # Entropy of the answer distribution (Confusion among options)
    # High entropy means the model considered other options as nearly equally valid.
    ans_tops = ans_token.get("top_logprobs", [])
    if ans_tops:
        # Convert logprobs to linear probabilities
        probs = np.exp([float(t["logprob"]) for t in ans_tops])
        # Normalize (since top_k is a subset of vocabulary)
        probs_norm = probs / np.sum(probs)
        # Calculate Shannon Entropy: H = -sum(p * log(p))
        feats["ans_entropy"] = -np.sum(probs_norm * np.log(probs_norm + 1e-9))
    else:
        raise ValueError("Entropy error")

    return feats
