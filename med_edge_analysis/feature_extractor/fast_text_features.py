"""
Fast text-based feature extraction using pure Python.

No heavy dependencies - just regex and string operations.
Designed to be orders of magnitude faster than NLP-based approaches.
"""

import re
from typing import Dict, Optional
from collections import Counter


# Precompiled patterns for speed
_SENTENCE_SPLIT = re.compile(r'[.!?]+')
_WORD_PATTERN = re.compile(r'\b[a-zA-Z]+\b')
_QUESTION_PATTERN = re.compile(r'\?')

# Keyword sets (lowercase)
HEDGE_WORDS = frozenset(
    {
        # core modal verbs
        "maybe",
        "possibly",
        "perhaps",
        "might",
        "may",
        "could",
        "can",
        "would",
        # probability / likelihood
        "probable",
        "probably",
        "possible",
        "possibly",
        "uncertain",
        "uncertainty",
        "unlikely",
        "likely",
        "questionable",
        "ambiguous",
        "ambiguity",
        "approximate",
        "approximately",
        "roughly",
        "broadly",
        # weak assertions
        "suggest",
        "suggests",
        "suggested",
        "suggesting",
        "indicate",
        "indicates",
        "indicated",
        "indicating",
        "imply",
        "implies",
        "implied",
        "implying",
        "hint",
        "hints",
        "hinted",
        "point",
        "points",
        "pointed",
        "lean",
        "leans",
        # appearance
        "appear",
        "appears",
        "appeared",
        "appearing",
        "seem",
        "seems",
        "seemed",
        "seeming",
        "suggestive",
        "apparently",
        # uncertainty / partial confidence
        "unclear",
        "unsure",
        "doubtful",
        "doubt",
        "presumed",
        "presume",
        "presumably",
        "suspected",
        "suspect",
        "suspects",
        "tentative",
        "tentatively",
        "inconclusive",
        "incomplete",
        "not_certain",
        "not_sure",  # these appear as "not sure" etc.
        # softeners
        "somewhat",
        "partially",
        "partly",
        "generally",
        "typically",  # hedging in med reasoning
        "often",
        "rarely",
        "occasionally",
        "commonly",
        "sometimes",
        # multiword hedge expressions (as single tokens for partial matching)
        "more_likely",
        "less_likely",
        "most_likely",
        "cannot_rule_out",
        "cant_rule_out",
        "cannot_exclude",
        "cant_exclude",
        "rule_out",  # hedge in med context
        "more_consistent",
        "less_consistent",
    }
)


CONFIDENCE_WORDS = frozenset({
    # strong certainty adverbs
    "definitely", "certainly", "clearly", "obviously",
    "undoubtedly", "absolutely", "surely", "evidently",
    "plainly", "indeed", "unquestionably",

    # strong verbs / modal certainty
    "must", "prove", "proves", "proven", "proved",
    "confirm", "confirms", "confirmed", "confirming",
    "establish", "establishes", "established",
    "demonstrate", "demonstrates", "demonstrated",

    # certainty adjectives
    "certain", "sure", "confident", "positive",
    "definite", "clear", "obvious", "evident",
    "conclusive", "decisive",

    # absolute qualifiers
    "always", "never", "inevitably",
    "undeniable", "unmistakable", "irrefutable",

    # strong correctness markers
    "correct", "exact", "precise",

    # factuality assertions
    "known", "established", "verified", "validated",
    "guaranteed", "assured",

    # consistency assertions that imply correctness
    "consistent", "incontrovertible"
})


NEGATION_WORDS = frozenset({
    # base negation
    "not", "no", "none", "never", "neither",
    "cannot", "can't", "dont", "don't",
    "doesnt", "doesn't", "didnt", "didn't",
    "wont", "won't", "wouldnt", "wouldn't",
    "shouldnt", "shouldn't",
    "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't",

    # explicit absence / lack
    "without", "absent", "lacking", "lack", "insufficient",
    "nonexistent", "nonreactive", "negative", "neg",
    "free_of",

    # medical negation markers
    "no_evidence", "no_signs", "no_symptoms",
    "no_indication", "no_change",
    "unremarkable",    # medico: "normal / nothing abnormal"

    # exclusion verbs (strong negation in clinical reasoning)
    "deny", "denies", "denied",
    "exclude", "excludes", "excluded", "excluding",
    "rule", "rules", "ruled", "ruling",
    "rule_out", "ruled_out", "ruling_out",

    # negative likelihood (these *are* negations)
    "unlikely", "improbable",

    # negation intensifiers
    "never_seen", "cannot_exclude", "cant_exclude",
    "unable", "unable_to",  # es: "unable to determine"
})


REASONING_CONNECTORS = frozenset({
    # causal (strong logical connectors)
    "because", "since", "therefore", "hence", "thus",
    "consequently", "accordingly", "so", "ergo",

    # contrast / concession
    "however", "although", "though", "whereas",
    "yet", "nonetheless", "nevertheless", "despite",
    "conversely", "alternatively",

    # additive / sequencing
    "moreover", "furthermore", "additionally",
    "besides", "afterward", "initially",
    "first", "second", "third", "finally",
    "subsequently", "previously",

    # conditional
    "if", "unless", "provided", "assuming",
    "otherwise", "rather",

    # justification / inference
    "indicating", "implying", "suggests", "indicates",
    "implies", "shows", "demonstrates", "meaning",
    "thereby", "confirms", "supports",

    # emphasis / focus (signal key reasoning steps)
    "notably", "importantly", "crucially", "specifically",
    "particularly", "especially", "primarily", "mainly",

    # conclusive markers
    "ultimately", "overall", "essentially", "clearly",
    "evidently", "obviously",

    # comparative
    "similarly", "likewise", "namely", "indeed",

    # reasoning meta-markers
    "considering", "given", "assuming",
})


DIFFERENTIAL_KEYWORDS = frozenset({
    # differential diagnosis
    "differential", "differentials", "ddx",

    # compare / contrast diagnoses
    "versus", "vs",
    "compared", "compared_to", "relative_to",

    # exclusion / ruling in vs out
    "rule", "ruled", "ruling",
    "exclude", "excludes", "excluded", "excluding",

    # consider alternatives
    "consider", "considers", "considered", "considering",
    "alternatively", "alternative",

    # likelihood comparison
    "more", "most", "less", "least",
    "more_likely", "less_likely",
    "more_consistent", "less_consistent",
    "more_common", "less_common",

    # frequency / typicality
    "common", "rare", "rarer", "rarely",
    "typical", "atypical", "classic", "classical",
    "characteristic", "characteristically",

    # compatibility
    "consistent", "inconsistent",
    "compatible", "incompatible",

    # directional judgement
    "favor", "favors", "favored", "favoring",
    "against", "support", "supports", "supported",
})


CONCLUSION_WORDS = frozenset({
    # explicit conclusion
    "answer", "conclusion", "final", "finally",
    "ultimately", "in_summary", "in_conclusion",

    # inferential closers
    "therefore", "thus", "hence",

    # decision / choice
    "diagnosis", "diagnosed",
    "correct", "best", "optimal", "preferred",
    "choice", "option", "select", "selected",
    "choose", "chosen",

    # likelihood-based close
    "likely", "most_likely", "least_likely",
    "consistent_with", "supports",

    # outcome phrases
    "result", "results_in", "leading_to",
})


MEDICAL_ACTION_WORDS = frozenset({
    # treatment
    "treat", "treatment", "therapy", "therapeutic",
    "manage", "management", "intervene", "intervention",

    # prescribing / giving meds
    "prescribe", "prescribed",
    "administer", "administered",
    "initiate", "initiated",
    "start", "started",

    # diagnostic process
    "diagnose", "diagnosed", "diagnosis", "diagnostic",
    "assess", "assessed", "evaluate", "evaluated",
    "test", "tested", "testing",
    "examine", "examined",
    "screen", "screened", "screening",

    # monitoring / follow-up
    "monitor", "monitored",
    "follow", "follow_up", "reassess",

    # referral
    "refer", "referred", "consult", "consulted",

    # ordering studies
    "order", "ordered", "perform", "performed",
    "obtain", "obtained",

    # imaging
    "image", "imaging",
    "scan", "scanned",
    "xray", "ct", "mri", "ultrasound",
    "echocardiogram", "ecg", "ekg",

    # labs and sampling
    "measure", "measured",
    "sample", "sampled",
    "draw", "drawn",
})



def extract_fast_text_features(text: str) -> Optional[Dict[str, float]]:
    """
    Extract text features using fast string operations.

    Args:
        text: The reasoning content text

    Returns:
        Dictionary of features, or None if text is empty
    """
    if not text or not text.strip():
        return _empty_features()

    text_lower = text.lower()

    # Extract all words (alphabetic only)
    words = _WORD_PATTERN.findall(text_lower)

    if not words:
        return _empty_features()

    total_words = len(words)
    unique_words = set(words)
    unique_word_count = len(unique_words)

    # Word frequency for analysis
    word_freq = Counter(words)

    # Sentence analysis
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    sentence_count = max(len(sentences), 1)

    # Count keywords by category
    hedge_count = sum(1 for w in words if w in HEDGE_WORDS)
    confidence_count = sum(1 for w in words if w in CONFIDENCE_WORDS)
    negation_count = sum(1 for w in words if w in NEGATION_WORDS)
    reasoning_count = sum(1 for w in words if w in REASONING_CONNECTORS)
    differential_count = sum(1 for w in words if w in DIFFERENTIAL_KEYWORDS)
    conclusion_count = sum(1 for w in words if w in CONCLUSION_WORDS)
    medical_action_count = sum(1 for w in words if w in MEDICAL_ACTION_WORDS)

    # Structure features
    question_count = len(_QUESTION_PATTERN.findall(text))

    # Calculate derived metrics
    avg_word_length = sum(len(w) for w in words) / total_words
    vocabulary_richness = unique_word_count / total_words  # Type-token ratio

    # Hapax legomena ratio (words appearing only once)
    hapax_count = sum(1 for w, c in word_freq.items() if c == 1)
    hapax_ratio = hapax_count / unique_word_count if unique_word_count > 0 else 0

    # Reasoning style scores (normalized by total words)
    hedge_ratio = hedge_count / total_words
    confidence_ratio = confidence_count / total_words
    negation_ratio = negation_count / total_words

    # Certainty score: confidence words vs hedge words
    certainty_denom = confidence_count + hedge_count
    certainty_score = (confidence_count - hedge_count) / certainty_denom if certainty_denom > 0 else 0

    # --- NEW FEATURES ---
    # Vocabulary growth rate: how fast does vocabulary saturate?
    # Split words into 3 parts, calculate cumulative unique words
    # Rate < 1 means vocabulary saturates (repetitive), > 1 means keeps growing (diverse)
    if total_words >= 6:
        third = total_words // 3
        words_part1 = words[:third]
        words_part2 = words[:2*third]
        words_part3 = words  # all words

        unique_part1 = len(set(words_part1))
        unique_part2 = len(set(words_part2))
        unique_part3 = len(set(words_part3))

        # Growth rate: ratio of actual unique words vs expected if linear
        # Expected at part 2: unique_part1 * 2, Expected at part 3: unique_part1 * 3
        expected_part2 = unique_part1 * 2
        expected_part3 = unique_part1 * 3
        vocab_growth_rate = ((unique_part2 / (expected_part2 + 1e-9)) +
                            (unique_part3 / (expected_part3 + 1e-9))) / 2
    else:
        vocab_growth_rate = 1.0  # neutral for short texts

    features = {
        # Basic counts (commented out - absolute values correlate with text length)
        # "fast_total_word_count": float(total_words),
        # "fast_sentence_count": float(sentence_count),
        # "fast_question_count": float(question_count),

        # Keyword densities (normalized)
        # "fast_hedge_count": float(hedge_count),  # use fast_hedge_ratio instead
        "fast_differential_density": differential_count / total_words,
        "fast_conclusion_density": conclusion_count / total_words,
        # "fast_medical_action_count": float(medical_action_count),  # low importance

        # Structure metrics
        "fast_avg_word_length": avg_word_length,
        "fast_vocabulary_richness": vocabulary_richness,
        "fast_hapax_ratio": hapax_ratio,

        # Normalized ratios
        "fast_hedge_ratio": hedge_ratio,
        "fast_confidence_ratio": confidence_ratio,
        "fast_negation_ratio": negation_ratio,
        "fast_question_density": question_count / sentence_count,

        # Composite scores
        "fast_certainty_score": certainty_score,
        "fast_analytical_score": (reasoning_count + differential_count) / total_words,

        # New features
        "fast_vocab_growth_rate": vocab_growth_rate,
    }

    return features


def _empty_features() -> Dict[str, float]:
    """Return feature dict with all zeros for empty/invalid text."""
    return {
        # Absolute counts commented out
        # "fast_total_word_count": 0.0,
        # "fast_sentence_count": 0.0,
        # "fast_question_count": 0.0,
        # "fast_hedge_count": 0.0,
        "fast_differential_density": 0.0,
        "fast_conclusion_density": 0.0,
        # "fast_medical_action_count": 0.0,
        "fast_avg_word_length": 0.0,
        "fast_vocabulary_richness": 0.0,
        "fast_hapax_ratio": 0.0,
        "fast_hedge_ratio": 0.0,
        "fast_confidence_ratio": 0.0,
        "fast_negation_ratio": 0.0,
        "fast_question_density": 0.0,
        "fast_certainty_score": 0.0,
        "fast_analytical_score": 0.0,
        "fast_vocab_growth_rate": 1.0,  # neutral default
    }


if __name__ == "__main__":
    test_text = """
    The patient presents with fever and cough. We need to consider pneumonia
    versus bronchitis. Given the chest X-ray findings, pneumonia is more likely.
    However, we cannot rule out tuberculosis completely. The WBC count of 15,000
    suggests bacterial infection. Should we start empiric antibiotics?
    I think amoxicillin would be appropriate. The diagnosis is most consistent
    with community-acquired pneumonia. Therefore, the answer is B.
    """

    features = extract_fast_text_features(test_text)

    if features:
        print("Extracted fast text features:")
        for k, v in sorted(features.items()):
            if isinstance(v, float) and v != int(v):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v:.0f}")
