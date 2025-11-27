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
    # causal
    "because", "since", "as", "given", "therefore",
    "hence", "thus", "consequently", "accordingly",
    "resulting", "resulting_in",

    # contrast / concession
    "however", "although", "though", "even_though",
    "whereas", "while", "yet", "nonetheless",
    "nevertheless", "despite", "in_spite_of",
    "on_the_other_hand",

    # additive / sequencing
    "moreover", "furthermore", "additionally",
    "besides", "also", "next", "then", "afterward",
    "initially", "first", "second", "third", "finally",

    # conditional
    "if", "unless", "provided", "assuming",
    "in_case", "otherwise", "rather",

    # justification / inference
    "indicating", "implying", "meaning",
    "thereby", "according_to",

    # reasoning meta-markers
    "considering", "based_on", "given_that",
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

    features = {
        # Basic counts
        "fast_total_word_count": float(total_words),
        "fast_unique_word_count": float(unique_word_count),
        "fast_sentence_count": float(sentence_count),
        "fast_question_count": float(question_count),

        # Keyword counts
        "fast_hedge_count": float(hedge_count),
        "fast_reasoning_connector_count": float(reasoning_count),
        "fast_differential_keyword_count": float(differential_count),
        "fast_conclusion_word_count": float(conclusion_count),
        "fast_medical_action_count": float(medical_action_count),

        # Structure metrics
        "fast_avg_word_length": avg_word_length,
        "fast_vocabulary_richness": vocabulary_richness,
        "fast_hapax_ratio": hapax_ratio,

        # Normalized ratios
        "fast_hedge_ratio": hedge_ratio,
        "fast_confidence_ratio": confidence_ratio,
        "fast_negation_ratio": negation_ratio,
        "fast_reasoning_density": reasoning_count / total_words,
        "fast_question_density": question_count / sentence_count,

        # Composite scores
        "fast_certainty_score": certainty_score,
        "fast_analytical_score": (reasoning_count + differential_count) / total_words,
    }

    return features


def _empty_features() -> Dict[str, float]:
    """Return feature dict with all zeros for empty/invalid text."""
    return {
        "fast_total_word_count": 0.0,
        "fast_unique_word_count": 0.0,
        "fast_sentence_count": 0.0,
        "fast_question_count": 0.0,
        "fast_hedge_count": 0.0,
        "fast_reasoning_connector_count": 0.0,
        "fast_differential_keyword_count": 0.0,
        "fast_conclusion_word_count": 0.0,
        "fast_medical_action_count": 0.0,
        "fast_avg_word_length": 0.0,
        "fast_vocabulary_richness": 0.0,
        "fast_hapax_ratio": 0.0,
        "fast_hedge_ratio": 0.0,
        "fast_confidence_ratio": 0.0,
        "fast_negation_ratio": 0.0,
        "fast_reasoning_density": 0.0,
        "fast_question_density": 0.0,
        "fast_certainty_score": 0.0,
        "fast_analytical_score": 0.0,
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
