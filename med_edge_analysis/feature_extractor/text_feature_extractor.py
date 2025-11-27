"""
Text-based feature extraction using MedSpaCy for clinical NLP.

Extracts semantic features from the reasoning text:
- Entity counts (medical concepts detected)
- Negation ratio (how many concepts are negated)
- Uncertainty ratio (how many concepts are uncertain)
- Historical/hypothetical mentions

These features complement the log-probability features by capturing
the semantic content of the reasoning, not just the model's confidence.
"""

from typing import Dict, Optional
from loguru import logger

# Lazy loading - only import medspacy when needed
_nlp = None
_medspacy_available = None


def _check_medspacy_available() -> bool:
    """Check if medspacy is installed and working."""
    global _medspacy_available

    if _medspacy_available is not None:
        return _medspacy_available

    try:
        import medspacy
        _medspacy_available = True
        logger.info("MedSpaCy is available")
    except ImportError:
        _medspacy_available = False
        logger.warning("MedSpaCy not installed. Text features will be skipped. "
                      "Install with: pip install medspacy")

    return _medspacy_available


def _get_nlp():
    """Lazy-load the MedSpaCy pipeline (singleton)."""
    global _nlp

    if _nlp is not None:
        return _nlp

    if not _check_medspacy_available():
        return None

    import medspacy

    logger.info("Loading MedSpaCy pipeline...")

    # Load default medspacy pipeline with context detection
    # This includes: tokenizer, sentencizer, target_matcher, context
    _nlp = medspacy.load(
        enable=["medspacy_pyrush", "medspacy_target_matcher", "medspacy_context"]
    )

    # Add rules for common medical concepts
    # MedSpaCy needs TargetRules to know what entities to look for
    from medspacy.ner import TargetRule

    target_matcher = _nlp.get_pipe("medspacy_target_matcher")

    # Add broad medical concept patterns
    # These patterns catch common medical terminology
    target_rules = [
        # Diseases/Conditions
        TargetRule("disease", "PROBLEM", pattern=[{"LOWER": {"REGEX": r"(itis|osis|emia|pathy|oma|ia)$"}}]),
        TargetRule("syndrome", "PROBLEM", pattern=[{"LOWER": {"REGEX": r"syndrome$"}}]),
        TargetRule("infection", "PROBLEM", pattern=[{"LOWER": "infection"}]),
        TargetRule("disease", "PROBLEM", pattern=[{"LOWER": "disease"}]),
        TargetRule("disorder", "PROBLEM", pattern=[{"LOWER": "disorder"}]),
        TargetRule("cancer", "PROBLEM", pattern=[{"LOWER": "cancer"}]),
        TargetRule("tumor", "PROBLEM", pattern=[{"LOWER": {"IN": ["tumor", "tumour"]}}]),
        TargetRule("failure", "PROBLEM", pattern=[{"LOWER": {"IN": ["failure", "insufficiency"]}}]),

        # Symptoms
        TargetRule("pain", "PROBLEM", pattern=[{"LOWER": "pain"}]),
        TargetRule("fever", "PROBLEM", pattern=[{"LOWER": "fever"}]),
        TargetRule("cough", "PROBLEM", pattern=[{"LOWER": "cough"}]),
        TargetRule("fatigue", "PROBLEM", pattern=[{"LOWER": "fatigue"}]),
        TargetRule("nausea", "PROBLEM", pattern=[{"LOWER": "nausea"}]),
        TargetRule("vomiting", "PROBLEM", pattern=[{"LOWER": "vomiting"}]),
        TargetRule("diarrhea", "PROBLEM", pattern=[{"LOWER": {"IN": ["diarrhea", "diarrhoea"]}}]),
        TargetRule("bleeding", "PROBLEM", pattern=[{"LOWER": "bleeding"}]),
        TargetRule("swelling", "PROBLEM", pattern=[{"LOWER": "swelling"}]),
        TargetRule("rash", "PROBLEM", pattern=[{"LOWER": "rash"}]),

        # Treatments/Drugs (common suffixes)
        TargetRule("medication", "TREATMENT", pattern=[{"LOWER": {"REGEX": r"(mycin|cillin|pril|sartan|statin|prazole|tidine|mab|nib|vir|pine|lol|zole|oxin|amine)$"}}]),
        TargetRule("therapy", "TREATMENT", pattern=[{"LOWER": "therapy"}]),
        TargetRule("treatment", "TREATMENT", pattern=[{"LOWER": "treatment"}]),
        TargetRule("surgery", "TREATMENT", pattern=[{"LOWER": "surgery"}]),
        TargetRule("procedure", "TREATMENT", pattern=[{"LOWER": "procedure"}]),

        # Tests/Findings
        TargetRule("test", "TEST", pattern=[{"LOWER": "test"}]),
        TargetRule("scan", "TEST", pattern=[{"LOWER": {"IN": ["scan", "ct", "mri", "xray", "x-ray"]}}]),
        TargetRule("biopsy", "TEST", pattern=[{"LOWER": "biopsy"}]),
        TargetRule("level", "TEST", pattern=[{"LOWER": "level"}]),  # e.g., "glucose level"

        # Common specific conditions (high frequency in medical QA)
        TargetRule("diabetes", "PROBLEM", pattern=[{"LOWER": "diabetes"}]),
        TargetRule("hypertension", "PROBLEM", pattern=[{"LOWER": {"IN": ["hypertension", "htn"]}}]),
        TargetRule("pneumonia", "PROBLEM", pattern=[{"LOWER": "pneumonia"}]),
        TargetRule("asthma", "PROBLEM", pattern=[{"LOWER": "asthma"}]),
        TargetRule("stroke", "PROBLEM", pattern=[{"LOWER": "stroke"}]),
        TargetRule("infarction", "PROBLEM", pattern=[{"LOWER": {"IN": ["infarction", "mi", "infarct"]}}]),
        TargetRule("anemia", "PROBLEM", pattern=[{"LOWER": {"IN": ["anemia", "anaemia"]}}]),
        TargetRule("sepsis", "PROBLEM", pattern=[{"LOWER": "sepsis"}]),
        TargetRule("uti", "PROBLEM", pattern=[{"LOWER": {"IN": ["uti", "cystitis", "pyelonephritis"]}}]),
    ]

    target_matcher.add(target_rules)

    logger.info(f"MedSpaCy pipeline loaded with {len(target_rules)} target rules")

    return _nlp


def extract_text_features(reasoning_content: str) -> Optional[Dict[str, float]]:
    """
    Extract semantic features from reasoning text using MedSpaCy.

    Args:
        reasoning_content: The text of the model's reasoning/chain-of-thought

    Returns:
        Dictionary of features, or None if MedSpaCy is not available

    Features extracted:
        - entity_count: Total medical entities detected
        - negation_count: Entities that are negated
        - negation_ratio: negation_count / entity_count
        - uncertainty_count: Entities marked as uncertain/possible
        - uncertainty_ratio: uncertainty_count / entity_count
        - historical_count: Entities in historical context
        - hypothetical_count: Entities in hypothetical context
        - family_count: Entities about family members
        - affirmative_count: Entities that are confirmed (not negated/uncertain)
        - affirmative_ratio: Ratio of confirmed findings
        - problem_count: Count of PROBLEM entities (diseases, symptoms)
        - treatment_count: Count of TREATMENT entities (drugs, procedures)
    """
    nlp = _get_nlp()

    if nlp is None:
        return None

    if not reasoning_content or not reasoning_content.strip():
        # Return zeros for empty text
        return _empty_features()

    # Process text
    doc = nlp(reasoning_content)

    # Count entities and their modifiers
    entity_count = len(doc.ents)

    if entity_count == 0:
        return _empty_features()

    # Initialize counters
    negation_count = 0
    uncertainty_count = 0
    historical_count = 0
    hypothetical_count = 0
    family_count = 0

    # Count by entity label
    problem_count = 0
    treatment_count = 0
    test_count = 0

    for ent in doc.ents:
        # Context modifiers
        if ent._.is_negated:
            negation_count += 1
        if ent._.is_uncertain:
            uncertainty_count += 1
        if ent._.is_historical:
            historical_count += 1
        if ent._.is_hypothetical:
            hypothetical_count += 1
        if ent._.is_family:
            family_count += 1

        # Entity types
        if ent.label_ == "PROBLEM":
            problem_count += 1
        elif ent.label_ == "TREATMENT":
            treatment_count += 1
        elif ent.label_ == "TEST":
            test_count += 1

    # Affirmative = not negated and not uncertain
    affirmative_count = entity_count - negation_count - uncertainty_count
    # Handle potential double-counting (entity can be both negated AND uncertain)
    affirmative_count = max(0, affirmative_count)

    # Calculate ratios (avoid division by zero)
    features = {
        # Raw counts
        "text_entity_count": float(entity_count),
        "text_negation_count": float(negation_count),
        "text_uncertainty_count": float(uncertainty_count),
        "text_historical_count": float(historical_count),
        "text_hypothetical_count": float(hypothetical_count),
        "text_family_count": float(family_count),
        "text_affirmative_count": float(affirmative_count),

        # Entity type counts
        "text_problem_count": float(problem_count),
        "text_treatment_count": float(treatment_count),
        "text_test_count": float(test_count),

        # Ratios
        "text_negation_ratio": negation_count / entity_count,
        "text_uncertainty_ratio": uncertainty_count / entity_count,
        "text_historical_ratio": historical_count / entity_count,
        "text_hypothetical_ratio": hypothetical_count / entity_count,
        "text_affirmative_ratio": affirmative_count / entity_count,

        # Composite scores
        # High = model is making confident assertions
        # Low = model is hedging or ruling things out
        "text_confidence_score": (entity_count - uncertainty_count) / entity_count,

        # Diagnostic reasoning style
        # High ratio = model using differential diagnosis (ruling out)
        "text_differential_ratio": (negation_count + uncertainty_count) / entity_count,
    }

    return features


def _empty_features() -> Dict[str, float]:
    """Return feature dict with all zeros for empty/invalid text."""
    return {
        "text_entity_count": 0.0,
        "text_negation_count": 0.0,
        "text_uncertainty_count": 0.0,
        "text_historical_count": 0.0,
        "text_hypothetical_count": 0.0,
        "text_family_count": 0.0,
        "text_affirmative_count": 0.0,
        "text_problem_count": 0.0,
        "text_treatment_count": 0.0,
        "text_test_count": 0.0,
        "text_negation_ratio": 0.0,
        "text_uncertainty_ratio": 0.0,
        "text_historical_ratio": 0.0,
        "text_hypothetical_ratio": 0.0,
        "text_affirmative_ratio": 0.0,
        "text_confidence_score": 0.0,
        "text_differential_ratio": 0.0,
    }


# For testing
if __name__ == "__main__":
    test_text = """
    We need to treat a pregnant woman with likely uncomplicated cystitis (UTI).
    The best treatment in pregnancy: nitrofurantoin (avoid in first trimester?
    Actually nitrofurantoin is safe except near term due to hemolytic anemia in newborn.
    At 22 weeks, nitrofurantoin is acceptable. Alternatives: ampicillin (but resistance high).
    Check other options: ciprofloxacin is contraindicated in pregnancy (fluoroquinolones).
    Doxycycline is tetracycline class, contraindicated.
    Patient denies fever and has no signs of pyelonephritis.
    History of recurrent UTIs. Family history of diabetes.
    """

    features = extract_text_features(test_text)

    if features:
        print("Extracted features:")
        for k, v in sorted(features.items()):
            print(f"  {k}: {v:.3f}")
    else:
        print("MedSpaCy not available - install with: pip install medspacy")
