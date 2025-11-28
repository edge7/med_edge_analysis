# From bigbio/med_qa
MED_QA_DATASET = "med_qa"

# From super-dainiu/medagents-benchmark
MEDAGENTS_DATASETS = [
    "AfrimedQA",
    "MedBullets",
    "MedExQA",
    "MedMCQA",
    "MedXpertQA-R",
    "MedXpertQA-U",
    "MMLU-Pro",
    "MMLU",
    "PubMedQA"
]

# All datasets combined
DATASETS = [MED_QA_DATASET] + MEDAGENTS_DATASETS

# Open-source models
MODELS = [
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B",
    "openai_gpt-oss-120b"
]

# Proprietary models (different data format - CSV, no logprobs)
PROPRIETARY_MODELS = [
    "openai_gpt-5.1"
]

# All models combined
ALL_MODELS = MODELS + PROPRIETARY_MODELS

# Split constants
SPLITS = ["train", "val", "test"]
