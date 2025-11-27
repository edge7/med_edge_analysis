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

# Model constants
MODELS = [
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B",
    "openai_gpt-oss-120b"
]

# Split constants
SPLITS = ["train", "val", "test"]
