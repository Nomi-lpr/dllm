import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset, concatenate_datasets

def log(msg):
    print(msg, flush=True)

# ----- C-Eval: 52 configs, dev/val -> train, test -> test -----
CEVAL_CONFIGS = [
    "accountant", "advanced_mathematics", "art_studies", "basic_medicine",
    "business_administration", "chinese_language_and_literature", "civil_servant",
    "clinical_medicine", "college_chemistry", "college_economics", "college_physics",
    "college_programming", "computer_architecture", "computer_network",
    "discrete_mathematics", "education_science", "electrical_engineer",
    "environmental_impact_assessment_engineer", "fire_engineer", "high_school_biology",
    "high_school_chemistry", "high_school_chinese", "high_school_geography",
    "high_school_history", "high_school_mathematics", "high_school_physics",
    "high_school_politics", "ideological_and_moral_cultivation", "law",
    "legal_professional", "logic", "mao_zedong_thought", "marxism", "metrology_engineer",
    "middle_school_biology", "middle_school_chemistry", "middle_school_geography",
    "middle_school_history", "middle_school_mathematics", "middle_school_physics",
    "middle_school_politics", "modern_chinese_history", "operating_system",
    "physician", "plant_protection", "probability_and_statistics",
    "professional_tour_guide", "sports_science", "tax_accountant",
    "teacher_qualification", "urban_and_rural_planner", "veterinary_medicine",
]

log("开始下载 C-Eval（52 个科目），按 train/test 分拆...")
ceval_train_list = []
ceval_test_list = []
for i, cfg in enumerate(CEVAL_CONFIGS):
    log(f"  C-Eval [{i+1}/{len(CEVAL_CONFIGS)}] {cfg}")
    ds = load_dataset("ceval/ceval-exam", cfg)
    # dev 与 val 都并入 train；test -> test
    if "dev" in ds:
        ceval_train_list.append(ds["dev"])
    if "validation" in ds:
        ceval_train_list.append(ds["validation"])
    if "test" in ds:
        ceval_test_list.append(ds["test"])

log("  合并 C-Eval 并写入 parquet...")
os.makedirs("data/c-eval", exist_ok=True)
if ceval_train_list:
    train_ds = concatenate_datasets(ceval_train_list)
    train_ds.to_parquet("data/c-eval/train-00000-of-00001.parquet")
    log(f"    train: {len(train_ds)} 条 -> data/c-eval/train-00000-of-00001.parquet")
if ceval_test_list:
    test_ds = concatenate_datasets(ceval_test_list)
    test_ds.to_parquet("data/c-eval/test-00000-of-00001.parquet")
    log(f"    test: {len(test_ds)} 条 -> data/c-eval/test-00000-of-00001.parquet")
log("C-Eval 完成!")

# ----- C-MMLU: 67 subjects, dev -> train, test -> test -----
task_list = ['agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics', 'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture', 'chinese_foreign_policy', 'chinese_history', 'chinese_literature', 
'chinese_teacher_qualification', 'clinical_knowledge', 'college_actuarial_science', 'college_education', 'college_engineering_hydrology', 'college_law', 'college_mathematics', 'college_medical_statistics', 'college_medicine', 'computer_science',
'computer_security', 'conceptual_physics', 'construction_project_management', 'economics', 'education', 'electrical_engineering', 'elementary_chinese', 'elementary_commonsense', 'elementary_information_and_technology', 'elementary_mathematics', 
'ethnology', 'food_science', 'genetics', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_geography', 'high_school_mathematics', 'high_school_physics', 'high_school_politics', 'human_sexuality',
'international_law', 'journalism', 'jurisprudence', 'legal_and_moral_basis', 'logical', 'machine_learning', 'management', 'marketing', 'marxist_theory', 'modern_chinese', 'nutrition', 'philosophy', 'professional_accounting', 'professional_law', 
'professional_medicine', 'professional_psychology', 'public_relations', 'security_study', 'sociology', 'sports_science', 'traditional_chinese_medicine', 'virology', 'world_history', 'world_religions']

CMMLU_REVISION = "refs/pr/6"
log("开始下载 C-MMLU（67 个科目），按 train/test 分拆...")
cmmlu_train_list = []
cmmlu_test_list = []
for i, k in enumerate(task_list):
    log(f"  C-MMLU [{i+1}/{len(task_list)}] {k}")
    ds = load_dataset("haonan-li/cmmlu", k, revision=CMMLU_REVISION)
    if "dev" in ds:
        cmmlu_train_list.append(ds["dev"])
    if "test" in ds:
        cmmlu_test_list.append(ds["test"])

log("  合并 C-MMLU 并写入 parquet...")
os.makedirs("data/cmmlu", exist_ok=True)
if cmmlu_train_list:
    train_ds = concatenate_datasets(cmmlu_train_list)
    train_ds.to_parquet("data/cmmlu/train-00000-of-00001.parquet")
    log(f"    train: {len(train_ds)} 条 -> data/cmmlu/train-00000-of-00001.parquet")
if cmmlu_test_list:
    test_ds = concatenate_datasets(cmmlu_test_list)
    test_ds.to_parquet("data/cmmlu/test-00000-of-00001.parquet")
    log(f"    test: {len(test_ds)} 条 -> data/cmmlu/test-00000-of-00001.parquet")
log("C-MMLU 完成!")
