# config.py
import os

# === 경로 설정 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# === 데이터 파일 ===
CSV_FILE = os.path.join(DATA_DIR, "final_rag_data.csv")
VAL_JSON_FILE = os.path.join(DATA_DIR, "val_QA.json") 
EVAL_OUTPUT_FILE = os.path.join(DATA_DIR, "multiturn_evaluation_results.csv")  
SUMMARY_CSV_FILE = os.path.join(DATA_DIR, "preprocessed_hanacard_summary.csv")

# === 벡터스토어 파일 ===
EMBEDDINGS_FILE = os.path.join(VECTORSTORE_DIR, "faiss_index.bin")
DOCUMENTS_FILE = os.path.join(VECTORSTORE_DIR, "documents.json")

# === 모델 설정 ===
EMBEDDING_MODEL = "nlpai-lab/KURE-v1"
LLM_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
SUMMARY_BASE_MODEL = "gogamza/kobart-base-v2"
LORA_PATH = os.path.join(MODELS_DIR, "kobart-lora-dialogue-summary")

# === 검색 및 대화 설정 ===
MAX_HISTORY_TURNS = 10
RETRIEVAL_K = 3
RETRIEVAL_K_DENSE = 20
CATEGORY_CONFIDENCE_THRESHOLD = 0.8
MIN_SIMILARITY_THRESHOLD = 0.35

# === Summarization Fine-tuning 설정 ===
SUMMARY_OUTPUT_DIR = LORA_PATH
SUMMARY_TRAIN_BATCH_SIZE = 8
SUMMARY_NUM_EPOCHS = 3
SUMMARY_MAX_INPUT_LENGTH = 1024
SUMMARY_MAX_TARGET_LENGTH = 128
SUMMARY_LEARNING_RATE = 2e-4
SUMMARY_GRAD_ACCUM_STEPS = 8
