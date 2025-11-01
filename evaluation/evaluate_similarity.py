import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from model import FreeCardConsultingChatbot
from config import CSV_FILE, VAL_JSON_FILE, EVAL_OUTPUT_FILE

def evaluate_multiturn_conversation(chatbot, sbert_model, sample_id, qa_turns):
    turn_results = []

    print(f"\n{'='*70}")
    print(f"📝 샘플 {sample_id} 평가 시작 (3턴 대화)")
    print(f"{'='*70}")

    for turn_num, qa_pair in enumerate(qa_turns, 1):
        user_question = qa_pair["Q"]
        real_answer = qa_pair["A"]

        print(f"\n🔹 Turn {turn_num}")
        print(f"고객: {user_question[:80]}...")

        model_output = chatbot.chat(user_question)
        model_answer = model_output["answer"]

        print(f"상담사(실제): {real_answer[:80]}...")
        print(f"상담사(모델): {model_answer[:80]}...")

        embeddings = sbert_model.encode([real_answer, model_answer])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

        print(f"📊 유사도: {similarity:.4f}")

        turn_results.append({
            "sample_id": sample_id,
            "turn": turn_num,
            "question": user_question,
            "real_answer": real_answer,
            "model_answer": model_answer,
            "similarity": similarity,
            "sources": str(model_output.get("sources", []))
        })

    chatbot.chat_history = []
    chatbot.summary_memory = ""

    print(f"\n✅ 샘플 {sample_id} 평가 완료")
    print(f"{'='*70}\n")

    return turn_results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 사용 장치: {device}")

    print(f"✅ 평가 데이터 로드 중: {VAL_JSON_FILE}")
    with open(VAL_JSON_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"✅ 평가 데이터 로드 완료: {len(eval_data)}개 샘플")

    print("✅ SBERT 모델 로드 중...")
    sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    print("✅ SBERT 모델 로드 완료")

    print("✅ 챗봇 초기화 중...")
    df = pd.read_csv(CSV_FILE)
    chatbot = FreeCardConsultingChatbot(df)
    print("✅ 챗봇 초기화 완료\n")

    all_results = []
    for sample_id, qa_turns in enumerate(eval_data, 1):
        turn_results = evaluate_multiturn_conversation(chatbot, sbert_model, sample_id, qa_turns)
        all_results.extend(turn_results)

    df_results = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(EVAL_OUTPUT_FILE), exist_ok=True)
    df_results.to_csv(EVAL_OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("\n✅ 평가 완료!")
    print(f"총 샘플 수: {len(eval_data)}개")
    print(f"총 턴 수: {len(df_results)}개")
    print(f"결과 저장: {EVAL_OUTPUT_FILE}\n")

    return df_results

if __name__ == "__main__":
    df_results = main()
