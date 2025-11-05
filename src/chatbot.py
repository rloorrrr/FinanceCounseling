import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch
import os
import numpy as np

from config import (
    EMBEDDINGS_FILE, DOCUMENTS_FILE, 
    EMBEDDING_MODEL, LLM_MODEL, SUMMARY_BASE_MODEL, LORA_PATH,
    MAX_HISTORY_TURNS, RETRIEVAL_K, RETRIEVAL_K_DENSE,
    CATEGORY_CONFIDENCE_THRESHOLD, MIN_SIMILARITY_THRESHOLD
)


class FreeCardConsultingChatbot:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL).to(self.device)

        self.llm_model_name = LLM_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, trust_remote_code=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        self.documents = []
        self.vectorstore = self._build_vectorstore()

        self.summary_tokenizer = AutoTokenizer.from_pretrained(SUMMARY_BASE_MODEL, use_fast=True)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_BASE_MODEL)
        self.summary_model = PeftModel.from_pretrained(base_model, LORA_PATH)
        self.summary_model = self.summary_model.merge_and_unload()
        self.summary_model.eval().to(self.device)

        self.chat_history = []
        self.summary_memory = ""
        self.max_history_turns = MAX_HISTORY_TURNS

    def _build_vectorstore(self):
        if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(DOCUMENTS_FILE):
            try:
                self.documents = pd.read_json(DOCUMENTS_FILE).to_dict('records')
                return faiss.read_index(EMBEDDINGS_FILE)
            except:
                pass

        documents, texts = [], []
        for i, row in self.df.iterrows():
            category = row.get("consulting_category", "기타")
            content = row.get("dialogue", "")
            if not isinstance(content, str) or not content.strip():
                continue
            combined_text = f"카테고리: {category}\n\n{content}"
            documents.append({
                "content": combined_text,
                "full_content": content,
                "metadata": {
                    "source_id": str(i),
                    "title": row.get("task_category", "기본요약"),
                    "category": category
                }
            })
            texts.append(combined_text)

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=32,
            normalize_embeddings=True
        )

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype("float32"))
        self.documents = documents

        os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
        faiss.write_index(index, EMBEDDINGS_FILE)
        pd.DataFrame(self.documents).to_json(DOCUMENTS_FILE, orient="records")

        return index

    def _retrieve_documents(self, query: str, k: int = RETRIEVAL_K) -> List[Dict]:
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        categories = list({doc["metadata"]["category"] for doc in self.documents})
        cat_embs = self.embedding_model.encode(categories, convert_to_numpy=True, normalize_embeddings=True)
        cat_scores = (cat_embs @ query_emb.T).reshape(-1)

        best_idx = int(cat_scores.argmax())
        inferred_category = categories[best_idx]
        confidence = float(cat_scores[best_idx])

        k_dense = RETRIEVAL_K_DENSE
        if confidence >= CATEGORY_CONFIDENCE_THRESHOLD:
            cat_indices = [i for i, d in enumerate(self.documents) if d["metadata"]["category"] == inferred_category]
            sub_texts = [self.documents[i]["content"] for i in cat_indices]
            sub_emb = self.embedding_model.encode(sub_texts, convert_to_numpy=True, normalize_embeddings=True)
            sims = (sub_emb @ query_emb.T).reshape(-1)
            top_idx = np.argsort(-sims)[:k_dense]
            selected = [(cat_indices[i], float(sims[i])) for i in top_idx]
        else:
            distances, indices = self.vectorstore.search(query_emb.astype("float32"), k_dense)
            selected = [(int(i), float(d)) for i, d in zip(indices[0], distances[0])]

        results = []
        for idx, score in selected[:k]:
            doc = self.documents[idx].copy()
            doc["similarity_score"] = score
            results.append(doc)

        if not results or max([r["similarity_score"] for r in results]) < MIN_SIMILARITY_THRESHOLD:
            return []
        return results

    def _get_recent_dialogue(self, start_idx, end_idx):
        text = ""
        for turn in self.chat_history[start_idx:end_idx]:
            text += f"Q: {turn['user']}\nA: {turn['assistant']}\n"
        return text.strip()

    def _summarize_text(self, text: str) -> str:
        prompt = f"{text.strip()}\n요약:"
        inputs = self.summary_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            ids = self.summary_model.generate(**inputs, max_new_tokens=60, num_beams=4, no_repeat_ngram_size=3)
        return self.summary_tokenizer.decode(ids[0], skip_special_tokens=True).strip()

    def _rewrite_query(self, user_query: str, summary: str) -> str:
        prompt = f"고객 상담 요약: {summary}\n고객 질문: {user_query}".strip()
        inputs = self.summary_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            ids = self.summary_model.generate(**inputs, max_new_tokens=60, num_beams=4, no_repeat_ngram_size=3)
        rewritten = self.summary_tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        return rewritten if len(rewritten) > 3 else user_query

    def _generate_response_with_summary(self, query: str, context_docs: List[Dict], summary: str) -> str:
        if not context_docs:
            context = "관련 매뉴얼을 찾지 못했습니다."
        else:
            context = "\n\n".join([f"[참고자료]\n{d['full_content']}" for d in context_docs])

        system_prompt = (
            "당신은 카드사의 전문 상담사입니다.\n"
            "참고자료 기반으로 정확하고 간결하게 답변하세요."
        )

        user_prompt = f"{context}\n\n고객 질문: {query}\n\n[이전 대화 요약]\n{summary}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.llm_model.generate(
                input_ids,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=128,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
            )

        reply = self.tokenizer.decode(output[0], skip_special_tokens=True)
        reply = reply.split("assistant")[-1].strip().lstrip("|]").strip()

        return reply

    def chat(self, user_input: str) -> Dict:
        print("\n[고객 질문]")
        print(user_input)

        turn = len(self.chat_history) + 1
        summary = self.summary_memory or "이전 대화 없음."

        if turn == 2:
            self.summary_memory = self._summarize_text(self._get_recent_dialogue(0, 1))
            summary = self.summary_memory
        elif turn > 2 and turn % 2 == 0:
            prev = self.summary_memory or "이전 대화 없음."
            text = prev + "\n\n" + self._get_recent_dialogue(turn - 3, turn - 1)
            self.summary_memory = self._summarize_text(text)
            summary = self.summary_memory

        rewritten = user_input if turn == 1 else self._rewrite_query(user_input, summary)
        docs = self._retrieve_documents(rewritten, k=RETRIEVAL_K)
        answer = self._generate_response_with_summary(user_input, docs, summary)

        self.chat_history.append({"user": user_input, "assistant": answer})
        if len(self.chat_history) > self.max_history_turns:
            self.chat_history = self.chat_history[-self.max_history_turns:]

        print("\n[모델 답변]")
        print(answer)

        return {
            "answer": answer,
            "rewritten_query": rewritten,
            "sources": [
                {
                    "title": d["metadata"]["title"],
                    "category": d["metadata"]["category"],
                    "source_id": d["metadata"]["source_id"],
                    "similarity": d.get("similarity_score", 0),
                }
                for d in docs
            ],
        }
