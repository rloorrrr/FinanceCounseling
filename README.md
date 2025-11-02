# FinanceCounseling

<br>

AI 기반 **금융 상담 챗봇 프로젝트**, **LUPANG**입니다.  
고객의 질문 의도와 대화 맥락을 이해해 금융 상담을 자연스럽게 이어갑니다.  
실제 금융 상담 데이터를 기반으로, **멀티턴 대화**에서 **정확한 정보 검색 + 자연스러운 응답 생성**이 가능합니다.

---

## 📑 목차

- [📌 프로젝트 소개](#-프로젝트-소개)
- [🎯 프로젝트 목표](#-프로젝트-목표)
- [📂 학습 데이터](#-학습-데이터)
- [✨ 주요 기능](#-주요-기능)
- [🏛️ 시스템 아키텍처](#-시스템-아키텍처)
- [🧠 모델 구성](#-모델-구성)
- [📂 프로젝트 구조](#-프로젝트-구조)
- [⚙️ 설치 및 환경 설정](#-설치-및-환경-설정)
- [🚀 실행 방법](#-실행-방법)
- [🧪 평가 및 분석](#-평가-및-분석)
- [💡 권장 환경](#-권장-환경)
---

### 📌 프로젝트 소개

**금융상담봇 LUPANG**은 사용자의 질문을 이해하고,  
실제 금융 상담 업무에 맞는 자연스러운 답변을 제공하는 **AI 상담 챗봇**입니다.

- 다중 턴 대화 중에도 **맥락을 유지**하며 정확한 응답 제공  
- 금융 민원 및 상담 데이터를 기반으로 **실무형 대화 모델** 구축  
- **RAG + LLM + 요약 모델**을 결합한 하이브리드 구조  

---

### 🎯 프로젝트 목표

- 💬 **정확하고 맥락을 이해한 응답 제공**  
- 🧠 **다중 턴 대화에서 문맥 유지 및 자연스러운 흐름 구현**  
- 📚 **실제 금융 민원 시나리오 기반 데이터 학습**  
- 🪶 **Instruction Tuning으로 고품질 응답 생성**

---

### 📂 학습 데이터

본 프로젝트는 **AI Hub 금융 민원 상담 데이터**를 기반으로 학습되었습니다.

> 📘 [AI Hub 금융 민원 상담 데이터 바로가기](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71844)

- 상담 시나리오 기반 LLM 사전학습  
- Instruction Tuning 데이터로 파인튜닝  
- KoBART + LoRA 기반 요약 모델 데이터 포함  

---

### ✨ 주요 기능

#### 금융 고객 상담 대화
- 금융 상담 데이터 기반 고객 질문 이해 및 응답 생성  
- 다중 턴 대화 관리 및 자연스러운 대화 흐름  

#### 멀티턴 요약 & 메모리
- KoBART + LoRA 요약 모델로 긴 대화 요약  
- Turn 조건 기반 자동 요약 및 문맥 유지  

#### RAG 기반 정보 검색
- SentenceTransformer(`KURE-v1`) 임베딩  
- FAISS 인덱스 기반 유사 문서 검색  
- Confidence 기반 카테고리 조건부 검색  

#### LLM 응답 생성
- `EXAONE-3.5` 모델 기반 자연스러운 상담사 스타일 응답  
- RAG 실패 시 fallback 정책 적용  

#### 개인정보 마스킹
- 이름, 카드번호, 계좌, 금액, 날짜 등 민감정보 자동 마스킹  
- `<NAME>`, `<CARD>`, `<DATE>`, `<NUM>` 형태로 변환  

#### 파인튜닝 파이프라인 제공
- `Gemma 2 9B`: 금융 상담 데이터 생성  
- `KoBART + LoRA`: 요약 모델 파인튜닝  
- 학습 및 추론 스크립트 포함  

---

### 🏛️ 시스템 아키텍처

```
[사용자 입력]
      ↓
[대화 히스토리 저장]
      ↓
[Turn 조건 기반 요약 처리]
  └─ KoBART + LoRA 요약
      ↓
[Query Rewrite (KoBART)]
      ↓
[RAG 검색]
  └─ SentenceTransformer + FAISS
      ↓
[선택 문서 Confidence 체크]
      ↓
[LLM 응답 생성 (EXAONE-3.5)]
      ↓
[후처리 & 응답 반환]
  └─ 개인정보 마스킹
```


<br>



### 🧠 모델 구성

**개인정보 마스킹 모듈 (Masking)**  
- `<NAME>`, `<CARD>`, `<DATE>`, `<NUM>` 토큰으로 정규화
- 상담 기록 내 개인정보 보호  

↓

**KoBART + LoRA (대화 요약 모델)**  
- Base: `gogamza/kobart-base-v2`
- Fine-tuning 방식: LoRA
- 긴 대화 요약 및 문맥 유지
- Query Rewrite 입력으로 사용  

↓

**SentenceTransformer (임베딩 모델)**  
- 모델: `nlpai-lab/KURE-v1`
- 한국어 금융 도메인 문장 임베딩
- 사용자 질의 및 금융 문서 의미 벡터화  

↓

**FAISS (벡터 검색 엔진)**  
- 방식: `IndexFlatIP` (Cosine 기반)
- 금융 문서 Top-k 검색 수행
- 저장 파일: `faiss_index.bin`, `documents.json`  

↓

**EXAONE-3.5 Instruct (LLM 응답 생성)**  
- 모델: `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`
- 카드사 상담사 톤의 답변 생성
- RAG 기반 응답 + fallback 처리  

↓

**최종 출력**  
- 금융 상담사 스타일 응답 제공
- 관련 문서 출처(Optional) 함께 노출

<br>


### 📂 프로젝트 구조

```
FinanceCounseling/
├── assets/                   # 서비스 로고 및 이미지 파일
│   ├── lupanglogo.png
│   └── lupangg.png
│
├── data/                     # 학습 및 검증 데이터
│   ├── final_rag_data.csv
│   ├── val_QA.json
│   └── preprocessed_hanacard_summary.csv
│
├── models/                   # KoBART LoRA 가중치
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── vectorstore/              # 벡터 데이터베이스
│   ├── faiss_index.bin       # 구글 드라이브에서 다운로드 필요 (용량 초과)
│   ├── documents.json
│   └── .gitkeep
│
├── src/
│   ├── chatbot.py            # RAG + LLM 오케스트레이션
│   └── config.py             # 환경 및 경로 설정
│
├── evaluation/
│   ├── analyze_results.py
│   └── evaluate_similarity.py
│
├── main.py                   # CLI 실행용
├── app.py                    # Gradio 실행용
├── sum_train.py              # KoBART LoRA 학습 코드
├── requirements.txt
└── README.md

```

<br>

### ⚙️ 설치 및 환경 설정

1️⃣ Python 가상환경 생성
```
python3 -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```
2️⃣ 필수 패키지 설치
```
pip install --upgrade pip
pip install -r requirements.txt
```

📝 데이터 준비
학습 및 운영 데이터: data/final_rag_data.csv
KoBART LoRA 요약 모델 가중치: models/kobart-lora-dialogue-summary/ (이미 저장됨)
FAISS 인덱스 파일: [Google Drive](https://drive.google.com/drive/folders/1wGRHYRE6s1Jpj4qkpUHwNCISuTNesyY4?usp=drive_link) 에서 faiss_index.bin 다운로드 후 vectorstore/ 경로에 저장
⚠️ vectorstore/faiss_index.bin 파일이 없으면 검색 기반 기능이 정상 동작하지 않습니다.

🔧 KoBART LoRA 모델 학습 (선택 사항)
```
python sum_train.py --dataset preprocessed_habacard_summary.csv
```
모델 학습 시 models/ 경로에 가중치 저장장
학습하지 않아도 이미 저장된 가중치로 챗봇 실행 가능

### 🚀 실행 방법
1) CLI 실행
```
python main.py
```


2) Gradio 웹 인터페이스 실행
```
python app.py
```


### 🧪 평가 및 분석
1) 유사도 평가
```
python evaluate_similarity.py
```
val_QA.json과 모델 응답의 유사도 비교
결과 CSV 파일로 저장


2) 결과 분석
```
python analyze_results.py
```


### 💡 권장 환경
PyTorch 2.0+ (torch>=2.0.0)

CUDA GPU (VRAM 8GB 이상 권장)

CPU 모드에서도 실행 가능하지만 속도 저하 가능


