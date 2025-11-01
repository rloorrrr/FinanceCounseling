"""
KoBART + LoRA Summarization Fine-tuning (config.py 사용)
"""
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import warnings

from src.config import SUMMARY_CSV_FILE, MODELS_DIR, SUMMARY_BASE_MODEL

warnings.filterwarnings("ignore", message="BPE merge indices")
warnings.filterwarnings("ignore", message="num_labels")
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

TRAIN_PATH = SUMMARY_CSV_FILE
OUTPUT_DIR = os.path.join(MODELS_DIR, "kobart-lora-dialogue-summary")

tokenizer = AutoTokenizer.from_pretrained(SUMMARY_BASE_MODEL, use_fast=True)
model = BartForConditionalGeneration.from_pretrained(SUMMARY_BASE_MODEL)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("csv", data_files={"train": TRAIN_PATH}, delimiter=",")
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

def preprocess_function(batch):
    inputs = tokenizer(
        batch["content"],
        max_length=1024,
        padding="max_length",
        truncation=True
    )
    labels = tokenizer(
        batch["title"],
        max_length=128,
        padding="max_length",
        truncation=True
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    do_train=True,
    do_eval=False,
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    gradient_accumulation_steps=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    tokenizer=tokenizer
)

print("LoRA Fine-tuning 시작")
trainer.train()

final_path = os.path.join(OUTPUT_DIR, "final")
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path, legacy_format=False)
print(f"LoRA 파인튜닝 완료 및 모델 저장 완료: {final_path}")


