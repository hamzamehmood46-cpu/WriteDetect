"""Fine-tune the WriteDetect AI Judge (Qwen2.5-0.5B-Instruct) with LoRA.

Run this on Google Colab's free T4 GPU, or on TTU HPCC, or any machine with a CUDA GPU.
It will NOT run in reasonable time on CPU - train it elsewhere, then copy the resulting
adapter folder into this project's models/qwen_judge_lora/ directory.

Steps:
  1. Run prepare_finetune_data.py locally first to produce finetune_data/ai_judge_train.jsonl
     and finetune_data/ai_judge_val.jsonl.
  2. Upload this script + both JSONL files into the same Colab working directory
     (or `!git clone` / `!gdown` them in, or use the Colab file upload panel).
  3. In Colab, first cell:
         !pip install -q transformers peft accelerate datasets
  4. Run this script:
         !python finetune_ai_judge.py
  5. Download qwen_judge_lora.zip (auto-saved at the end), unzip it, and copy the contents
     into <project>/models/qwen_judge_lora/ so llm_utils.py can find it.
"""
import shutil

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILE = "finetune_data/ai_judge_train.jsonl"
VAL_FILE = "finetune_data/ai_judge_val.jsonl"
OUTPUT_DIR = "qwen_judge_lora"
MAX_LENGTH = 768

DEVICE_AVAILABLE = torch.cuda.is_available()
if not DEVICE_AVAILABLE:
    print(
        "WARNING: no CUDA GPU detected. This script will be extremely slow on CPU. "
        "Run it on Google Colab (Runtime > Change runtime type > T4 GPU) or TTU HPCC instead."
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if DEVICE_AVAILABLE else torch.float32,
    device_map="auto" if DEVICE_AVAILABLE else None,
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()  # required so gradients flow through frozen base layers under gradient checkpointing
model.print_trainable_parameters()


def format_and_tokenize(example):
    messages = example["messages"]
    prompt_messages = messages[:-1]  # system + user, no assistant reply

    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    full_ids = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH)
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=MAX_LENGTH)

    # Mask out the prompt tokens so loss is only computed on the assistant's reply.
    labels = list(full_ids["input_ids"])
    prompt_len = min(len(prompt_ids["input_ids"]), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    full_ids["labels"] = labels
    return full_ids


dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
tokenized = dataset.map(format_and_tokenize, remove_columns=dataset["train"].column_names)


def collate(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids, attention_mask, labels = [], [], []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(b["attention_mask"] + [0] * pad_len)
        labels.append(b["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    bf16=DEVICE_AVAILABLE,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=collate,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nSaved LoRA adapter to {OUTPUT_DIR}/")

shutil.make_archive("qwen_judge_lora", "zip", OUTPUT_DIR)
print("Zipped adapter to qwen_judge_lora.zip")
print(
    "Download this zip, unzip it, and copy its contents into "
    "<project>/models/qwen_judge_lora/ in the WriteDetect repo."
)

try:
    from google.colab import files

    files.download("qwen_judge_lora.zip")
except ImportError:
    pass
