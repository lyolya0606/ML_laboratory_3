from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch

# === Настройки ===
MODEL_NAME = "ai-forever/rugpt3medium_based_on_gpt2"
TXT_PATH = "sektor_gaza_lyrics_clean.txt"
OUTPUT_DIR = "gpt2-lora-sektor-gaza"

# === Загружаем токенизатор и модель ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=quant_config
)

# === Подготавливаем модель для LoRA ===
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "q_proj", "v_proj"],  # зависит от модели
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# === Чтение и объединение строк по 4 ===
with open(TXT_PATH, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]  # убираем пустые строки

chunks = ["\n".join(lines[i:i+4]) for i in range(0, len(lines), 4) if len(lines[i:i+4]) == 4]

dataset = Dataset.from_dict({"text": chunks})
dataset = dataset.select(range(min(len(chunks), 2000)))  # ограничим размер

# === Разбиваем на train/test ===
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# === Токенизация ===
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

tokenized_train = train_dataset.map(tokenize, batched=True)
tokenized_eval = eval_dataset.map(tokenize, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Аргументы обучения ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    learning_rate=8e-5,
    logging_dir="./logs",
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="no",
    load_best_model_at_end=False,
    overwrite_output_dir=True
)

# === Обучение ===
trainer = Trainer(
    model=model,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# === Сохраняем модель и токенизатор ===
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
