from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch

# === Настройки ===
MODEL_NAME = "gpt2"  # или "tiiuae/falcon-rw-1b" для более сильной модели
TXT_PATH = "sektor_gaza_lyrics_clean.txt"
OUTPUT_DIR = "gpt2-lora-sektor-gaza"

# === Загружаем токенизатор и модель ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    load_in_8bit=True  # для экономии памяти
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

# === Загружаем датасет ===
with open(TXT_PATH, "r", encoding="utf-8") as f:
    texts = f.read().split("\n\n")

dataset = Dataset.from_dict({"text": texts})

# === Токенизация ===
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

tokenized = dataset.map(tokenize, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Обучение ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # Увеличиваем количество эпох для более глубокого обучения
    num_train_epochs=60,  # Увеличиваем до 10-20 эпох, в зависимости от времени и ресурсов
    per_device_train_batch_size=32,  # Если память позволяет, можно увеличить
    per_device_eval_batch_size=32,
    logging_dir='./logs',
    save_strategy="epoch",  # Сохраняем модель после каждой эпохи
    save_steps=500,  # Сохраняем модель каждые 500 шагов
    logging_steps=100,  # Логируем прогресс каждые 100 шагов
    warmup_steps=500,  # Количество шагов для прогрева learning rate
    # evaluation_strategy="epoch",  # Оценка модели после каждой эпохи
    learning_rate=5e-5,  # Можно уменьшить learning rate для стабильности
    weight_decay=0.01,  # Для предотвращения переобучения
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# Сохраняем модель и токенизатор
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
