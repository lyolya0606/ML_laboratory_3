from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# === Пути ===
MODEL_DIR = "gpt2-lora-sektor-gaza"  # путь к твоей обученной модели

# === Устройство ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Загрузка токенизатора и модели ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    "ai-forever/rugpt3medium_based_on_gpt2",
    device_map="auto",
    load_in_8bit=True  # если обучал в 8-битах
)

model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()
model.to(device)

# === Генерация текста ===
def generate(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=100,
        top_p=0.95,
        temperature=1.1,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# === Пример вызова ===
if __name__ == "__main__":
    while True:
        prompt = input("\nВведи начальную строку текста ('exit' для выхода): ").strip()
        if prompt.lower() == "exit":
            break
        result = generate(prompt)
        print("\n📜 Результат:")
        print(result)
