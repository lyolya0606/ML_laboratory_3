import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_text(model_path, prompt, max_length=100, temperature=0.9, top_k=50, top_p=0.95, num_return_sequences=1):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id
    )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


if __name__ == "__main__":
    # === Настройки ===
    model_path = "gpt2-lora-sektor-gaza"
    prompt = "ночь опускается"
    max_length = 400
    temperature = 0.9
    top_k = 50
    top_p = 0.95
    num_return_sequences = 3

    generated_texts = generate_text(
        model_path=model_path,
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences
    )

    for i, text in enumerate(generated_texts, 1):
        print(f"\n{text}\n")