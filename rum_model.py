from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

output = generator(
    "Из-за леса выезжает конная милиция",
    max_length=400,
    num_return_sequences=2,
    temperature=1,   # Повысь случайность
    top_p=0.9,         # Сужаем выбор до 90% вероятности
    top_k=50           # Ограничиваем 50 самыми вероятными словами
)
print(output[0]['generated_text'])
