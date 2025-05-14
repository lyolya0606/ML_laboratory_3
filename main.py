# import pandas as pd
#
# # Загружаем Parquet-файл
# df = pd.read_parquet("0000.parquet")
#
# # Посмотрим на колонки
# print(df.columns)
#
# # Извлекаем тексты песен, допустим колонка называется 'lyrics'
# lyrics_list = df["text"].dropna().tolist()
#
# # Сохраняем в txt
# with open("sektor_gaza_lyrics.txt", "w", encoding="utf-8") as f:
#     f.write("\n\n".join(lyrics_list))

with open("sektor_gaza_lyrics.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Заменяем U+2005 на обычный пробел
clean_text = text.replace('\u2005', ' ')

# Сохраняем обратно
with open("sektor_gaza_lyrics_clean.txt", "w", encoding="utf-8") as f:
    f.write(clean_text)

