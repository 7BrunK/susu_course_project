import os
import re
from collections import Counter

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords

from pymorphy3 import MorphAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# Пути
# ==========================================

PATH = "data/corpus/support_measures_clean.txt"

RESULTS_DIR = "analysis_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================
# Загрузка корпуса
# ==========================================

with open(PATH, "r", encoding="utf-8") as file:
    text = file.read()

print("=" * 100)
print("ИСХОДНЫЙ ТЕКСТ")
print("=" * 100)

print("Количество символов:", len(text))
print(text[:1000])


# ==========================================
# Очистка текста
# ==========================================

clean_text = text.lower()

clean_text = re.sub(r"[^а-яё\s]", " ", clean_text)
clean_text = re.sub(r"\s+", " ", clean_text)

print("\n")
print("=" * 100)
print("ОЧИЩЕННЫЙ ТЕКСТ")
print("=" * 100)

print(clean_text[:1000])


# ==========================================
# Токенизация
# ==========================================

tokens = clean_text.split()

print("\nКоличество слов после токенизации:", len(tokens))


# ==========================================
# Лемматизация
# ==========================================

morph = MorphAnalyzer()

lemmas = [
    morph.parse(word)[0].normal_form
    for word in tokens
]

print("\nПервые 50 лемм:")
print(lemmas[:50])

# ==========================================
# Подсчёт частоты слов
# ==========================================

word_freq = Counter(lemmas)

freq_df = pd.DataFrame(
    word_freq.most_common(30),
    columns=["Слово", "Частота"]
)

freq_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "word_frequency.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n")
print("=" * 100)
print("ТОП-30 НАИБОЛЕЕ ЧАСТОТНЫХ СЛОВ")
print("=" * 100)

print(freq_df)

plt.figure(figsize=(12, 6))

plt.bar(
    freq_df["Слово"],
    freq_df["Частота"]
)

plt.title("30 наиболее частотных слов")
plt.xlabel("Слова")
plt.ylabel("Частота")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "word_frequency.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# Облако слов до удаления стоп-слов
# ==========================================

wordcloud_before = WordCloud(
    width=1200,
    height=600,
    background_color="white"
).generate(" ".join(lemmas))

plt.figure(figsize=(14, 7))
plt.imshow(wordcloud_before)
plt.axis("off")
plt.title("Облако слов до удаления стоп-слов")

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "wordcloud_before_stopwords.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# Удаление стоп-слов
# ==========================================

nltk.download("stopwords")

russian_stopwords = set(
    stopwords.words("russian")
)

filtered_lemmas = [
    word
    for word in lemmas
    if word not in russian_stopwords
    and len(word) > 2
]

print(
    "\nКоличество слов после удаления стоп-слов:",
    len(filtered_lemmas)
)


# ==========================================
# Частотность после удаления стоп-слов
# ==========================================

filtered_freq = Counter(filtered_lemmas)

filtered_freq_df = pd.DataFrame(
    filtered_freq.most_common(30),
    columns=["Слово", "Частота"]
)

filtered_freq_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "word_frequency_no_stopwords.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n")
print("=" * 100)
print("ТОП-30 СЛОВ ПОСЛЕ УДАЛЕНИЯ СТОП-СЛОВ")
print("=" * 100)

print(filtered_freq_df)

plt.figure(figsize=(12, 6))

plt.bar(
    filtered_freq_df["Слово"],
    filtered_freq_df["Частота"]
)

plt.title(
    "30 наиболее частотных слов после удаления стоп-слов"
)

plt.xlabel("Слова")
plt.ylabel("Частота")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "word_frequency_no_stopwords.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# Облако слов после удаления стоп-слов
# ==========================================

wordcloud_after = WordCloud(
    width=1200,
    height=600,
    background_color="white"
).generate(" ".join(filtered_lemmas))

plt.figure(figsize=(14, 7))
plt.imshow(wordcloud_after)
plt.axis("off")
plt.title("Облако слов после удаления стоп-слов")

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "wordcloud_after_stopwords.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# Подготовка документов корпуса
# ==========================================

documents = []

for block in re.split(r"\n\s*\n", text):

    block = block.strip()

    if len(block) > 100:
        documents.append(block)

print("\nКоличество документов корпуса:",
      len(documents))


# ==========================================
# Функция предобработки
# ==========================================

def preprocess(document):

    document = document.lower()

    document = re.sub(
        r"[^а-яё\s]",
        " ",
        document
    )

    document = re.sub(
        r"\s+",
        " ",
        document
    )

    words = document.split()

    lemmas = [
        morph.parse(word)[0].normal_form
        for word in words
        if word not in russian_stopwords
        and len(word) > 2
    ]

    return " ".join(lemmas)


processed_docs = [
    preprocess(doc)
    for doc in documents
]


# ==========================================
# TF-IDF
# ==========================================

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(
    processed_docs
)

feature_names = (
    vectorizer.get_feature_names_out()
)

mean_tfidf = np.asarray(
    tfidf_matrix.mean(axis=0)
).flatten()

tfidf_df = pd.DataFrame({
    "Слово": feature_names,
    "TF-IDF": mean_tfidf
})

tfidf_df = tfidf_df.sort_values(
    by="TF-IDF",
    ascending=False
)

tfidf_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "tfidf_words.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n")
print("=" * 100)
print("ТОП-30 СЛОВ ПО TF-IDF")
print("=" * 100)

print(tfidf_df.head(30))

top_tfidf = tfidf_df.head(20)

plt.figure(figsize=(12, 6))

plt.bar(
    top_tfidf["Слово"],
    top_tfidf["TF-IDF"]
)

plt.title(
    "20 слов с максимальным TF-IDF"
)

plt.xlabel("Слова")
plt.ylabel("TF-IDF")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "tfidf_top_words.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# Информационный поиск
# ==========================================

def search(query, top_k=5):

    processed_query = preprocess(query)

    query_vector = vectorizer.transform(
        [processed_query]
    )

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for rank, idx in enumerate(top_indices, start=1):

        fragment = (
            documents[idx]
            .replace("\n", " ")
            .replace("\r", " ")
        )

        fragment = re.sub(
            r"\s+",
            " ",
            fragment
        )

        results.append({
            "Ранг": rank,
            "Документ": idx,
            "Сходство": round(
                similarities[idx],
                4
            ),
            "Фрагмент": fragment[:500]
        })

    return pd.DataFrame(results)


# ==========================================
# Пример информационного поиска
# ==========================================

query = (
    "льготный кредит для промышленного "
    "предприятия"
)

results = search(query)

results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "search_results.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n")
print("=" * 100)
print("РЕЗУЛЬТАТЫ ИНФОРМАЦИОННОГО ПОИСКА")
print("=" * 100)

print(
    results[
        ["Документ", "Сходство"]
    ]
)

print("\n")
print("=" * 100)
print("РЕЗУЛЬТАТЫ ИНФОРМАЦИОННОГО ПОИСКА")
print("=" * 100)

print(results[["Ранг", "Сходство"]])

for _, row in results.iterrows():

    print("\n" + "=" * 100)

    print("Ранг:", row["Ранг"])

    if "Документ" in results.columns:
        print("Документ:", row["Документ"])

    print("Сходство:", row["Сходство"])

    print()
    print(row["Фрагмент"])


# ==========================================
# Завершение
# ==========================================

print("\n")
print("=" * 100)
print("АНАЛИЗ ЗАВЕРШЁН")
print("=" * 100)

print(
    "Все результаты сохранены в папку:"
)

print(
    os.path.abspath(RESULTS_DIR)
)