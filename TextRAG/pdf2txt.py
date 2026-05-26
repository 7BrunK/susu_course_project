import pdfplumber
import re
from pathlib import Path

# Пути к файлам
pdf_path = "support_measures.pdf"
output_path_txt = "data/corpus/support_measures_clean.txt"
output_path_md = "data/corpus/support_measures_clean.md"

# Повторяющиеся заголовки PDF
repeating_headers = [
    r'Меры поддержки бизнеса\s*в Челябинской области',
    r'Федеральные институты поддержки бизнеса',
    r'Региональные институты поддержки бизнеса',
]

# Заголовки разделов
section_titles = [
    'Кредиты',
    'Поручительства',
    'Лизинг',
    'Финансовая поддержка',
    'Гранты',
    'Субсидии',
    'Услуги',
    'Льготы',
    'Социальные контракты',
    'Налоговые льготы',
    'Инвестиционный налоговый вычет',
    'Инвестиционный налоговый кредит',
    'Соглашение о защите и поощрении капиталовложений',
    'Льготы по аренде земельных участков',
    'Льготы по аренде имущества',
]

# Текущий активный заголовок
current_section = None

pages = []

with pdfplumber.open(pdf_path) as pdf:

    for page in pdf.pages[2:]:

        text = page.extract_text()

        # Пропуск пустых страниц
        if not text:
            continue

        # ---------------------------------
        # Удаление URL и QR-ссылок
        # ---------------------------------
        text = re.sub(r'http[s]?://\S+', ' ', text)
        text = re.sub(r'www\.\S+', ' ', text)
        text = re.sub(r'qrcoder\.ru\S*', ' ', text)

        # ---------------------------------
        # Удаление путей Windows
        # ---------------------------------
        text = re.sub(
            r'[A-Za-z]:\\\S+',
            ' ',
            text
        )

        text = re.sub(
            r'/[^\s]+?\.(png|jpg|jpeg|gif|bmp|webp)',
            ' ',
            text,
            flags=re.IGNORECASE
        )

        # ---------------------------------
        # Удаление повторяющихся заголовков
        # ---------------------------------
        for header in repeating_headers:
            text = re.sub(
                header,
                ' ',
                text,
                flags=re.IGNORECASE
            )

        # ---------------------------------
        # Удаление номеров страниц
        # ---------------------------------
        text = re.sub(
            r'^\s*\d+\s*$',
            ' ',
            text,
            flags=re.MULTILINE
        )

        # ---------------------------------
        # Удаление служебных символов
        # ---------------------------------
        text = text.replace('\xa0', ' ')
        text = text.replace('•', ' ')
        text = text.replace('▪', ' ')
        text = text.replace('■', ' ')
        text = text.replace('●', ' ')

        # ---------------------------------
        # Склейка переносов внутри слов
        # ---------------------------------
        text = re.sub(
            r'(\w)-\n(\w)',
            r'\1\2',
            text
        )

        cleaned_lines = []

        for line in text.split('\n'):

            line = line.strip()

            # Пропуск пустых строк
            if not line:
                continue

            # Пропуск мусорных строк
            if re.fullmatch(r'[-_=+*#.:;]+', line):
                continue

            normalized_line = re.sub(r'\s+', ' ', line)

            # ---------------------------------
            # Проверка заголовка
            # ---------------------------------
            if normalized_line in section_titles:

                # Добавляем заголовок
                # только если он новый
                if normalized_line != current_section:
                    current_section = normalized_line
                    cleaned_lines.append(f'\n## {normalized_line}\n')

                # Сам текст заголовка повторно не добавляем
                continue

            cleaned_lines.append(normalized_line)

        # ---------------------------------
        # Формирование текста страницы
        # ---------------------------------
        page_text = '\n'.join(cleaned_lines)

        # ---------------------------------
        # Нормализация пробелов
        # ---------------------------------
        page_text = re.sub(r'[ \t]+', ' ', page_text)

        # ---------------------------------
        # Удаление лишних пустых строк
        # ---------------------------------
        page_text = re.sub(r'\n{3,}', '\n\n', page_text)

        pages.append(page_text.strip())

# ---------------------------------
# Объединение страниц
# ---------------------------------
full_text = '\n\n'.join(pages)

# ---------------------------------
# Финальная очистка
# ---------------------------------
full_text = re.sub(r' +', ' ', full_text)
full_text = re.sub(r'\n{3,}', '\n\n', full_text)

# ---------------------------------
# Создание директории
# ---------------------------------
Path("data/corpus").mkdir(parents=True, exist_ok=True)

# ---------------------------------
# Сохранение результата
# ---------------------------------
for output_path in (output_path_txt, output_path_md):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text.strip())

print("Подготовка RAG-корпуса завершена")