import os
import json
import random
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Пути к данным
# =========================================================

IMAGES_DIR = "data/images/train"
LABELS_FILE = "data/labels/train_labels.json"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# Загрузка аннотаций
# Этап 3.3.1 — Баланс классов
# =========================================================

def load_annotations(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_class_distribution(data):
    class_counter = Counter()

    for item in data:
        labels = item.get("labels", [])

        for label in labels:
            category = label.get("category")

            if category is not None:
                class_counter[category] += 1

    return class_counter


def plot_class_distribution(class_counter):
    classes = list(class_counter.keys())
    counts = list(class_counter.values())

    plt.figure(figsize=(14, 7))
    plt.bar(classes, counts)

    plt.xticks(rotation=45)
    plt.xlabel("Класс")
    plt.ylabel("Количество объектов")
    plt.title("Распределение объектов по классам")

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "3_3_1_class_distribution.png")
    )
    plt.close()

    df = pd.DataFrame({
        "class": classes,
        "count": counts
    })

    df.to_csv(
        os.path.join(RESULTS_DIR, "3_3_1_class_distribution.csv"),
        index=False
    )


# =========================================================
# Этап 3.3.2 — Примеры изображений
# =========================================================

def draw_bounding_boxes(image, labels):
    for label in labels:
        if "box2d" not in label:
            continue

        box = label["box2d"]

        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])

        category = label.get("category", "object")

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            category,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return image


def save_sample_images(data, images_dir, sample_count=4):
    samples = random.sample(data, sample_count)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, item in zip(axes, samples):
        image_name = item["name"]
        image_path = os.path.join(images_dir, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels = item.get("labels", [])

        image = draw_bounding_boxes(image, labels)

        ax.imshow(image)
        ax.set_title(image_name)
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULTS_DIR, "3_3_2_sample_images.png")
    )

    plt.close()


# =========================================================
# Этап 3.3.3 — Анализ качества изображений
# =========================================================

def analyze_image_quality(data, images_dir):
    widths = []
    heights = []
    brightness_values = []

    for item in data[:1000]:
        image_name = item["name"]
        image_path = os.path.join(images_dir, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        h, w = image.shape[:2]

        widths.append(w)
        heights.append(h)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        brightness_values.append(brightness)

    plt.figure(figsize=(12, 6))

    plt.hist(brightness_values, bins=30)

    plt.xlabel("Средняя яркость")
    plt.ylabel("Количество изображений")
    plt.title("Распределение яркости изображений")

    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULTS_DIR, "3_3_3_image_quality.png")
    )

    plt.close()

    quality_df = pd.DataFrame({
        "width": widths,
        "height": heights,
        "brightness": brightness_values
    })

    quality_df.to_csv(
        os.path.join(RESULTS_DIR, "3_3_3_image_quality.csv"),
        index=False
    )


# =========================================================
# Этап 3.3.4 — Анализ аннотаций
# =========================================================

def analyze_annotations(data):
    object_counts = []

    bbox_widths = []
    bbox_heights = []

    for item in data:
        labels = item.get("labels", [])

        object_counts.append(len(labels))

        for label in labels:
            if "box2d" not in label:
                continue

            box = label["box2d"]

            width = box["x2"] - box["x1"]
            height = box["y2"] - box["y1"]

            bbox_widths.append(width)
            bbox_heights.append(height)

    plt.figure(figsize=(12, 6))

    plt.hist(object_counts, bins=30)

    plt.xlabel("Количество объектов")
    plt.ylabel("Количество изображений")
    plt.title("Распределение количества объектов на изображениях")

    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULTS_DIR, "3_3_4_annotations.png")
    )

    plt.close()

    annotations_df = pd.DataFrame({
        "bbox_width": bbox_widths,
        "bbox_height": bbox_heights
    })

    annotations_df.to_csv(
        os.path.join(RESULTS_DIR, "3_3_4_annotations.csv"),
        index=False
    )


# =========================================================
# Этап 3.3.5 — Анализ качества разметки
# =========================================================

def analyze_label_quality(data):
    invalid_boxes = 0
    total_boxes = 0

    for item in data:
        labels = item.get("labels", [])

        for label in labels:
            if "box2d" not in label:
                continue

            total_boxes += 1

            box = label["box2d"]

            width = box["x2"] - box["x1"]
            height = box["y2"] - box["y1"]

            if width <= 0 or height <= 0:
                invalid_boxes += 1

    quality_df = pd.DataFrame({
        "total_boxes": [total_boxes],
        "invalid_boxes": [invalid_boxes],
        "invalid_ratio_percent": [
            (invalid_boxes / total_boxes) * 100
            if total_boxes > 0 else 0
        ]
    })

    quality_df.to_csv(
        os.path.join(RESULTS_DIR, "3_3_5_label_quality.csv"),
        index=False
    )


# =========================================================
# Главный запуск
# =========================================================

def main():
    data = load_annotations(LABELS_FILE)

    class_counter = extract_class_distribution(data)

    plot_class_distribution(class_counter)

    save_sample_images(data, IMAGES_DIR)

    analyze_image_quality(data, IMAGES_DIR)

    analyze_annotations(data)

    analyze_label_quality(data)


if __name__ == "__main__":
    main()