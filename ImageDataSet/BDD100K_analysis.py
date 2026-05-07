import os
import json
import random
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATASET_PATH = "bdd100k"
LABELS_PATH = os.path.join(DATASET_PATH, "labels", "bdd100k_labels_images_train.json")
IMAGES_PATH = os.path.join(DATASET_PATH, "images", "100k", "train")

RESULTS_PATH = "results"
os.makedirs(RESULTS_PATH, exist_ok=True)


def load_annotations():
    # Этап 3.3.1 — загрузка и подготовка данных
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    return annotations


def analyze_class_balance(annotations):
    # Этап 3.3.1 — анализ баланса классов

    class_counter = Counter()

    for item in annotations:
        labels = item.get("labels", [])

        for label in labels:
            category = label.get("category")

            if category is not None:
                class_counter[category] += 1

    df_classes = pd.DataFrame(
        class_counter.items(),
        columns=["class", "count"]
    ).sort_values(by="count", ascending=False)

    plt.figure(figsize=(14, 8))
    plt.bar(df_classes["class"], df_classes["count"])
    plt.xticks(rotation=90)
    plt.xlabel("Класс")
    plt.ylabel("Количество объектов")
    plt.title("Распределение объектов по классам")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "1_balance_classes.png"))
    plt.close()


def visualize_examples(annotations, num_images=4):
    # Этап 3.3.2 — визуализация примеров изображений

    selected = random.sample(annotations, num_images)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, item in zip(axes, selected):
        image_name = item["name"]
        image_path = os.path.join(IMAGES_PATH, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels = item.get("labels", [])

        for label in labels:
            box2d = label.get("box2d")

            if box2d is None:
                continue

            x1 = int(box2d["x1"])
            y1 = int(box2d["y1"])
            x2 = int(box2d["x2"])
            y2 = int(box2d["y2"])

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        ax.imshow(image)
        ax.set_title(image_name)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "2_examples_images.png"))
    plt.close()


def analyze_image_quality(annotations):
    # Этап 3.3.3 — анализ качества изображений

    widths = []
    heights = []
    brightness_values = []

    sample_annotations = random.sample(annotations, min(500, len(annotations)))

    for item in sample_annotations:
        image_name = item["name"]
        image_path = os.path.join(IMAGES_PATH, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        height, width = image.shape[:2]

        widths.append(width)
        heights.append(height)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness_values.append(np.mean(gray))

    df_quality = pd.DataFrame({
        "width": widths,
        "height": heights,
        "brightness": brightness_values
    })

    plt.figure(figsize=(10, 6))
    plt.hist(df_quality["brightness"], bins=30)

    plt.xlabel("Средняя яркость")
    plt.ylabel("Количество изображений")
    plt.title("Распределение яркости изображений")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "3_image_quality.png"))
    plt.close()


def analyze_annotations(annotations):
    # Этап 3.3.4 — анализ аннотаций

    bbox_widths = []
    bbox_heights = []

    for item in annotations:
        labels = item.get("labels", [])

        for label in labels:
            box2d = label.get("box2d")

            if box2d is None:
                continue

            width = box2d["x2"] - box2d["x1"]
            height = box2d["y2"] - box2d["y1"]

            bbox_widths.append(width)
            bbox_heights.append(height)

    df_boxes = pd.DataFrame({
        "bbox_width": bbox_widths,
        "bbox_height": bbox_heights
    })

    plt.figure(figsize=(8, 8))
    plt.scatter(
        df_boxes["bbox_width"],
        df_boxes["bbox_height"],
        alpha=0.3
    )

    plt.xlabel("Ширина bounding box")
    plt.ylabel("Высота bounding box")
    plt.title("Размеры bounding box объектов")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "4_annotations_analysis.png"))
    plt.close()


def analyze_annotation_quality(annotations):
    # Этап 3.3.5 — анализ качества разметки

    stats = {
        "total_images": 0,
        "images_without_labels": 0,
        "objects_without_box": 0,
        "total_objects": 0
    }

    for item in annotations:
        stats["total_images"] += 1

        labels = item.get("labels", [])

        if len(labels) == 0:
            stats["images_without_labels"] += 1

        for label in labels:
            stats["total_objects"] += 1

            if label.get("box2d") is None:
                stats["objects_without_box"] += 1

    df_stats = pd.DataFrame([stats])

    df_stats.to_csv(
        os.path.join(RESULTS_PATH, "5_annotation_quality.csv"),
        index=False
    )


def main():
    annotations = load_annotations()

    analyze_class_balance(annotations)
    visualize_examples(annotations)
    analyze_image_quality(annotations)
    analyze_annotations(annotations)
    analyze_annotation_quality(annotations)


if __name__ == "__main__":
    main()