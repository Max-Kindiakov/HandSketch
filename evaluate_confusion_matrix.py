import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import json
from tqdm import tqdm

# --- Імпорт конфігурації ---
try:
    from configuration import CNNModel, val_test_transform, NUM_CLASSES
except ImportError:
    print("Помилка: Не вдалося імпортувати з configuration.py.")
    print("Переконайтесь, що файл configuration.py існує та не містить синтаксичних помилок.")
    exit(1)

# --- Константи та шляхи ---
MODEL_PATH = "logs/best_model.pth"
CLASS_INFO_PATH = "logs/class_info.json"
VAL_DATA_DIR = "./dataset/Validation" # Шлях до валідаційних даних
BATCH_SIZE = 64 # Можна змінити, якщо потрібно
OUTPUT_IMAGE_PATH = "logs/confusion_matrix.png" # Куди зберегти зображення матриці

# --- Визначення пристрою ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Використовується GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Використовується CPU")

def load_class_names():
    """Завантажує імена класів з class_info.json."""
    try:
        with open(CLASS_INFO_PATH, 'r', encoding='utf-8') as f:
            class_info = json.load(f)
        if 'classes' not in class_info:
            raise ValueError("Ключ 'classes' відсутній у файлі class_info.json")
        class_names = class_info['classes']
        if len(class_names) != NUM_CLASSES:
            print(f"ПОПЕРЕДЖЕННЯ: Кількість класів у class_info.json ({len(class_names)}) "
                  f"не співпадає з NUM_CLASSES ({NUM_CLASSES}) у configuration.py.")
        return class_names
    except FileNotFoundError:
        print(f"ПОМИЛКА: Файл '{CLASS_INFO_PATH}' не знайдено. Неможливо отримати імена класів.")
        return None
    except Exception as e:
        print(f"Помилка завантаження імен класів: {e}")
        return None

def plot_confusion_matrix(cm, class_names, output_path):
    """Візуалізує та зберігає матрицю помилок."""
    plt.figure(figsize=(15, 12) if len(class_names) > 20 else (10,8) ) # Розмір залежить від кількості класів
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Справжній клас')
    plt.xlabel('Передбачений клас')
    plt.title('Матриця помилок (Confusion Matrix)')
    plt.tight_layout() # Для кращого розміщення елементів
    try:
        plt.savefig(output_path)
        print(f"Матрицю помилок збережено як: {output_path}")
    except Exception as e:
        print(f"Помилка збереження матриці помилок: {e}")

def evaluate_and_get_predictions(model, data_loader, device):
    """Оцінює модель та повертає списки справжніх та передбачених міток."""
    model.eval()
    all_labels = []
    all_predictions = []

    progress_bar = tqdm(data_loader, desc="Оцінка моделі", unit="batch")

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    return all_labels, all_predictions

def main():
    print("--- Побудова матриці помилок для навченої моделі ---")

    # 1. Завантаження імен класів
    class_names = load_class_names()
    if class_names is None:
        return

    # 2. Завантаження валідаційного датасету
    if not os.path.isdir(VAL_DATA_DIR):
        print(f"ПОМИЛКА: Папка валідаційних даних не знайдена: {VAL_DATA_DIR}")
        return

    try:
        validation_dataset = datasets.ImageFolder(root=VAL_DATA_DIR, transform=val_test_transform)
        validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        print(f"Знайдено {len(validation_dataset)} зображень для валідації у '{VAL_DATA_DIR}'.")
        # Перевірка, чи класи датасету відповідають завантаженим
        if validation_dataset.classes != class_names:
            print("ПОПЕРЕДЖЕННЯ: Порядок класів у валідаційному датасеті не співпадає з порядком у class_info.json!")
            print(f"Датасет класи: {validation_dataset.classes}")
            print(f"JSON класи:   {class_names}")
            # Це може призвести до неправильної інтерпретації матриці, якщо не виправити
    except Exception as e:
        print(f"Помилка завантаження валідаційних даних: {e}")
        return

    # 3. Завантаження навченої моделі
    if not os.path.exists(MODEL_PATH):
        print(f"ПОМИЛКА: Файл моделі '{MODEL_PATH}' не знайдено.")
        return

    try:
        model = CNNModel(num_classes=NUM_CLASSES) # Використовуємо NUM_CLASSES з конфігурації
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')) # Завантажуємо на CPU спочатку
        model.to(device)
        print(f"Модель '{MODEL_PATH}' успішно завантажено.")
    except Exception as e:
        print(f"Помилка завантаження моделі: {e}")
        return

    # 4. Отримання передбачень моделі на валідаційному наборі
    print("Отримання передбачень моделі...")
    true_labels, pred_labels = evaluate_and_get_predictions(model, validation_loader, device)

    if not true_labels or not pred_labels:
        print("Не вдалося отримати мітки або передбачення. Перевірте дані та модель.")
        return

    # 5. Побудова матриці помилок
    print("Побудова матриці помилок...")
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names))))

    # 6. Візуалізація та збереження матриці помилок
    plot_confusion_matrix(cm, class_names, OUTPUT_IMAGE_PATH)

    print("--- Завершено ---")

if __name__ == '__main__':
    main()
