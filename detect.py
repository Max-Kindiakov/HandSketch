
"""
Скрипт для детектування та розпізнавання рукописних символів на зображенні.
Використовує навчену модель CNN та обробку зображень за допомогою OpenCV.
Завантажує порядок класів з 'logs/class_info.json'.
"""

import cv2
import torch
import torch.nn as nn # Потрібно для type hinting у detect_characters
import numpy as np
import os
from PIL import Image
import torch.nn.functional as F # Для функції padding
import json # <--- Імпортуємо бібліотеку JSON
import torchvision.transforms as transforms # <--- Потрібно для transforms.Pad

# Імпортуємо конфігурацію: Модель та ТРАНСФОРМАЦІЮ ДЛЯ ВАЛІДАЦІЇ/ТЕСТУВАННЯ
try:
    from configuration import CNNModel, val_test_transform, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH
except ImportError:
    print("Помилка: Не вдалося імпортувати з configuration.py.")
    print("Переконайтесь, що файл configuration.py існує та не містить синтаксичних помилок.")
    exit(1)

# --- Визначення Пристрою (GPU/CPU) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Detect.py: Використовується GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Detect.py: Використовується CPU")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> ПОЧАТОК ЗМІН <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# --- Завантаження інформації про класи з JSON ---
CLASS_INFO_PATH = "logs/class_info.json" # Шлях до файлу з інформацією про класи
CLASS_MAP = {} # Словник для зберігання мапування індекс -> клас

try:
    print(f"Завантаження інформації про класи з: {CLASS_INFO_PATH}")
    with open(CLASS_INFO_PATH, 'r', encoding='utf-8') as f:
        class_info = json.load(f)

    # Перевіряємо наявність ключа 'classes' у файлі
    if 'classes' not in class_info:
        raise ValueError("Ключ 'classes' відсутній у файлі class_info.json")

    # Список імен класів у порядку їх індексів (0, 1, 2...)
    CLASS_LIST = class_info['classes']

    # Створюємо мапування індекс -> ім'я класу
    CLASS_MAP = {idx: class_name for idx, class_name in enumerate(CLASS_LIST)}
    print("Мапування індекс->клас успішно завантажено:")
    # Виведемо лише перші 10 та останні 5 для стислості, якщо їх багато
    if len(CLASS_MAP) > 15:
         print({k: v for k, v in list(CLASS_MAP.items())[:10]})
         print("...")
         print({k: v for k, v in list(CLASS_MAP.items())[-5:]})
    else:
         print(CLASS_MAP)

    # Перевірка відповідності кількості завантажених класів до NUM_CLASSES
    loaded_classes_count = len(CLASS_MAP)
    if loaded_classes_count != NUM_CLASSES:
        print("-" * 30)
        print(f"!!! КРИТИЧНЕ ПОПЕРЕДЖЕННЯ !!!")
        print(f"Кількість класів, завантажених з '{CLASS_INFO_PATH}' ({loaded_classes_count}),")
        print(f"не відповідає значенню NUM_CLASSES ({NUM_CLASSES}) у configuration.py!")
        print("Це може призвести до НЕПРАВИЛЬНОГО розпізнавання.")
        print("Переконайтеся, що використовується модель, навчена з тим самим набором класів,")
        print("який описано у '{CLASS_INFO_PATH}', або перетренуйте модель.")
        print("-" * 30)
        # Не зупиняємо програму, але попередження дуже важливе
        # Можна розкоментувати exit(), якщо потрібна зупинка
        # exit(1)
    else:
         print(f"Кількість завантажених класів ({loaded_classes_count}) відповідає NUM_CLASSES.")

except FileNotFoundError:
    print("-" * 30)
    print(f"!!! КРИТИЧНА ПОМИЛКА !!! Файл '{CLASS_INFO_PATH}' не знайдено.")
    print("Неможливо визначити правильний порядок класів для розпізнавання.")
    print("Запустіть тренування (train.py), щоб створити цей файл.")
    print("-" * 30)
    exit(1) # Зупиняємо програму, бо без мапування вона марна
except json.JSONDecodeError as e:
    print("-" * 30)
    print(f"!!! КРИТИЧНА ПОМИЛКА !!! Помилка декодування JSON у файлі '{CLASS_INFO_PATH}': {e}")
    print("Перевірте вміст файлу. Можливо, він пошкоджений.")
    print("-" * 30)
    exit(1)
except ValueError as e:
     print("-" * 30)
     print(f"!!! КРИТИЧНА ПОМИЛКА !!! Неправильний формат файлу '{CLASS_INFO_PATH}': {e}")
     print("-" * 30)
     exit(1)
except Exception as e:
    print("-" * 30)
    print(f"!!! НЕОЧІКУВАНА ПОМИЛКА під час завантаження інформації про класи: {e}")
    print("-" * 30)
    exit(1)

# --- Видаляємо старе, жорстко закодоване мапування ---
# CLASS_MAP = { ... } # ВИДАЛЕНО

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> КІНЕЦЬ ЗМІН <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def map_class_to_char(class_idx: int) -> str:
    """
    Перетворює індекс класу, отриманий від моделі, на відповідний символ,
    використовуючи динамічно завантажене мапування CLASS_MAP.
    """
    return CLASS_MAP.get(class_idx, '?') # Повертає '?' якщо індекс не знайдено

# --- Покращена функція трансформації ROI зі збереженням пропорцій ---
def preprocess_roi(roi_image: np.ndarray) -> torch.Tensor:
    """
    Обробляє вирізане зображення символу (ROI):
    1. Конвертує в PIL Image.
    2. Додає відступи (padding), щоб зробити зображення квадратним, зберігаючи пропорції.
    3. Застосовує стандартні трансформації (Resize, ToTensor, Normalize).
    4. Додає batch dimension.
    """
    roi_pil = Image.fromarray(roi_image).convert('L')
    width, height = roi_pil.size
    if width == 0 or height == 0: # Захист від порожніх ROI
         raise ValueError("ROI має нульову ширину або висоту.")
    target_size = max(width, height)
    padding_left = (target_size - width) // 2
    padding_right = target_size - width - padding_left
    padding_top = (target_size - height) // 2
    padding_bottom = target_size - height - padding_top

    # fill=0 для чорного фону, fill=255 для білого. Оскільки thresh інвертований (символ білий),
    # фон буде чорним (0), тому додаємо чорні рамки (fill=0).
    # Якщо б thresh не був інвертований, використовували б fill=255.
    padding_transform = transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0)
    roi_padded = padding_transform(roi_pil)

    # Використовуємо трансформації з configuration.py
    # val_test_transform має містити Resize, ToTensor, Normalize
    # Переконаємось, що він не містить Grayscale, оскільки ми робимо convert('L')
    # Краще зібрати потрібні трансформації тут, щоб бути впевненим
    final_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=transforms.InterpolationMode.BICUBIC), # Якісніша інтерполяція
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    roi_tensor = final_transform(roi_padded)
    return roi_tensor.unsqueeze(0)

# --- Пошук та Сортування Контурів ---
def get_characters_contours(canvas: np.ndarray) -> tuple[list, np.ndarray, list]:
    """
    Знаходить контури символів на полотні, сортує їх у порядку читання
    (зліва направо, зверху вниз) та визначає приблизні місця розриву рядків.
    """
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # Використовуємо адаптивний поріг - може бути кращим для неоднорідного фону/освітлення
    # _, thresh = cv2.threshold(gray_canvas, 127, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.adaptiveThreshold(gray_canvas, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2) # Параметри (blockSize, C) можна налаштувати

    kernel = np.ones((3, 3), np.uint8) # Зменшимо ядро для діляції
    dilated_thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_h, min_w = 8, 8 # Зменшимо мінімальний розмір контуру
    bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] >= min_w and cv2.boundingRect(c)[3] >= min_h]

    if not bounding_boxes:
        return [], thresh, []

    try:
        # Використовуємо медіану висоти - стійкіше до викидів
        median_height = np.median([box[3] for box in bounding_boxes])
        tolerance = median_height * 0.6 # Трохи збільшимо допуск
    except Exception:
        tolerance = 10

    bounding_boxes.sort(key=lambda box: (round(box[1] / tolerance) * tolerance, box[0]))

    row_breaks = []
    if len(bounding_boxes) > 1:
        for i in range(len(bounding_boxes) - 1):
            y_curr = bounding_boxes[i][1]
            h_curr = bounding_boxes[i][3]
            x_next = bounding_boxes[i+1][0]
            y_next = bounding_boxes[i+1][1]
            # Якщо наступний бокс нижче поточного (більше ніж tolerance)
            if y_next > y_curr + tolerance:
                 # І якщо він не значно правіше кінця поточного рядка (дозволяємо невеликий відступ)
                 # Це більш складна умова, спростимо: якщо Y значно більший - це новий рядок
                 row_breaks.append(i)
            # Додаткова умова: якщо наступний бокс на тій же Y-лінії (в межах tolerance),
            # але дуже далеко по X - це може бути пробіл між словами (не обробляємо зараз)

    return bounding_boxes, thresh, row_breaks


# --- Основна Функція Детектування ---
def detect_characters(model: nn.Module, canvas: np.ndarray) -> list[str]:
    """
    Виконує детектування та розпізнавання символів на зображенні.
    """
    # Перевірка, чи CLASS_MAP завантажено
    if not CLASS_MAP:
         print("!!! ПОМИЛКА в detect_characters: CLASS_MAP порожній. Неможливо продовжити.")
         return ['[ПОМИЛКА КЛАСІВ]']

    model.eval()
    model.to(device)
    results = []
    sorted_boxes, thresh, row_breaks = get_characters_contours(canvas)

    if not sorted_boxes:
        return []

    processed_row_breaks = set() # Для уникнення подвійних пробілів

    for i, box in enumerate(sorted_boxes):
        x, y, w, h = box
        # Вирізаємо ROI з бінаризованого зображення (thresh)
        # Додамо невеликий запас (padding) навколо ROI перед препроцесингом,
        # щоб уникнути обрізання країв символу при діляції/сортуванні.
        pad = 2
        y_start, y_end = max(0, y - pad), min(thresh.shape[0], y + h + pad)
        x_start, x_end = max(0, x - pad), min(thresh.shape[1], x + w + pad)
        roi = thresh[y_start:y_end, x_start:x_end]

        if roi.size == 0 or np.all(roi == 0): # Пропускаємо порожні або повністю чорні ROI
            print(f"Пропущено порожній або чорний ROI для боксу {i}: {box}")
            continue

        try:
            roi_tensor = preprocess_roi(roi)
            roi_tensor = roi_tensor.to(device)
        except ValueError as e: # Обробка помилки від preprocess_roi
             print(f"Помилка препроцесингу ROI для боксу {i} ({box}): {e}. Пропускаємо.")
             results.append('?')
             continue
        except Exception as e:
            print(f"Неочікувана помилка препроцесингу ROI для боксу {i} ({box}): {e}. Пропускаємо.")
            results.append('?')
            continue

        with torch.no_grad():
            output = model(roi_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, detected_idx = torch.max(probabilities, 1)
            char = map_class_to_char(detected_idx.item())
            confidence_score = confidence.item()

        # Додаємо розпізнаний символ (можна фільтрувати за confidence_score)
        if confidence_score < 0.3: # Приклад фільтрації низької впевненості
             print(f"Низька впевненість ({confidence_score:.2f}) для символу '{char}' (бокс {i}). Заміна на '?'.")
             results.append('?')
        else:
             results.append(char)
             # print(f"Бокс {i}: '{char}' (впевненість: {confidence_score:.2f})") # Для дебагу

        # Вставляємо пробіл ПІСЛЯ символу, якщо тут був визначений розрив рядка
        if i in row_breaks and i not in processed_row_breaks:
             results.append(" ")
             processed_row_breaks.add(i)

    return results

# --- Блок для тестування скрипту ---
if __name__ == "__main__":
    print("\n--- Тестування detect.py ---")
    MODEL_PATH = "logs/best_model.pth"
    IMAGES_DIR = "images"
    results_dict = {}

    # Перевірка наявності CLASS_MAP (вже має бути завантажено або програма вийшла)
    if not CLASS_MAP:
         print("Помилка: CLASS_MAP не було завантажено. Зупинка тестування.")
         exit(1)

    # 1. Завантаження моделі
    print(f"Завантаження моделі з: {MODEL_PATH}")
    try:
        model = CNNModel(num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.to(device)
        model.eval()
        print("Модель успішно завантажено.")
    except FileNotFoundError:
        print(f"Помилка: Файл моделі '{MODEL_PATH}' не знайдено.")
        print("Запустіть тренування (train.py) для створення моделі.")
        exit(1)
    except Exception as e:
        print(f"Помилка під час завантаження моделі: {e}")
        exit(1)

    # 2. Обробка зображень
    if os.path.exists(IMAGES_DIR):
        print(f"\nОбробка зображень з папки: {IMAGES_DIR}")
        image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]
        if not image_files:
            print("У папці 'images' не знайдено файлів зображень.")
        else:
            for filename in image_files:
                file_path = os.path.join(IMAGES_DIR, filename)
                print(f"  Обробка: {filename}")
                try:
                    canvas_img = cv2.imread(file_path)
                    if canvas_img is None:
                        print(f"    Попередження: Не вдалося прочитати файл {filename}.")
                        continue
                    detected_chars = detect_characters(model, canvas_img)
                    results_dict[filename] = "".join(detected_chars)
                except Exception as e:
                    print(f"    Помилка під час обробки {filename}: {e}")
                    results_dict[filename] = "[ПОМИЛКА ОБРОБКИ]"
    else:
        print(f"\nПапка '{IMAGES_DIR}' не знайдена. Тестування на зображеннях не проводиться.")

    # 3. Виведення результатів
    print("\n--- Результати Детектування ---")
    if results_dict:
        for filename, characters in results_dict.items():
            print(f"{filename}: {characters}")
    else:
        print("Не було оброблено жодного зображення.")

    print("\n--- Тестування завершено ---")
    # input("Натисніть Enter для виходу...")