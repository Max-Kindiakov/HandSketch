
"""
Вікно Tkinter для відображення результатів розпізнавання символів,
їх корекції та збереження для донавчання.
"""

import cv2
import torch
import tkinter as tk
from tkinter import ttk # Використовуємо ttk для кращого вигляду
import os
import pyperclip
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np

# Імпортуємо необхідне з інших модулів
try:
    # Модель потрібна для завантаження стану
    from configuration import CNNModel, NUM_CLASSES
    # Функції для отримання символів та контурів
    from detect import detect_characters, get_characters_contours, device as detect_device # Імпортуємо device з detect.py
except ImportError as e:
    print(f"Помилка імпорту в detect_screen.py: {e}")
    print("Переконайтесь, що файли configuration.py та detect.py знаходяться поруч.")
    exit(1)

# Глобальна змінна для зберігання посилання на вікно
detect_window_ref = None
model_instance = None
model_loaded = False

def load_detection_model():
    """Завантажує модель один раз."""
    global model_instance, model_loaded
    if not model_loaded:
        model_path = "logs/best_model.pth"
        if not os.path.exists(model_path):
            print(f"Помилка в detect_screen: Файл моделі '{model_path}' не знайдено.")
            tk.messagebox.showerror("Помилка Моделі", f"Файл моделі '{model_path}' не знайдено.\nЗапустіть тренування.")
            return None
        try:
            print(f"Detect_screen: Завантаження моделі з {model_path}")
            model_instance = CNNModel(num_classes=NUM_CLASSES)
            # Завжди завантажуємо на CPU спочатку
            model_instance.load_state_dict(torch.load(model_path, map_location='cpu'))
            model_instance.to(detect_device) # Переміщуємо на пристрій, визначений в detect.py
            model_instance.eval()
            model_loaded = True
            print("Detect_screen: Модель успішно завантажено.")
            return model_instance
        except Exception as e:
            print(f"Помилка завантаження моделі в detect_screen: {e}")
            tk.messagebox.showerror("Помилка Моделі", f"Не вдалося завантажити модель:\n{e}")
            return None
    return model_instance

def format_characters(char_list):
    """Форматує список символів у рядок, ігноруючи None."""
    return "".join([char for char in char_list if char is not None])

def copy_characters_to_clipboard(char_list):
    """Копіює поточний текст у буфер обміну."""
    text = format_characters(char_list)
    try:
        pyperclip.copy(text)
        print("Текст скопійовано в буфер обміну.")
    except Exception as e:
        print(f"Помилка копіювання в буфер обміну: {e}")
        # Можна показати повідомлення користувачу, якщо pyperclip не встановлено або не працює
        tk.messagebox.showwarning("Помилка Копіювання", "Не вдалося скопіювати текст.\nМожливо, бібліотека pyperclip не встановлена або не підтримується.")

def update_characters_label(label_widget, char_list):
    """Оновлює текстову мітку з поточними символами."""
    label_widget.config(text=f"Розпізнано: {format_characters(char_list)}")

# --- Функції для керування GUI елементами ---

class CharacterWidgetManager:
    """Клас для управління віджетами символів та їх даними."""
    def __init__(self, root_frame, canvas_image):
        self.root_frame = root_frame # Frame всередині Canvas для скролінгу
        self.canvas_image = canvas_image # Оригінальне BGR зображення
        self.widgets = [] # Список кортежів (frame, entry_widget, char_label, img_label, delete_button)
        self.data = [] # Список кортежів (current_char, original_box)

    def add_character(self, char, box):
        """Додає віджет для символу."""
        frame = ttk.Frame(self.root_frame, width=120, height=210, relief="groove", borderwidth=2)
        frame.pack_propagate(False)
        frame.pack(side=tk.LEFT, padx=5, pady=5, anchor="n")

        # --- Відображення зображення символу ---
        x, y, w, h = box
        # Додаємо невеликі відступи для кращого візуального контексту
        pad = 5
        x_padded, y_padded = max(x - pad, 0), max(y - pad, 0)
        h_canvas, w_canvas = self.canvas_image.shape[:2]
        w_padded = min(w + 2 * pad, w_canvas - x_padded)
        h_padded = min(h + 2 * pad, h_canvas - y_padded)
        char_img_bgr = self.canvas_image[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]

        # Зміна розміру зі збереженням пропорцій для відображення
        display_h = 80
        aspect_ratio = w_padded / h_padded
        display_w = int(display_h * aspect_ratio)
        if display_w > 80: # Обмеження ширини
             display_w = 80
             display_h = int(display_w / aspect_ratio)

        if char_img_bgr.size == 0: # Якщо ROI порожній
            print(f"Попередження: Порожній ROI для боксу {box}")
            char_img_resized_rgb = np.ones((display_h, display_w, 3), dtype=np.uint8) * 127 # Сірий фон
        else:
            char_img_resized_bgr = cv2.resize(char_img_bgr, (display_w, display_h), interpolation=cv2.INTER_AREA)
            char_img_resized_rgb = cv2.cvtColor(char_img_resized_bgr, cv2.COLOR_BGR2RGB) # Конвертація в RGB для PIL

        char_img_pil = Image.fromarray(char_img_resized_rgb)
        char_img_tk = ImageTk.PhotoImage(char_img_pil)

        img_label = ttk.Label(frame, image=char_img_tk)
        img_label.image = char_img_tk # Зберігаємо посилання!
        img_label.pack(pady=(10, 5))

        # --- Мітка з розпізнаним символом ---
        char_label = ttk.Label(frame, text=f"({char})", font=("Arial", 14))
        char_label.pack(pady=(0, 5))

        # --- Поле для введення/корекції ---
        validate_cmd = frame.register(lambda P: len(P) <= 1) # Дозволяємо лише 1 символ (будь-який)
        entry = ttk.Entry(frame, font=("Arial", 12), width=5, justify='center', validate="key", validatecommand=(validate_cmd, "%P"))
        entry.insert(0, char if char != '?' else '') # Вставляємо розпізнаний символ, якщо він не '?'
        entry.pack(pady=5)

        # --- Кнопка видалення ---
        delete_button = ttk.Button(frame, text="Видалити", command=lambda idx=len(self.widgets): self._delete_widget(idx))
        delete_button.pack(side=tk.BOTTOM, padx=5, pady=5)

        self.widgets.append((frame, entry, char_label, img_label, delete_button))
        self.data.append({"char": char, "box": box})
        self._update_bindings()


    def add_space(self):
         """Додає візуальний елемент для пробілу."""
         frame = ttk.Frame(self.root_frame, width=60, height=210, relief="groove", borderwidth=2)
         frame.pack_propagate(False)
         frame.pack(side=tk.LEFT, padx=5, pady=5, anchor="n")

         ttk.Label(frame, text="[Пробіл]", font=("Arial", 10), anchor="center").pack(expand=True)

         delete_button = ttk.Button(frame, text="X", width=3, command=lambda idx=len(self.widgets): self._delete_widget(idx))
         delete_button.pack(side=tk.BOTTOM, pady=5)

         self.widgets.append((frame, None, None, None, delete_button)) # Немає Entry, CharLabel, ImgLabel
         self.data.append({"char": " ", "box": None}) # Box не потрібен для пробілу
         self._update_bindings()


    def _delete_widget(self, index):
        """Видаляє віджет та пов'язані дані за індексом."""
        if 0 <= index < len(self.widgets):
            widget_tuple = self.widgets.pop(index)
            deleted_data = self.data.pop(index)

            frame = widget_tuple[0]
            frame.destroy() # Видаляємо фрейм з усіма його дочірніми віджетами

            print(f"Видалено елемент {index}: '{deleted_data['char']}'")
            self._update_bindings() # Оновлюємо команди кнопок видалення
            # Потрібно оновити мітку з повним текстом зовні
            update_characters_label(characters_label_ref, self.get_current_characters())
        else:
            print(f"Помилка видалення: Неправильний індекс {index}")

    def _update_bindings(self):
         """Оновлює індекси в командах кнопок видалення."""
         for i, widget_tuple in enumerate(self.widgets):
             delete_button = widget_tuple[4]
             if delete_button:
                 delete_button.config(command=lambda idx=i: self._delete_widget(idx))


    def get_current_data_for_saving(self):
        """Повертає список даних для збереження."""
        results = []
        for i, data_item in enumerate(self.data):
            current_char = data_item["char"]
            box = data_item["box"]
            widget_tuple = self.widgets[i]
            entry = widget_tuple[1] # Отримуємо Entry віджет

            if current_char == " ": # Пропускаємо пробіли
                continue

            corrected_char = entry.get().strip()
            if not corrected_char: # Пропускаємо, якщо поле порожнє
                print(f"Пропущено збереження для індексу {i}: поле порожнє.")
                continue

            # Конвертуємо літери у верхній регістр для назви папки
            save_folder_char = corrected_char.upper()

            if len(save_folder_char) != 1:
                print(f"Попередження: Некоректний символ '{corrected_char}' для збереження (індекс {i}). Пропускаємо.")
                continue

            results.append({
                "folder_char": save_folder_char, # Символ для назви папки
                "original_box": box,           # Оригінальний bounding box
            })
        return results

    def get_current_characters(self):
        """Повертає поточний список символів (з урахуванням видалень)."""
        return [item["char"] for item in self.data]

# Глобальне посилання на мітку для оновлення
characters_label_ref = None

def save_corrected_results(manager: CharacterWidgetManager):
    """Зберігає зображення виправлених символів у датасет."""
    data_to_save = manager.get_current_data_for_saving()
    canvas_image = manager.canvas_image # Оригінальне BGR зображення

    if not data_to_save:
        print("Немає даних для збереження.")
        tk.messagebox.showinfo("Збереження", "Немає виправлених символів для збереження.")
        return

    saved_count = 0
    errors_count = 0
    base_dataset_dir = "dataset/Train" # Зберігаємо одразу в тренувальний набір

    for item in data_to_save:
        folder_char = item["folder_char"]
        box = item["original_box"]
        target_dir = os.path.join(base_dataset_dir, folder_char)

        try:
            os.makedirs(target_dir, exist_ok=True)

            x, y, w, h = box
            # Вирізаємо ROI з ОРИГІНАЛЬНОГО BGR зображення
            roi_bgr = canvas_image[y:y+h, x:x+w]

            if roi_bgr.size == 0:
                print(f"Помилка збереження: Порожній ROI для '{folder_char}' з боксом {box}")
                errors_count += 1
                continue

            # Додаємо білі рамки (padding) - 7 пікселів
            border_size = 7
            roi_bgr_with_border = cv2.copyMakeBorder(
                roi_bgr,
                top=border_size, bottom=border_size, left=border_size, right=border_size,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 255, 255] # Білий колір для рамки
            )

            # Конвертація в RGB для збереження через PIL
            roi_rgb_with_border = cv2.cvtColor(roi_bgr_with_border, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(roi_rgb_with_border)

            # Генеруємо унікальне ім'я файлу
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{timestamp}_{folder_char}.png" # Використовуємо PNG для кращої якості без втрат
            save_path = os.path.join(target_dir, filename)

            pil_image.save(save_path)
            saved_count += 1

        except Exception as e:
            print(f"Помилка збереження символу '{folder_char}' в {target_dir}: {e}")
            errors_count += 1

    print(f"Збереження завершено. Успішно: {saved_count}, Помилки: {errors_count}")
    tk.messagebox.showinfo("Збереження", f"Збереження завершено.\nУспішно: {saved_count}\nПомилки: {errors_count}")


def on_close_detect_window():
    """Обробник закриття вікна детектування."""
    global detect_window_ref
    if detect_window_ref:
        detect_window_ref.destroy()
    detect_window_ref = None # Скидаємо посилання

def detect_screen(root, canvas_bgr_image, reopen=False):
    """Основна функція для створення та показу вікна детектування."""
    global detect_window_ref, characters_label_ref

    # --- Керування станом вікна ---
    if reopen and detect_window_ref is not None:
        print("Перевідкриття вікна детектування...")
        on_close_detect_window()

    if detect_window_ref is not None and detect_window_ref.winfo_exists():
        print("Вікно детектування вже відкрито. Фокусування.")
        detect_window_ref.focus_set()
        detect_window_ref.lift()
        return

    # --- Завантаження моделі ---
    model = load_detection_model()
    if model is None:
        return # Помилка завантаження моделі, виходимо

    # --- Отримання результатів розпізнавання ---
    print("Запуск детектування символів...")
    try:
        # 1. Отримуємо розпізнані символи (список рядків, включаючи " ")
        recognized_characters = detect_characters(model, canvas_bgr_image)
        print(f"Розпізнано: {''.join(recognized_characters)}")

        # 2. Отримуємо відсортовані bounding boxes
        sorted_boxes, _, _ = get_characters_contours(canvas_bgr_image)

        # 3. Зіставлення: Створюємо список боксів, що відповідають НЕ-пробільним символам
        aligned_boxes = []
        box_idx = 0
        for char in recognized_characters:
            if char != " ":
                if box_idx < len(sorted_boxes):
                    aligned_boxes.append(sorted_boxes[box_idx])
                    box_idx += 1
                else:
                    # Це не повинно трапитися, якщо detect_characters працює правильно
                    print(f"Помилка зіставлення: Не вистачає боксів для символу '{char}'")
                    # Додамо фіктивний бокс, щоб уникнути падіння
                    aligned_boxes.append((0,0,10,10)) # Або інша обробка помилки

        if len(aligned_boxes) != len([c for c in recognized_characters if c != ' ']):
             print("Попередження: Кількість зіставлених боксів не співпадає з кількістю не-пробільних символів!")

    except Exception as e:
        print(f"Помилка під час детектування або зіставлення: {e}")
        tk.messagebox.showerror("Помилка Детектування", f"Сталася помилка:\n{e}")
        return

    # --- Створення GUI ---
    detect_window = tk.Toplevel(root)
    detect_window_ref = detect_window
    detect_window.title("Результати розпізнавання - HandSketch")
    detect_window.geometry("900x400") # Збільшимо розмір
    detect_window.minsize(600, 300)
    detect_window.focus_set()

    # --- Фрейм для меню знизу ---
    menu_frame = ttk.Frame(detect_window, padding="5")
    menu_frame.pack(fill=tk.X, side=tk.BOTTOM)

    # --- Головний контейнер з прокруткою ---
    main_frame = ttk.Frame(detect_window)
    main_frame.pack(fill=tk.BOTH, expand=True)

    scroll_canvas = tk.Canvas(main_frame)
    scrollbar_x = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=scroll_canvas.xview)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll_canvas.configure(xscrollcommand=scrollbar_x.set)

    # Фрейм всередині Canvas, куди додаватимуться віджети символів
    scrollable_frame = ttk.Frame(scroll_canvas)
    scroll_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    # Зв'язуємо оновлення розміру scrollable_frame з областю прокрутки Canvas
    scrollable_frame.bind("<Configure>", lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))

    # --- Менеджер віджетів ---
    widget_manager = CharacterWidgetManager(scrollable_frame, canvas_bgr_image)

    # --- Заповнення GUI елементами ---
    box_iter = iter(aligned_boxes)
    for char in recognized_characters:
        if char == " ":
            widget_manager.add_space()
        else:
            try:
                box = next(box_iter)
                widget_manager.add_character(char, box)
            except StopIteration:
                print("Помилка: Закінчилися бокси при ітерації.")
                break # Зупиняємо додавання, якщо бокси скінчилися

    # --- Додавання елементів у нижнє меню ---
    ttk.Button(menu_frame, text="Зберегти Виправлення", command=lambda: save_corrected_results(widget_manager)).pack(side=tk.LEFT, padx=5)
    ttk.Button(menu_frame, text="Копіювати Текст", command=lambda: copy_characters_to_clipboard(widget_manager.get_current_characters())).pack(side=tk.LEFT, padx=5)

    # Мітка для відображення повного тексту (глобальне посилання)
    characters_label_ref = ttk.Label(menu_frame, text="", anchor="w")
    characters_label_ref.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
    update_characters_label(characters_label_ref, widget_manager.get_current_characters()) # Початкове оновлення

    # --- Обробник закриття вікна ---
    detect_window.protocol("WM_DELETE_WINDOW", on_close_detect_window)

# --- Захист від прямого запуску ---
if __name__ == "__main__":
    print("Цей файл є частиною GUI і не призначений для прямого запуску.")
    print("Запустіть 'main.py'.")
    # input("Натисніть Enter для виходу...")
