"""
Головний файл програми HandSketch.
Ініціалізує GUI, обробляє введення користувача (миша/камера),
запускає розпізнавання та тренування моделі.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, filedialog
import threading
import os
import time
from datetime import datetime
from PIL import Image, ImageTk

# --- Імпорт локальних модулів ---
try:
    from video_stream import video_stream
    from detect_screen import detect_screen
    from train import start_training
except ImportError as e:
    print(f"Помилка імпорту в main.py: {e}")
    print("Переконайтесь, що всі файли (.py) знаходяться в одному каталозі.")
    exit(1)

# --- Глобальні налаштування та змінні ---
BEST_MODEL_PATH = "logs/best_model.pth"
INITIAL_TRAINING_TRIGGERED = False # Прапорець, що тренування вже запускалось
APP_RUNNING = True # Прапорець для керування потоками

# --- Перевірка наявності моделі при старті ---
model_exists = os.path.exists(BEST_MODEL_PATH)
if not model_exists and not INITIAL_TRAINING_TRIGGERED:
    # Замість автоматичного запуску, покажемо попередження
    print(f"Попередження: Файл найкращої моделі '{BEST_MODEL_PATH}' не знайдено.")

# --- Ініціалізація Головного Вікна Tkinter ---
root = tk.Tk()
root.title("HandSketch - Розпізнавання Рукописного Тексту")
if os.name == "nt": # Встановлення іконки для Windows
    try:
        root.iconbitmap("icon.ico") # Переконайтесь, що файл icon.ico існує
    except tk.TclError:
        print("Попередження: Файл icon.ico не знайдено або має неправильний формат.")
# Забороняємо зміну розміру вікна
root.resizable(False, False)

# --- Налаштування Камери ---
allow_camera = False
video = None
try:
    # Спробуємо відкрити камеру, щоб перевірити її наявність
    test_video = cv2.VideoCapture(0)
    if test_video.isOpened():
        print("Камеру виявлено.")
        test_video.release() # Закриваємо тестове з'єднання
        # Запитуємо дозвіл користувача
        allow_camera = messagebox.askyesno("Дозвіл на використання камери",
                                           "Дозволити програмі HandSketch доступ до камери?")
        if allow_camera:
            video = cv2.VideoCapture(0) # Відкриваємо камеру для використання
            if not video.isOpened(): # Додаткова перевірка
                 allow_camera = False
                 messagebox.showerror("Помилка Камери", "Не вдалося відкрити камеру для використання.")
        else:
            print("Користувач відхилив використання камери.")
    else:
        print("Камеру не виявлено або не вдалося відкрити.")

except Exception as e:
    print(f"Помилка під час ініціалізації камери: {e}")
    messagebox.showerror("Помилка Камери", f"Сталася помилка під час доступу до камери:\n{e}")

# Визначення розмірів полотна
if allow_camera and video.isOpened():
    # Беремо розміри з камери
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Розміри камери: {width}x{height}")
else:
    # Розміри за замовчуванням, якщо камера недоступна
    width = 960
    height = 540
    print(f"Камера недоступна. Встановлено розміри полотна за замовчуванням: {width}x{height}")

# --- Створення Полотна (Canvas) ---
# Використовуємо NumPy array як буфер для малювання
canvas = np.ones((height, width, 3), dtype="uint8") * 255 # Білий фон (BGR)

# --- Налаштування GUI Елементів ---
style = ttk.Style()
style.configure("TButton", padding=5, relief="flat", background="#ccc")
style.map("TButton", background=[('active', '#eee')])

# Фрейм для полотна
canvas_frame = ttk.Frame(root, width=width, height=height)
canvas_frame.pack()
# Мітка для відображення полотна
canvas_label = ttk.Label(canvas_frame)
canvas_label.pack()
# Мітка для курсору камери (спочатку прихована)
cursor_label = tk.Label(canvas_frame, width=1, height=1, bg="red") # Зробимо курсор помітнішим

# --- Змінні для Керування Малюванням ---
mouse_mode_var = tk.BooleanVar(value=not allow_camera) # Режим миші активний, якщо камери немає
mouse_drawing = False
mouse_erasing = False
prev_x, prev_y = None, None # Глобальні для збереження попередніх координат

# --- Функції Обробники Подій ---

def update_canvas():
    """Оновлює зображення на мітці Tkinter з NumPy canvas."""
    if not APP_RUNNING: return # Зупиняємо оновлення, якщо програма закривається

    # Конвертуємо BGR (OpenCV) в RGB (PIL/Tkinter)
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(canvas_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    # Оновлюємо мітку
    canvas_label.imgtk = imgtk # Зберігаємо посилання, щоб уникнути збирання сміття
    canvas_label.configure(image=imgtk)

    # Плануємо наступне оновлення
    root.after(15, update_canvas) # Оновлення приблизно 60 FPS (1000/15)

def mouse_press(event):
    """Обробляє натискання кнопки миші."""
    global mouse_drawing, mouse_erasing, prev_x, prev_y
    if mouse_mode_var.get(): # Працює лише в режимі миші
        if event.num == 1: # Ліва кнопка - малювання
            mouse_drawing = True
            mouse_erasing = False
            prev_x, prev_y = event.x, event.y
        elif event.num == 3: # Права кнопка - стирання
            mouse_drawing = False
            mouse_erasing = True

def mouse_release(event):
    """Обробляє відпускання кнопки миші."""
    global mouse_drawing, mouse_erasing, prev_x, prev_y
    if mouse_mode_var.get():
        mouse_drawing = False
        mouse_erasing = False
        prev_x, prev_y = None, None # Скидаємо координати

def mouse_motion(event):
    """Обробляє рух миші з натиснутою кнопкою."""
    global mouse_drawing, mouse_erasing, prev_x, prev_y
    if mouse_mode_var.get(): # Працює лише в режимі миші
        x, y = event.x, event.y
        # Перевірка виходу за межі полотна
        if 0 <= x < width and 0 <= y < height:
            if mouse_drawing:
                if prev_x is not None and prev_y is not None:
                    # Малюємо лінію чорним кольором (0, 0, 0)
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), thickness=5, lineType=cv2.LINE_AA) # Згладжування
                prev_x, prev_y = x, y
            elif mouse_erasing:
                # Стираємо білим кольором (255, 255, 255)
                cv2.circle(canvas, (x, y), radius=15, color=(255, 255, 255), thickness=-1) # Заповнене коло

def save_image():
    """Зберігає поточне зображення полотна у файл."""
    # Перевіряємо, чи є що зберігати
    if not np.any(canvas != 255):
        messagebox.showinfo("Збереження", "Полотно порожнє. Немає що зберігати.")
        return

    # Створюємо папку, якщо її немає
    os.makedirs("images", exist_ok=True)

    # Діалог збереження файлу
    file_path = filedialog.asksaveasfilename(
        title="Зберегти зображення як...",
        defaultextension=".png", # Використовуємо PNG для кращої якості
        initialdir="images",
        initialfile=f"handsketch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
    )

    if file_path:
        try:
            # !!! Конвертуємо BGR в RGB перед збереженням через PIL !!!
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            img_to_save = Image.fromarray(canvas_rgb)
            img_to_save.save(file_path)
            print(f"Зображення успішно збережено як: {file_path}")
            messagebox.showinfo("Збереження", f"Зображення успішно збережено:\n{file_path}")
        except Exception as e:
            print(f"Помилка під час збереження зображення: {e}")
            messagebox.showerror("Помилка Збереження", f"Не вдалося зберегти зображення:\n{e}")

def toggle_mode():
    """Перемикає режим введення між камерою та мишею."""
    if not allow_camera: # Якщо камери немає, перемикання неможливе
        messagebox.showwarning("Режим Камери Недоступний", "Камера не виявлена або дозвіл не надано.")
        return

    current_mode = mouse_mode_var.get()
    mouse_mode_var.set(not current_mode) # Інвертуємо режим

    if mouse_mode_var.get():
        camera_mode_label.config(text="Режим: МИША")
        cursor_label.place_forget() # Ховаємо курсор камери
        print("Перемкнено в режим МИШІ.")
    else:
        camera_mode_label.config(text="Режим: КАМЕРА")
        print("Перемкнено в режим КАМЕРИ.")
        # Курсор камери покаже `video_stream`, коли виявить руку

def start_training_in_thread():
    """Запускає процес тренування в окремому потоці."""
    global INITIAL_TRAINING_TRIGGERED, model_exists
    INITIAL_TRAINING_TRIGGERED = True # Відмічаємо, що тренування було ініційоване

    # Вимикаємо кнопки, що залежать від тренування/моделі
    detect_button.config(state=tk.DISABLED)
    train_button.config(text="Тренування...", state=tk.DISABLED)
    root.update_idletasks() # Оновлюємо GUI

    print("Запуск тренування в окремому потоці...")
    try:
        # Запускаємо тренування
        start_training() # Ця функція тепер блокує до завершення (або виводить в консоль)
        print("Процес тренування завершено (або запущено у фоні, якщо він так реалізований).")
        # Перевіряємо, чи з'явилася модель після тренування
        model_exists = os.path.exists(BEST_MODEL_PATH)
        if model_exists:
             print(f"Файл моделі '{BEST_MODEL_PATH}' знайдено після тренування.")
             # Вмикаємо кнопку детектування
             detect_button.config(state=tk.NORMAL)
             messagebox.showinfo("Тренування Завершено",
                                 f"Тренування завершено.\nМодель '{BEST_MODEL_PATH}' створено або оновлено.")
        else:
             print(f"Попередження: Файл моделі '{BEST_MODEL_PATH}' не знайдено після тренування.")
             messagebox.showwarning("Тренування Завершено",
                                    f"Тренування завершено, але файл моделі\n'{BEST_MODEL_PATH}'\nне знайдено. Можливо, сталася помилка.")

    except Exception as e:
        print(f"Помилка під час виконання start_training: {e}")
        messagebox.showerror("Помилка Тренування", f"Сталася помилка під час тренування:\n{e}")
    finally:
        # Завжди вмикаємо кнопку тренування назад
        train_button.config(text="Train Again", state=tk.NORMAL)


def train_again():
    """Запитує підтвердження та запускає тренування."""
    if messagebox.askyesno("Повторне Тренування",
                           "Ви впевнені, що хочете запустити тренування моделі?\n"
                           "Це може зайняти тривалий час, і програма може не відповідати.\n"
                           "Рекомендується перезапустити програму після завершення тренування."):
        # Запускаємо тренування в окремому потоці, щоб GUI не блокувався
        training_thread = threading.Thread(target=start_training_in_thread, daemon=True)
        training_thread.start()

def clear_canvas_action():
     """Очищує полотно."""
     canvas.fill(255)
     print("Полотно очищено.")

def on_close_app():
    """Обробляє закриття головного вікна."""
    global APP_RUNNING
    print("Закриття програми HandSketch...")
    APP_RUNNING = False # Сигнал для зупинки циклів/потоків

    # Даємо трохи часу потокам завершитися (опціонально)
    # time.sleep(0.1)

    if video is not None and video.isOpened():
        print("Звільнення ресурсів камери...")
        video.release()

    # Закриваємо OpenCV вікна (якщо video_stream їх не закрив)
    cv2.destroyAllWindows()

    print("Знищення головного вікна...")
    root.destroy()
    print("Програму завершено.")


# --- Створення Меню Кнопок ---
menu_frame = ttk.Frame(root, padding="5")
menu_frame.pack(fill=tk.X)

# Кнопка Збереження
save_button = ttk.Button(menu_frame, text="Зберегти Зображення", command=save_image)
save_button.pack(side=tk.LEFT, padx=5, pady=2)

# Кнопка Детектування
detect_button = ttk.Button(
    menu_frame,
    text="Розпізнати Символи",
    command=lambda: detect_screen(root, canvas, reopen=True) if np.any(canvas != 255) else messagebox.showinfo("Розпізнавання", "Полотно порожнє."),
    state=tk.NORMAL if model_exists else tk.DISABLED # Вимкнено, якщо моделі немає
)
detect_button.pack(side=tk.LEFT, padx=5, pady=2)

# Кнопка Очищення
clear_button = ttk.Button(menu_frame, text="Очистити Полотно", command=clear_canvas_action)
clear_button.pack(side=tk.LEFT, padx=5, pady=2)

# Кнопка Тренування
train_button = ttk.Button(menu_frame, text="Train Again", command=train_again)
train_button.pack(side=tk.LEFT, padx=5, pady=2)

# Кнопка Перемикання Режиму (якщо камера доступна)
if allow_camera:
    toggle_button = ttk.Button(menu_frame, text="Перемкнути Режим", command=toggle_mode)
    toggle_button.pack(side=tk.LEFT, padx=5, pady=2)
    initial_mode_text = "Режим: КАМЕРА" if not mouse_mode_var.get() else "Режим: МИША"
    camera_mode_label = ttk.Label(menu_frame, text=initial_mode_text)
    camera_mode_label.pack(side=tk.LEFT, padx=10, pady=3)

# --- Прив'язка Обробників Миші до Мітки Полотна ---
canvas_label.bind("<ButtonPress-1>", mouse_press)   # Ліва кнопка
canvas_label.bind("<ButtonPress-3>", mouse_press)   # Права кнопка
canvas_label.bind("<ButtonRelease-1>", mouse_release) # Ліва кнопка
canvas_label.bind("<ButtonRelease-3>", mouse_release) # Права кнопка
canvas_label.bind("<Motion>", mouse_motion)          # Рух миші

# --- Обробник Закриття Вікна ---
root.protocol("WM_DELETE_WINDOW", on_close_app)

# --- Запуск Потоку Відео (якщо камера доступна) ---
if allow_camera and video is not None and video.isOpened():
    print("Запуск потоку обробки відео з камери...")
    # Використовуємо daemon=True, щоб потік автоматично завершився при виході з програми
    video_thread = threading.Thread(
        target=video_stream,
        args=(video, root, canvas, mouse_mode_var, cursor_label),
        daemon=True
    )
    video_thread.start()
else:
    print("Потік відео не запущено (камера недоступна або відхилена).")

# --- Запуск Головного Циклу Оновлення GUI ---
print("Запуск головного циклу GUI...")
update_canvas() # Запускаємо перший раз
root.mainloop()

print("Головний цикл GUI завершено.")
