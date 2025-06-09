
"""
Обробляє відеопотік з камери, розпізнає жести рук за допомогою MediaPipe
та керує малюванням/стиранням на полотні або викликом екрану детектування.
"""

import cv2
import numpy as np
import mediapipe as mp
import time # Для стабілізації жестів

# Імпортуємо посилання на вікно та функцію виклику з detect_screen
try:
    # Нам потрібна лише функція detect_screen та посилання на вікно
    from detect_screen import detect_screen, detect_window_ref
except ImportError as e:
    print(f"Помилка імпорту в video_stream.py: {e}")
    print("Переконайтесь, що файл detect_screen.py знаходиться поруч.")
    exit(1)


# Ініціалізація MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
HandLandmark = mp_hands.HandLandmark # Зручний доступ до індексів

# --- Параметри для стабілізації жестів ---
GESTURE_CONFIRMATION_FRAMES = 3 # Кількість кадрів для підтвердження жесту
gesture_counters = {"write": 0, "erase": 0, "approve": 0}
active_gesture = None # Поточний активний жест

# --- Допоміжні функції для визначення жестів ---

def get_landmark_coords(hand_landmarks, landmark_index, frame_shape):
    """Отримує координати (x, y) landmark'а у пікселях."""
    landmark = hand_landmarks.landmark[landmark_index]
    height, width, _ = frame_shape
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return x, y

def is_writing_gesture(hand_landmarks, frame_shape):
    """
    Перевіряє жест 'письма' (зближення великого та вказівного пальців).
    Простіша та потенційно надійніша логіка.
    """
    try:
        thumb_tip = hand_landmarks.landmark[HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_TIP]
        index_mcp = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_MCP] # Основа вказівного пальця

        # Розрахунок відстані між кінчиками великого та вказівного пальців
        distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

        # Поріг відстані (емпіричний, може потребувати налаштування)
        # Відстань нормалізована (0 до ~1.4), поріг залежить від розміру руки на екрані.
        # Спробуємо порівняти з відстанню між кінчиком і основою вказівного пальця.
        index_length = np.sqrt((index_tip.x - index_mcp.x)**2 + (index_tip.y - index_mcp.y)**2)
        distance_threshold = index_length * 0.3 # Наприклад, 30% довжини пальця

        # Додаткова умова: кінчик вказівного пальця нижче середнього суглоба
        index_pip = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_PIP]
        is_index_down = index_tip.y > index_pip.y # Y збільшується вниз

        return distance < distance_threshold and is_index_down
    except IndexError:
        return False

def is_erasing_gesture(hand_landmarks, frame_shape):
    """
    Перевіряє жест 'стирання' (розкрита долоня або кулак).
    Логіка схожа на оригінальну, але використовує іменовані landmark'и.
    """
    try:
        # Перевіряємо, чи кінчики пальців нижче середніх суглобів
        fingers_down = [
            hand_landmarks.landmark[HandLandmark.INDEX_FINGER_TIP].y > hand_landmarks.landmark[HandLandmark.INDEX_FINGER_PIP].y,
            hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_PIP].y,
            hand_landmarks.landmark[HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[HandLandmark.RING_FINGER_PIP].y,
            hand_landmarks.landmark[HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[HandLandmark.PINKY_PIP].y
        ]
        # Додамо великий палець: кінчик нижче другого суглоба
        thumb_down = hand_landmarks.landmark[HandLandmark.THUMB_TIP].y > hand_landmarks.landmark[HandLandmark.THUMB_IP].y

        # Вважаємо жестом, якщо всі або майже всі пальці зігнуті/опущені
        return sum(fingers_down) >= 3 and thumb_down # Наприклад, хоча б 3 пальці + великий
    except IndexError:
        return False

def is_approve_gesture(hand_landmarks, frame_shape):
    """
    Перевіряє жест 'підтвердження'.
    Спробуємо простіший варіант: підняті вказівний та середній пальці ("V" або "peace").
    """
    try:
        # Кінчики вказівного та середнього пальців мають бути вище своїх основ
        index_up = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[HandLandmark.INDEX_FINGER_MCP].y
        middle_up = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_MCP].y

        # Кінчики безіменного та мізинця мають бути нижче своїх основ (зігнуті)
        ring_down = hand_landmarks.landmark[HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[HandLandmark.RING_FINGER_MCP].y
        pinky_down = hand_landmarks.landmark[HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[HandLandmark.PINKY_MCP].y

        return index_up and middle_up and ring_down and pinky_down
    except IndexError:
        return False

# --- Основна функція відеопотоку ---

def video_stream(video, root=None, canvas=None, mouse_mode_var=None, cursor_label=None):
    """
    Обробляє відеопотік, розпізнає жести та взаємодіє з canvas та GUI.
    """
    global active_gesture, gesture_counters

    # Створюємо об'єкт Hands з трохи вищими порогами для стабільності
    with mp_hands.Hands(
            model_complexity=0, # 0 для швидкості, 1 для більшої точності
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6) as hands:

        prev_x, prev_y = None, None # Локальні для цього потоку

        while True:
            ret, frame = video.read()
            if not ret:
                print("Помилка читання кадру з камери або кінець відео.")
                break

            # Віддзеркалення кадру для інтуїтивного керування
            frame = cv2.flip(frame, 1)
            frame_shape = frame.shape

            # Перевірка режиму (камера/миша)
            if mouse_mode_var is not None and mouse_mode_var.get():
                # У режимі миші ми не обробляємо руки, але маємо очистити стан
                active_gesture = None
                gesture_counters = {k: 0 for k in gesture_counters}
                prev_x, prev_y = None, None
                if cursor_label:
                    cursor_label.place_forget() # Ховаємо курсор камери
                cv2.imshow("Frame", frame) # Все одно показуємо кадр
                if cv2.waitKey(1) & 0xFF == ord('q'): # Додамо вихід по 'q'
                   break
                # Перевірка закриття вікна OpenCV
                if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue # Переходимо до наступної ітерації

            # --- Обробка в режимі камери ---
            # Конвертація в RGB для MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False # Оптимізація: позначити як read-only
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True # Повернути можливість запису

            current_gesture_detected = None # Який жест виявлено в цьому кадрі

            # Обробка результатів MediaPipe
            if results.multi_hand_landmarks:
                # Беремо першу знайдену руку
                hand_landmark = results.multi_hand_landmarks[0]

                # Малювання скелету руки (опціонально, для візуалізації)
                mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

                # Отримання координат курсору (кінчик вказівного пальця)
                cursor_x, cursor_y = get_landmark_coords(hand_landmark, HandLandmark.INDEX_FINGER_TIP, frame_shape)

                # Оновлення положення GUI курсору (якщо є)
                if cursor_label:
                    # Зауваження: Пряме оновлення GUI з іншого потоку не є ідеально безпечним.
                    cursor_label.place(x=cursor_x - cursor_label.winfo_width() // 2, # Центрування
                                       y=cursor_y - cursor_label.winfo_height() // 2)

                # --- Розпізнавання жестів ---
                # Перевіряємо лише якщо вікно детектування не відкрито
                if detect_window_ref is None:
                    write_detected = is_writing_gesture(hand_landmark, frame_shape)
                    erase_detected = is_erasing_gesture(hand_landmark, frame_shape)
                    approve_detected = is_approve_gesture(hand_landmark, frame_shape)

                    # Визначаємо, який жест (якщо є) має пріоритет
                    if write_detected:
                        current_gesture_detected = "write"
                    elif erase_detected:
                        current_gesture_detected = "erase"
                    elif approve_detected:
                        current_gesture_detected = "approve"
                    # Якщо жоден не активний, current_gesture_detected залишиться None

                # --- Стабілізація жестів ---
                if current_gesture_detected == active_gesture:
                    # Якщо поточний збігається з активним, збільшуємо лічильник
                    if active_gesture is not None:
                        gesture_counters[active_gesture] = min(
                            gesture_counters[active_gesture] + 1,
                            GESTURE_CONFIRMATION_FRAMES # Обмежуємо зверху
                        )
                else:
                    # Якщо жест змінився або зник, скидаємо лічильники
                    gesture_counters = {k: 0 for k in gesture_counters}
                    active_gesture = current_gesture_detected # Оновлюємо активний жест

                # --- Виконання дій на основі ПІДТВЕРДЖЕНОГО жесту ---
                gesture_confirmed = (active_gesture is not None and
                                     gesture_counters[active_gesture] >= GESTURE_CONFIRMATION_FRAMES)

                if gesture_confirmed:
                    if active_gesture == "write":
                        if prev_x is not None and prev_y is not None:
                            cv2.line(canvas, (prev_x, prev_y), (cursor_x, cursor_y), (0, 0, 0), 5)
                        prev_x, prev_y = cursor_x, cursor_y
                    elif active_gesture == "erase":
                        cv2.circle(canvas, (cursor_x, cursor_y), 15, (255, 255, 255), -1)
                        prev_x, prev_y = None, None # Не малюємо лінію після стирання
                    elif active_gesture == "approve":
                        # Викликаємо екран детектування, якщо є що детектувати
                        if root and canvas is not None and np.any(canvas != 255):
                            print("Жест 'approve' підтверджено. Виклик detect_screen.")
                            # Викликаємо в головному потоці через 'after' для безпеки GUI
                            root.after(0, lambda: detect_screen(root, canvas, reopen=False))
                        # Скидаємо лічильник, щоб не викликати постійно
                        gesture_counters["approve"] = 0
                        active_gesture = None # Скидаємо активний жест після дії
                        prev_x, prev_y = None, None
                    # Якщо жест підтверджено, але це не write/erase/approve,
                    # просто скидаємо prev_x, prev_y
                    elif active_gesture not in ["write", "erase"]:
                         prev_x, prev_y = None, None

                else:
                    # Якщо жест не підтверджено (або немає активного жесту)
                    prev_x, prev_y = None, None

            else:
                # Руки не знайдено
                prev_x, prev_y = None, None
                active_gesture = None
                gesture_counters = {k: 0 for k in gesture_counters}
                if cursor_label:
                    cursor_label.place_forget() # Ховаємо курсор

            # Відображення кадру
            cv2.imshow("Frame", frame)

            # Обробка натискання клавіш та закриття вікна
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): # Вихід по 'q'
                break
            # Перевірка, чи вікно OpenCV все ще відкрите
            # Це надійніше, ніж перевіряти key == -1
            if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
                 print("Вікно OpenCV було закрито.")
                 break

    # Звільнення ресурсів
    print("Зупинка відеопотоку та звільнення ресурсів.")
    video.release()
    cv2.destroyAllWindows()
    if cursor_label:
        # Остаточно ховаємо курсор при виході
        # Використовуємо 'after', щоб гарантовано виконати в головному потоці
        if root and root.winfo_exists():
             root.after(0, cursor_label.place_forget)


# --- Блок для самостійного тестування ---
if __name__ == "__main__":
    print("Запуск video_stream.py в тестовому режимі...")
    # Створюємо 'заглушки' для залежностей від GUI
    class DummyTkWidget:
        def get(self): return False # Імітуємо режим камери
        def place(self, x, y): pass
        def place_forget(self): pass
        def winfo_width(self): return 10 # Приблизний розмір курсора
        def winfo_height(self): return 10
        def winfo_exists(self): return True # Імітуємо існування вікна

    class DummyTkRoot:
         def after(self, ms, func):
             # В тестовому режимі просто викликаємо функцію негайно
             # УВАГА: Це не імітує реальну асинхронність GUI!
             try:
                 func()
             except Exception as e:
                 print(f"Помилка при виклику функції через dummy 'after': {e}")
         def winfo_exists(self): return True

    test_video = cv2.VideoCapture(0)
    test_canvas = np.ones((480, 640, 3), dtype="uint8") * 255 # Створюємо тестове полотно
    dummy_root = DummyTkRoot()
    dummy_cursor = DummyTkWidget()
    dummy_mouse_mode = DummyTkWidget()

    if not test_video.isOpened():
        print("Помилка: Камера не знайдена або не може бути відкрита.")
    else:
        try:
            video_stream(
                video=test_video,
                root=dummy_root, # Передаємо заглушку
                canvas=test_canvas,
                mouse_mode_var=dummy_mouse_mode, # Передаємо заглушку
                cursor_label=dummy_cursor # Передаємо заглушку
            )
        except Exception as e:
            print(f"Під час тестового запуску сталася помилка: {e}")
        finally:
             # Переконуємося, що ресурси звільнені, навіть якщо була помилка
             if test_video.isOpened():
                 test_video.release()
             cv2.destroyAllWindows()
    print("Тестовий режим video_stream.py завершено.")
    # input("Натисніть Enter для виходу...")