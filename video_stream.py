"""
Оптимізований модуль для обробки відеопотоку, розпізнавання жестів
та керування малюванням для систем з обмеженими ресурсами.
"""

import cv2
import numpy as np
import mediapipe as mp
import time

try:
    from detect_screen import detect_screen, detect_window_ref
except ImportError as e:
    print(f"Помилка імпорту в video_stream.py: {e}")
    exit(1)

# --- Ініціалізація MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
HandLandmark = mp_hands.HandLandmark

# --- Параметри оптимізації та стабілізації ---
# Частота перевірки жесту (1 раз на 5 кадрів). Зменшує навантаження.
GESTURE_CHECK_RATE = 5
# Кількість кадрів для підтвердження зміни жесту. Запобігає мерехтінню.
GESTURE_CONFIRMATION_FRAMES = 3

# --- Допоміжні функції для визначення жестів ---

def is_writing_gesture(hand_landmarks):
    """Жест "Письмо": витягнутий лише вказівний палець."""
    try:
        index_up = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[HandLandmark.INDEX_FINGER_PIP].y
        middle_down = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_PIP].y
        ring_down = hand_landmarks.landmark[HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[HandLandmark.RING_FINGER_PIP].y
        pinky_down = hand_landmarks.landmark[HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[HandLandmark.PINKY_PIP].y
        return index_up and middle_down and ring_down and pinky_down
    except:
        return False

def is_erasing_gesture(hand_landmarks):
    """Жест "Стирання": розкрита долоня, всі пальці випрямлені."""
    try:
        thumb_up = hand_landmarks.landmark[HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[HandLandmark.THUMB_IP].x
        index_up = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[HandLandmark.INDEX_FINGER_PIP].y
        middle_up = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_PIP].y
        ring_up = hand_landmarks.landmark[HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[HandLandmark.RING_FINGER_PIP].y
        pinky_up = hand_landmarks.landmark[HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[HandLandmark.PINKY_PIP].y
        return thumb_up and index_up and middle_up and ring_up and pinky_up
    except:
        return False

def is_approve_gesture(hand_landmarks):
    """Жест "Підтвердження": V-знак."""
    try:
        index_up = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[HandLandmark.INDEX_FINGER_MCP].y
        middle_up = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_MCP].y
        ring_down = hand_landmarks.landmark[HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[HandLandmark.RING_FINGER_MCP].y
        pinky_down = hand_landmarks.landmark[HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[HandLandmark.PINKY_MCP].y
        return index_up and middle_up and ring_down and pinky_down
    except:
        return False

# --- Основна функція відеопотоку ---

def video_stream(video, root, canvas, mouse_mode_var, cursor_label):
    active_gesture = None
    gesture_counters = {"write": 0, "erase": 0, "approve": 0}
    prev_x, prev_y = None, None
    frame_count = 0

    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while root.winfo_exists():
            if mouse_mode_var.get():
                time.sleep(0.1) # Зменшуємо навантаження в режимі миші
                continue

            ret, frame = video.read()
            if not ret:
                print("Помилка читання кадру з камери.")
                break

            frame = cv2.flip(frame, 1)
            frame_shape = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            frame_count += 1
            current_gesture_detected = None

            if results.multi_hand_landmarks:
                hand_landmark = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

                cursor_x = int(hand_landmark.landmark[HandLandmark.INDEX_FINGER_TIP].x * frame_shape[1])
                cursor_y = int(hand_landmark.landmark[HandLandmark.INDEX_FINGER_TIP].y * frame_shape[0])

                if cursor_label:
                    cursor_label.place(x=cursor_x - 5, y=cursor_y - 5)

                if frame_count % GESTURE_CHECK_RATE == 0:
                    if is_writing_gesture(hand_landmark):
                        current_gesture_detected = "write"
                    elif is_erasing_gesture(hand_landmark):
                        current_gesture_detected = "erase"
                    elif is_approve_gesture(hand_landmark):
                        current_gesture_detected = "approve"

                    if current_gesture_detected != active_gesture:
                        gesture_counters = {k: 0 for k in gesture_counters}
                        active_gesture = current_gesture_detected

                    if active_gesture:
                        gesture_counters[active_gesture] += 1

                # Дії виконуються на основі підтвердженого жесту
                if active_gesture and gesture_counters[active_gesture] >= GESTURE_CONFIRMATION_FRAMES:
                    if detect_window_ref is None:
                        if active_gesture == "write":
                            if prev_x is not None:
                                cv2.line(canvas, (prev_x, prev_y), (cursor_x, cursor_y), (0, 0, 0), 5)
                            prev_x, prev_y = cursor_x, cursor_y
                        elif active_gesture == "erase":
                            cv2.circle(canvas, (cursor_x, cursor_y), 20, (255, 255, 255), -1)
                            prev_x, prev_y = None, None
                        elif active_gesture == "approve":
                            if np.any(canvas != 255):
                                root.after(0, lambda: detect_screen(root, canvas, reopen=False))
                            active_gesture = None # Скидаємо жест після дії
                            gesture_counters = {k: 0 for k in gesture_counters}
                    else:
                        prev_x, prev_y = None, None
                else:
                    prev_x, prev_y = None, None
            else:
                if cursor_label:
                    cursor_label.place_forget()
                prev_x, prev_y = None, None
                active_gesture = None
                gesture_counters = {k: 0 for k in gesture_counters}

            #Показ кадру можна закоментувати для ще більшої продуктивності
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()
    print("Відеопотік завершено.")
