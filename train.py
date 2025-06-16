
"""
Скрипт для тренування моделі розпізнавання рукописних символів.

Виконує цикл навчання, валідацію після кожної епохи,
зберігає найкращу модель, інформацію про класи та використовує GPU, якщо доступно.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # Для логування в TensorBoard
from torchvision import datasets
import argparse
import os
import time
import json
from tqdm import tqdm

# Імпортуємо конфігурацію (модель та трансформації)
try:
    from configuration import CNNModel, train_transform, val_test_transform, NUM_CLASSES
except ImportError:
    print("Помилка: Не вдалося імпортувати з configuration.py.")
    print("Переконайтесь, що файл configuration.py існує та не містить синтаксичних помилок.")
    exit(1)

# --- Налаштування Аргументів Командного Рядка ---
parser = argparse.ArgumentParser(description="Налаштування тренування моделі розпізнавання символів")
parser.add_argument("-e", "--epochs", type=int, default=25, help="Кількість епох для тренування")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="Розмір батчу")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Швидкість навчання (learning rate)")
parser.add_argument("-w", "--workers", type=int, default=2, help="Кількість паралельних процесів для завантаження даних (рекомендовано <= 2 для T4)")
parser.add_argument("--data_dir", type=str, default="./dataset", help="Шлях до кореневої папки датасету (з підпапками Train та Validation)")
parser.add_argument("--log_dir", type=str, default="./logs", help="Шлях до папки для збереження логів та моделі")
parser.add_argument("--force_cpu", action="store_true", help="Примусово використовувати CPU, навіть якщо GPU доступний")
args = parser.parse_args()

# --- Визначення Пристрою (GPU/CPU) ---
if not args.force_cpu and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"Використовується GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Використовується CPU")

num_workers = min(args.workers, 2)
print(f"Кількість workers для DataLoader: {num_workers}")

# --- Функція Тренування однієї епохи ---
def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch_num, total_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(data_loader, desc=f"Епоха {epoch_num+1}/{total_epochs} [Тренування]", unit="batch")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        progress_bar.set_postfix(loss=loss.item())
    epoch_loss = running_loss / total_samples
    epoch_acc = 100.0 * correct_predictions / total_samples
    return epoch_loss, epoch_acc

# --- Функція Валідації/Оцінки ---
def evaluate_model(model, criterion, data_loader, device, phase="Валідація"):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(data_loader, desc=f"{phase}", unit="batch", leave=False)
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item())
    epoch_loss = running_loss / total_samples
    epoch_acc = 100.0 * correct_predictions / total_samples
    return epoch_loss, epoch_acc

# --- Основна Функція Тренування ---
def start_training():
    print("\n--- Початок процесу тренування ---")
    start_time = time.time()

    # --- 1. Перевірка та Створення Директорій ---
    train_dir = os.path.join(args.data_dir, "Train")
    val_dir = os.path.join(args.data_dir, "Validation")
    log_dir = args.log_dir
    model_save_path = os.path.join(log_dir, "best_model.pth")
    # Шлях для збереження інформації про класи
    class_info_path = os.path.join(log_dir, 'class_info.json') # <--- Визначаємо шлях

    if not os.path.isdir(train_dir):
        print(f"Помилка: Папка тренувальних даних не знайдена: {train_dir}")
        return
    if not os.path.isdir(val_dir):
        print(f"Помилка: Папка валідаційних даних не знайдена: {val_dir}")
        return

    os.makedirs(log_dir, exist_ok=True)
    print(f"Логи, модель та інформація про класи будуть збережені в: {log_dir}")

    # --- 2. Завантаження Даних ---
    print("\nЗавантаження датасетів...")
    try:
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        validation_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>> ПОЧАТОК ЗМІН <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # !!! ОТРИМАННЯ ТА ЗБЕРЕЖЕННЯ ІНФОРМАЦІЇ ПРО КЛАСИ !!!
        print("-" * 30)
        print("Визначення порядку класів (з ImageFolder):")
        actual_classes = train_dataset.classes # Список імен класів у правильному порядку
        actual_class_to_idx = train_dataset.class_to_idx # Словник клас->індекс
        print(f"  Знайдені класи (папки): {actual_classes}")
        print(f"  Мапування клас -> індекс: {actual_class_to_idx}")
        print(f"  Кількість знайдених класів: {len(actual_classes)}")
        print("-" * 30)

        # Перевірка відповідності кількості класів з конфігурацією
        if len(actual_classes) != NUM_CLASSES:
             print(f"!!! КРИТИЧНЕ ПОПЕРЕДЖЕННЯ !!!")
             print(f"Кількість знайдених класів ({len(actual_classes)}) не співпадає з NUM_CLASSES ({NUM_CLASSES}) у configuration.py!")
             print("Це може призвести до помилок під час тренування або розпізнавання.")
             print("Перевірте папки у '{train_dir}' та значення NUM_CLASSES.")
             # Можна розкоментувати наступний рядок, щоб зупинити процес
             # exit(1)

        # Зберігаємо інформацію у файл JSON
        class_info_to_save = {
            'classes': actual_classes, # Список імен класів у правильному порядку індексів 0, 1, ...
            'class_to_idx': actual_class_to_idx # Словник для довідки
        }
        try:
            with open(class_info_path, 'w', encoding='utf-8') as f:
                json.dump(class_info_to_save, f, ensure_ascii=False, indent=4)
            print(f"Інформацію про класи успішно збережено у: '{class_info_path}'")
        except IOError as e:
            print(f"!!! ПОМИЛКА !!! Не вдалося зберегти інформацію про класи у '{class_info_path}': {e}")
        except Exception as e:
             print(f"!!! НЕОЧІКУВАНА ПОМИЛКА під час збереження class_info.json: {e}")
             # exit(1)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>> КІНЕЦЬ ЗМІН <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    except FileNotFoundError as e:
         print(f"Помилка: Не знайдено папку з даними: {e}")
         print("Перевірте правильність шляху у --data_dir та структуру папок Train/Validation.")
         return
    except Exception as e:
        print(f"Помилка під час завантаження даних з ImageFolder: {e}")
        print("Перевірте структуру папок у 'dataset/Train' та 'dataset/Validation'.")
        print("Кожна підпапка повинна відповідати класу, і містити зображення.")
        return

    print(f"\nЗнайдено {len(train_dataset)} зображень для тренування.")
    print(f"Знайдено {len(validation_dataset)} зображень для валідації.")

    # Створення DataLoader'ів
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)

    print(f"\nРозмір батчу: {args.batch_size}, Кількість епох: {args.epochs}, Швидкість навчання: {args.learning_rate}")

    # --- 3. Ініціалізація Моделі, Функції Втрат, Оптимізатора ---
    print("\nІніціалізація моделі...")
    model = CNNModel(num_classes=NUM_CLASSES).to(device) # Використовуємо NUM_CLASSES з конфігурації
    print(f"Архітектура моделі (очікує {NUM_CLASSES} класів):")
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # --- 4. Налаштування Логування (TensorBoard) ---
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard_logs'))
    print(f"\nЛоги TensorBoard зберігаються в: {os.path.join(log_dir, 'tensorboard_logs')}")
    print("Щоб переглянути: запустіть 'tensorboard --logdir=./logs/tensorboard_logs' в терміналі.")

    # --- 5. Цикл Навчання та Валідації ---
    best_val_accuracy = 0.0
    print("\n--- Початок циклу навчання ---")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, args.epochs)
        val_loss, val_acc = evaluate_model(model, criterion, validation_loader, device, phase="Валідація")

        print(f"\nЕпоха {epoch+1}/{args.epochs}:")
        print(f"  Тренування: Loss: {train_loss:.4f}, Точність: {train_acc:.2f}%")
        print(f"  Валідація: Loss: {val_loss:.4f}, Точність: {val_acc:.2f}%")

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(val_loss)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            try:
                torch.save(model.state_dict(), model_save_path)
                print(f"  -> Збережено нову найкращу модель з точністю {best_val_accuracy:.2f}% у '{model_save_path}'")
            except Exception as e:
                print(f"!!! ПОМИЛКА при збереженні моделі у '{model_save_path}': {e}")

    print("\n--- Тренування завершено ---")
    writer.close()

    # --- 6. Фінальна Оцінка на Валідаційному Наборі (з найкращою моделлю) ---
    print(f"\nЗавантаження найкращої моделі з '{model_save_path}' для фінальної оцінки...")
    try:
        # Створюємо новий екземпляр моделі на випадок, якщо стара модель змінилася
        final_model = CNNModel(num_classes=NUM_CLASSES).to(device)
        final_model.load_state_dict(torch.load(model_save_path, map_location=device))
        print("Найкращу модель успішно завантажено.")

        final_val_loss, final_val_acc = evaluate_model(final_model, criterion, validation_loader, device, phase="Фінальна Оцінка (Найкраща Модель)")
        print(f"\n--- Фінальна Оцінка Найкращої Моделі ---")
        print(f"  Валідаційний Loss: {final_val_loss:.4f}")
        print(f"  Валідаційна Точність: {final_val_acc:.2f}%")
    except FileNotFoundError:
         print(f"Помилка: Не вдалося знайти збережену модель '{model_save_path}'. Пропускаємо фінальну оцінку.")
    except Exception as e:
         print(f"Помилка під час завантаження моделі або фінальної оцінки: {e}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nЗагальний час тренування: {total_time // 60:.0f} хв {total_time % 60:.0f} сек")
    print("--- Завершення Роботи ---")

if __name__ == "__main__":
    start_training()
    # input("Натисніть Enter для виходу...") # Закоментовано для автоматизації
