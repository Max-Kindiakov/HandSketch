
"""
Конфігураційний файл для моделі розпізнавання рукописних символів.

Містить визначення архітектури CNN моделі та трансформацій для даних.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# --- Константи ---
# Кількість класів для розпізнавання: 10 цифр + 26 літер + 3 спецсимволи (@&#)
NUM_CLASSES = 40
# Розмір вхідного зображення (використовуємо 32x32, як в оригінальних даних)
IMG_HEIGHT = 32
IMG_WIDTH = 32
# Кількість каналів вхідного зображення (1 для чорно-білого)
INPUT_CHANNELS = 1

# --- Архітектура Моделі ---
class CNNModel(nn.Module):
    """
    Згорткова нейронна мережа для розпізнавання рукописних символів.
    Приймає на вхід зображення 32x32.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super(CNNModel, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 8x8 -> 4x4
        )

        # Розрахунок розміру входу для повнозв'язного шару
        # Після трьох MaxPool(2,2) розмір карти ознак буде IMG_SIZE / (2^3)
        # Тобто 32 / 8 = 4. Отже, розмір 4x4.
        # Кількість каналів на виході останнього згорткового блоку = 256.
        self.flattened_size = 256 * (IMG_HEIGHT // 8) * (IMG_WIDTH // 8) # 256 * 4 * 4 = 4096

        self.fc_block = nn.Sequential(
            nn.Linear(self.flattened_size, 512), # Збільшимо розмір першого FC шару для кращої ємності
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Dropout для регуляризації
            nn.Linear(512, num_classes) # Вихідний шар з правильною кількістю класів
        )

        # Можна додати ініціалізацію ваг для кращої збіжності (опціонально)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прохід даних через модель."""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # Розгортання тензора перед повнозв'язними шарами
        x = x.view(x.size(0), -1) # Автоматично визначає розмір flattened_size
        x = self.fc_block(x)
        return x

    def _initialize_weights(self):
        """Ініціалізація ваг моделі."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) # Або kaiming_normal_ теж підійде
                nn.init.constant_(m.bias, 0)

# --- Трансформації Даних ---

# Трансформації для тренувального набору (з аугментацією)
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=INPUT_CHANNELS),
    # Додаємо аугментацію: невеликі повороти, зсуви, масштабування
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    # Переконаємось, що розмір правильний після афінних перетворень
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(), # Перетворення в тензор і масштабування в [0, 1]
    transforms.Normalize((0.5,), (0.5,)) # Нормалізація до [-1, 1]
])

# Трансформації для валідаційного/тестового набору (без аугментації)
# Важливо, щоб валідаційні дані проходили ту ж предобробку, що й тренувальні,
# окрім аугментації.
val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=INPUT_CHANNELS),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# --- Захист від прямого запуску ---
if __name__ == "__main__":
    print("Цей файл містить конфігурацію моделі та трансформацій.")
    print("Для тренування або тестування моделі запустіть відповідні скрипти (напр., 'train.py').")

    # Приклад використання: створення моделі та перевірка вихідного розміру
    print("\nСтворення екземпляру моделі...")
    try:
        model = CNNModel(num_classes=NUM_CLASSES)
        print("Модель успішно створено.")
        # Створимо фіктивний вхідний тензор
        # batch_size=4, канали=1, висота=32, ширина=32
        dummy_input = torch.randn(4, INPUT_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        print(f"Розмір вхідного тензора: {dummy_input.shape}")
        # Пропустимо тензор через модель
        output = model(dummy_input)
        print(f"Розмір вихідного тензора: {output.shape}")
        # Перевіримо, чи відповідає кількість класів
        assert output.shape == (4, NUM_CLASSES), f"Помилка: очікувався вихідний розмір (4, {NUM_CLASSES}), отримано {output.shape}"
        print("Тест розмірності пройшов успішно!")
    except Exception as e:
        print(f"Помилка під час створення або тестування моделі: {e}")

    # input("Натисніть Enter для виходу...") # Закоментовано для автоматизації