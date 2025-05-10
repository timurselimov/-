# Импорт библиотек
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- ШАГ 1: Загрузка и подготовка данных ---
# Настройки
digits_to_use = [5, 8, 1, 3, 7]  # Вариант 1
samples_per_digit_train = 20
samples_per_digit_test = 10

# Загрузка файла MNIST (предположим, он находится в том же каталоге)
data = pd.read_csv("mnist_test.csv", header=None)

# Фильтрация нужных цифр
filtered_data = data[data[0].isin(digits_to_use)]

# Формирование обучающей и тестовой выборок
train_samples = []
test_samples = []

for digit in digits_to_use:
    digit_data = filtered_data[filtered_data[0] == digit]
    train_samples.append(digit_data.iloc[:samples_per_digit_train])
    test_samples.append(digit_data.iloc[samples_per_digit_train:samples_per_digit_train + samples_per_digit_test])

train_data = pd.concat(train_samples).sample(frac=1, random_state=42)  # перемешать
test_data = pd.concat(test_samples).sample(frac=1, random_state=42)

# Отделение меток и признаков
X_train = train_data.iloc[:, 1:].values / 255.0
y_train = train_data.iloc[:, 0].values

X_test = test_data.iloc[:, 1:].values / 255.0
y_test = test_data.iloc[:, 0].values

# Перевод меток в индексы классов (0–4)
label_to_index = {label: idx for idx, label in enumerate(digits_to_use)}
y_train_indexed = np.array([label_to_index[label] for label in y_train])
y_test_indexed = np.array([label_to_index[label] for label in y_test])

# --- ШАГ 2: Обучение нейронной сети (персептрон) ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(digits_to_use), activation='softmax')  # 5 классов
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train_indexed, epochs=30, validation_data=(X_test, y_test_indexed))

# --- ШАГ 3: Визуализация ошибки и точности ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title("Ошибка обучения")
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title("Точность распознавания")
plt.xlabel("Эпоха")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# --- ШАГ 4: Проверка одного изображения ---
idx = 0
sample = X_test[idx].reshape(28, 28)
plt.imshow(sample, cmap='gray')
plt.title(f"Истинный класс: {y_test[idx]}")
plt.axis('off')
plt.show()

pred = model.predict(X_test[idx].reshape(1, 784))
predicted_label = digits_to_use[np.argmax(pred)]
print(f"Предсказанный класс: {predicted_label}")

# --- ШАГ 5: Общая точность распознавания ---
train_pred = model.predict(X_train).argmax(axis=1)
test_pred = model.predict(X_test).argmax(axis=1)

train_accuracy = accuracy_score(y_train_indexed, train_pred)
test_accuracy = accuracy_score(y_test_indexed, test_pred)

print(f"Точность на обучающей выборке: {train_accuracy * 100:.2f}%")
print(f"Точность на тестовой выборке: {test_accuracy * 100:.2f}%")
