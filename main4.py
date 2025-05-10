import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- ПАРАМЕТРЫ ---
digits_to_use = [5, 8, 1, 3, 7]
samples_train = 200
samples_test = 100

# --- ЗАГРУЗКА ДАННЫХ ---
data = pd.read_csv("mnist_test.csv", header=None)

# --- ФИЛЬТРАЦИЯ И ПОДГОТОВКА ---
train_data = []
test_data = []

for digit in digits_to_use:
    subset = data[data[0] == digit]
    train_data.append(subset.iloc[:samples_train])
    test_data.append(subset.iloc[samples_train:samples_train + samples_test])

train_df = pd.concat(train_data).sample(frac=1, random_state=42)
test_df = pd.concat(test_data).sample(frac=1, random_state=42)

X_train = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
X_test = test_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0

# Преобразование меток в индексы классов (0–4)
label_map = {digit: idx for idx, digit in enumerate(digits_to_use)}
y_train = train_df.iloc[:, 0].map(label_map).values
y_test = test_df.iloc[:, 0].map(label_map).values

# --- ПОСТРОЕНИЕ СВЁРТОЧНОЙ СЕТИ ---
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(digits_to_use), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- ОБУЧЕНИЕ ---
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# --- ГРАФИКИ ---
plt.figure(figsize=(12, 5))

# LOSS
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title("Функция потерь")
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.legend()

# ACCURACY
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title("Точность распознавания")
plt.xlabel("Эпоха")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# --- МАТРИЦА НЕТОчНОСТЕЙ ---
y_pred = model.predict(X_test).argmax(axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=digits_to_use)
disp.plot(cmap='Blues')
plt.title("Матрица неточностей")
plt.show()

# --- ИТОГОВЫЕ ОЦЕНКИ ---
final_loss, final_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Финальная точность на тесте: {final_accuracy * 100:.2f}%")
print(f"Финальная ошибка (loss): {final_loss:.4f}")
