import os
import cv2
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers

base = 'dataset'
class_dirs = [name for name in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, name))]

X = []
y = []
for idx, class_name in enumerate(class_dirs):
    files = sorted([
        f for f in os.listdir(os.path.join(base, class_name))
        if f.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])[:250]
    for file_name in files:
        img = cv2.imread(os.path.join(base, class_name, file_name), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (48, 48))
        X.append(img.astype('float32') / 255.0)
        y.append(idx)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
idx = np.arange(len(X))
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

model = keras.Sequential([
    layers.Input(shape=(48, 48, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dropout(0.35),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(len(class_dirs), activation='softmax'),
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.08,
    width_shift_range=0.08,
    height_shift_range=0.08,
    brightness_range=[0.9, 1.1],
    shear_range=0.08,
    fill_mode='nearest',
)
val_test_datagen = keras.preprocessing.image.ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=64, shuffle=True)
val_generator = val_test_datagen.flow(X_val, y_val, batch_size=64, shuffle=False)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=12,
    steps_per_epoch=max(1, math.ceil(len(X_train) / 64)),
    validation_steps=max(1, math.ceil(len(X_val) / 64)),
    verbose=0,
)

probs = model.predict(X_test, verbose=0)
preds = np.argmax(probs, axis=1)
acc = accuracy_score(y_test, preds)
print(f'Plain CNN test accuracy: {acc:.4f}')
print(f'Last validation accuracy: {history.history["val_accuracy"][-1]:.4f}')
