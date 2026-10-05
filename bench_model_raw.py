import os
import cv2
import numpy as np
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
        X.append(img.astype('float32'))
        y.append(idx)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
idx = np.arange(len(X))
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# MobileNetV2 with proper preprocessing
base_model = keras.applications.MobileNetV2(input_shape=(48, 48, 3), include_top=False, weights='imagenet', pooling='avg')
base_model.trainable = False
inputs = layers.Input((48, 48, 3))
x = base_model(inputs, training=False)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.35)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(len(class_dirs), activation='softmax')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(keras.applications.mobilenet_v2.preprocess_input(X_train), y_train, validation_data=(keras.applications.mobilenet_v2.preprocess_input(X_val), y_val), epochs=6, batch_size=64, verbose=0)
preds = np.argmax(model.predict(keras.applications.mobilenet_v2.preprocess_input(X_test), verbose=0), axis=1)
print('mobilenetv2 accuracy=', round(accuracy_score(y_test, preds), 4))
