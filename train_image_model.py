import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

img_size = 224
batch = 32

train_path = "dataset/train"
test_path = "dataset/test"

# =========================================================
# DATA AUGMENTATION
# =========================================================

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=8,
    zoom_range=0.15,
    brightness_range=[0.9,1.1],
    width_shift_range=0.05,
    height_shift_range=0.05
)

test_gen = ImageDataGenerator(rescale=1./255)

# LOAD DATA

train_data = train_gen.flow_from_directory(
    train_path,
    target_size=(img_size,img_size),
    batch_size=batch,
    class_mode='binary',
    shuffle=True
)

test_data = test_gen.flow_from_directory(
    test_path,
    target_size=(img_size,img_size),
    batch_size=batch,
    class_mode='binary',
    shuffle=False
)

# CNN MODEL

model = models.Sequential([

    layers.Conv2D(32,(3,3),activation='relu',
                  input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128,activation='relu'),

    layers.Dropout(0.5),

    layers.Dense(1,activation='sigmoid')
])

# COMPILE MODEL

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================================================
# CLASS WEIGHTS
# =========================================================

class_weight = {
    0: 3.0,
    1: 1.0
}

# TRAIN MODEL

print("Training started...")

history = model.fit(
    train_data,
    epochs=15,
    validation_data=test_data,
    class_weight=class_weight
)

# SAVE MODEL

model.save("xray_model.keras")
print("Training done & BEST model saved")

# =========================================================
# TRAINING & TESTING ACCURACY / LOSS
# =========================================================

train_acc = history.history['accuracy'][-1]
train_loss = history.history['loss'][-1]

test_acc = history.history['val_accuracy'][-1]
test_loss = history.history['val_loss'][-1]

print("\n================ RESULTS ================")

print(f"Training Accuracy : {train_acc:.4f}")
print(f"Training Loss     : {train_loss:.4f}")

print(f"\nTesting Accuracy  : {test_acc:.4f}")
print(f"Testing Loss      : {test_loss:.4f}")

# =========================================================
# PREDICTIONS
# =========================================================

pred_probs = model.predict(test_data)

predictions = (pred_probs > 0.5).astype("int32").flatten()

true_labels = test_data.classes

# =========================================================
# PERFORMANCE METRICS
# =========================================================

accuracy = accuracy_score(true_labels, predictions)

precision = precision_score(true_labels, predictions)

recall = recall_score(true_labels, predictions)

f1 = f1_score(true_labels, predictions)

print("\n================ PERFORMANCE METRICS ================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(true_labels, predictions)

print("\n================ CONFUSION MATRIX ================")

print(cm)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n================ CLASSIFICATION REPORT ================")

print(classification_report(
    true_labels,
    predictions,
    target_names=list(test_data.class_indices.keys())
))

# =========================================================
# ACCURACY GRAPH
# =========================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(history.history['accuracy'],
         label='Training Accuracy')

plt.plot(history.history['val_accuracy'],
         label='Validation Accuracy')

plt.title('Accuracy Graph')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

# =========================================================
# LOSS GRAPH
# =========================================================

plt.subplot(1,2,2)

plt.plot(history.history['loss'],
         label='Training Loss')

plt.plot(history.history['val_loss'],
         label='Validation Loss')

plt.title('Loss Graph')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()