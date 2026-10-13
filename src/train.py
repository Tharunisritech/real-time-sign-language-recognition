import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------
# Settings
# -----------------------------
DATASET_PATH = r"..\dataset\archive (1)\asl_alphabet_train\asl_alphabet_train"

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

# -----------------------------
# Load training data
# -----------------------------
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Save class names
class_names = train_dataset.class_names

print("\nClasses found:")
print(class_names)

print("\nNumber of classes:", len(class_names))

# -----------------------------
# Improve performance
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# Build CNN model
# -----------------------------
model = keras.Sequential([
    layers.Rescaling(1.0 / 255, input_shape=(64, 64, 3)),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation="softmax")
])

# -----------------------------
# Compile model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Show model information
# -----------------------------
model.summary()

# -----------------------------
# Train model
# -----------------------------
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# -----------------------------
# Save model
# -----------------------------
model.save("../model/sign_language_model.keras")

print("\nModel saved successfully!")
print("Location: model/sign_language_model.keras")

# -----------------------------
# Final results
# -----------------------------
final_train_accuracy = history.history["accuracy"][-1]
final_val_accuracy = history.history["val_accuracy"][-1]

print("\nFinal training accuracy:",
      round(final_train_accuracy * 100, 2), "%")

print("Final validation accuracy:",
      round(final_val_accuracy * 100, 2), "%")