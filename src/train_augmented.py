import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = r"..\dataset\archive (1)\asl_alphabet_train\asl_alphabet_train"

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

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

# Get class names
class_names = train_dataset.class_names

print("\nClasses found:")
print(class_names)

print("\nNumber of classes:", len(class_names))

# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================
#
# These changes make training images slightly different so
# the model can become more tolerant of webcam conditions.
#
# We are NOT using horizontal flipping because flipping a
# sign-language image can change the meaning or appearance
# of some signs.
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)

# ============================================================
# BUILD CNN MODEL
# ============================================================

model = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),

        # Data augmentation
        data_augmentation,

        # Normalize pixels from 0-255 to 0-1
        layers.Rescaling(1.0 / 255),

        # CNN block 1
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        # CNN block 2
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        # CNN block 3
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        # CNN block 4
        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        # Classifier
        layers.Flatten(),

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(len(class_names), activation="softmax"),
    ]
)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ============================================================
# SHOW MODEL
# ============================================================

model.summary()

# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    restore_best_weights=True
)

# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# ============================================================
# SAVE MODEL
# ============================================================

model.save("../model/sign_language_model_augmented.keras")

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    "Model saved to:"
    " model/sign_language_model_augmented.keras"
)

# ============================================================
# FINAL RESULTS
# ============================================================

final_train_accuracy = history.history["accuracy"][-1]
final_val_accuracy = history.history["val_accuracy"][-1]

print(
    "\nFinal training accuracy:",
    round(final_train_accuracy * 100, 2),
    "%"
)

print(
    "Final validation accuracy:",
    round(final_val_accuracy * 100, 2),
    "%"
)

print("========================================")