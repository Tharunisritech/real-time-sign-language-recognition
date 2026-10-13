import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "../model/sign_language_model.keras"
TEST_PATH = r"../dataset/archive (1)/asl_alphabet_test/asl_alphabet_test"

# -----------------------------
# Load trained model
# -----------------------------
print("Loading trained model...")

model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# These are the same classes used by the training folders.
class_names = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
]

# -----------------------------
# Test images
# -----------------------------
files = sorted(
    [f for f in os.listdir(TEST_PATH) if f.lower().endswith(".jpg")]
)

correct = 0
total = 0

print("\nStarting predictions...\n")

for filename in files:

    image_path = os.path.join(TEST_PATH, filename)

    # Load image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(64, 64)
    )

    # Convert image to array
    image_array = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]

    # Get actual class from filename
    actual_class = filename.split("_")[0]

    if actual_class == "nothing":
        actual_class = "nothing"

    elif actual_class == "space":
        actual_class = "space"

    correct_prediction = predicted_class == actual_class

    if correct_prediction:
        correct += 1
        result = "CORRECT"
    else:
        result = "WRONG"

    total += 1

    print(
        f"{filename:20} "
        f"Actual: {actual_class:8} "
        f"Predicted: {predicted_class:8} "
        f"{result}"
    )

# -----------------------------
# Final accuracy
# -----------------------------
accuracy = (correct / total) * 100

print("\n--------------------------------")
print("TEST RESULTS")
print("--------------------------------")
print(f"Correct predictions: {correct}")
print(f"Total test images:   {total}")
print(f"Test accuracy:       {accuracy:.2f}%")
print("--------------------------------")