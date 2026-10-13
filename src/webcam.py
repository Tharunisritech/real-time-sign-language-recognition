import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "../model/sign_language_model.keras"
HAND_MODEL_PATH = "../model/hand_landmarker.task"


# ============================================================
# LOAD SIGN LANGUAGE MODEL
# ============================================================

print("Loading sign language model...")

model = keras.models.load_model(MODEL_PATH)

print("Sign language model loaded!")


# ============================================================
# CLASS NAMES
# These match the folders used during training
# ============================================================

class_names = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space"
]


# ============================================================
# CREATE MEDIAPIPE HAND LANDMARKER
# ============================================================

print("Loading hand detector...")

base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

hand_detector = vision.HandLandmarker.create_from_options(options)

print("Hand detector loaded!")


# ============================================================
# START WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print()
print("Webcam started.")
print("Show your hand to the camera.")
print("Press Q to quit.")
print("Press S to save the detected hand image.")
print()


# ============================================================
# PREDICTION SMOOTHING
# ============================================================

recent_predictions = deque(maxlen=10)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    # Mirror only for display.
    # Original frame is used for prediction.
    display_frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # --------------------------------------------------------
    # Convert OpenCV BGR -> RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hand
    result = hand_detector.detect(mp_image)

    # --------------------------------------------------------
    # HAND DETECTED
    # --------------------------------------------------------

    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        # Get bounding box coordinates from hand landmarks
        x_values = [landmark.x * width for landmark in landmarks]
        y_values = [landmark.y * height for landmark in landmarks]

        x_min = int(min(x_values))
        x_max = int(max(x_values))
        y_min = int(min(y_values))
        y_max = int(max(y_values))

        # ----------------------------------------------------
        # Add padding around hand
        # ----------------------------------------------------

        hand_width = x_max - x_min
        hand_height = y_max - y_min

        box_size = int(max(hand_width, hand_height) * 1.5)

        # Avoid an extremely small crop
        box_size = max(box_size, 100)

        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        crop_x1 = center_x - box_size // 2
        crop_y1 = center_y - box_size // 2
        crop_x2 = crop_x1 + box_size
        crop_y2 = crop_y1 + box_size

        # Keep crop inside original image
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(width, crop_x2)
        crop_y2 = min(height, crop_y2)

        # ----------------------------------------------------
        # Crop the hand
        # ----------------------------------------------------

        hand_crop = frame[
            crop_y1:crop_y2,
            crop_x1:crop_x2
        ]

        if hand_crop.size > 0:

            # ------------------------------------------------
            # Resize for CNN
            # ------------------------------------------------

            image = cv2.resize(
                hand_crop,
                (64, 64)
            )

            # Convert BGR -> RGB
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            # Convert to float32
            image = image.astype(np.float32)

            # Add batch dimension
            image = np.expand_dims(image, axis=0)

            # ------------------------------------------------
            # CNN prediction
            # ------------------------------------------------

            predictions = model.predict(
                image,
                verbose=0
            )[0]

            predicted_index = int(np.argmax(predictions))

            predicted_class = class_names[predicted_index]

            confidence = float(
                predictions[predicted_index] * 100
            )

            # ------------------------------------------------
            # Smooth predictions
            # ------------------------------------------------

            recent_predictions.append(predicted_class)

            stable_prediction = max(
                set(recent_predictions),
                key=recent_predictions.count
            )

            # ------------------------------------------------
            # Convert crop coordinates for mirrored display
            # ------------------------------------------------

            display_x1 = width - crop_x2
            display_x2 = width - crop_x1

            # ------------------------------------------------
            # Draw hand bounding box
            # ------------------------------------------------

            cv2.rectangle(
                display_frame,
                (display_x1, crop_y1),
                (display_x2, crop_y2),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Display prediction
            # ------------------------------------------------

            cv2.putText(
                display_frame,
                f"Prediction: {stable_prediction}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"Confidence: {confidence:.1f}%",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                "Hand detected",
                (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Show cropped hand preview
            # ------------------------------------------------

            small_hand = cv2.resize(
                hand_crop,
                (160, 160)
            )

            # Corrected: 160 pixels wide, not 170
            display_frame[
                10:170,
                width - 160:width
            ] = small_hand

    else:

        # ----------------------------------------------------
        # NO HAND DETECTED
        # ----------------------------------------------------

        recent_predictions.clear()

        cv2.putText(
            display_frame,
            "No hand detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.putText(
            display_frame,
            "Show your hand clearly",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------------
    # SHOW WEBCAM
    # --------------------------------------------------------

    cv2.imshow(
        "Real-Time Sign Language Recognition",
        display_frame
    )

    key = cv2.waitKey(1) & 0xFF

    # Q = quit
    if key == ord("q"):
        break

    # S = save hand crop
    if key == ord("s") and result.hand_landmarks:

        try:
            cv2.imwrite(
                "../model/webcam_test.jpg",
                hand_crop
            )

            print(
                "Saved hand image: "
                "../model/webcam_test.jpg"
            )

        except Exception as e:
            print(
                "Could not save image:",
                e
            )


# ============================================================
# CLEAN UP
# ============================================================

cap.release()
hand_detector.close()
cv2.destroyAllWindows()

print("Webcam closed.")