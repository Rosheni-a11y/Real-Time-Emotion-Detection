# --- Imports ---
import cv2                              # OpenCV: webcam access, face detection, drawing
import numpy as np                      # numerical arrays (for reshaping + averaging predictions)
import time                            # for measuring frames per second (FPS)
from collections import deque          # a fixed-length buffer for prediction smoothing
from tensorflow.keras.models import load_model         # to load the trained model

# --- Load the trained model ---
# load_model rebuilds the exact network you trained (layers + learned weights)
# from the file you saved at the end of train_model.py.
model = load_model('emotion_model.h5')

# --- Emotion labels ---
# These MUST be in the same order the model learned them. flow_from_directory
# assigned class indices alphabetically, so index 0 = angry, 1 = disgust, etc.
# The model's output is a list of 7 probabilities lined up with this list.
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Must match training: the model was trained on 48x48 grayscale images.
IMG_SIZE = 48

# How many pixels of extra context to include around each detected face.
PADDING = 15

# --- Prediction smoothing ---
# A single prediction can jump around frame-to-frame, making the label flicker.
# Instead of trusting one frame, we keep the last few prediction vectors and
# average them. Averaging the 7 probabilities (then taking the argmax) gives a
# steadier label than averaging the final labels would.
# NOTE: this is one shared buffer, so it assumes ONE main face in view. With
# several faces their predictions would mix together.
SMOOTHING_WINDOW = 8
pred_history = deque(maxlen=SMOOTHING_WINDOW)

# --- Face detector ---
# A Haar Cascade is a lightweight, pre-trained face detector that ships with
# OpenCV. cv2.data.haarcascades is the folder where OpenCV keeps these files,
# so we don't have to hard-code a path to it.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# --- Open the webcam ---
# 0 = the default camera. VideoCapture gives us frames one at a time.
cap = cv2.VideoCapture(0)

# Fail early with a clear message if the camera can't be opened.
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Is another app using it?")

print("Webcam started. Press 'q' to quit.")

# Timestamp of the previous frame, used to calculate FPS each loop.
prev_time = time.time()

# --- Main loop ---
# Each pass through this loop grabs one frame, finds faces, predicts emotions,
# and draws the results. It repeats fast enough to look like live video.
while True:
    # Read one frame. `ok` is False if the camera dropped the frame.
    ok, frame = cap.read()
    if not ok:
        break

    # Mirror the frame left-to-right. Webcams normally show a non-mirrored image,
    # which feels backwards (you move left, the image moves right). Flipping makes
    # it behave like a mirror, which is what people expect.
    frame = cv2.flip(frame, 1)

    # The full frame size, used below to keep the padded crop inside the image.
    frame_h, frame_w = frame.shape[:2]

    # The model works on grayscale, and the face detector is faster on grayscale,
    # so convert the color (BGR) frame to gray once and reuse it.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detectMultiScale returns a box (x, y, width, height) for each face it finds.
    #   scaleFactor=1.1  -> shrink the image 10% per pass to catch faces of different sizes
    #   minNeighbors=5   -> how many overlapping detections are needed to keep a box
    #                       (higher = fewer false positives, but may miss real faces)
    #   minSize=(30,30)  -> smaller minimum so faces further from the camera still register
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Handle each detected face separately.
    for (x, y, w, h) in faces:
        # Add padding around the box for more context, but clamp to the frame
        # edges with max()/min() so we never index outside the image.
        x1 = max(0, x - PADDING)
        y1 = max(0, y - PADDING)
        x2 = min(frame_w, x + w + PADDING)
        y2 = min(frame_h, y + h + PADDING)

        # Crop the padded face out of the grayscale image.
        face = gray[y1:y2, x1:x2]

        # --- Preprocess exactly like training ---
        # 1) resize to 48x48 so it matches the model's input size
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        # 2) rescale pixels from 0-255 to 0-1 (train_model.py used rescale=1/255)
        face = face.astype('float32') / 255.0
        # 3) reshape to the 4D shape the model expects: (batch, height, width, channels)
        #    1 image, 48x48, 1 grayscale channel.
        face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        # --- Predict ---
        # predict returns one row of 7 probabilities (they sum to 1).
        # verbose=0 keeps it from printing a progress bar every frame.
        preds = model.predict(face, verbose=0)[0]

        # Add this frame's probabilities to the history, then average the whole
        # buffer. Early on the buffer holds fewer than 8 entries; that's fine,
        # np.mean just averages whatever is there.
        pred_history.append(preds)
        avg_preds = np.mean(pred_history, axis=0)

        index = int(np.argmax(avg_preds))   # position of the highest averaged probability
        emotion = EMOTIONS[index]           # look up the matching label
        confidence = avg_preds[index] * 100 # turn that probability into a percentage

        # --- Draw the face box ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- Draw the label with a filled background for readability ---
        label = f"{emotion} {confidence:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # Measure how big the text will be so the background box fits it exactly.
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Filled green rectangle sitting just above the face box (the -1 thickness
        # tells OpenCV to fill it instead of drawing only an outline).
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 8),
            (x1 + text_w + 8, y1),
            (0, 255, 0), -1
        )

        # Black text on top of the green background reads clearly in any lighting.
        cv2.putText(
            frame, label, (x1 + 4, y1 - baseline - 4),
            font, font_scale, (0, 0, 0), thickness
        )

    # --- FPS counter ---
    # FPS = how many frames we process per second = 1 / (time for this frame).
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
    prev_time = curr_time

    # Draw it in the top-left corner.
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
    )

    # Show the annotated frame in a window.
    cv2.imshow('Emotion Detector', frame)

    # waitKey(1) waits 1ms for a keypress; this also lets the window refresh.
    # & 0xFF is a portability mask; ord('q') is the key code for 'q'.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
# Release the camera and close the window so they're free for other programs.
cap.release()
cv2.destroyAllWindows()
