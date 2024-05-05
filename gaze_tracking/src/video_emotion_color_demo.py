from statistics import mode
import cv2
from keras.models import load_model
import numpy as np
import time
from utils.datasets import get_labels
from utils.inference import detect_faces, draw_text, draw_bounding_box, apply_offsets
from utils.preprocessor import preprocess_input

# Parameters for loading data and images
detection_model_path = './utils/trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = './utils/trained_models/emotion_models/fer2013_mini_XCEPTION.110-0.65.hdf5'
emotion_labels = get_labels('fer2013')

# Hyper-parameters for bounding boxes shape
frame_window = 200
emotion_offsets = (20, 40)

# Loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# Getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Starting lists for calculating modes
emotion_window = []

# Starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
window_width = 1280
window_height = 720
cv2.resizeWindow('window_frame', window_width, window_height)  # Resize window

# Initialize variables for emotion prediction timing
last_prediction_time = time.time()

# Initialize a dictionary to store emotion windows for each face
emotion_windows = {}
cv2.namedWindow('window_frame')

while True:
    # Read frame from the video capture
    ret, bgr_image = video_capture.read()
    if not ret:
        continue

    # Convert to grayscale for face detection
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detect_faces(face_detection, gray_image)

    # Create a copy of the RGB image for drawing
    rgb_image = np.copy(bgr_image)

    # Iterate over detected faces
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        # Create a unique key for each face based on its coordinates
        face_key = tuple(face_coordinates)

        # Check if this face has an associated emotion window
        if face_key not in emotion_windows:
            emotion_windows[face_key] = []

        # Predict emotion (every 0.5 seconds)
        current_time = time.time()
        if current_time - last_prediction_time >= 0:
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]

            emotion_windows[face_key].append(emotion_text)

            if len(emotion_windows[face_key]) > frame_window:
                emotion_windows[face_key].pop(0)

            last_prediction_time = current_time  # Update last prediction time

        # Calculate mode for this specific face's emotion window
        try:
            emotion_mode = mode(emotion_windows[face_key])
        except:
            continue

        # Assign color based on emotion
        color = (0, 255, 0)  # Default color for neutral emotion
        if emotion_mode == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_mode == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_mode == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_mode == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        elif emotion_mode == 'disgust':
            color = emotion_probability * np.asarray((0, 255, 255))
        elif emotion_mode == 'fear':
            color = emotion_probability * np.asarray((0, 255, 255))

        # Draw bounding box and text directly on the RGB image
        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(rgb_image, emotion_mode, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the processed frame
    cv2.imshow('window_frame', rgb_image)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
