import cv2
import dlib
import numpy as np
from utils import GazeTracking
from statistics import mode
from keras.models import load_model
from utils.datasets import get_labels
from utils.inference import detect_faces, apply_offsets
from utils.preprocessor import preprocess_input
import time

# Initialize dlib's face detector and gaze tracker
detector = dlib.get_frontal_face_detector()
gaze_trackers = []  # Initialize an empty list for gaze trackers

# Load emotion detection model and labels
detection_model_path = './utils/trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = './utils/trained_models/emotion_models/fer2013_mini_XCEPTION.110-0.65.hdf5'
emotion_labels = get_labels('fer2013')
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]

# Open webcam
webcam = cv2.VideoCapture(0)

gw_values = []
facial_emotion_CIs = []

# Define the interval for frame processing (1 second in this example)
processing_interval = 0.01  # in seconds

# Time when the last frame was processed
last_process_time = time.time()



while True:
    # Read frame from webcam
    ret, frame = webcam.read()
    if not ret:
        break

    if time.time() - last_process_time >= processing_interval:
        last_process_time = time.time()  # Update last processing time
    # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using dlib for gaze tracking
        faces_dlib = detector(gray)

    # Ensure the number of gaze trackers matches the number of detected faces
# Ensure the number of gaze trackers matches the number of detected faces
        if len(faces_dlib) != len(gaze_trackers):
            gaze_trackers = [GazeTracking() for _ in range(len(faces_dlib))]
            gw_values = [0.0] * len(faces_dlib)  # Initialize gw_values for each detected face
            facial_emotion_CIs = [0.0] * len(faces_dlib)  # Initialize facial_emotion_CIs for each detected face


    # Process faces for gaze tracking using dlib
        for i, face in enumerate(faces_dlib):
            # Get face bounding box coordinates
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())

        # Crop the face region from the frame
            face_frame = frame[y:y+h, x:x+w]

        # Refresh gaze tracking with the face region
            gaze_trackers[i].refresh(face_frame)

        # Get gaze direction and display text
            gaze_text = ""
        # Determine if the student is looking at the center
            if gaze_trackers[i].is_center():
            # Increase GW value
                gw_values[i] += 0.01
                gaze_text = "centre"
            elif gaze_trackers[i].is_left():
                gw_values[i] -= 0.01
                gaze_text = "left"
            elif gaze_trackers[i].is_right():
                gw_values[i] -= 0.01
                gaze_text = "right"
            elif gaze_trackers[i].is_blinking():
                gw_values[i] -= 0.01
                gaze_text = "blinking"       
            else:
                gw_values[i] = 0.5
                gaze_text = "not detected"

            gw_values[i] = np.clip(gw_values[i], 0.0, 1.0)

        # Display gaze direction and pupil coordinates on the frame
            cv2.putText(frame, gaze_text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, f"GW: {gw_values[i]:.2f}", (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


            try:
            
                        # Convert face frame to grayscale
                face_frame_gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)

            # Resize and preprocess face for emotion detection
                face_roi = cv2.resize(face_frame_gray, emotion_target_size)
                face_roi = np.expand_dims(face_roi, axis=-1)  # Add a new axis for grayscale
                face_roi = preprocess_input(face_roi)  # Preprocess the input data

            # Perform emotion prediction
                emotion_prediction = emotion_classifier.predict(np.array([face_roi]))
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg]

             # Assign color based on emotion
                # emotion_probability= somme ( emotion_prediction[i]*poid d'emotion )
                # print(emotion_prediction)
                color = (0, 255, 0)  # Default color for neutral emotion
                if emotion_text == 'angry':
                    color = (0, 0, 255)  # Red for angry
                     
                    facial_emotion_CIs[i] = emotion_probability*0.25*gw_values[i]
                elif emotion_text == 'sad':
                    color = (255, 0, 0)  # Blue for sad
                    facial_emotion_CIs[i] = emotion_probability*0.4*gw_values[i]
                elif emotion_text == 'happy':
                    color = (255, 255, 0)  # Yellow for happy
                    facial_emotion_CIs[i] = emotion_probability*0.6*gw_values[i]
                elif emotion_text == 'surprise':
                    color = (0, 255, 255)  # Cyan for surprise
                    facial_emotion_CIs[i] = emotion_probability*0.5*gw_values[i]
                elif emotion_text == 'fear':
                    color = (255, 255, 255)  # White for fear
                    facial_emotion_CIs[i] = emotion_probability*0.3*gw_values[i]
                else :
                    color = (0, 255, 0)  # Default color for neutral emotion
                    facial_emotion_CIs[i] = emotion_probability*0.9*gw_values[i]    

            # Draw bounding box and emotion label on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


             # Display facial emotion concentration index (CI) on the frame
                cv2.putText(frame, f"Facial Emotion CI: {facial_emotion_CIs[i]:.2f}", (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            except Exception as e:
                print(f"Error processing face: {e}")

    # Display annotated frame
        cv2.imshow('Gaze and Emotion Detection', frame)

    # Check for 'Esc' key press to exit
        if cv2.waitKey(1) == 27:
            break

# Release webcam and close OpenCV windows
webcam.release()
cv2.destroyAllWindows()
