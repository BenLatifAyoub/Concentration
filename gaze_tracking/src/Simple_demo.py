import cv2
import dlib
from utils import GazeTracking

def main():
    # Initialize dlib's face detector and the gaze tracker
    detector = dlib.get_frontal_face_detector()
    gaze_trackers = []

    # Open webcam
    webcam = cv2.VideoCapture(0)

    skip_frames = 0
    max_skip_frames = 5  # Adjust this value based on performance requirements

    last_annotated_frame = None  # Variable to store the last annotated frame

    while True:
        # Read frame from webcam
        _, frame = webcam.read()

        # Implement frame skipping
        if skip_frames < max_skip_frames:
            skip_frames += 1
            if last_annotated_frame is not None:
                cv2.imshow("Gaze Tracking Demo", last_annotated_frame)
            # Break loop if 'Esc' key is pressed
            if cv2.waitKey(1) == 27:
                break
            continue  # Skip processing this frame

        # Reset skip_frames counter
        skip_frames = 0

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = detector(gray)

        # Update or create gaze trackers for each detected face
        if len(faces) != len(gaze_trackers):
            gaze_trackers = [GazeTracking() for _ in range(len(faces))]

        # Process each detected face
        for i, face in enumerate(faces):
            # Get face bounding box coordinates
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())

            # Crop the face region from the frame
            face_frame = frame[y:y+h, x:x+w]

            # Refresh gaze tracking with the face region
            gaze_trackers[i].refresh(face_frame)

            # Get annotated frame from gaze tracking
            annotated_face_frame = gaze_trackers[i].annotated_frame()

            # Determine gaze direction and display text
            print("gazeee",gaze_trackers[i].horizontal_ratio())
            text = ""
            if gaze_trackers[i].is_blinking():
                text = "Blinking"
            elif gaze_trackers[i].is_right():
                text = "Right"
            elif gaze_trackers[i].is_left():
                text = "Left"
            elif gaze_trackers[i].is_center():
                text = "Center"
            else:
                text = "Gaze not detected"

            # Display gaze direction text on the annotated face frame
            display_text(annotated_face_frame, text)

            # Display left and right pupil coordinates
            left_pupil = gaze_trackers[i].pupil_left_coords()
            right_pupil = gaze_trackers[i].pupil_right_coords()
            cv2.putText(annotated_face_frame, f"Left pupil:  {left_pupil}", (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(annotated_face_frame, f"Right pupil: {right_pupil}", (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

            # Overlay the annotated face frame back onto the original frame
            frame[y:y+h, x:x+w] = annotated_face_frame

        # Store the last annotated frame
        last_annotated_frame = frame.copy()

        # Display the main frame with annotated faces
        cv2.imshow("Gaze Tracking Demo", frame)

        # Break loop if 'Esc' key is pressed
        if cv2.waitKey(1) == 27:
            break

    # Release the webcam and close OpenCV windows
    webcam.release()
    cv2.destroyAllWindows()

def display_text(frame, text):
    """ Display text on the frame. """
    cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)


if __name__ == "__main__":
    main()
