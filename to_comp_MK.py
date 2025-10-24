import cv2
import mediapipe as mp

def start_camera(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return None
    return cap

def check_angle():
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = start_camera()
    if cap is None:
        return

    with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip frame for mirror effect (optional)
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame
            results = holistic.process(rgb)

            # Draw pose landmarks (arms included)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Draw hands
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Show webcam
            cv2.imshow("Arm Detection", frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_angle()
