import cv2
import mediapipe as mp
import math

def start_camera(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return None
    return cap

def detect_left_hand_gesture(img):
    """
    Detect hand and count fingers using MediaPipe (simple version).
    Returns: image_with_annotations, gesture_label, confidence
    """
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    #convert image for mediapipe
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #the hand dectecction setup
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    #simple process frame 
    results = hands.process(rgb)

    #is there a hand
    if not results.multi_hand_landmarks:
        hands.close()
        return img, None

    #important hand positions
    hand_landmarks = results.multi_hand_landmarks[0]

    # Label the important parts of the hand on top of the image
    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # count fingers
    #tip points for thumb + 4 fingers
    tip_ids = [4, 8, 12, 16, 20]
    fingers_up = 0

    # Get all landmark positions
    h, w, _ = img.shape
    landmarks = []
    for i, lm in enumerate(hand_landmarks.landmark):
        x, y = int(lm.x * w), int(lm.y * h)
        landmarks.append((x, y))

    #thumb --> compare x positions --> essentially left vs right 
    if landmarks[tip_ids[0]][0] < landmarks[tip_ids[0] - 1][0]:
        fingers_up += 1

    #other fingers --> essentially ensure that the tip of the finger is above lower joint of the finger.
    for tip in tip_ids[1:]:
        if landmarks[tip][1] < landmarks[tip - 2][1]:
            fingers_up += 1

    #fingers up = gesture to put into the image when fully completed
    if fingers_up >= 4:
        gesture = "BREAK"
    else:
        gesture = "GO"

    #write text
    cv2.putText(img, f"Hand: {gesture}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    #return image, what the gesture is, the confidence that I got the gesture right
    hands.close()
    return img, gesture


    
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

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            h, w, _ = frame.shape  # Frame dimensions

            # Draw landmarks
           
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                # Get right shoulder (chest anchor)
                right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                cx, cy = int(right_shoulder.x * w), int(right_shoulder.y * h)
                

                # Get right elbow (arm direction)
                right_elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
                ex, ey = int(right_elbow.x * w), int(right_elbow.y * h)
                angle_rad = math.atan2(cx - ex, ey - cy)
                # Kaegan this is the angle : 180degrees = max right, 90 = middle, 0 = max left
                angle_deg = math.degrees(angle_rad)

                

                cv2.putText(frame, f"Angle: {angle_deg:.1f} deg", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # This is the break and Go part of the function: 
            frame, hand_action = detect_left_hand_gesture(frame)
            if hand_action:
                print(hand_action)  # prints BREAK or GO in console

            # Show the frame
            cv2.imshow("Arm Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break    

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    check_angle()
