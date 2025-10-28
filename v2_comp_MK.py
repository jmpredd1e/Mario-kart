import cv2
import mediapipe as mp
import math

def start_camera(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return None
    return cap


def get_hand_info(hands_results, width, height):
    """Find the center of each hand and check if it is open or closed."""
    hand_data = []
    
    if not hands_results.multi_hand_landmarks:
        return hand_data

    for hand_landmarks in hands_results.multi_hand_landmarks:
        landmarks = []
        for lm in hand_landmarks.landmark:
            x = int(lm.x * width)
            y = int(lm.y * height)
            landmarks.append((x, y))

        # find center of hand
        total_x = 0
        total_y = 0
        for point in landmarks:
            total_x += point[0]
            total_y += point[1]

        cx = total_x // len(landmarks)
        cy = total_y // len(landmarks)

        # count fingers
        tip_ids = [4, 8, 12, 16, 20]
        fingers_up = 0

        # thumb check
        if landmarks[tip_ids[0]][0] < landmarks[tip_ids[0] - 1][0]:
            fingers_up += 1

        # other fingers
        for i in range(1, 5):
            tip = tip_ids[i]
            if landmarks[tip][1] < landmarks[tip - 2][1]:
                fingers_up += 1

        # open if 4 or more fingers up
        if fingers_up >= 4:
            is_open = True
        else:
            is_open = False

        hand_data.append((cx, cy, is_open))

    return hand_data


def compute_wheel_angle(hand1, hand2):
    """Find steering wheel angle based on hand positions."""
    x1, y1, _ = hand1
    x2, y2, _ = hand2

    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dy, dx))
    return angle


def check_angle():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = start_camera()
    if cap is None:
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=4,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # draw jazz handz 
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            hand_data = get_hand_info(results, width, height)

            #make lists for each side
            left_side = []
            right_side = []
            for hand in hand_data:
                x, y, open_state = hand
                if x < width // 2:
                    left_side.append(hand)
                else:
                    right_side.append(hand)

            # --- PLAYER 1 --> left side ---
            if len(left_side) >= 2:
                hand1 = left_side[0]
                hand2 = left_side[1]
                angle1 = compute_wheel_angle(hand1, hand2)
                if hand1[2] and hand2[2]:
                    state1 = "STOP"
                else:
                    state1 = "GO"
                cv2.putText(frame, "P1 Angle: " + str(round(angle1, 1)), (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "P1 State: " + state1, (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print("Player 1 -> Angle:", round(angle1, 1), "State:", state1)
            else:
                cv2.putText(frame, "P1: Not detected", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- PLAYER 2 (right side) ---
            if len(right_side) >= 2:
                hand1 = right_side[0]
                hand2 = right_side[1]
                angle2 = compute_wheel_angle(hand1, hand2)
                if hand1[2] and hand2[2]:
                    state2 = "STOP"
                else:
                    state2 = "GO"
                cv2.putText(frame, "P2 Angle: " + str(round(angle2, 1)), (width // 2 + 30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, "P2 State: " + state2, (width // 2 + 30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                print("Player 2 -> Angle:", round(angle2, 1), "State:", state2)
            else:
                cv2.putText(frame, "P2: Not detected", (width // 2 + 30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # show the frame
            cv2.imshow("Multiplayer Steering", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    check_angle()
