import cv2
import mediapipe as mp
import math
import time  # used for timing checks

# ---------------------------------------------------
# Function to start the camera 
# ---------------------------------------------------
def start_camera(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return None
    return cap


# ---------------------------------------------------
# Function to get info about each detected hand
# ---------------------------------------------------
def get_hand_info(hands_results, width, height):
    """
    Finds center position (x, y) of each hand and checks if it is open or closed.
    Returns a list of tuples (center_x, center_y, is_open)
    """
    hand_data = []

    #if no hands visisble than return empty list
    if not hands_results.multi_hand_landmarks:
        return hand_data

    #go through all hands 
    for hand_landmarks in hands_results.multi_hand_landmarks:
        #mark landmarks 
        landmarks = []
        for lm in hand_landmarks.landmark:
            x = int(lm.x * width)
            y = int(lm.y * height)
            landmarks.append((x, y))

        #find center of hand --> neet tip this is just the average of all the landmarks --> that's right the more you know Mr.Rose the more you know 
        total_x = 0
        total_y = 0
        for point in landmarks:
            total_x += point[0]
            total_y += point[1]

        cx = total_x // len(landmarks)
        cy = total_y // len(landmarks)

        #how many fingers are up
        tip_ids = [4, 8, 12, 16, 20]
        fingers_up = 0

        #check thumbs
        if landmarks[tip_ids[0]][0] < landmarks[tip_ids[0] - 1][0]:
            fingers_up += 1

        #check other fingers --> are my hands open or closed 
        for i in range(1, 5):
            tip = tip_ids[i]
            if landmarks[tip][1] < landmarks[tip - 2][1]:
                fingers_up += 1

        #open if fingers up 
        if fingers_up >= 4:
            is_open = True
        else:
            is_open = False

        #append hand data
        hand_data.append((cx, cy, is_open))

    return hand_data


# ---------------------------------------------------
# Function to calculate the steering wheel angle
# ---------------------------------------------------
def compute_wheel_angle(hand1, hand2):
    """
    Take two hand positions and return the steering angle between them.
    """
    x1, y1, _ = hand1
    x2, y2, _ = hand2

    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dy, dx))
    return angle


# ---------------------------------------------------
# Main function that checks both players' angles and states
# ---------------------------------------------------
def check_angle():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = start_camera()
    if cap is None:
        return

    #most recent valid readings for each player
    last_angle_p1 = 0
    last_state_p1 = "STOP"
    last_angle_p2 = 0
    last_state_p2 = "STOP"

    #track the time now you don't need a watch 
    last_update_time = time.time()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=4,  #borth players allowed or something like that 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            #conversion
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            #draw deteced handz
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            #get hand info 
            hand_data = get_hand_info(results, width, height)

            #left and right halves of the screen --> seperarate 
            left_side = []
            right_side = []
            for hand in hand_data:
                x, y, is_open = hand
                if x < width // 2:
                    left_side.append(hand)   #player 1
                else:
                    right_side.append(hand)  #player 2

            #update every 10 milliseconds
            current_time = time.time()
            if (current_time - last_update_time) >= 0.01:
                # --- PLAYER 1 --> left side --- 
                if len(left_side) >= 2:
                    hand1 = left_side[0]
                    hand2 = left_side[1]
                    angle1 = compute_wheel_angle(hand1, hand2)

                    # If both hands open → STOP, else GO
                    if hand1[2] and hand2[2]:
                        state1 = "STOP"
                    else:
                        state1 = "GO"

                    # Save the most recent valid values
                    last_angle_p1 = angle1
                    last_state_p1 = state1
                #keep last known value

                # --- PLAYER 2 --> right side --- 
                if len(right_side) >= 2:
                    hand1 = right_side[0]
                    hand2 = right_side[1]
                    angle2 = compute_wheel_angle(hand1, hand2)

                    if hand1[2] and hand2[2]:
                        state2 = "STOP"
                    else:
                        state2 = "GO"

                    last_angle_p2 = angle2
                    last_state_p2 = state2
                #keep last known value

                #update time 
                last_update_time = current_time

            # ------------------------------
            # DISPLAY VALUES ON SCREEN
            # ------------------------------
            cv2.putText(frame, "P1 Angle: " + str(round(last_angle_p1, 1)), (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "P1 State: " + last_state_p1, (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, "P2 Angle: " + str(round(last_angle_p2, 1)), (width // 2 + 30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "P2 State: " + last_state_p2, (width // 2 + 30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            #shows the frame thingy
            cv2.imshow("Multiplayer Steering (Stable)", frame)

            #printing values console --> KAEGAN FOR YOUR OWN SANITY COMMIT THIS OUT IT WILL KILL YOUR COMPUTER
            print("Player 1 -> Angle:", round(last_angle_p1, 1), "State:", last_state_p1,
                  " | Player 2 -> Angle:", round(last_angle_p2, 1), "State:", last_state_p2)

            #q = quit hooray
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------
# Run the program
# ---------------------------------------------------
if __name__ == "__main__":
    check_angle()
