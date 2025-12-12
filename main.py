import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Canvas to draw on
canvas = None

# Previous point for drawing
prev_x, prev_y = None, None

# Finger state detection
def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other 4 fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tips[id]].y < hand_landmarks.landmark[tips[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                # Get finger states
                finger_state = fingers_up(handLms)

                # Index finger tip
                h, w, c = frame.shape
                cx = int(handLms.landmark[8].x * w)
                cy = int(handLms.landmark[8].y * h)

                # Drawing mode: Only index finger up
                if finger_state[1] == 1 and sum(finger_state) == 1:
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

                    if prev_x is None:
                        prev_x, prev_y = cx, cy

                    cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 0, 255), 8)
                    prev_x, prev_y = cx, cy

                else:
                    prev_x, prev_y = None, None

        # Merge canvas with webcam frame
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

        frame = cv2.bitwise_and(frame, inv)
        frame = cv2.bitwise_or(frame, canvas)

        cv2.putText(frame, "Press C to clear | Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Finger Draw", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('c'):
            canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()
