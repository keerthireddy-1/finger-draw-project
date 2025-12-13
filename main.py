import cv2
import numpy as np
import mediapipe as mp
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# ML MODEL FOR DIGIT RECOGNITION
# -----------------------------
digits = load_digits()
model = KNeighborsClassifier(n_neighbors=3)
model.fit(digits.data, digits.target)

def recognize_digit(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    resized = cv2.resize(thresh, (8, 8))
    resized = resized / 16.0
    data = resized.reshape(1, -1)
    prediction = model.predict(data)
    return str(prediction[0])

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# VARIABLES
# -----------------------------
canvas = None
prev_x, prev_y = None, None
draw_color = (0, 0, 255)  # RED
recognized_text = ""

# -----------------------------
# FINGER DETECTIONFUNCTION
# -----------------------------
def fingers_up(hand):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    fingers.append(1 if hand.landmark[tips[0]].x <
                   hand.landmark[tips[0] - 1].x else 0)

    # Other fingers
    for i in range(1, 5):
        fingers.append(1 if hand.landmark[tips[i]].y <
                       hand.landmark[tips[i] - 2].y else 0)
    return fingers

# -----------------------------
# CAMERA
# -----------------------------
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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                finger_state = fingers_up(hand)
                h, w, _ = frame.shape
                cx = int(hand.landmark[8].x * w)
                cy = int(hand.landmark[8].y * h)

                # DRAW MODE: only index finger
                if finger_state[1] == 1 and sum(finger_state) == 1:
                    cv2.circle(frame, (cx, cy), 8, draw_color, cv2.FILLED)

                    if prev_x is not None:
                        cv2.line(canvas, (prev_x, prev_y),
                                 (cx, cy), draw_color, 6)
                    prev_x, prev_y = cx, cy

                # COLOR SELECTION
                elif sum(finger_state) == 2:
                    draw_color = (255, 0, 0)   # BLUE
                elif sum(finger_state) == 3:
                    draw_color = (0, 255, 0)   # GREEN
                elif sum(finger_state) == 4:
                    draw_color = (0, 0, 255)   # RED
                elif sum(finger_state) == 5:
                    canvas = np.zeros_like(frame)  # CLEAR
                    prev_x = prev_y = None
                else:
                    prev_x = prev_y = None

        # MERGE CANVAS
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, inv)
        frame = cv2.bitwise_or(frame, canvas)

        # UI TEXT
        cv2.putText(frame, "Index: Draw | 2-4 Fingers: Color | 5: Clear",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(frame, "Press D: Detect Digit | C: Clear | Q: Quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(frame, f"Recognized: {recognized_text}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,255), 3)

        cv2.imshow("Finger Draw Advanced", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros_like(frame)
            recognized_text = ""
        elif key == ord('d'):
            recognized_text = recognize_digit(canvas)

cap.release()
cv2.destroyAllWindows()
