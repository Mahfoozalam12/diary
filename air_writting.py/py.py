import cv2
import mediapipe as mp
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

# Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

canvas = None

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    
    if canvas is None:
        canvas = np.zeros_like(frame)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    index_finger_tip = None
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            
            h, w, _ = frame.shape
            tip = handLms.landmark[8]  # index finger tip
            index_finger_tip = (int(tip.x * w), int(tip.y * h))
    
    # Draw when finger detected
    if index_finger_tip:
        cv2.circle(canvas, index_finger_tip, 5, (0, 255, 0), -1)
    
    # Merge canvas and frame
    output = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    cv2.imshow("AIR WRITING", output)
    
    key = cv2.waitKey(1)
    
    # Press 'c' to clear
    if key & 0xFF == ord('c'):
        canvas = np.zeros_like(frame)
    
    # Press ESC to exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()