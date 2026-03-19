import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

# Download model if not exists
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", model_path)

# Hand connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# Global for results
hand_results = None

def result_callback(result, output_image, timestamp_ms):
    global hand_results
    hand_results = result

# Initialize landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=result_callback
)

landmarker = HandLandmarker.create_from_options(options)

# Initialize camera
cap = cv2.VideoCapture(0)

canvas = None
timestamp = 0

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    
    if canvas is None:
        canvas = np.zeros_like(frame)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker.detect_async(mp_image, timestamp)
    timestamp += 1
    
    index_finger_tip = None
    
    if hand_results and hand_results.hand_landmarks:
        for handLms in hand_results.hand_landmarks:
            h, w, _ = frame.shape
            # Draw landmarks
            for i, lm in enumerate(handLms):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)
            # Draw connections
            for connection in HAND_CONNECTIONS:
                start = handLms[connection[0]]
                end = handLms[connection[1]]
                cv2.line(frame, (int(start.x*w), int(start.y*h)), (int(end.x*w), int(end.y*h)), (255,0,255), 2)
            
            # Index finger tip
            tip = handLms[8]
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

landmarker.close()
cap.release()
cv2.destroyAllWindows()