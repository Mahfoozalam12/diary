"""
YouTube Gesture Controller - MediaPipe 0.10.x Compatible Version
"""

import cv2
import pyautogui
import time
import math
import os
import urllib.request
import numpy as np
from collections import deque
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

# MediaPipe 0.10.x imports
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core import base_options
from mediapipe import Image

# Configuration
class Config:
    CAMERA_INDEX = 0
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    DETECTION_CONFIDENCE = 0.7
    TRACKING_CONFIDENCE = 0.7
    MAX_NUM_HANDS = 1
    GESTURE_HOLD_TIME = 0.4
    COOLDOWN_TIME = 1.0
    FONT = cv2.FONT_HERSHEY_SIMPLEX


class GestureType(Enum):
    NONE = "None"
    OPEN_PALM = "Open Palm"
    FIST = "Fist"
    TWO_FINGERS = "Volume Up"
    RING_PINKY = "Volume Down"
    PINKY_ONLY = "Next Video"
    ALL_EXCEPT_PINKY = "Previous Video"


class GestureRecognizer:
    def __init__(self):
        # Setup model path
        self.model_path = self._get_or_download_model()
        self.frame_timestamp = 0
        
        # Create options with model
        options = HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=Config.MAX_NUM_HANDS,
            min_hand_detection_confidence=Config.DETECTION_CONFIDENCE,
            min_hand_presence_confidence=Config.TRACKING_CONFIDENCE,
            min_tracking_confidence=Config.TRACKING_CONFIDENCE
        )
        
        self.landmarker = HandLandmarker.create_from_options(options)
        print(f"HandLandmarker initialized successfully")
    
    def _get_or_download_model(self) -> str:
        """Get model path, downloading if necessary"""
        # Check multiple possible locations
        possible_paths = [
            os.path.join(mp.__path__[0], 'modules', 'hand_landmarker', 'hand_landmarker.task'),
            os.path.join(os.path.dirname(__file__), 'hand_landmarker.task'),
            os.path.join(os.path.expanduser('~'), '.mediapipe', 'hand_landmarker.task'),
            'hand_landmarker.task',  # Current directory
        ]
        
        # Check if model exists in any location
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found model at: {path}")
                return path
        
        # Download model if not found
        model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        download_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        
        print(f"Model not found. Downloading from {model_url}...")
        print(f"This may take a moment...")
        
        try:
            urllib.request.urlretrieve(model_url, download_path)
            print(f"Model downloaded to: {download_path}")
            return download_path
        except Exception as e:
            print(f"Failed to download model: {e}")
            # Try to use the legacy API as fallback
            raise RuntimeError("Could not find or download hand_landmarker model")
    
    def is_finger_extended(self, landmarks, tip_idx: int, pip_idx: int) -> bool:
        """Check if finger is extended using landmark points"""
        wrist = landmarks[0]
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        
        tip_dist = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
        pip_dist = math.sqrt((pip.x - wrist.x)**2 + (pip.y - wrist.y)**2)
        return tip_dist > pip_dist
    
    def recognize_gesture(self, hand_landmarks) -> GestureType:
        """Recognize gesture from hand landmarks"""
        # MediaPipe 0.10.x returns NormalizedLandmarkList
        landmarks = hand_landmarks
        
        # Finger indices: tip, pip
        thumb_ext = self.is_finger_extended(landmarks, 4, 2)
        index_ext = self.is_finger_extended(landmarks, 8, 6)
        middle_ext = self.is_finger_extended(landmarks, 12, 10)
        ring_ext = self.is_finger_extended(landmarks, 16, 14)
        pinky_ext = self.is_finger_extended(landmarks, 20, 18)
        
        extended = [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]
        count = sum(extended)
        
        if count >= 4: return GestureType.OPEN_PALM
        if count == 0 or (count == 1 and thumb_ext): return GestureType.FIST
        if index_ext and middle_ext and not ring_ext and not pinky_ext: return GestureType.TWO_FINGERS
        if not index_ext and not middle_ext and ring_ext and pinky_ext: return GestureType.RING_PINKY
        if not thumb_ext and not index_ext and not middle_ext and not ring_ext and pinky_ext: return GestureType.PINKY_ONLY
        if thumb_ext and index_ext and middle_ext and ring_ext and not pinky_ext: return GestureType.ALL_EXCEPT_PINKY
        
        return GestureType.NONE
    
    def process_frame(self, frame):
        """Process frame using MediaPipe 0.10.x Tasks API"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image - fix for newer versions
        try:
            mp_image = Image(image_format=Image.ImageFormat.SRGB, data=rgb_frame)
        except:
            # Alternative method for some versions
            mp_image = mp.Image(data=rgb_frame, image_format=mp.ImageFormat.SRGB)
        
        # Detect hands
        self.frame_timestamp += 33  # Assume 30fps = 33ms per frame
        results = self.landmarker.detect_for_video(mp_image, self.frame_timestamp)
        
        return results
    
    def draw_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks on frame manually"""
        h, w, _ = frame.shape
        landmarks = hand_landmarks
        
        # Draw connections (simplified)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        # Convert normalized coordinates to pixel coordinates
        points = []
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        # Draw lines
        for start, end in connections:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
        
        return frame


class CommandExecutor:
    def __init__(self):
        self.last_action_time = 0
        self.last_gesture = None
        self.command_map = {
            GestureType.OPEN_PALM: (['space'], "Play"),
            GestureType.FIST: (['space'], "Pause"),
            GestureType.TWO_FINGERS: (['up'], "Volume Up"),
            GestureType.RING_PINKY: (['down'], "Volume Down"),
            GestureType.PINKY_ONLY: (['shift', 'n'], "Next"),
            GestureType.ALL_EXCEPT_PINKY: (['shift', 'p'], "Previous"),
        }
    
    def can_execute(self, gesture: GestureType) -> bool:
        return (time.time() - self.last_action_time) >= Config.COOLDOWN_TIME
    
    def execute(self, gesture: GestureType) -> bool:
        if gesture == GestureType.NONE or gesture not in self.command_map:
            return False
        if not self.can_execute(gesture):
            return False
        
        keys, desc = self.command_map[gesture]
        try:
            if len(keys) == 1:
                pyautogui.press(keys[0])
            else:
                pyautogui.hotkey(*keys)
            self.last_action_time = time.time()
            self.last_gesture = gesture
            print(f"Executed: {desc}")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False


class GestureController:
    def __init__(self):
        print("Initializing YouTube Gesture Controller...")
        print(f"MediaPipe version: {mp.__version__ if hasattr(mp, '__version__') else 'unknown'}")
        self.recognizer = GestureRecognizer()
        self.executor = CommandExecutor()
        self.current_gesture = GestureType.NONE
        self.gesture_start_time = 0
        self.state = "IDLE"
        self.confidence = 0.0
    
    def run(self):
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        
        print("YouTube Gesture Controller Started")
        print("Make sure YouTube is open and focused in your browser")
        print("Press 'Q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            results = self.recognizer.process_frame(frame)
            
            detected = GestureType.NONE
            
            # MediaPipe 0.10.x results format
            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                detected = self.recognizer.recognize_gesture(hand_landmarks)
                frame = self.recognizer.draw_landmarks(frame, hand_landmarks)
            
            # Simple state machine
            current_time = time.time()
            
            if detected != self.current_gesture:
                self.current_gesture = detected
                self.gesture_start_time = current_time
                self.confidence = 0.0
                self.state = "DETECTING" if detected != GestureType.NONE else "IDLE"
            else:
                if detected != GestureType.NONE:
                    hold_time = current_time - self.gesture_start_time
                    self.confidence = min(hold_time / Config.GESTURE_HOLD_TIME, 1.0)
                    
                    if hold_time >= Config.GESTURE_HOLD_TIME and self.state != "EXECUTED":
                        if self.executor.execute(detected):
                            self.state = "EXECUTED"
                    elif self.state == "EXECUTED" and self.executor.can_execute(detected):
                        self.state = "IDLE"
                        self.gesture_start_time = current_time
            
            # Draw UI
            h, w = frame.shape[:2]
            color = (0, 255, 0) if self.state == "EXECUTED" else (0, 255, 255) if self.confidence > 0.5 else (128, 128, 128)
            
            cv2.putText(frame, f"Gesture: {self.current_gesture.value}", (10, 30), Config.FONT, 0.7, color, 2)
            cv2.putText(frame, f"State: {self.state}", (10, 60), Config.FONT, 0.7, color, 2)
            
            # Progress bar
            bar_width = 200
            filled = int(bar_width * self.confidence)
            cv2.rectangle(frame, (10, 80), (10 + bar_width, 100), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 80), (10 + filled, 100), color, -1)
            
            # Instructions
            instructions = [
                "Open Palm=Play | Fist=Pause | 2Fingers=Vol+",
                "Ring+Pinky=Vol- | Pinky=Next | All-Pinky=Prev"
            ]
            for i, text in enumerate(instructions):
                cv2.putText(frame, text, (10, h - 40 + i * 25), Config.FONT, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('YouTube Gesture Controller', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Controller stopped")


if __name__ == "__main__":
    pyautogui.FAILSAFE = True
    controller = GestureController()
    controller.run()