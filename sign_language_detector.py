import cv2
import mediapipe as mp
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import gc
import psutil
import os

# Set process priority
try:
    p = psutil.Process(os.getpid())
    p.nice(-10)  # Set high priority
except:
    pass

def get_finger_state(hand_landmarks):
    """Determine if each finger is extended or not - optimized version"""
    try:
        finger_tips = [8, 12, 16, 20]
        finger_bases = [5, 9, 13, 17]
        
        # Simplified orientation check using numpy 1.17.3 compatible operations
        wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
        middle_base = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y])
        is_vertical = abs(middle_base[1] - wrist[1]) > abs(middle_base[0] - wrist[0])
        
        # Check thumb with simplified logic
        thumb_tip = hand_landmarks.landmark[4]
        thumb_base = hand_landmarks.landmark[2]
        thumb_extended = (thumb_tip.y < thumb_base.y) if is_vertical else (thumb_tip.x > thumb_base.x)
        
        # Check other fingers with simplified logic
        fingers_extended = []
        for tip, base in zip(finger_tips, finger_bases):
            tip_point = hand_landmarks.landmark[tip]
            base_point = hand_landmarks.landmark[base]
            is_extended = (tip_point.y < base_point.y) if is_vertical else (tip_point.x > thumb_base.x)
            fingers_extended.append(is_extended)
        
        return thumb_extended, fingers_extended
    except Exception as e:
        print(f"Error in get_finger_state: {str(e)}")
        return False, [False, False, False, False]

def detect_letter(hand_landmarks):
    """Simplified letter detection logic"""
    try:
        if not hand_landmarks:
            return None

        thumb_extended, fingers_extended = get_finger_state(hand_landmarks)
        index, middle, ring, pinky = fingers_extended
        
        # Simplified letter detection
        # A: Thumb up, other fingers down
        if thumb_extended and not any(fingers_extended):
            return 'A'

        # B: All fingers up
        if thumb_extended and all(fingers_extended):
            return 'B'

        # C: All fingers up
        if thumb_extended and all(fingers_extended):
            return 'C'

        # D: Index finger up
        if thumb_extended and index and not middle and not ring and not pinky:
            return 'D'

        return None
    except Exception as e:
        print(f"Error in detect_letter: {str(e)}")
        return None

def main():
    # Initialize camera
    camera = PiCamera()
    camera.resolution = (320, 240)  # Lower resolution for better performance
    camera.framerate = 15  # Reduced framerate
    camera.exposure_mode = 'night'  # Better for indoor use
    camera.awb_mode = 'auto'
    raw_capture = PiRGBArray(camera, size=(320, 240))
    
    # Initialize MediaPipe with optimized settings for 0.8.8
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3
    )
    
    # Create window with OpenCV 3.4.3.18 compatible flags
    cv2.namedWindow('Sign Language Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sign Language Detector', 640, 480)
    
    frame_counter = 0
    print("Starting sign language detection...")
    print("Press 'q' to quit")
    
    try:
        for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            frame_counter += 1
            image = frame.array
            
            # Process hands with numpy 1.17.3 compatible operations
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)
            detected_letter = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    detected_letter = detect_letter(hand_landmarks)
                    if detected_letter:
                        # Draw landmarks using OpenCV 3.4.3.18 compatible method
                        for connection in mp_hands.HAND_CONNECTIONS:
                            start_idx = connection[0]
                            end_idx = connection[1]
                            start_point = (int(hand_landmarks.landmark[start_idx].x * image.shape[1]),
                                        int(hand_landmarks.landmark[start_idx].y * image.shape[0]))
                            end_point = (int(hand_landmarks.landmark[end_idx].x * image.shape[1]),
                                       int(hand_landmarks.landmark[end_idx].y * image.shape[0]))
                            cv2.line(image, start_point, end_point, (0, 255, 0), 2)
            
            # Add letter to frame if detected
            if detected_letter:
                cv2.putText(
                    image,
                    f"Letter: {detected_letter}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            
            # Display the frame
            cv2.imshow('Sign Language Detector', image)
            
            # Clear the stream
            raw_capture.truncate(0)
            raw_capture.seek(0)  # Added for compatibility
            
            # Force garbage collection less frequently
            if frame_counter % 30 == 0:
                gc.collect()
            
            # Check for quit command with OpenCV 3.4.3.18 compatible key check
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up
        camera.close()
        cv2.destroyAllWindows()
        # Ensure proper cleanup for OpenCV 3.4.3.18
        cv2.waitKey(1)

if __name__ == '__main__':
    main() 