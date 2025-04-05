#!/usr/bin/env python3

import cv2
import numpy as np
import mediapipe as mp
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
import psutil
import os
import gc

# Set process priority to improve performance
try:
    p = psutil.Process(os.getpid())
    p.nice(-10)  # Higher priority
except:
    pass

# Helper function to draw text with background
def draw_text_with_background(image, text, position, font_scale=0.56, color=(0, 0, 255), thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Draw background rectangle
    cv2.rectangle(
        image, 
        (position[0] - 5, position[1] - text_size[1] - 5), 
        (position[0] + text_size[0] + 5, position[1] + 5), 
        (255, 255, 255), 
        -1
    )
    
    # Draw text
    cv2.putText(
        image,
        text,
        position,
        font,
        font_scale,
        color,
        thickness
    )

def get_finger_state(hand_landmarks, image=None, debug=False):
    """Determine if each finger is extended or not"""
    try:
        finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
        finger_bases = [5, 9, 13, 17]  # Index, middle, ring, pinky bases
        finger_names = ["Index", "Middle", "Ring", "Pinky"]
        
        # Get wrist and middle base positions
        wrist = hand_landmarks.landmark[0]
        middle_base = hand_landmarks.landmark[9]
        is_vertical = abs(middle_base.y - wrist.y) > abs(middle_base.x - wrist.x)
        
        # Check thumb (different logic based on hand orientation)
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]  # Add thumb base
        thumb_cmc = hand_landmarks.landmark[1]
        
        # Determine if vertical or horizontal orientation
        if is_vertical:
            # For vertical hand, improved check that looks at the actual direction the thumb is pointing
            # Not just its position relative to the IP joint
            thumb_direction_x = thumb_tip.x - thumb_mcp.x
            if wrist.x < middle_base.x:  # Left hand
                thumb_extended = thumb_direction_x < -0.03  # Thumb points left
            else:  # Right hand
                thumb_extended = thumb_direction_x > 0.03  # Thumb points right
        else:
            # For horizontal hand, check if thumb is above or below its base
            thumb_extended = (thumb_tip.y < thumb_mcp.y)
        
        # Check other fingers
        fingers_extended = []
        for i, (tip, base) in enumerate(zip(finger_tips, finger_bases)):
            tip_point = hand_landmarks.landmark[tip]
            base_point = hand_landmarks.landmark[base]
            
            # Different logic based on hand orientation
            if is_vertical:
                is_extended = (tip_point.y < base_point.y)  # Finger is up
            else:
                if wrist.x < middle_base.x:  # Hand pointing right
                    is_extended = (tip_point.x > base_point.x)
                else:  # Hand pointing left
                    is_extended = (tip_point.x < base_point.x)
                    
            fingers_extended.append(is_extended)
            
            # Debug visualization
            if debug and image is not None:
                img_h, img_w = image.shape[:2]
                tip_px = int(tip_point.x * img_w), int(tip_point.y * img_h)
                base_px = int(base_point.x * img_w), int(base_point.y * img_h)
                
                # Highlight finger tip and base
                cv2.circle(image, tip_px, 5, (0, 255, 255) if is_extended else (0, 0, 255), -1)
                cv2.circle(image, base_px, 5, (255, 0, 0), -1)
                
                # Draw status text
                status = "UP" if is_extended else "DOWN"
                draw_text_with_background(
                    image, 
                    f"{finger_names[i]}: {status}", 
                    (10, 70 + i * 30), 
                    color=(0, 255, 0) if is_extended else (0, 0, 255)
                )
        
        # Debug thumb with more info
        if debug and image is not None:
            img_h, img_w = image.shape[:2]
            thumb_tip_px = int(thumb_tip.x * img_w), int(thumb_tip.y * img_h)
            thumb_ip_px = int(thumb_ip.x * img_w), int(thumb_ip.y * img_h)
            
            cv2.circle(image, thumb_tip_px, 5, (0, 255, 255) if thumb_extended else (0, 0, 255), -1)
            cv2.circle(image, thumb_ip_px, 5, (255, 0, 0), -1)
            
            status = "OUT" if thumb_extended else "IN"
            thumb_direction = thumb_tip.x - thumb_mcp.x if is_vertical else thumb_tip.y - thumb_mcp.y
            draw_text_with_background(
                image, 
                f"Thumb: {status} ({thumb_direction:.2f})", 
                (10, 40), 
                color=(0, 255, 0) if thumb_extended else (0, 0, 255)
            )
            
            # Orientation indicator
            orientation = "Vertical" if is_vertical else "Horizontal"
            draw_text_with_background(
                image, 
                f"Hand: {orientation}", 
                (10, 200), 
                color=(255, 0, 0)
            )
        
        return thumb_extended, fingers_extended
    except Exception as e:
        print(f"Error in get_finger_state: {str(e)}")
        return False, [False, False, False, False]

def detect_letter(hand_landmarks, image=None, debug=False):
    """Detect ASL letters based on hand position using the reference project logic"""
    try:
        if not hand_landmarks:
            return None

        # Get finger positions
        posList = []
        for id, lm in enumerate(hand_landmarks.landmark):
            if image is not None:
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
            else:
                cx, cy = lm.x, lm.y
            posList.append([id, cx, cy])
        
        if len(posList) == 0:
            return None
            
        # Define finger indices - following the reference project's approach
        finger_mcp = [5, 9, 13, 17]  # Base of fingers
        finger_dip = [6, 10, 14, 18]  # First joint
        finger_pip = [7, 11, 15, 19]  # Second joint
        finger_tip = [8, 12, 16, 20]  # Fingertips
        
        # Check finger states (adaptation of the laptop code)
        fingers = []
        
        for id in range(4):
            if(posList[finger_tip[id]][1] + 25 < posList[finger_dip[id]][1] and posList[16][2] < posList[20][2]):
                fingers.append(0.25)  # Partially bent
            elif(posList[finger_tip[id]][2] > posList[finger_dip[id]][2]):
                fingers.append(0)     # Closed
            elif(posList[finger_tip[id]][2] < posList[finger_pip[id]][2]): 
                fingers.append(1)     # Extended
            elif(posList[finger_tip[id]][1] > posList[finger_pip[id]][1] and posList[finger_tip[id]][1] > posList[finger_dip[id]][1]): 
                fingers.append(0.5)   # Half bent
        
        # Debug visualization
        if debug and image is not None:
            finger_names = ["Index", "Middle", "Ring", "Pinky"]
            finger_states = {0: "CLOSED", 0.25: "PARTIALLY BENT", 0.5: "HALF BENT", 1: "EXTENDED"}
            
            for i, state in enumerate(fingers):
                state_text = finger_states.get(state, "UNKNOWN")
                draw_text_with_background(
                    image, 
                    f"{finger_names[i]}: {state_text}", 
                    (10, 70 + i * 30), 
                    color=(0, 255, 0) if state == 1 else (0, 0, 255)
                )
            
            # Debug thumb state
            thumb_extended = posList[3][2] > posList[4][2]
            status = "IN" if thumb_extended else "OUT"
            draw_text_with_background(
                image, 
                f"Thumb: {status}", 
                (10, 40), 
                color=(0, 255, 0) if not thumb_extended else (0, 0, 255)
            )
        
        # Detect letters using reference project's conditions
        result = None
        
        # A - Thumb folded, fingers closed
        if (posList[3][2] > posList[4][2]) and (posList[3][1] > posList[6][1]) and (posList[4][2] < posList[6][2]) and fingers.count(0) == 4:
            result = 'A'
            
        # B - Thumb tucked, fingers extended
        elif (posList[3][1] > posList[4][1]) and fingers.count(1) == 4:
            result = 'B'
        
        # C - Curved hand shape
        elif (posList[3][1] > posList[6][1]) and fingers.count(0.5) >= 1 and (posList[4][2] > posList[8][2]):
            result = 'C'
            
        # D - Index extended, others closed, thumb position specific
        elif (fingers[0] == 1) and fingers.count(0) == 3 and (posList[3][1] > posList[4][1]):
            result = 'D'
        
        # E - All fingers closed, thumb position specific
        elif (posList[3][1] < posList[6][1]) and fingers.count(0) == 4 and posList[12][2] < posList[4][2]:
            result = 'E'

        # F - Three fingers extended, index down, thumb position
        elif (fingers.count(1) == 3) and (fingers[0] == 0) and (posList[3][2] > posList[4][2]):
            result = 'F'

        # G - Index partially bent, others closed
        elif (fingers[0] == 0.25) and fingers.count(0) == 3:
            result = 'G'

        # H - Index and middle partially bent, others closed
        elif (fingers[0] == 0.25) and (fingers[1] == 0.25) and fingers.count(0) == 2:
            result = 'H'
        
        # I - Pinky extended, others closed, thumb position
        elif (posList[4][1] < posList[6][1]) and fingers.count(0) == 3:
            if (len(fingers) == 4 and fingers[3] == 1):
                result = 'I'
        
        # K - Index and middle extended with specific thumb position
        elif (posList[4][1] < posList[6][1] and posList[4][1] > posList[10][1] and fingers.count(1) == 2):
            result = 'K'
            
        # L - Index extended, others closed, thumb position
        elif (fingers[0] == 1) and fingers.count(0) == 3 and (posList[3][1] < posList[4][1]):
            result = 'L'
        
        # M - All fingers closed, thumb position near ring finger
        elif (posList[4][1] < posList[16][1]) and fingers.count(0) == 4:
            result = 'M'
        
        # N - All fingers closed, thumb position near middle finger
        elif (posList[4][1] < posList[12][1]) and fingers.count(0) == 4:
            result = 'N'
            
        # O - Thumb and fingers form a circle
        elif (posList[4][2] < posList[8][2]) and (posList[4][2] < posList[12][2]) and (posList[4][2] < posList[16][2]) and (posList[4][2] < posList[20][2]):
            result = 'O'
        
        # P - Index and middle extended, thumb in specific position
        elif (fingers[2] == 0) and (posList[4][2] < posList[12][2]) and (posList[4][2] > posList[6][2]):
            if (len(fingers) == 4 and fingers[3] == 0):
                result = 'P'
        
        # Q - Fingers down, specific thumb position
        elif (fingers[1] == 0) and (fingers[2] == 0) and (fingers[3] == 0) and (posList[8][2] > posList[5][2]) and (posList[4][2] < posList[1][2]):
            result = 'Q'
        
        # R - Index and middle extended, specific positioning
        elif (posList[8][1] < posList[12][1]) and (fingers.count(1) == 2) and (posList[9][1] > posList[4][1]):
            result = 'R'
            
        # S - All fingers closed, specific thumb position
        elif (posList[4][1] > posList[12][1]) and posList[4][2] < posList[12][2] and fingers.count(0) == 4:
            result = 'S'
            
        # T - All fingers closed, thumb between index and middle
        elif (posList[4][1] > posList[12][1]) and posList[4][2] < posList[6][2] and fingers.count(0) == 4:
            result = 'T'
        
        # U - Index and middle extended parallel, thumb position
        elif (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 2 and posList[3][2] > posList[4][2] and (posList[8][1] - posList[11][1]) <= 50):
            result = 'U'
            
        # V - Index and middle extended in V shape
        elif (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 2 and posList[3][2] > posList[4][2]):
            result = 'V'
        
        # W - Index, middle, and ring extended
        elif (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 3):
            result = 'W'
        
        # X - Index half bent, others closed
        elif (fingers[0] == 0.5 and fingers.count(0) == 3 and posList[4][1] > posList[6][1]):
            result = 'X'
        
        # Y - Pinky extended, thumb extended, others closed
        elif (fingers.count(0) == 3) and (posList[3][1] < posList[4][1]):
            if (len(fingers) == 4 and fingers[3] == 1):
                result = 'Y'
                
        return result
        
    except Exception as e:
        print(f"Error in detect_letter: {str(e)}")
        return None

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.4,  # Reduced from 0.5 for better detection
        min_tracking_confidence=0.4    # Reduced from 0.5 for more stable tracking
    )
    
    # Initialize camera
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 15
    camera.exposure_mode = 'auto'
    camera.awb_mode = 'auto'
    raw_capture = PiRGBArray(camera, size=(640, 480))
    
    # Allow camera to warm up
    time.sleep(2)
    
    # Counter for memory management
    frame_counter = 0
    debug_mode = True  # Enable debug visualization
    
    print("Sign Language Detector Started")
    print("Press 'q' to quit, 'd' to toggle debug mode")
    
    # Create window
    cv2.namedWindow('Sign Language Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sign Language Detector', 640, 480)
    
    # Main detection loop
    try:
        for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            image = frame.array
            frame_counter += 1
            
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = hands.process(rgb_image)
            detected_letter = None
            
            # Add title and instructions
            draw_text_with_background(
                image, 
                "ASL Sign Language Detector", 
                (image.shape[1]//2 - 120, 25), 
                font_scale=0.7,
                color=(255, 0, 0)
            )
            
            # Add debug message
            if debug_mode:
                draw_text_with_background(image, "DEBUG MODE ON", (10, 20), color=(0, 0, 255))
                draw_text_with_background(image, "Press 'd' to toggle debug", (10, image.shape[0] - 20), color=(0, 0, 255))
                
            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    for connection in mp_hands.HAND_CONNECTIONS:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        
                        # Get coordinates
                        start_point = (
                            int(hand_landmarks.landmark[start_idx].x * image.shape[1]),
                            int(hand_landmarks.landmark[start_idx].y * image.shape[0])
                        )
                        end_point = (
                            int(hand_landmarks.landmark[end_idx].x * image.shape[1]),
                            int(hand_landmarks.landmark[end_idx].y * image.shape[0])
                        )
                        
                        # Draw thicker line
                        cv2.line(image, start_point, end_point, (0, 255, 0), 3)
                    
                    # Draw landmarks as larger circles for better visibility
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        h, w, _ = image.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(image, (cx, cy), 4, (255, 0, 255), -1)
                    
                    # Detect letter
                    detected_letter = detect_letter(hand_landmarks, image if debug_mode else None, debug_mode)
            
            # Display detected letter
            if detected_letter:
                cv2.putText(
                    image,
                    f"Sign: {detected_letter}",
                    (image.shape[1]//2 - 40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    2
                )
            
            # Display the frame
            cv2.imshow('Sign Language Detector', image)
            
            # Clear buffer for next frame
            raw_capture.truncate(0)
            raw_capture.seek(0)
            
            # Periodic garbage collection
            if frame_counter % 30 == 0:
                gc.collect()
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        
    finally:
        # Cleanup
        camera.close()
        hands.close()
        cv2.destroyAllWindows()
        # Additional cleanup for OpenCV on Raspberry Pi
        cv2.waitKey(1)

if __name__ == "__main__":
    main() 