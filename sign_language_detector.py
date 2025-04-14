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
def draw_text_with_background(image, text, position, font_scale=0.7, color=(0, 0, 255), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Calculate background rectangle
    padding = 5
    bg_rect_pt1 = (position[0] - padding, position[1] - text_size[1] - padding)
    bg_rect_pt2 = (position[0] + text_size[0] + padding, position[1] + padding)
    
    # Draw white background rectangle
    cv2.rectangle(image, bg_rect_pt1, bg_rect_pt2, (255, 255, 255), -1)
    
    # Draw text
    cv2.putText(image, text, position, font, font_scale, color, thickness)

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
            return None, 0.0, []

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
            return None, 0.0, []
            
        # Define finger indices
        finger_mcp = [5, 9, 13, 17]  # Base of fingers
        finger_dip = [6, 10, 14, 18]  # First joint
        finger_pip = [7, 11, 15, 19]  # Second joint
        finger_tip = [8, 12, 16, 20]  # Fingertips
        
        # Check finger states
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
        
        # Store ALL matches and their confidence levels
        all_matches = []
        
        # Initialize confidence scores for all letters
        letter_scores = {
            'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0,
            'F': 0.0, 'G': 0.0, 'H': 0.0, 'I': 0.0, 'K': 0.0,
            'L': 0.0, 'M': 0.0, 'R': 0.0, 'S': 0.0, 'U': 0.0,
            'V': 0.0, 'W': 0.0, 'Y': 0.0
        }
        
        # Calculate base confidence for each letter
        # A
        if fingers.count(0) == 4:
            # For A: thumb should be sticking out to the side and up
            letter_scores['A'] = 0.95 if (posList[4][2] < posList[6][2] and  # Thumb tip above index base
                                         posList[4][1] > posList[6][1]) else 0.3  # Thumb to the right of index base
        
        # B
        if fingers.count(1) >= 3:
            letter_scores['B'] = 0.95 if (posList[3][1] > posList[4][1]) and fingers.count(1) == 4 else 0.4
        
        # C
        if fingers.count(0.5) >= 1:
            letter_scores['C'] = 0.90 if (posList[3][1] > posList[6][1]) and (posList[4][2] > posList[8][2]) else 0.3
        
        # D
        if fingers[0] == 1:
            letter_scores['D'] = 0.95 if fingers.count(0) == 3 and (posList[3][1] > posList[4][1]) else 0.3
        
        # E
        if fingers.count(0) == 4:
            letter_scores['E'] = 0.95 if (posList[3][1] < posList[6][1]) and posList[12][2] < posList[4][2] else 0.3
        
        # F
        if fingers.count(1) >= 2:
            letter_scores['F'] = 0.90 if (fingers.count(1) == 3) and (fingers[0] == 0) and (posList[3][2] > posList[4][2]) else 0.3
        
        # G
        if fingers[0] == 0.25:
            letter_scores['G'] = 0.90 if fingers.count(0) == 3 else 0.3
        
        # H
        if fingers[0] == 0.25 or fingers[1] == 0.25:
            letter_scores['H'] = 0.90 if fingers[0] == 0.25 and fingers[1] == 0.25 and fingers.count(0) == 2 else 0.3
        
        # I
        if fingers.count(0) >= 2:
            letter_scores['I'] = 0.95 if (posList[4][1] < posList[6][1]) and fingers.count(0) == 3 and len(fingers) == 4 and fingers[3] == 1 else 0.3
        
        # K
        if fingers.count(1) >= 1:
            letter_scores['K'] = 0.90 if (posList[4][1] < posList[6][1] and posList[4][1] > posList[10][1] and fingers.count(1) == 2) else 0.3
        
        # L
        if fingers[0] == 1:
            letter_scores['L'] = 0.95 if fingers.count(0) == 3 and (posList[3][1] < posList[4][1]) else 0.3
        
        # M
        if fingers.count(0) >= 3:
            letter_scores['M'] = 0.95 if (posList[4][1] < posList[16][1]) and fingers.count(0) == 4 else 0.3
        
        # R
        if fingers.count(1) >= 1:
            letter_scores['R'] = 0.90 if (posList[8][1] < posList[12][1]) and (fingers.count(1) == 2) and (posList[9][1] > posList[4][1]) else 0.3
        
        # S
        if fingers.count(0) == 4:
            # Get key points
            thumb_tip = posList[4]
            thumb_ip = posList[3]  # Thumb IP joint
            index_tip = posList[8]
            middle_tip = posList[12]
            index_base = posList[6]
            
            # For S: thumb should wrap over closed fist
            thumb_wrapped = (
                thumb_tip[2] < thumb_ip[2] and      # Thumb tip should be above its IP joint
                thumb_tip[2] < middle_tip[2] and    # Thumb tip above middle finger
                thumb_tip[1] > middle_tip[1] and    # Thumb tip should be to the right of middle finger
                thumb_tip[2] > index_base[2] - 40   # But not too high above index base
            )
            
            letter_scores['S'] = 0.95 if thumb_wrapped else 0.3
        
        # U
        if fingers.count(1) >= 1:
            letter_scores['U'] = 0.90 if (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 2 and posList[3][2] > posList[4][2] and (posList[8][1] - posList[11][1]) <= 50) else 0.3
        
        # V
        if fingers.count(1) >= 1:
            letter_scores['V'] = 0.90 if (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 2 and posList[3][2] > posList[4][2]) else 0.3
        
        # W
        if fingers.count(1) >= 2:
            letter_scores['W'] = 0.90 if (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 3) else 0.3
        
        # Y
        if fingers.count(0) >= 2:
            letter_scores['Y'] = 0.95 if fingers.count(0) == 3 and (posList[3][1] < posList[4][1]) and len(fingers) == 4 and fingers[3] == 1 else 0.3
        
        # Convert scores to list format
        all_matches = [(letter, score) for letter, score in letter_scores.items()]
        all_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return best match and all matches
        best_match = all_matches[0]
        return best_match[0], best_match[1], all_matches
        
    except Exception as e:
        print(f"Error in detect_letter: {str(e)}")
        return None, 0.0, []

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
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
    debug_mode = True
    
    print("Sign Language Detector Started")
    print("Press 'q' to quit, 'd' to toggle debug mode")
    
    # Create window
    cv2.namedWindow('Sign Language Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sign Language Detector', 640, 480)
    
    # Variables to store detection history
    last_detection_time = time.time()
    detection_history = []
    
    try:
        for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            image = frame.array
            frame_counter += 1
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)
            detected_letter = None
            confidence_level = 0.0
            
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
                        
                        # Draw thinner white lines
                        cv2.line(image, start_point, end_point, (255, 255, 255), 1)
                    
                    # Draw landmarks as larger circles for better visibility
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        h, w, _ = image.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(image, (cx, cy), 4, (255, 0, 255), -1)  # Keep magenta dots
                    
                    # Detect letter with all matches
                    detected_letter, confidence_level, all_matches = detect_letter(hand_landmarks, image if debug_mode else None, debug_mode)
                    
                    # Display all letter confidences
                    y_offset = 0
                    for l, score in sorted([('A', confidence_level if detected_letter == 'A' else 0.3),
                                         ('B', confidence_level if detected_letter == 'B' else 0.3),
                                         ('C', confidence_level if detected_letter == 'C' else 0.3),
                                         ('D', confidence_level if detected_letter == 'D' else 0.3),
                                         ('E', confidence_level if detected_letter == 'E' else 0.3),
                                         ('F', confidence_level if detected_letter == 'F' else 0.3),
                                         ('G', confidence_level if detected_letter == 'G' else 0.3),
                                         ('H', confidence_level if detected_letter == 'H' else 0.3),
                                         ('I', confidence_level if detected_letter == 'I' else 0.3),
                                         ('K', confidence_level if detected_letter == 'K' else 0.3),
                                         ('L', confidence_level if detected_letter == 'L' else 0.3),
                                         ('M', confidence_level if detected_letter == 'M' else 0.3),
                                         ('R', confidence_level if detected_letter == 'R' else 0.3),
                                         ('S', confidence_level if detected_letter == 'S' else 0.3),
                                         ('U', confidence_level if detected_letter == 'U' else 0.3),
                                         ('V', confidence_level if detected_letter == 'V' else 0.3),
                                         ('W', confidence_level if detected_letter == 'W' else 0.3),
                                         ('Y', confidence_level if detected_letter == 'Y' else 0.3)],
                                        key=lambda x: x[1], reverse=True):
                        if score > 0.3:  # Only show letters with confidence above 0.3
                            color = (0, 255, 0) if score >= 0.95 else (255, 165, 0) if score >= 0.9 else (128, 128, 128)
                            draw_text_with_background(
                                image,
                                f"{l}: {score*100:.1f}%",
                                (image.shape[1] - 150, 50 + y_offset),
                                color=color
                            )
                            y_offset += 30
            
            else:
                # No hand detected
                if debug_mode:
                    draw_text_with_background(
                        image,
                        "No hand detected",
                        (image.shape[1] - 190, 80),  # Right-aligned
                        font_scale=0.6,
                        color=(0, 0, 255)
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
        cv2.waitKey(1)

if __name__ == "__main__":
    main() 