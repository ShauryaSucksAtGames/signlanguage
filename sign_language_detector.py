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
        
        # Initialize confidence scores for all letters (excluding N, O, and S)
        letter_scores = {
            'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0,
            'F': 0.0, 'G': 0.0, 'H': 0.0, 'I': 0.0, 'K': 0.0,
            'L': 0.0, 'M': 0.0, 'P': 0.0, 'R': 0.0, 'T': 0.0,
            'U': 0.0, 'V': 0.0, 'W': 0.0, 'Y': 0.0
        }
        
        # A - Thumb out to side
        if fingers.count(0) == 4:
            letter_scores['A'] = 0.95 if (posList[4][2] < posList[6][2] and  # Thumb tip above index base
                                         posList[4][1] > posList[6][1]) else 0.3  # Thumb to the right of index base
        
        # B - All fingers extended
        if fingers.count(1) >= 3:
            letter_scores['B'] = 0.95 if (posList[3][1] > posList[4][1]) and fingers.count(1) == 4 else 0.4
        
        # C - Curved hand
        if fingers.count(0.5) >= 1:
            letter_scores['C'] = 0.90 if (posList[3][1] > posList[6][1]) and (posList[4][2] > posList[8][2]) else 0.3
        
        # D - Index up, others down
        if fingers[0] == 1:
            letter_scores['D'] = 0.95 if fingers.count(0) == 3 and (posList[3][1] > posList[4][1]) else 0.3
        
        # E - All fingers curled
        if fingers.count(0) == 4:
            letter_scores['E'] = 0.95 if (posList[3][1] < posList[6][1]) and posList[12][2] < posList[4][2] else 0.3
        
        # F - Index and thumb touching
        if fingers.count(1) >= 2:
            letter_scores['F'] = 0.90 if (fingers.count(1) == 3) and (fingers[0] == 0) and (posList[3][2] > posList[4][2]) else 0.3
        
        # G - Index pointing at thumb
        if fingers[0] == 0.25:
            letter_scores['G'] = 0.90 if fingers.count(0) == 3 else 0.3
        
        # H - Index and middle parallel
        if fingers[0] == 0.25 or fingers[1] == 0.25:
            letter_scores['H'] = 0.90 if fingers[0] == 0.25 and fingers[1] == 0.25 and fingers.count(0) == 2 else 0.3
        
        # I - Pinky up
        if fingers.count(0) >= 2:
            letter_scores['I'] = 0.95 if (posList[4][1] < posList[6][1]) and fingers.count(0) == 3 and len(fingers) == 4 and fingers[3] == 1 else 0.3
        
        # K - Index and middle up, spread
        if fingers.count(1) >= 1:
            letter_scores['K'] = 0.90 if (posList[4][1] < posList[6][1] and posList[4][1] > posList[10][1] and fingers.count(1) == 2) else 0.3
        
        # L - L shape with thumb and index
        if fingers[0] == 1:
            letter_scores['L'] = 0.95 if fingers.count(0) == 3 and (posList[3][1] < posList[4][1]) else 0.3
        
        # M - Improved detection
        if fingers.count(0) == 4:
            # Thumb between ring and pinky
            thumb_tip = posList[4]
            ring_base = posList[14]
            pinky_base = posList[18]
            
            thumb_between_ring_pinky = (
                thumb_tip[1] > ring_base[1] and
                thumb_tip[1] < pinky_base[1] and
                thumb_tip[2] > ring_base[2]
            )
            
            letter_scores['M'] = 0.95 if thumb_between_ring_pinky else 0.3
        
        # P - Thumb between middle and ring fingers
        if fingers[2] == 0:
            letter_scores['P'] = 0.90 if (posList[4][2] < posList[12][2]) and (posList[4][2] > posList[6][2]) and len(fingers) == 4 and fingers[3] == 0 else 0.3
        
        # R - Cross fingers
        if fingers.count(1) >= 1:
            letter_scores['R'] = 0.90 if (posList[8][1] < posList[12][1]) and (fingers.count(1) == 2) and (posList[9][1] > posList[4][1]) else 0.3
        
        # T - Thumb between index and middle
        if fingers.count(0) >= 3:
            letter_scores['T'] = 0.95 if (posList[4][1] > posList[12][1]) and posList[4][2] < posList[6][2] and fingers.count(0) == 4 else 0.3
        
        # U - Index and middle parallel
        if fingers.count(1) >= 1:
            letter_scores['U'] = 0.90 if (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 2 and posList[3][2] > posList[4][2] and (posList[8][1] - posList[11][1]) <= 50) else 0.3
        
        # V - Index and middle spread
        if fingers.count(1) >= 1:
            letter_scores['V'] = 0.90 if (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 2 and posList[3][2] > posList[4][2] and (posList[8][1] - posList[12][1]) > 50) else 0.3
        
        # W - Three fingers up
        if fingers.count(1) >= 2:
            letter_scores['W'] = 0.90 if (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 3) else 0.3
        
        # Y - Thumb and pinky out
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
                    
                    # Display all matches if in debug mode
                    if debug_mode and all_matches:
                        # Calculate total height needed
                        total_height = len(all_matches) * 25 + 20  # Add padding
                        
                        # Draw a semi-transparent background for the confidence list
                        overlay = image.copy()
                        cv2.rectangle(
                            overlay,
                            (image.shape[1] - 200, 60),  # Move to right side
                            (image.shape[1] - 10, 60 + total_height),
                            (255, 255, 255),
                            -1
                        )
                        image = cv2.addWeighted(overlay, 0.8, image, 0.2, 0)
                        
                        # Display each match with confidence percentage
                        for i, (letter, conf) in enumerate(all_matches):
                            # Convert confidence to percentage
                            percentage = conf * 100
                            
                            # Use different colors based on confidence level
                            if conf >= 0.95:
                                color = (0, 255, 0)  # Green for high confidence
                            elif conf >= 0.90:
                                color = (0, 165, 255)  # Orange for medium confidence
                            elif conf >= 0.5:
                                color = (0, 0, 255)  # Red for low confidence
                            else:
                                color = (128, 128, 128)  # Gray for very low confidence
                            
                            # Draw confidence bar
                            bar_start = (image.shape[1] - 180, 77 + i * 25)
                            bar_end = (int(bar_start[0] + 150 * (conf)), bar_start[1])
                            cv2.rectangle(
                                image,
                                bar_start,
                                (image.shape[1] - 30, bar_start[1] + 15),
                                (200, 200, 200),
                                1
                            )
                            if conf > 0:
                                cv2.rectangle(
                                    image,
                                    bar_start,
                                    bar_end,
                                    color,
                                    -1
                                )
                            
                            # Draw letter and percentage
                            draw_text_with_background(
                                image, 
                                f"{letter}: {percentage:.1f}%",
                                (image.shape[1] - 190, 75 + i * 25),
                                font_scale=0.5,
                                color=color,
                                thickness=1
                            )
            
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