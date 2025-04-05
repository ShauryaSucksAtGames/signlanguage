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
            return None, 0.0

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
            return None, 0.0
            
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
        
        # Calculate hand orientation
        wrist = posList[0]
        middle_base = posList[9]
        is_vertical = abs(middle_base[2] - wrist[2]) > abs(middle_base[1] - wrist[1])
        
        # Calculate thumb state more precisely
        thumb_in = posList[3][2] > posList[4][2]  # Thumb tucked in
        thumb_out = not thumb_in  # Thumb extended out
        
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
            status = "IN" if thumb_in else "OUT"
            draw_text_with_background(
                image, 
                f"Thumb: {status}", 
                (10, 40), 
                color=(0, 255, 0) if not thumb_in else (0, 0, 255)
            )
            
            # Show orientation
            orientation = "Vertical" if is_vertical else "Horizontal"
            draw_text_with_background(
                image, 
                f"Hand: {orientation}", 
                (10, 230), 
                color=(255, 0, 0)
            )
        
        # Detect letters using reference project's conditions
        result = None
        confidence_level = 0.0
        
        # For performance on Raspberry Pi, use a more efficient approach to letter detection
        # Instead of multiple complex condition checks, create a signature-based method
        
        # Create a simplified hand signature based on finger states and thumb position
        # Format: [thumb_position, index, middle, ring, pinky, is_vertical]
        signature = [1 if thumb_out else 0] + [int(f * 2) for f in fingers] + [1 if is_vertical else 0]
        
        # A - Thumb in, all fingers closed, vertical orientation with specific thumb position
        if fingers.count(0) == 4 and thumb_in and (posList[3][1] > posList[6][1]) and (posList[4][2] < posList[6][2]):
            result = 'A'
            confidence_level = 0.95
        
        # B - ALL four fingers extended, thumb in, vertical orientation
        elif is_vertical and fingers.count(1) == 4 and thumb_in:
            result = 'B'
            confidence_level = 0.95
        
        # C - Curved hand shape (at least 2 fingers curved, not closed)
        elif fingers.count(0.5) >= 2 and fingers.count(0) <= 1 and thumb_out:
            # Additional check to ensure it's a proper C shape
            if (posList[4][1] > posList[8][1]) and (posList[4][2] > posList[8][2]):
                result = 'C'
                confidence_level = 0.90
        
        # D - Index extended, others closed, thumb in, vertical
        elif is_vertical and fingers[0] == 1 and fingers.count(0) == 3 and thumb_in:
            result = 'D'
            confidence_level = 0.95
        
        # E - All fingers closed, thumb position specific, vertical
        elif is_vertical and fingers.count(0) == 4 and posList[3][1] < posList[6][1]:
            result = 'E'
            confidence_level = 0.95
        
        # F - Index closed, middle/ring/pinky extended, thumb in
        elif fingers[0] == 0 and fingers.count(1) == 3 and thumb_in:
            result = 'F'
            confidence_level = 0.95
        
        # G - Index partially bent, ONLY in horizontal orientation
        elif not is_vertical and (fingers[0] == 0.25 or fingers[0] == 0.5):
            # Count partially bent fingers - G should have exactly one
            partially_bent_count = sum(1 for f in fingers if f == 0.25 or f == 0.5)
            
            if partially_bent_count == 1:  # Only index is partially bent
                result = 'G'
                confidence_level = 0.95
        
        # H - Only in horizontal orientation, index and middle extended
        elif not is_vertical and fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
            # Improved H detection - more lenient with distance check
            # Check finger distance - must be reasonably close but not super strict
            index_middle_dist = abs(posList[8][1] - posList[12][1])
            # Also check that fingers are at similar height level
            index_middle_y_dist = abs(posList[8][2] - posList[12][2])
            
            # More lenient H detection condition
            if index_middle_dist < 80 and index_middle_y_dist < 40:  # Increased distance threshold
                # Verify it's not G by ensuring index finger is fully extended, not bent
                if fingers[0] == 1:  # Index is fully extended
                    result = 'H'
                    confidence_level = 0.95
        
        # I - Pinky extended, others closed, vertical orientation
        elif is_vertical and fingers[3] == 1 and fingers.count(0) == 3:
            # Make sure the pinky is clearly extended and other fingers are clearly closed
            # Also verify thumb is tucked in
            pinky_extended = posList[20][2] < posList[17][2]  # Pinky tip is above pinky base
            
            # Strict thumb check - ensure thumb is clearly tucked in
            thumb_clearly_in = thumb_in and (posList[4][2] > posList[5][2])  # Thumb below index base
            
            if pinky_extended and thumb_clearly_in:
                result = 'I'
                confidence_level = 0.95
        
        # K - Index and middle extended, thumb in, vertical orientation, fingers separated
        elif is_vertical and fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and thumb_in:
            # Ensure fingers are separated enough - key difference from U and H
            index_middle_dist = abs(posList[8][1] - posList[12][1])
            # Additional check to ensure it's not confused with V
            if index_middle_dist > 40 and index_middle_dist < 80:
                result = 'K'
                confidence_level = 0.90
        
        # L - Index extended, thumb out, others closed, either orientation
        elif fingers[0] == 1 and fingers.count(0) >= 2 and thumb_out:
            # L shape requires index extended and thumb out to side
            result = 'L'
            confidence_level = 0.95
        
        # O - Fingers form a circle - specific thumb-to-index distance
        elif fingers.count(0) >= 3:
            # Check thumb and index tip distance to form a circle
            thumb_to_index = abs(posList[4][1] - posList[8][1]) + abs(posList[4][2] - posList[8][2])
            if thumb_to_index < 60 and abs(posList[4][2] - posList[8][2]) < 30:
                # Make sure this isn't S or T by verifying thumb position
                if abs(posList[4][1] - posList[5][1]) > 20:
                    result = 'O'
                    confidence_level = 0.80
        
        # P - Two cases: (1) Traditional P or (2) Ring/pinky extended P
        # Traditional P - Index extended, thumb in
        # if is_vertical and fingers[0] == 1 and fingers.count(0) >= 2 and thumb_in:
        #     result = 'P'
        #     confidence_level = 0.90
        # # Alternate P - Ring/pinky extended, index/middle closed, thumb out, horizontal
        # elif not is_vertical and fingers[0] == 0 and fingers[1] == 0 and fingers[2] == 1 and fingers[3] == 1 and thumb_out:
        #     result = 'P'
        #     confidence_level = 0.95
        
        # Q detection removed as requested
        
        # R - Index and middle extended, nearly crossed, vertical orientation
        elif is_vertical and fingers[0] == 1 and fingers[1] == 1 and fingers.count(0) >= 2:
            # Improved R detection to match the image
            # R has two fingers extended (index and middle) in vertical orientation
            
            # Less strict requirements for finger crossing
            index_middle_y_dist = abs(posList[8][2] - posList[12][2])
            index_middle_x_dist = abs(posList[8][1] - posList[12][1])
            
            # Check if fingers are close together, but not requiring them to be crossed
            # For R, allow the fingers to be almost parallel (similar to U but with thumb in)
            # Check fingers are extended
            index_extended = posList[8][2] < posList[5][2]
            middle_extended = posList[12][2] < posList[9][2]
            
            # For the R case shown in the image - two fingers extended vertically
            if index_extended and middle_extended and fingers[2] == 0 and fingers[3] == 0:
                # Key difference from V - fingers are closer together in R
                if index_middle_x_dist < 70 and index_middle_y_dist < 60:
                    # Additional check for thumb position
                    result = 'R'
                    confidence_level = 0.95
        
        # S - Fist with thumb over fingers
        elif fingers.count(0) == 4 and thumb_in:
            # Improved S detection based on the image
            # S is a fist with thumb wrapped across the front of the fingers
            
            # For S, thumb should cross in front of the fingers
            # The key is to detect a closed fist with thumb positioned properly
            
            # More lenient check that doesn't require specific thumb positioning
            # Just verify all fingers are closed and thumb is positioned in front
            fingers_closed = fingers.count(0) == 4
            
            # For S, we don't need such strict thumb position checks
            # Just verify the thumb is in front of the fingers but not forming a circle (like O)
            if fingers_closed:
                # Verify it's not being confused with E or A
                thumb_to_index = abs(posList[4][1] - posList[8][1]) + abs(posList[4][2] - posList[8][2])
                
                # Make sure this isn't O (which would have thumb very close to index)
                if thumb_to_index > 50 and thumb_to_index < 200:
                    # For S, thumb should be in front of fingers, but not too far away
                    if posList[4][2] < posList[8][2]:  # Thumb is above index
                        result = 'S'
                        confidence_level = 0.95
        
        # T - Thumb positioned between index and middle on closed fist
        elif fingers.count(0) == 4:
            # Check if thumb is between index and middle finger bases
            index_pos = posList[5]
            middle_pos = posList[9]
            thumb_pos = posList[4]
            
            if (thumb_pos[1] > index_pos[1] and thumb_pos[1] < middle_pos[1]):
                # Verify it's not O
                thumb_to_index = abs(posList[4][1] - posList[8][1]) + abs(posList[4][2] - posList[8][2])
                if thumb_to_index > 70:
                    result = 'T'
                    confidence_level = 0.85
        
        # U - Two fingers close together, vertical orientation
        elif is_vertical and fingers[0] == 1 and fingers[1] == 1 and fingers.count(0) == 2:
            # Improved U detection based on the image
            # U has index and middle extended vertically and close together
            
            # Check fingers are close together
            index_middle_dist = abs(posList[8][1] - posList[12][1])
            index_middle_y_dist = abs(posList[8][2] - posList[12][2])
            
            # U detection - fingers must be close together (not as strict as before)
            if index_middle_dist < 60 and index_middle_y_dist < 40:
                # Key distinction from K - doesn't require thumb to be in
                result = 'U'
                confidence_level = 0.95
        
        # V - Two fingers separated, vertical, thumb out
        elif is_vertical and fingers[0] == 1 and fingers[1] == 1 and fingers.count(0) == 2 and thumb_out:
            # Check fingers are widely separated in V shape
            index_middle_dist = abs(posList[8][1] - posList[12][1])
            index_middle_y_dist = abs(posList[8][2] - posList[12][2])
            
            # V requires clearly separated fingers - increase threshold to avoid confusion with K
            if index_middle_dist >= 80 and index_middle_y_dist < 50:
                # Critical distinction from K - thumb must be out
                result = 'V'
                confidence_level = 0.95
        
        # W - Three fingers extended (index, middle, ring), pinky closed, vertical
        elif is_vertical and fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
            # Check fingers are spread apart in W shape
            index_middle_dist = abs(posList[8][1] - posList[12][1])
            middle_ring_dist = abs(posList[12][1] - posList[16][1])
            
            if index_middle_dist > 30 and middle_ring_dist > 30:
                result = 'W'
                confidence_level = 0.95
        
        # X - Index half bent, others closed
        elif fingers[0] == 0.5 and fingers.count(0) >= 2:
            # Make sure it's not confused with O
            thumb_to_index = abs(posList[4][1] - posList[8][1]) + abs(posList[4][2] - posList[8][2])
            if thumb_to_index > 70:
                result = 'C'  # Changed from 'X' to 'C' as requested
                confidence_level = 0.80
        
        # Y - Pinky extended, thumb out, other fingers closed
        elif fingers[3] == 1 and thumb_out:
            # Improved Y detection with focus on pinky extended and thumb out
            # Less restrictive conditions to ensure Y is detected when it should be
            
            # Check pinky is clearly extended
            pinky_extended = posList[20][2] < posList[17][2]
            
            # Verify at least 2 fingers are closed (allow some flexibility)
            closed_count = sum(1 for f in fingers if f == 0)
            if closed_count >= 2 and pinky_extended:
                # Make sure thumb is clearly extended outward
                thumb_extended_distance = abs(posList[4][1] - posList[0][1])
                if thumb_extended_distance > 40:  # Reduced threshold for easier detection
                    result = 'Y' 
                    confidence_level = 0.95
        
        return result, confidence_level
        
    except Exception as e:
        print(f"Error in detect_letter: {str(e)}")
        return None, 0.0

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
                        
                        # Draw thicker line
                        cv2.line(image, start_point, end_point, (0, 255, 0), 3)
                    
                    # Draw landmarks as larger circles for better visibility
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        h, w, _ = image.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(image, (cx, cy), 4, (255, 0, 255), -1)
                    
                    # Detect letter
                    detected_letter, confidence_level = detect_letter(hand_landmarks, image if debug_mode else None, debug_mode)
                    
                    # Display the detected letter immediately
                    if detected_letter:
                        cv2.putText(
                            image,
                            f"Sign: {detected_letter} ({confidence_level:.2f})",
                            (image.shape[1]//2 - 80, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
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