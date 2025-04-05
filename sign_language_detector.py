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
        
        # Improved thumb detection
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]  # Inner thumb joint
        thumb_mcp = hand_landmarks.landmark[2]  # Base of thumb
        thumb_cmc = hand_landmarks.landmark[1]  # Thumb CMC joint
        
        # Calculate directional vector from base to tip for thumb
        thumb_vec_x = thumb_tip.x - thumb_mcp.x
        thumb_vec_y = thumb_tip.y - thumb_mcp.y
        
        # Determine if vertical or horizontal orientation
        if is_vertical:
            # For vertical hand, more complex thumb detection
            if wrist.x < middle_base.x:  # Left hand
                thumb_extended = (thumb_tip.x < thumb_mcp.x)
            else:  # Right hand
                thumb_extended = (thumb_tip.x > thumb_mcp.x)
        else:
            # For horizontal hand, check if thumb is pointing in the opposite direction than fingers
            if wrist.x < middle_base.x:  # Hand pointing right
                thumb_extended = (thumb_tip.y < thumb_mcp.y)
            else:  # Hand pointing left
                thumb_extended = (thumb_tip.y < thumb_mcp.y)
        
        # Special case adjustments for common problems
        index_dip = hand_landmarks.landmark[7]  # Index finger DIP joint
        pinky_tip = hand_landmarks.landmark[20]  # Pinky tip
        
        # Check if thumb is across palm (important for several signs)
        distance_thumb_to_pinky = np.sqrt((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2)
        is_thumb_across_palm = distance_thumb_to_pinky < 0.25  # Threshold for thumb across palm
        
        # Check other fingers (improved)
        fingers_extended = []
        for i, (tip, base) in enumerate(zip(finger_tips, finger_bases)):
            tip_point = hand_landmarks.landmark[tip]
            base_point = hand_landmarks.landmark[base]
            
            # Get middle joint for this finger (for bend detection)
            mid_joint_idx = tip - 2  # PIP joint for each finger
            mid_joint = hand_landmarks.landmark[mid_joint_idx]
            
            # Calculate bend angle
            vec1 = np.array([base_point.x - mid_joint.x, base_point.y - mid_joint.y])
            vec2 = np.array([tip_point.x - mid_joint.x, tip_point.y - mid_joint.y])
            
            # Normalize vectors
            if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                vec1 = vec1 / np.linalg.norm(vec1)
                vec2 = vec2 / np.linalg.norm(vec2)
                
                # Dot product for angle
                dot_product = min(1.0, max(-1.0, np.dot(vec1, vec2)))
                angle = np.arccos(dot_product) * 180 / np.pi
            else:
                angle = 0
                
            # Different logic based on hand orientation
            if is_vertical:
                # For vertical hand, check both position and angle
                position_check = (tip_point.y < base_point.y - 0.02)  # Added threshold
                is_extended = position_check and angle < 45  # Must be relatively straight
            else:
                if wrist.x < middle_base.x:  # Hand pointing right
                    position_check = (tip_point.x > base_point.x + 0.02)
                else:  # Hand pointing left
                    position_check = (tip_point.x < base_point.x - 0.02)
                is_extended = position_check and angle < 45
                    
            fingers_extended.append(is_extended)
            
            # Debug visualization
            if debug and image is not None:
                img_h, img_w = image.shape[:2]
                tip_px = int(tip_point.x * img_w), int(tip_point.y * img_h)
                base_px = int(base_point.x * img_w), int(base_point.y * img_h)
                mid_px = int(mid_joint.x * img_w), int(mid_joint.y * img_h)
                
                # Highlight finger tip, base and mid joint
                cv2.circle(image, tip_px, 5, (0, 255, 255) if is_extended else (0, 0, 255), -1)
                cv2.circle(image, base_px, 5, (255, 0, 0), -1)
                cv2.circle(image, mid_px, 5, (0, 165, 255), -1)  # Orange for mid joint
                
                # Draw status text
                status = "UP" if is_extended else "DOWN"
                draw_text_with_background(
                    image, 
                    f"{finger_names[i]}: {status} ({angle:.0f}°)", 
                    (10, 70 + i * 30), 
                    color=(0, 255, 0) if is_extended else (0, 0, 255)
                )
        
        # Debug thumb
        if debug and image is not None:
            img_h, img_w = image.shape[:2]
            thumb_tip_px = int(thumb_tip.x * img_w), int(thumb_tip.y * img_h)
            thumb_base_px = int(thumb_mcp.x * img_w), int(thumb_mcp.y * img_h)
            
            cv2.circle(image, thumb_tip_px, 5, (0, 255, 255) if thumb_extended else (0, 0, 255), -1)
            cv2.circle(image, thumb_base_px, 5, (255, 0, 0), -1)
            
            status = "OUT" if thumb_extended else "IN"
            draw_text_with_background(
                image, 
                f"Thumb: {status} {'ACROSS' if is_thumb_across_palm else ''}", 
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
        
        return thumb_extended, fingers_extended, is_thumb_across_palm
    except Exception as e:
        print(f"Error in get_finger_state: {str(e)}")
        return False, [False, False, False, False], False

def detect_letter(hand_landmarks, image=None, debug=False):
    """Detect ASL letters based on simplified finger positions"""
    try:
        if not hand_landmarks:
            return None

        thumb_extended, fingers_extended, thumb_across_palm = get_finger_state(hand_landmarks, image, debug)
        index, middle, ring, pinky = fingers_extended
        
        # Get hand orientation
        wrist = hand_landmarks.landmark[0]
        middle_base = hand_landmarks.landmark[9]
        is_vertical = abs(middle_base.y - wrist.y) > abs(middle_base.x - wrist.x)
        
        # Get fingertip positions for distance calculations
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        # Calculate distances between fingertips and joints
        thumb_index_distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        index_middle_distance = np.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)
        thumb_pinky_distance = np.sqrt((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2)
        
        # Get joint positions
        index_pip = hand_landmarks.landmark[6]
        index_dip = hand_landmarks.landmark[7]
        middle_pip = hand_landmarks.landmark[10]
        
        # Calculate fingertip-to-palm distances (for 3D depth estimation)
        index_base = hand_landmarks.landmark[5]
        thumb_base = hand_landmarks.landmark[2]
        
        # A - Fist with thumb out to side
        if thumb_extended and not any(fingers_extended) and not thumb_across_palm:
            return 'A'
        
        # B - All fingers up, thumb folded across palm
        if all(fingers_extended) and not thumb_extended and thumb_across_palm:
            # Make sure fingers are close together
            if index_middle_distance < 0.08:
                return 'B'
        
        # C - Curved hand shape (thumb and fingers in C curve)
        # C has all fingers closed but curved, thumb extended in a C shape
        if not any(fingers_extended) and thumb_extended and not thumb_across_palm:
            # Check distance from thumb to index for C shape
            if 0.08 < thumb_index_distance < 0.2:
                return 'C'
        
        # D - Index up, others closed, thumb touching middle
        if index and not middle and not ring and not pinky and thumb_extended:
            # D has thumb touching middle finger
            thumb_middle_distance = np.sqrt((thumb_tip.x - middle_pip.x)**2 + (thumb_tip.y - middle_pip.y)**2)
            if thumb_middle_distance < 0.1:
                return 'D'
        
        # E - All fingers curled in, thumb across palm
        if not any(fingers_extended) and not thumb_extended and thumb_across_palm:
            return 'E'
        
        # F - Index touching thumb, other fingers up
        if not index and middle and ring and pinky:
            # Check if index and thumb are touching
            if thumb_index_distance < 0.1:
                return 'F'
        
        # G - Index pointing outward, thumb across palm
        if index and not middle and not ring and not pinky and not thumb_extended:
            # For G, index points horizontally while thumb is across
            # Check if index is more horizontal than vertical
            index_direction = abs(index_tip.x - index_base.x) > abs(index_tip.y - index_base.y)
            if index_direction and thumb_across_palm:
                return 'G'
        
        # H - Index and middle extended side by side
        if index and middle and not ring and not pinky:
            # H has two fingers together, not spread
            if index_middle_distance < 0.08:
                return 'H'
        
        # I - Pinky up, other fingers closed
        if not index and not middle and not ring and pinky:
            return 'I'
        
        # K - Index and middle up in V, thumb touches middle base
        if index and middle and not ring and not pinky and thumb_extended:
            # K has index and middle up in a V with thumb touching hand
            if index_middle_distance > 0.08:  # Spread V
                # Check thumb position more precisely
                thumb_to_middle_base = np.sqrt((thumb_tip.x - middle_base.x)**2 + (thumb_tip.y - middle_base.y)**2)
                if thumb_to_middle_base < 0.12:
                    return 'K'
        
        # L - Index and thumb out in L shape
        if index and not middle and not ring and not pinky and thumb_extended:
            # L has thumb and index at approx. 90 degrees
            v1 = np.array([index_tip.x - index_base.x, index_tip.y - index_base.y])
            v2 = np.array([thumb_tip.x - index_base.x, thumb_tip.y - index_base.y])
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm > 0 and v2_norm > 0:
                dot_product = np.dot(v1, v2) / (v1_norm * v2_norm)
                angle = np.arccos(max(-1.0, min(1.0, dot_product))) * 180 / np.pi
                if 70 < angle < 110:  # Approximately 90 degrees
                    return 'L'
        
        # M - Thumb between fingers
        if not index and not middle and not ring and not pinky and thumb_across_palm:
            # More specific check for M vs. E (M has thumb visible between fingers)
            # M has thumb tucked under fingers, but visible between ring and pinky
            thumb_position = (thumb_tip.x > ring_tip.x) and (thumb_tip.x < pinky_tip.x)
            if thumb_position:
                return 'M'
        
        # N - Thumb between finger
        if not index and not middle and not ring and not pinky and thumb_across_palm:
            # N has thumb tucked under fingers, but visible between middle and ring
            thumb_position = (thumb_tip.x > middle_tip.x) and (thumb_tip.x < ring_tip.x)
            if thumb_position:
                return 'N'
        
        # R - Cross fingers
        if index and middle and not ring and not pinky:
            # R has index crossing over middle
            finger_crossed = (index_tip.x > middle_tip.x) if (wrist.x < middle_base.x) else (index_tip.x < middle_tip.x)
            if finger_crossed:
                return 'R'
        
        # T - Thumb between index and middle
        if not index and not middle and not ring and not pinky and thumb_extended:
            # T has thumb between index and middle
            thumb_between = (thumb_tip.x > index_tip.x) and (thumb_tip.x < middle_tip.x)
            if thumb_between:
                return 'T'
        
        # U - Index and middle parallel
        if index and middle and not ring and not pinky and not thumb_extended:
            # U has two fingers parallel
            if index_middle_distance < 0.08:
                return 'U'
        
        # V - Index and middle in V
        if index and middle and not ring and not pinky:
            # V has fingers spread
            if index_middle_distance > 0.08:
                return 'V'
        
        # W - Index, middle, ring extended
        if index and middle and ring and not pinky:
            return 'W'
        
        # Y - Thumb and pinky extended
        if not index and not middle and not ring and pinky and thumb_extended:
            return 'Y'

        return None
    except Exception as e:
        print(f"Error in detect_letter: {str(e)}")
        return None

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
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
            
            # Add debug message
            if debug_mode:
                draw_text_with_background(image, "DEBUG MODE ON", (10, 20), color=(0, 0, 255))
            
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
                        
                        # Draw line
                        cv2.line(image, start_point, end_point, (0, 255, 0), 2)
                    
                    # Detect letter
                    detected_letter = detect_letter(hand_landmarks, image if debug_mode else None, debug_mode)
            
            # Display detected letter
            if detected_letter:
                cv2.putText(
                    image,
                    f"Sign: {detected_letter}",
                    (image.shape[1]//2 - 50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
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