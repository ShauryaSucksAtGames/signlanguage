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
        thumb_cmc = hand_landmarks.landmark[1]
        
        # Determine if vertical or horizontal orientation
        if is_vertical:
            # For vertical hand, check if thumb tip is to the left or right of its base
            thumb_extended = (thumb_tip.x < thumb_ip.x) if wrist.x < middle_base.x else (thumb_tip.x > thumb_ip.x)
        else:
            # For horizontal hand, check if thumb is above or below its base
            thumb_extended = (thumb_tip.y < thumb_ip.y)
        
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
        
        # Debug thumb
        if debug and image is not None:
            img_h, img_w = image.shape[:2]
            thumb_tip_px = int(thumb_tip.x * img_w), int(thumb_tip.y * img_h)
            thumb_ip_px = int(thumb_ip.x * img_w), int(thumb_ip.y * img_h)
            
            cv2.circle(image, thumb_tip_px, 5, (0, 255, 255) if thumb_extended else (0, 0, 255), -1)
            cv2.circle(image, thumb_ip_px, 5, (255, 0, 0), -1)
            
            status = "OUT" if thumb_extended else "IN"
            draw_text_with_background(
                image, 
                f"Thumb: {status}", 
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
    """Detect ASL letters based on simplified finger positions"""
    try:
        if not hand_landmarks:
            return None

        thumb_extended, fingers_extended = get_finger_state(hand_landmarks, image, debug)
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
        
        # Calculate distances between fingertips
        thumb_index_distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        index_middle_distance = np.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)
        middle_ring_distance = np.sqrt((middle_tip.x - ring_tip.x)**2 + (middle_tip.y - ring_tip.y)**2)
        ring_pinky_distance = np.sqrt((ring_tip.x - pinky_tip.x)**2 + (ring_tip.y - pinky_tip.y)**2)
        
        # Get joint positions
        index_pip = hand_landmarks.landmark[6]
        middle_pip = hand_landmarks.landmark[10]
        ring_pip = hand_landmarks.landmark[14]
        pinky_pip = hand_landmarks.landmark[18]
        
        # Calculate fingertip-to-pip distances for bend detection
        index_bend = np.sqrt((index_tip.x - index_pip.x)**2 + (index_tip.y - index_pip.y)**2)
        middle_bend = np.sqrt((middle_tip.x - middle_pip.x)**2 + (middle_tip.y - middle_pip.y)**2)
        
        # Get bases
        index_base = hand_landmarks.landmark[5]
        pinky_base = hand_landmarks.landmark[17]
        
        # A - Thumb to side, fingers in a fist
        if thumb_extended and not any(fingers_extended):
            return 'A'
        
        # B - Fingers straight up, thumb across palm
        if all(fingers_extended) and not thumb_extended:
            # Fingers must be close together
            if index_middle_distance < 0.08 and middle_ring_distance < 0.08 and ring_pinky_distance < 0.08:
                return 'B'
        
        # C - Hand curved in C shape, thumb and fingers form opening
        if not any(fingers_extended) and thumb_extended:
            # Measure curvature and opening
            # C has fingers curved but not fully closed (partial bend)
            # Measure the distance from thumb to pinky
            thumb_pinky_distance = np.sqrt((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2)
            if 0.1 < thumb_pinky_distance < 0.3:
                return 'C'
        
        # D - Index up, others curled, thumb touches middle finger
        if index and not middle and not ring and not pinky and thumb_extended:
            # D requires thumb to touch middle finger
            thumb_middle_distance = np.sqrt((thumb_tip.x - middle_pip.x)**2 + (thumb_tip.y - middle_pip.y)**2)
            if thumb_middle_distance < 0.1:
                return 'D'
        
        # E - All fingers curled, thumb across palm
        if not any(fingers_extended) and not thumb_extended:
            return 'E'
        
        # F - Index and thumb touch, other fingers up
        if not index and middle and ring and pinky and thumb_extended:
            # Check if index and thumb are touching
            if thumb_index_distance < 0.1:
                return 'F'
        
        # G - Thumb and index pointing forward, fingers curled
        if index and not middle and not ring and not pinky and not thumb_extended:
            # For G, index points horizontally (sideways)
            index_horizontal = abs(index_tip.x - index_base.x) > abs(index_tip.y - index_base.y)
            if index_horizontal:
                return 'G'
        
        # H - Index and middle extended side by side horizontally
        if index and middle and not ring and not pinky:
            # Must be horizontal and fingers close together
            if not is_vertical and index_middle_distance < 0.08:
                return 'H'
        
        # I - Pinky up, others closed
        if not index and not middle and not ring and pinky and not thumb_extended:
            return 'I'
        
        # J - Like I but with a motion (can't detect motion in static images)
        # We can approximate J as I with thumb slightly extended
        if not index and not middle and not ring and pinky and thumb_extended:
            # Only if it's not identified as Y (which also has thumb and pinky out)
            # For J, the hand would typically be more rotated
            if not is_vertical:
                return 'J'
        
        # K - Index and middle up in V shape, thumb touches hand
        if index and middle and not ring and not pinky and thumb_extended:
            # K has index and middle spread in V shape with thumb touching between them
            if index_middle_distance > 0.08:
                thumb_middle_base_distance = np.sqrt((thumb_tip.x - middle_base.x)**2 + (thumb_tip.y - middle_base.y)**2)
                if thumb_middle_base_distance < 0.1:
                    return 'K'
        
        # L - L shape with thumb and index
        if index and not middle and not ring and not pinky and thumb_extended:
            # L has thumb and index at approx. 90 degrees
            v1 = np.array([index_tip.x - index_base.x, index_tip.y - index_base.y])
            v2 = np.array([thumb_tip.x - index_base.x, thumb_tip.y - index_base.y])
            angle = np.abs(np.cross(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            if angle > 0.7:  # Close to 90 degrees
                return 'L'
        
        # M - All fingers curled, thumb between fingers
        if not any(fingers_extended) and not thumb_extended:
            # Can't easily distinguish from E in static frame
            # Technically M has thumb tucked between fingers
            return 'M'
        
        # N - All fingers curled, thumb between fingers (different position)
        # N is very similar to M in static frame
        
        # O - Fingers and thumb form round O
        if not any(fingers_extended) and thumb_extended:
            # Detecting circular O shape
            # Check that thumb is closer to index than in C
            if thumb_index_distance < 0.1:
                return 'O'
        
        # P - Thumb out, index pointing down, others up
        if not index and middle and ring and pinky and thumb_extended:
            # P has index pointing horizontally
            if not is_vertical:
                return 'P'
        
        # Q - Similar to G but with hand pointing down
        if index and not middle and not ring and not pinky and not thumb_extended:
            # For Q, index points downward, which is hard to distinguish in static frame
            # We'll prioritize G for now, could add hand orientation checks
            pass
        
        # R - Cross fingers (index and middle crossed)
        if index and middle and not ring and not pinky:
            # R has index crossing over middle
            finger_crossed = (index_tip.x > middle_tip.x) != (wrist.x < middle_base.x)
            if finger_crossed:
                return 'R'
        
        # S - Fist with thumb in front of fingers
        if not any(fingers_extended) and not thumb_extended:
            # S is hard to distinguish from E and M in static frame
            # Technically S has thumb wrapped in front of fingers
            pass
        
        # T - Thumb between index and middle
        if not index and not middle and not ring and not pinky and thumb_extended:
            # T has thumb between index and middle fingers
            return 'T'
        
        # U - Index and middle together pointing up
        if index and middle and not ring and not pinky and not thumb_extended:
            # U has two fingers parallel and close
            if index_middle_distance < 0.08:
                return 'U'
        
        # V - Index and middle in V shape
        if index and middle and not ring and not pinky:
            # V has fingers spread
            if index_middle_distance > 0.08:
                # Make sure not K (which has thumb extended)
                if not thumb_extended:
                    return 'V'
                else:
                    # Could be V or K, check thumb position more precisely
                    thumb_middle_base_distance = np.sqrt((thumb_tip.x - middle_base.x)**2 + (thumb_tip.y - middle_base.y)**2)
                    if thumb_middle_base_distance > 0.1:
                        return 'V'
        
        # W - Index, middle, ring extended
        if index and middle and ring and not pinky:
            # W has three fingers spread
            if (index_middle_distance > 0.05) and (middle_ring_distance > 0.05):
                return 'W'
        
        # X - Index bent, like halfway between X and E
        if not any(fingers_extended):
            # X has index finger slightly bent
            # Hard to detect in static frame but we could check partial extension
            pass
        
        # Y - Thumb and pinky extended
        if not index and not middle and not ring and pinky and thumb_extended:
            # Y has thumb and pinky out, must be vertical
            if is_vertical:
                return 'Y'
        
        # Z - Requires motion (can't detect in static frame)
        # Z traces the shape of Z in the air
        
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