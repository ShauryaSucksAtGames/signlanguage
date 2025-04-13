#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
import time
import gc
from picamera import PiCamera
from picamera.array import PiRGBArray

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

def detect_letter(posList, fingers):
    """
    Detect ASL letter M based on hand landmarks
    Returns: (detected_letter, confidence_level)
    """
    letter_scores = {
        'M': 0.3
    }
    
    # M
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
    
    # Find best match
    best_letter = max(letter_scores.items(), key=lambda x: x[1])
    return best_letter[0], best_letter[1]

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
    
    print("Sign Language Detector (M) Started")
    print("Press 'q' to quit, 'd' to toggle debug mode")
    
    # Create window
    cv2.namedWindow('Sign Language Detector (M)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sign Language Detector (M)', 640, 480)
    
    try:
        for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            image = frame.array
            frame_counter += 1
            
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = hands.process(rgb_image)
            
            # Add title
            draw_text_with_background(
                image,
                "ASL Sign Language Detector (M)",
                (image.shape[1]//2 - 150, 25),
                font_scale=0.7,
                color=(255, 0, 0)
            )
            
            # Add debug message
            if debug_mode:
                draw_text_with_background(image, "DEBUG MODE ON", (10, 20), color=(0, 0, 255))
                draw_text_with_background(image, "Press 'd' to toggle debug", (10, image.shape[0] - 20), color=(0, 0, 255))
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Get landmark positions
                    h, w, _ = image.shape
                    posList = []
                    for landmark in hand_landmarks.landmark:
                        posList.append([
                            int(landmark.x * w),
                            int(landmark.y * h),
                            int(landmark.z * w)
                        ])
                    
                    # Detect finger states
                    fingers = []
                    if posList[8][1] > posList[6][1]:  # Index
                        fingers.append(0)
                    else:
                        fingers.append(1)
                    if posList[12][1] > posList[10][1]:  # Middle
                        fingers.append(0)
                    else:
                        fingers.append(1)
                    if posList[16][1] > posList[14][1]:  # Ring
                        fingers.append(0)
                    else:
                        fingers.append(1)
                    if posList[20][1] > posList[18][1]:  # Pinky
                        fingers.append(0)
                    else:
                        fingers.append(1)
                    
                    # Display finger states in debug mode
                    if debug_mode:
                        states = ["Index:", "Middle:", "Ring:", "Pinky:"]
                        for i, (state, name) in enumerate(zip(fingers, states)):
                            status = "CLOSED" if state == 0 else "EXTENDED"
                            color = (0, 0, 255) if state == 0 else (0, 255, 0)
                            draw_text_with_background(
                                image,
                                f"{name} {status}",
                                (10, 50 + i * 30),
                                color=color
                            )
                    
                    # Detect letter
                    letter, confidence = detect_letter(posList, fingers)
                    
                    # Display M confidence
                    color = (0, 255, 0) if confidence >= 0.95 else (255, 165, 0) if confidence >= 0.9 else (128, 128, 128)
                    draw_text_with_background(
                        image,
                        f"M: {confidence*100:.1f}%",
                        (image.shape[1] - 150, 50),
                        color=color
                    )
            
            # Display the frame
            cv2.imshow('Sign Language Detector (M)', image)
            
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