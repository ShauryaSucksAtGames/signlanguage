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
    Detect ASL letters M, N, O, S, T based on hand landmarks
    Returns: (detected_letter, confidence_level)
    """
    letter_scores = {
        'M': 0.3,
        'N': 0.3,
        'O': 0.3,
        'S': 0.3,
        'T': 0.3
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
    
    # N
    if fingers.count(0) == 4:
        # Thumb between index and middle
        thumb_tip = posList[4]
        index_base = posList[6]
        middle_base = posList[10]
        
        thumb_between_index_middle = (
            thumb_tip[1] > index_base[1] and
            thumb_tip[1] < middle_base[1] and
            thumb_tip[2] > index_base[2]
        )
        
        letter_scores['N'] = 0.95 if thumb_between_index_middle else 0.3
    
    # O
    # Get key points
    thumb_tip = posList[4]
    index_tip = posList[8]
    middle_tip = posList[12]
    ring_tip = posList[16]
    pinky_tip = posList[20]
    
    # Get base joints
    thumb_base = posList[2]
    index_base = posList[5]
    
    # Check alignment between each pair of adjacent fingers
    alignments = [
        abs(index_tip[0] - middle_tip[0]) < 50,  # index-middle
        abs(middle_tip[0] - ring_tip[0]) < 50,   # middle-ring
        abs(ring_tip[0] - pinky_tip[0]) < 50,    # ring-pinky
    ]
    
    # Count how many adjacent fingers are aligned
    aligned_count = sum(1 for aligned in alignments if aligned)
    tips_aligned = aligned_count >= 2  # At least 3 fingers aligned (needs 2 adjacent pairs)
    
    # Thumb should be meeting middle finger
    thumb_middle_meeting = (
        abs(thumb_tip[0] - middle_tip[0]) < 60 and  # x-coordinate
        abs(thumb_tip[1] - middle_tip[1]) < 60      # y-coordinate
    )
    
    # Tips should be higher than bases (curved forward)
    fingers_curved = (
        thumb_tip[2] < thumb_base[2] and
        index_tip[2] < index_base[2]
    )
    
    letter_scores['O'] = 0.95 if (tips_aligned and thumb_middle_meeting and fingers_curved) else 0.3
    
    # S
    if fingers.count(0) == 4:
        # Thumb wrapped over closed fist
        thumb_tip = posList[4]
        middle_tip = posList[12]
        index_base = posList[6]
        
        thumb_wrapped = (
            thumb_tip[1] > middle_tip[1] and  # Thumb past middle finger
            thumb_tip[2] < middle_tip[2] and  # Thumb above middle finger
            thumb_tip[2] > index_base[2]      # But below index base
        )
        
        letter_scores['S'] = 0.95 if thumb_wrapped else 0.3
    
    # T
    if fingers.count(0) == 4:
        # Thumb between index and middle, more vertical
        thumb_tip = posList[4]
        thumb_ip = posList[3]
        index_base = posList[6]
        middle_base = posList[10]
        
        thumb_position = (
            thumb_tip[1] > index_base[1] and
            thumb_tip[1] < middle_base[1] and
            thumb_tip[2] < thumb_ip[2] and    # Thumb pointing more upward
            abs(thumb_tip[2] - index_base[2]) < 30  # Close to index height
        )
        
        letter_scores['T'] = 0.95 if thumb_position else 0.3
    
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
    
    print("Sign Language Detector (M,N,O,S,T) Started")
    print("Press 'q' to quit, 'd' to toggle debug mode")
    
    # Create window
    cv2.namedWindow('Sign Language Detector (M,N,O,S,T)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sign Language Detector (M,N,O,S,T)', 640, 480)
    
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
                "ASL Sign Language Detector (M,N,O,S,T)",
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
                    
                    # Display all letter confidences
                    y_offset = 0
                    for l, score in sorted([('M', confidence if letter == 'M' else 0.3),
                                         ('N', confidence if letter == 'N' else 0.3),
                                         ('O', confidence if letter == 'O' else 0.3),
                                         ('S', confidence if letter == 'S' else 0.3),
                                         ('T', confidence if letter == 'T' else 0.3)],
                                        key=lambda x: x[1], reverse=True):
                        color = (0, 255, 0) if score >= 0.95 else (255, 165, 0) if score >= 0.9 else (128, 128, 128)
                        draw_text_with_background(
                            image,
                            f"{l}: {score*100:.1f}%",
                            (image.shape[1] - 150, 50 + y_offset),
                            color=color
                        )
                        y_offset += 30
            
            # Display the frame
            cv2.imshow('Sign Language Detector (M,N,O,S,T)', image)
            
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