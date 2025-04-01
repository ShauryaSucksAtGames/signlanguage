import cv2
import mediapipe as mp
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import argparse
import gc
import psutil
import os

# Add command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--headless', action='store_true', help='Run in headless mode without display')
parser.add_argument('--memory-limit', type=int, default=80, help='Memory usage limit percentage')
args = parser.parse_args()

def check_memory_usage():
    """Check if memory usage is above limit"""
    memory_percent = psutil.Process(os.getpid()).memory_percent()
    if memory_percent > args.memory_limit:
        print(f"\nWarning: High memory usage ({memory_percent:.1f}%)")
        gc.collect()  # Force garbage collection
        return True
    return False

# Initialize MediaPipe Hands with optimized settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,  # Lowered for better performance
    min_tracking_confidence=0.3,   # Lowered for better performance
    model_complexity=0  # Use lighter model
)
mp_draw = mp.solutions.drawing_utils

try:
    # Initialize the camera with lower resolution
    camera = PiCamera()
    camera.resolution = (320, 240)  # Reduced resolution
    camera.framerate = 15          # Reduced framerate
    raw_capture = PiRGBArray(camera, size=(320, 240))

    # Allow the camera to warm up
    time.sleep(1)

    print("Starting sign language detection...")
    print("Press Ctrl+C to exit")
    
    # Frame counter for skipping frames
    frame_counter = 0
    last_gc_time = time.time()

    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        current_time = time.time()
        
        # Periodic memory check and cleanup (every 30 seconds)
        if current_time - last_gc_time > 30:
            if check_memory_usage():
                last_gc_time = current_time

        # Skip every other frame for better performance
        frame_counter += 1
        if frame_counter % 2 != 0:
            raw_capture.truncate(0)
            continue

        # Get the frame
        image = frame.array
        
        # Convert the BGR image to RGB (smaller image = faster conversion)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            # Process the image and detect hands
            results = hands.process(rgb_image)
            
            # Draw hand landmarks and detect letter
            detected_letter = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    detected_letter = detect_letter(hand_landmarks)
                    if detected_letter:
                        if not args.headless:
                            mp_draw.draw_landmarks(
                                image, 
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS
                            )
            
            # Display the detected letter
            if detected_letter:
                print(f"Detected Letter: {detected_letter}", end='\r')
                if not args.headless:
                    cv2.putText(
                        image,
                        f"Letter: {detected_letter}",
                        (250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
            
            # Display the frame if not in headless mode
            if not args.headless:
                cv2.imshow("Sign Language Detector", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"\nError processing frame: {str(e)}")
            continue
        
        finally:
            # Clear the stream in preparation for the next frame
            raw_capture.truncate(0)
            # Clear unused variables
            del rgb_image
            if 'results' in locals():
                del results

except KeyboardInterrupt:
    print("\nStopping sign language detection...")

except Exception as e:
    print(f"\nFatal error: {str(e)}")

finally:
    # Clean up
    try:
        camera.close()
    except:
        pass
    
    if not args.headless:
        cv2.destroyAllWindows()
    
    try:
        hands.close()
    except:
        pass
    
    # Final garbage collection
    gc.collect()
    print("Cleanup complete")

def get_finger_state(hand_landmarks):
    """Determine if each finger is extended or not - optimized version"""
    # Use only essential landmarks
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    finger_bases = [5, 9, 13, 17]  # Index, Middle, Ring, Pinky bases
    
    # Check thumb (simplified)
    thumb_extended = hand_landmarks.landmark[4].x > hand_landmarks.landmark[2].x
    
    # Check other fingers (simplified)
    fingers_extended = []
    for tip, base in zip(finger_tips, finger_bases):
        fingers_extended.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y)
    
    return thumb_extended, fingers_extended

def detect_letter(hand_landmarks):
    if not hand_landmarks:
        return None

    thumb_extended, fingers_extended = get_finger_state(hand_landmarks)
    index, middle, ring, pinky = fingers_extended

    # Simplified detection logic with fewer checks
    # A: Thumb up, other fingers down
    if thumb_extended and not any(fingers_extended):
        return 'A'

    # B: All fingers up
    if thumb_extended and all(fingers_extended):
        return 'B'

    # C: Curved hand (simplified check)
    if thumb_extended and all(fingers_extended):
        if hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y * 0.8:
            return 'C'

    # D: Index finger up
    if thumb_extended and index and not middle and not ring and not pinky:
        return 'D'

    # E: All fingers down
    if not thumb_extended and not any(fingers_extended):
        return 'E'

    # F: All fingers up and spread (simplified check)
    if thumb_extended and all(fingers_extended):
        if (hand_landmarks.landmark[8].x - hand_landmarks.landmark[20].x) > 0.15:
            return 'F'

    # G: Index finger pointing
    if thumb_extended and index and not middle and not ring and not pinky:
        return 'G'

    # H: Index and middle fingers up
    if thumb_extended and index and middle and not ring and not pinky:
        return 'H'

    # I: Pinky up
    if thumb_extended and not index and not middle and not ring and pinky:
        return 'I'

    # K: Index and middle fingers spread
    if thumb_extended and index and middle and not ring and not pinky:
        if (hand_landmarks.landmark[12].x - hand_landmarks.landmark[8].x) > 0.08:
            return 'K'

    # L: Index and thumb up
    if thumb_extended and index and not middle and not ring and not pinky:
        return 'L'

    # O: All fingers curved (simplified check)
    if thumb_extended and all(fingers_extended):
        if hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y * 0.8:
            return 'O'

    # R: Index and middle fingers crossed
    if thumb_extended and index and middle and not ring and not pinky:
        if hand_landmarks.landmark[12].x < hand_landmarks.landmark[8].x:
            return 'R'

    # S: Fist
    if not thumb_extended and not any(fingers_extended):
        return 'S'

    # T: Thumb between index and middle
    if thumb_extended and index and middle and not ring and not pinky:
        return 'T'

    # U: Index and middle fingers up together
    if thumb_extended and index and middle and not ring and not pinky:
        if (hand_landmarks.landmark[12].x - hand_landmarks.landmark[8].x) < 0.08:
            return 'U'

    # V: Index and middle fingers spread
    if thumb_extended and index and middle and not ring and not pinky:
        if (hand_landmarks.landmark[12].x - hand_landmarks.landmark[8].x) > 0.08:
            return 'V'

    # W: Three fingers up
    if thumb_extended and index and middle and ring and not pinky:
        return 'W'

    # X: Index finger bent
    if thumb_extended and index and not middle and not ring and not pinky:
        if hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y * 0.8:
            return 'X'

    # Y: Thumb and pinky out
    if thumb_extended and not index and not middle and not ring and pinky:
        return 'Y'

    return None 