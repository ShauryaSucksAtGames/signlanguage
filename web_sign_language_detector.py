from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import threading
from queue import Queue
import json
import socket
import traceback
import gc
import psutil
import os

app = Flask(__name__)

# Global variables for frame sharing
frame_queue = Queue(maxsize=1)  # Reduced queue size
latest_letter = None
camera = None
raw_capture = None
error_message = None
frame_counter = 0  # Added frame counter for garbage collection

# Set process priority
try:
    p = psutil.Process(os.getpid())
    p.nice(-10)  # Set high priority
except:
    pass

def get_ip_address():
    """Get the local IP address of the Raspberry Pi"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception as e:
        print(f"Warning: Could not get IP address: {str(e)}")
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

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

def process_frames():
    """Background thread for processing camera frames"""
    global latest_letter, camera, raw_capture, error_message, frame_counter
    
    # Initialize MediaPipe with optimized settings for 0.8.8
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3
    )

    while True:  # Main loop to handle camera reconnection
        try:
            # Initialize camera with lower resolution
            if camera is None:
                camera = PiCamera()
                camera.resolution = (320, 240)  # Lower resolution for better performance
                camera.framerate = 15  # Reduced framerate
                camera.exposure_mode = 'night'  # Better for indoor use
                camera.awb_mode = 'auto'
                raw_capture = PiRGBArray(camera, size=(320, 240))
                time.sleep(1)  # Camera warm-up
                print("Camera initialized successfully")

            for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
                try:
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
                    
                    # Update latest letter
                    latest_letter = detected_letter
                    
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
                    
                    # Put frame in queue, remove old frame if queue is full
                    if frame_queue.full():
                        frame_queue.get()
                    frame_queue.put(image)
                    
                    # Clear the stream
                    raw_capture.truncate(0)
                    raw_capture.seek(0)  # Added for compatibility
                    
                    # Force garbage collection less frequently
                    if frame_counter % 30 == 0:  # Reduced frequency
                        gc.collect()
                    
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    raw_capture.truncate(0)
                    raw_capture.seek(0)  # Added for compatibility
                    continue
                
        except Exception as e:
            print(f"Camera error: {str(e)}")
            print("Attempting to reconnect camera in 5 seconds...")
            try:
                if camera:
                    camera.close()
                    camera = None
                if raw_capture:
                    raw_capture = None
            except:
                pass
            time.sleep(5)
            continue

def generate_frames():
    """Generator function for streaming frames"""
    while True:
        try:
            if not frame_queue.empty():
                frame = frame_queue.get()
                # Use OpenCV 3.4.3.18 compatible encoding with numpy 1.17.3
                frame = frame.astype(np.uint8)  # Ensure correct data type
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.02)  # Increased sleep time to reduce CPU usage
        except Exception as e:
            print(f"Error generating frames: {str(e)}")
            time.sleep(0.2)
            continue

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_letter')
def get_letter():
    """Return the latest detected letter"""
    return json.dumps({'letter': latest_letter if latest_letter else ''})

@app.route('/get_status')
def get_status():
    """Return the current status and any error messages"""
    return json.dumps({
        'status': 'running',
        'error': error_message if error_message else ''
    })

if __name__ == '__main__':
    try:
        # Start frame processing in background thread
        processing_thread = threading.Thread(target=process_frames)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Get local IP address
        ip_address = get_ip_address()
        print(f"\nAccess the web interface at: http://{ip_address}:5000")
        print("Press Ctrl+C to exit\n")
        
        # Run Flask app with reduced threads
        app.run(host='0.0.0.0', port=5000, threaded=False)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print(traceback.format_exc())
    finally:
        try:
            if camera:
                camera.close()
        except:
            pass 