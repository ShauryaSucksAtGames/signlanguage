# ASL Sign Language Detector

A real-time American Sign Language (ASL) detection system using computer vision and machine learning, designed to run on a Raspberry Pi with a camera module.

## Features

- Real-time hand tracking and ASL letter detection
- Support for 18 ASL letters: A, B, C, D, E, F, G, H, I, K, L, M, R, S, U, V, W, Y
- Debug mode for visualizing hand landmarks and finger states
- Optimized for Raspberry Pi performance

## Requirements

- Raspberry Pi (tested on Raspberry Pi 4)
- Raspberry Pi Camera Module
- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- picamera

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/signlanguage.git
cd signlanguage
```

2. Install required packages:
```bash
pip3 install opencv-python mediapipe numpy picamera "picamera[array]" psutil
```

## Usage

1. Connect your Raspberry Pi camera module and enable it in raspi-config

2. Run the detector:
```bash
python3 sign_language_detector.py
```

3. Controls:
   - Press 'd' to toggle debug mode (shows finger states and confidence levels)
   - Press 'q' to quit the application

## Supported Letters

The system currently detects the following ASL letters:
- A: Closed fist with thumb to the side
- B: All fingers extended upward
- C: Curved hand shape
- D: Index finger up, others closed
- E: All fingers closed
- F: Three fingers extended
- G: Index and thumb pointing
- H: Two fingers pointing sideways
- I: Pinky extended
- K: Index and middle finger in 'V' with thumb between
- L: L-shape with thumb and index finger
- M: Closed fingers over thumb
- R: Crossed fingers
- S: Closed fist with thumb over fingers
- U: Two fingers extended together
- V: V-shape with index and middle finger
- W: Three fingers extended in W shape
- Y: Thumb and pinky extended

## Debug Mode

Debug mode provides visual feedback including:
- Hand landmark points
- Finger state indicators (CLOSED, EXTENDED, HALF BENT, PARTIALLY BENT)
- Confidence scores for each detected letter
- Hand orientation (Vertical/Horizontal)

## Performance Notes

- The system is optimized for Raspberry Pi performance with automatic process priority adjustment
- Frame rate is set to 15 FPS for optimal balance between performance and accuracy
- Garbage collection is performed periodically to manage memory usage

## Known Limitations

- Requires good lighting conditions
- Hand should be clearly visible in frame
- Best results when hand is oriented vertically
- Detection accuracy may vary based on hand position and lighting

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements 