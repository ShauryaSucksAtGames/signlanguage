# ASL Sign Language Detector

A real-time American Sign Language (ASL) letter detection system using a Raspberry Pi and MediaPipe. This project uses computer vision to detect and display ASL letters in real-time.

## Features

- Real-time ASL letter detection
- Confidence level display for each detected letter
- Debug mode for detailed finger state visualization
- Optimized for Raspberry Pi performance
- Support for 19 static ASL letters

## Supported Letters

The system currently supports the following ASL letters:

- A: Fist with thumb to the side
- B: All fingers extended, thumb tucked
- C: Curved hand shape
- D: Index up, others down
- E: All fingers curled
- F: Index and thumb touching
- G: Index pointing at thumb
- H: Index and middle parallel
- I: Pinky up
- K: Index and middle up, spread
- L: L shape with thumb and index
- M: Thumb between ring and pinky
- P: Thumb between middle and ring fingers
- R: Crossed fingers
- T: Thumb between index and middle
- U: Index and middle parallel
- V: Index and middle spread
- W: Three fingers up
- Y: Thumb and pinky out

Notes:
- J and Z require motion and are not supported (this detector only handles static signs)
- N, O, and S have been removed from detection

## Requirements

- Raspberry Pi 4 (recommended)
- Raspberry Pi Camera Module
- Python 3.7+
- MediaPipe
- OpenCV
- Picamera

## Installation

1. First, follow the setup instructions in `RASPBERRY_PI_SETUP.md` to configure your Raspberry Pi.

2. Clone this repository:
```bash
git clone https://github.com/yourusername/signlanguage.git
cd signlanguage
```

3. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

4. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the detector:
```bash
python sign_language_detector.py
```

2. Press 'd' to toggle debug mode, which shows:
   - Finger states (extended/closed)
   - Confidence levels for all possible letters
   - Visual confidence bars

3. Press 'q' to quit the application.

## Performance Optimization

The code includes several optimizations for Raspberry Pi:
- Reduced camera resolution (640x480)
- Lower framerate (15 FPS)
- Efficient memory management
- Periodic garbage collection
- Optimized hand landmark detection

## Troubleshooting

If you encounter issues:
1. Ensure the camera is properly connected and enabled
2. Check that all dependencies are installed
3. Verify the virtual environment is activated
4. Try adjusting the camera position or lighting conditions

## License

This project is licensed under the MIT License - see the LICENSE file for details. 