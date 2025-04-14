# ASL Sign Language Detector for Raspberry Pi 4

A real-time American Sign Language (ASL) detector optimized for Raspberry Pi 4 with 2GB RAM running 32-bit Raspberry Pi OS (Buster). This application uses MediaPipe for hand tracking and rule-based detection to recognize ASL letters.

## Overview

This project combines pre-trained machine learning with rule-based recognition to efficiently detect American Sign Language (ASL) hand signs on resource-constrained Raspberry Pi hardware. It focuses on static fingerspelling letters and prioritizes real-time performance.

### How It Works

1. **Hand Detection & Tracking**: Uses MediaPipe's pre-trained ML model to detect 21 landmarks on the hand
2. **Feature Extraction**: Analyzes finger positions, orientations, and relationships
3. **Rule-Based Detection**: Applies customized geometric rules for each ASL letter
4. **Real-time Visualization**: Displays the detected sign with confidence level

## Supported ASL Signs

This detector recognizes the following ASL letters:
- A, B, C, D, E, F, G, H, I, K, L, O, R, S, T, U, V, W, Y

Notes:
- J and Z require motion and are not supported (this detector only handles static signs)
- Q, P, M, and N have been removed from detection

## Getting Started

### Prerequisites

Before proceeding with this guide, follow the instructions in [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md) to:
1. Set up your Raspberry Pi with the correct OS
2. Configure networking and camera
3. Optimize the Pi's performance
4. Clone this repository and create a virtual environment

### Python Dependencies Installation

With your virtual environment activated, install dependencies in this specific order:

```bash
# First install numpy (must be installed BEFORE other packages)
pip3 install numpy==1.17.3

# Install matplotlib (with specific version compatible with numpy 1.17.3)
pip3 install matplotlib==3.3.4

# Install protobuf (required for MediaPipe)
pip3 install protobuf==3.20.0

# Install opencv-headless (specific version)
pip3 install opencv-python-headless==3.4.3.18

# Install remaining dependencies
pip3 install -r requirements.txt
```

### Running the ASL Detector

To start the sign language detector:

```bash
# Activate the virtual environment
source sign_env/bin/activate

# Run the detector
python3 sign_language_detector.py
```

### Key Controls
- Press 'q' to quit the program
- Press 'd' to toggle debug mode on/off

## Debug Mode

Debug mode displays extensive information to help improve sign detection:
- **Finger Status**: Shows each finger's state (EXTENDED, CLOSED, PARTIALLY BENT, HALF BENT)
- **Thumb Position**: Shows if thumb is IN or OUT
- **Hand Orientation**: Shows if the hand is in Vertical or Horizontal orientation
- **Visual Markers**: Highlights the detected hand landmarks with colored points and lines

## Sign Detection Guidelines

For optimal detection accuracy:

1. **Lighting**: Use consistent, well-lit environments
2. **Hand Position**: Keep your hand about 12-24 inches from the camera
3. **Orientation**: Match the expected orientation for each sign:
   - Most letters: Vertical orientation
   - G, H: Horizontal orientation
4. **Clear Formation**: Make deliberate, clear hand shapes
5. **Fingers**: Pay attention to proper finger positioning:
   - For 'Y': Clearly extend thumb and pinky outward
   - For 'I': Keep pinky extended, thumb tucked, other fingers closed
   - For 'R': Keep index and middle fingers extended but close together

## Troubleshooting

If you encounter issues with sign detection:

1. **Enable Debug Mode**: Press 'd' to see finger states and hand orientation
2. **Check Lighting**: Ensure consistent, adequate lighting on your hand
3. **Adjust Hand Position**: Try different distances and angles
4. **Clear Hand Shapes**: Make deliberate, clear hand shapes for each sign
5. **Camera Focus**: Ensure the camera can clearly see your hand details

For system-related issues, refer to the Troubleshooting section in [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md).

## Technical Details

See [DOCUMENTATION.md](DOCUMENTATION.md) for detailed technical information about:
- Detection algorithms and implementation
- System architecture
- Performance considerations
- Future development possibilities

## Limitations

1. **Static Signs Only**: Only detects non-moving signs (excludes J, Z which require motion)
2. **Single Hand**: Only one hand is detected at a time
3. **Lighting Dependent**: Performance degrades in poor lighting
4. **Frame Rate**: Real-time but limited to ~15fps on Raspberry Pi 4
5. **Similar Signs**: Some visually similar signs (like K/V, H/U) may be confused

## Credits

This project uses:
- **MediaPipe**: Google's ML framework for hand tracking
- **OpenCV**: For image processing and visualization
- **NumPy**: For numerical operations
- **PiCamera**: For Raspberry Pi camera integration 