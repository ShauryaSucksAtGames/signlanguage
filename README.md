# Raspberry Pi Sign Language Detector

This project implements real-time sign language alphabet detection using a Raspberry Pi 4 and Pi Camera. It uses MediaPipe Hands for hand tracking and OpenCV for image processing.

## Hardware Requirements
- Raspberry Pi 4 (2GB RAM)
- Pi Camera v1.3
- Raspberry Pi OS (Legacy) - Buster 32-bit

## Software Requirements
- Python 3.7+
- OpenCV
- MediaPipe
- PiCamera module

## Installation

1. First, ensure your Raspberry Pi is running Raspberry Pi OS (Legacy) - Buster 32-bit.

2. Install the required packages:
```bash
pip3 install -r requirements.txt
```

3. Enable the camera interface:
```bash
sudo raspi-config
```
Navigate to "Interface Options" > "Camera" and enable it.

4. Reboot your Raspberry Pi:
```bash
sudo reboot
```

## Usage

1. Run the script:
```bash
python3 sign_language_detector.py
```

2. The program will:
   - Open your Pi Camera feed
   - Detect hand landmarks in real-time
   - Display the detected sign language letter on the right side of the screen
   - Show hand tracking visualization

3. Press 'q' to quit the program

## Notes
- The current implementation includes basic detection for letters 'A' and 'B'
- You can expand the `LETTER_MAPPINGS` dictionary and `detect_letter()` function to include more letters
- The detection is based on hand landmark positions and angles
- For best results, ensure good lighting and keep your hand clearly visible to the camera

## Performance Tips
- The script is optimized for Raspberry Pi 4 with 2GB RAM
- If you experience performance issues, you can:
  - Reduce the camera resolution in the script
  - Lower the framerate
  - Adjust the detection confidence thresholds 