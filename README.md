# Sign Language Detector for Raspberry Pi 4

A sign language detector optimized for Raspberry Pi 4 with 2GB RAM running 32-bit Raspberry Pi OS (Buster). This application uses MediaPipe for accurate hand tracking and sign language detection.

## Hardware Requirements
- Raspberry Pi 4 (2GB RAM)
- Pi Camera Module v1.3 or v2.0
- Display (for direct viewing)

## System Requirements
- Raspberry Pi OS (Legacy) - Buster 32-bit

## Installation

1. Update your system:
```bash
sudo apt-get update
sudo apt-get upgrade
```

2. Install required system dependencies:
```bash
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y libjasper1
sudo apt-get install -y libqtgui4
sudo apt-get install -y libqt4-test
sudo apt-get install -y libhdf5-dev
sudo apt-get install -y libhdf5-serial-dev
sudo apt-get install -y libharfbuzz0b
sudo apt-get install -y libwebp6
sudo apt-get install -y libtiff5
sudo apt-get install -y libjasper-dev
sudo apt-get install -y libilmbase23
sudo apt-get install -y libopenexr23
sudo apt-get install -y libgstreamer1.0-0
sudo apt-get install -y libavcodec58
sudo apt-get install -y libavformat58
sudo apt-get install -y libswscale5
```

3. Set up Python virtual environment:
```bash
# Install venv
sudo apt-get install -y python3-venv

# Create virtual environment
python3 -m venv sign_env
source sign_env/bin/activate

# Upgrade pip
pip3 install --upgrade pip setuptools wheel
```

4. Install Python dependencies (CRITICAL: follow this exact order):
```bash
# Upgrade pip, setuptools, and wheel
pip3 install --upgrade pip setuptools wheel

# First install numpy (must be installed BEFORE other packages)
pip3 install numpy==1.17.3

# Install matplotlib (with specific version compatible with numpy 1.17.3)
pip3 install matplotlib==3.3.4

# Install protobuf (required for MediaPipe)
pip3 install protobuf==3.20.0

# Install opencv-headless (specific version)
pip3 install opencv-python-headless==3.4.3.18

# Install remaining dependencies from requirements.txt
pip3 install -r requirements.txt
```

5. Enable the camera interface:
```bash
sudo raspi-config
```
Navigate to "Interface Options" > "Camera" and enable it.

6. Reboot your Pi:
```bash
sudo reboot
```

## Usage

Run the sign language detector:
```bash
source sign_env/bin/activate
python3 sign_language_detector.py
```

The program will:
- Open a window showing the camera feed
- Track your hand using MediaPipe
- Recognize sign language gestures

### Key Controls
- Press 'q' to quit the program
- Press 'd' to toggle debug mode on/off

### Debug Mode
Debug mode shows:
- The status of each finger (UP/DOWN)
- Hand orientation (Vertical/Horizontal)
- Highlights finger tips and bases
- Helps you understand why signs may not be recognized correctly

## Features
- MediaPipe hand tracking for accurate gesture recognition
- Real-time processing optimized for Raspberry Pi
- Low memory usage
- Simple and intuitive interface
- Debug mode for troubleshooting

## Troubleshooting

If you encounter errors:

1. Numpy/MediaPipe errors:
   - Make sure to install dependencies in the exact order shown above
   - Try removing the virtual environment and creating a new one

2. Camera issues:
   - Ensure the camera is properly connected and enabled in raspi-config
   - Check for sufficient lighting when using the detector

3. Display issues:
   - If using over SSH, make sure X11 forwarding is enabled
   - For direct display, ensure your display is properly connected

4. Recognition accuracy issues:
   - Use debug mode ('d' key) to see the finger detection status
   - Adjust your hand position based on the debug information
   - Try different lighting conditions
   - Make clear and distinct hand shapes

## Notes
This application is specifically optimized for Raspberry Pi 4 with 2GB RAM. It uses MediaPipe's hand tracking which is efficient enough to run on this hardware while providing accurate hand gesture recognition. 