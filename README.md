# Raspberry Pi Sign Language Detector

This project implements real-time sign language alphabet detection using a Raspberry Pi 4 and Pi Camera. It uses MediaPipe Hands for hand tracking and OpenCV headless for image processing.

## Hardware Requirements
- Raspberry Pi 4 (2GB RAM)
- Pi Camera v1.3
- Raspberry Pi OS (Legacy) - Buster 32-bit

## Software Requirements
- Python 3.7+
- OpenCV 3.4.3.18 (headless)
- MediaPipe for Raspberry Pi
- PiCamera module
- Flask (for web interface)

## Installation

1. First, ensure your Raspberry Pi is running Raspberry Pi OS (Legacy) - Buster 32-bit.

2. Install system dependencies:
```bash
sudo apt-get update
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

# Remove existing venv if any
rm -rf virt

# Create new venv
python3 -m venv virt
source virt/bin/activate

# Upgrade pip
pip3 install --upgrade pip setuptools wheel
```

4. Install Python dependencies (EXACT ORDER IS IMPORTANT):
```bash
# First install numpy
pip3 install numpy==1.17.3

# Install protobuf (required for mediapipe)
pip3 install protobuf==3.20.0

# Install matplotlib (required by mediapipe)
pip3 install matplotlib==3.3.4

# Install OpenCV headless
pip3 install opencv-python-headless==3.4.3.18

# Install remaining dependencies
pip3 install -r requirements.txt
```

5. Enable the camera interface:
```bash
sudo raspi-config
```
Navigate to "Interface Options" > "Camera" and enable it.

6. Reboot your Raspberry Pi:
```bash
sudo reboot
```

## Usage

### VNC Version (Local Display)
1. Activate virtual environment:
```bash
source virt/bin/activate
```

2. Run the script:
```bash
python3 sign_language_detector.py
```

3. The program will:
   - Open your Pi Camera feed in a window
   - Detect hand landmarks in real-time
   - Display the detected sign language letter
   - Show hand tracking visualization

4. Press 'q' to quit the program

### Web Interface Version (Remote Access)
1. Activate virtual environment:
```bash
source virt/bin/activate
```

2. Run the web interface:
```bash
python3 web_sign_language_detector.py
```

3. Access the web interface:
   - Open a web browser on any device connected to your network
   - Navigate to `http://<raspberry-pi-ip>:5000`
   - Replace `<raspberry-pi-ip>` with your Raspberry Pi's IP address

4. Press Ctrl+C in the terminal to quit the program

## Notes
- The current implementation includes basic detection for letters A, B, C, and D
- The detection is based on hand landmark positions and angles
- For best results:
  - Ensure good lighting
  - Keep your hand clearly visible to the camera
  - Position your hand at a comfortable distance from the camera

## Performance Tips
- The script is optimized for Raspberry Pi 4 with 2GB RAM
- If you experience performance issues:
  - The camera resolution is already set to 320x240 for optimal performance
  - The framerate is set to 15 FPS to balance performance and smoothness
  - The JPEG quality is set to 80% for efficient streaming
  - Garbage collection runs every 30 frames to manage memory
- For better performance:
  - Close other applications while running this program
  - Ensure your Raspberry Pi is well-ventilated
  - Consider using a USB fan for extended use

## Troubleshooting

### Common Installation Issues
- If you get numpy/matplotlib errors:
  ```bash
  # Deactivate and remove virtual environment
  deactivate
  rm -rf virt
  
  # Create new virtual environment
  python3 -m venv virt
  source virt/bin/activate
  
  # Install dependencies in correct order
  pip3 install --upgrade pip setuptools wheel
  pip3 install numpy==1.17.3
  pip3 install protobuf==3.20.0
  pip3 install matplotlib==3.3.4
  pip3 install opencv-python-headless==3.4.3.18
  pip3 install -r requirements.txt
  ```

- If you get "module compiled against API version" error:
  - Follow the installation steps in EXACT order
  - Make sure to install numpy FIRST
  - Then install matplotlib BEFORE other packages
  - Clean and recreate virtual environment if needed

### Other Issues
- If you get other dependency errors:
  - Make sure you've installed all system dependencies listed above
  - Try running `sudo apt-get update` and `sudo apt-get upgrade`
  - Check if your Python virtual environment is activated
- If the camera is not detected:
  - Check if the camera is properly connected
  - Verify camera is enabled in raspi-config
  - Try rebooting the Raspberry Pi
- If the web interface is not accessible:
  - Check if the Raspberry Pi is connected to the network
  - Verify the IP address shown in the terminal
  - Make sure no firewall is blocking port 5000