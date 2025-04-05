# Raspberry Pi Initial Setup Guide

This guide walks you through the complete setup process for a Raspberry Pi 4 before installing the ASL Sign Language Detector. After completing these steps, you'll be ready to proceed with the specific installation instructions in the README.md.

## Table of Contents
1. [Hardware Requirements](#hardware-requirements)
2. [Operating System Installation](#operating-system-installation)
3. [Network and Software Setup](#network-and-software-setup)
4. [Camera Configuration](#camera-configuration)
5. [Performance Optimization](#performance-optimization)
6. [Project Setup](#project-setup)
7. [Troubleshooting](#troubleshooting)

## Hardware Requirements

You'll need the following components:
- Raspberry Pi 4 (2GB RAM recommended)
- microSD card (16GB or larger, Class 10 recommended)
- Raspberry Pi Camera Module v1.3 or v2
- Power supply (official 5.1V/3A USB-C recommended)
- HDMI cable and compatible display (optional if using headless setup)
- USB keyboard and mouse (optional if using headless setup)
- Case with cooling (recommended but optional)

## Operating System Installation

### 1. Download and Install Raspberry Pi OS Buster (Legacy)

The ASL Sign Language Detector is optimized for Raspberry Pi OS Buster (32-bit legacy):

1. Download the Raspberry Pi Imager from: https://www.raspberrypi.org/software/
2. Install and launch the Raspberry Pi Imager
3. Click "CHOOSE OS"
4. Select "Raspberry Pi OS (other)"
5. Select "Raspberry Pi OS (Legacy)"
6. Ensure you select the 32-bit version

### 2. Configure and Write the OS

1. Insert your microSD card into your computer
2. In Pi Imager, click "CHOOSE STORAGE" and select your microSD card
3. **Before clicking "WRITE"**, click the gear icon (⚙️) in the bottom right corner to access advanced options
4. In the advanced options menu:
   - Check "Set hostname" (e.g., "signlanguagepi")
   - Check "Enable SSH" and select "Use password authentication"
   - Check "Set username and password" (create your credentials)
   - Check "Configure wireless LAN" and enter your WiFi SSID and password
   - Select your country/region from the list
5. Click "SAVE" to apply these settings
6. Now click "WRITE" and confirm
7. Wait for the process to complete
8. When finished, eject the card safely

### 3. First Boot

1. Insert the microSD card into your Raspberry Pi
2. Connect the camera module (see Camera Configuration section below)
3. Connect power to boot your Pi
4. Wait about 2 minutes for the Pi to boot and connect to WiFi
5. Find your Pi's IP address using one of these methods:
   - Check your router's connected devices list
   - Use a network scanner app like "Fing" on your phone
   - Try connecting to `raspberrypi.local` (or your custom hostname if set) if your computer supports mDNS

## Network and Software Setup

### 1. Connect via SSH

1. On your computer, open a terminal/command prompt
2. Connect via SSH using the username you set in the advanced options:
   ```
   ssh [USERNAME]@[YOUR_PI_IP_ADDRESS]
   ```

### 2. System Updates

Ensure your system is fully updated:

```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get dist-upgrade -y
sudo reboot
```

Reconnect via SSH after the reboot.

### 3. Install Required System Packages

Install the packages needed for the ASL project:

```bash
sudo apt-get install -y git
sudo apt-get install -y libatlas-base-dev libjasper1 libqtgui4 libqt4-test
sudo apt-get install -y libhdf5-dev libhdf5-serial-dev libharfbuzz0b libwebp6 libtiff5
sudo apt-get install -y libjasper-dev libilmbase23 libopenexr23 libgstreamer1.0-0
sudo apt-get install -y libavcodec58 libavformat58 libswscale5
sudo apt-get install -y python3-venv
```

### 4. Enable Remote Desktop (Optional)

If you plan to use the Pi headless (without a direct display):

1. Enable VNC Server:
   ```bash
   sudo raspi-config
   ```
2. Navigate to "Interface Options" > "VNC" > "Yes"
3. Navigate to "System Options" > "Boot / Auto Login" > "Desktop Autologin"
4. Select "Finish" and reboot when prompted
5. Reconnect via SSH after reboot
6. On your computer, download VNC Viewer from: https://www.realvnc.com/en/connect/download/viewer/
7. Connect to your Pi's IP address

## Camera Configuration

### 1. Connect the Camera Module

1. **Power off your Raspberry Pi**: `sudo shutdown -h now`
2. Locate the camera connector (between HDMI and Ethernet ports)
3. Gently pull up the black tab to release the connector
4. Insert the camera ribbon cable with the blue side facing away from the HDMI ports
5. Push the black tab down to secure the connection
6. Power on your Pi

### 2. Enable the Camera

1. Connect via SSH
2. Run:
   ```bash
   sudo raspi-config
   ```
3. Navigate to "Interface Options" > "Camera" > "Yes"
4. Select "Finish" and reboot when prompted

### 3. Test the Camera

After reconnecting via SSH:

```bash
raspistill -o test.jpg
```

This should take a photo and save it as test.jpg. If using VNC, you can view this image.

## Performance Optimization

For best performance with the ASL Sign Language Detector:

### 1. Allocate Appropriate GPU Memory

```bash
sudo raspi-config
```
- Navigate to "Performance Options" > "GPU Memory"
- Enter "128" (this reserves 128MB for the GPU)
- Select "Finish" and reboot when prompted

### 2. Reduce Background Services

```bash
sudo systemctl disable bluetooth.service
sudo systemctl disable avahi-daemon.service
sudo reboot
```

## Project Setup

Now you're ready to set up the ASL Sign Language Detector:

### 1. Clone the Repository

After reconnecting via SSH:

```bash
# Navigate to home directory
cd ~

# Clone the repository
git clone https://github.com/your-username/signlanguage.git
# (Replace with the actual repository URL)

# Navigate into the project directory
cd signlanguage
```

### 2. Create a Virtual Environment

```bash
# Create a Python virtual environment
python3 -m venv sign_env

# Activate the environment
source sign_env/bin/activate
```

At this point, you are ready to proceed with the specific installation steps in the README.md file, starting from the "Install Python dependencies" section.

## Troubleshooting

### Camera Not Working

If the camera isn't working properly:

1. Check physical connection:
   - Ensure the ribbon cable is properly seated
   - Blue side should face away from HDMI ports
   
2. Verify camera is enabled:
   ```bash
   sudo raspi-config
   ```
   - Navigate to "Interface Options" > "Camera" > "Yes"

3. Test with different commands:
   ```bash
   libcamera-still -o test2.jpg
   # or
   vcgencmd get_camera
   # Should show "supported=1 detected=1"
   ```

### Display Issues with VNC

If VNC shows a "Cannot currently show the desktop" error:

1. Edit the VNC configuration:
   ```bash
   sudo nano /root/.vnc/config.d/vncserver-x11
   ```
2. Change the line:
   ```
   Authentication=VncAuth
   ```
   to:
   ```
   Authentication=SystemAuth
   ```
3. Add the line:
   ```
   Localhost=0
   ```
4. Save and exit (Ctrl+X, then Y)
5. Restart VNC:
   ```bash
   sudo systemctl restart vncserver-x11-serviced
   ```

### Performance Issues

If the Pi is running slowly:

1. Check temperature:
   ```bash
   vcgencmd measure_temp
   ```
   - If over 80°C, improve cooling

2. Monitor CPU usage:
   ```bash
   top
   ```
   - Look for processes using high CPU

3. Check available memory:
   ```bash
   free -h
   ```

---

After completing this setup process, proceed to the README.md for the specific ASL Sign Language Detector installation instructions starting from the Python dependencies section. 