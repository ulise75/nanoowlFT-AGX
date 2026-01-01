# Remote Camera Access with NanoOWL

Access the live camera detection feed from any device on your network.

## Quick Start

### 1. Start the Web Server on Jetson

```bash
cd /home/sombras4/Downloads/nanoowlOrin/examples

# Basic usage (default: GPU, port 5000)
python3 camera_web_server.py

# Custom settings
python3 camera_web_server.py --device cuda --prompt "a person,a bottle,a phone" --port 8080
```

### 2. Access from Remote PC

Open a web browser and go to:
```
http://<jetson-ip-address>:5000
```

**Find your Jetson IP:**
```bash
hostname -I | awk '{print $1}'
```

## Features

- ✅ Live video stream with real-time detections
- ✅ Runs on GPU for ~7 FPS performance
- ✅ Web-based interface (no software needed on client)
- ✅ Works on any device with a browser (PC, tablet, phone)
- ✅ Customizable detection prompts
- ✅ Low latency MJPEG streaming

## Options

```bash
python3 camera_web_server.py [OPTIONS]

Options:
  --camera ID          Camera device ID (default: 0)
  --width WIDTH        Camera width (default: 640)
  --height HEIGHT      Camera height (default: 480)
  --prompt "text"      Detection prompts (default: "a person,a face,a hand")
  --threshold VALUE    Confidence threshold (default: 0.15)
  --device cuda/cpu    Device to run on (default: cuda)
  --host HOST          Host to bind to (default: 0.0.0.0 = all interfaces)
  --port PORT          Web server port (default: 5000)
```

## Examples

### Detect people and objects
```bash
python3 camera_web_server.py --prompt "a person,a bottle,a cup,a phone,a laptop"
```

### Lower resolution for better FPS
```bash
python3 camera_web_server.py --width 320 --height 240
```

### Run on different port
```bash
python3 camera_web_server.py --port 8080
```

### More sensitive detection
```bash
python3 camera_web_server.py --threshold 0.1
```

## Disable Graphical Desktop to Save Memory

To free up ~500MB-1GB of memory, disable the desktop:

### Check current status
```bash
./scripts/manage_desktop.sh status
```

### Disable desktop (persists after reboot)
```bash
./scripts/manage_desktop.sh disable
sudo reboot
```

### Stop desktop temporarily (just for this session)
```bash
./scripts/manage_desktop.sh stop-now
```

### Re-enable desktop
```bash
./scripts/manage_desktop.sh enable
sudo reboot
```

**Note:** After disabling desktop, you'll need to access Jetson via SSH. The web server will still work perfectly!

## Network Configuration

### Access from local network
```
http://192.168.1.XXX:5000
```

### Port forwarding for internet access
1. Configure router to forward port 5000 to Jetson's IP
2. Access via: `http://your-public-ip:5000`

**Security Note:** For internet access, consider adding authentication or using a VPN.

## Troubleshooting

### Can't connect from remote PC
```bash
# Check if server is running
ps aux | grep camera_web_server

# Check firewall (Ubuntu)
sudo ufw status
sudo ufw allow 5000/tcp  # If firewall is active

# Test locally first
curl http://localhost:5000
```

### Server crashes or out of memory
```bash
# Disable desktop to free memory
./scripts/manage_desktop.sh disable
sudo reboot

# Or use CPU instead of GPU
python3 camera_web_server.py --device cpu
```

### Low FPS
- Lower resolution: `--width 320 --height 240`
- Reduce detection classes
- Close other applications
- Ensure using GPU: `--device cuda`

## Performance

**With GPU (CUDA):**
- Resolution 640x480: ~7 FPS
- Resolution 320x240: ~12-15 FPS

**With Desktop Disabled:**
- Additional ~500MB-1GB memory available
- Better thermal performance
- More stable for 24/7 operation

## Run as Service (Auto-start on boot)

Create systemd service file:
```bash
sudo nano /etc/systemd/system/nanoowl-camera.service
```

Add:
```ini
[Unit]
Description=NanoOWL Camera Web Server
After=network.target

[Service]
Type=simple
User=sombras4
WorkingDirectory=/home/sombras4/Downloads/nanoowlOrin/examples
ExecStart=/usr/bin/python3 camera_web_server.py --device cuda --port 5000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable nanoowl-camera.service
sudo systemctl start nanoowl-camera.service

# Check status
sudo systemctl status nanoowl-camera.service
```

## Stop the Server

Press `Ctrl+C` in the terminal where the server is running.
