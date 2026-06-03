# Object Detection & Tracking System

A real-time object detection and multi-object tracking system powered by **YOLO** (Ultralytics) and **SORT** (Simple Online and Realtime Tracking). Features both a terminal-based interface and a full-featured **web dashboard**.

![Dashboard Preview](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/flask-3.0%2B-lightgrey)

---

## ✨ Features

### Detection & Tracking
- **YOLO-based detection** — supports YOLO11 (Nano, Small, Medium) and YOLOv8 (Nano, Small) models
- **SORT tracking** — Kalman filter + Hungarian algorithm for robust multi-object tracking
- **80 COCO classes** — person, vehicles, animals, indoor objects, and more
- **Class filtering** — detect only the object categories you care about
- **Confidence threshold** — adjustable per session

### Web Dashboard (`http://127.0.0.1:5000`)
- **Live MJPEG video stream** with real-time detection/tracking overlays
- **Interactive controls** — select model, device, confidence, video source
- **Video file upload** with drag-and-drop support
- **Statistics panel** — live FPS, track count, detection count, frame counter
- **📊 FPS Over Time chart** — real-time line chart showing performance trend
- **📈 Detections Over Time chart** — detection count trend visualization
- **🎯 Class Filter** — searchable multi-select checkboxes for all 80 COCO classes
- **Class filter presets** — built-in presets (People, Vehicles, Animals, Street, Indoor, Sport) + custom saved presets with export/import as JSON
- **📋 Event Log** — timestamped log of new tracks and lost tracks
- **🌙/☀️ Theme toggle** — dark and light mode with system preference detection
- **Responsive layout** — adapts to different screen sizes

### Terminal CLI (`main.py`)
- Keyboard controls: play/pause, reset tracker, quit
- Output video recording
- Resize factor for display
- Configurable tracking parameters (max age, IoU threshold)

---

## 🏗️ Project Structure

```
├── web_app.py          # Flask web server & dashboard backend
├── main.py             # Terminal/CLI interface
├── detector.py         # YOLO object detector wrapper
├── tracker.py          # SORT tracker implementation
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Web dashboard frontend (HTML + CSS + JS)
└── uploads/            # Uploaded video files (created on first upload)
```

### Module Breakdown

| Module | Description |
|--------|-------------|
| `detector.py` | Wraps Ultralytics YOLO — handles model loading, inference, class filtering, and result parsing |
| `tracker.py` | Full SORT implementation with KalmanBoxTracker, IoU computation, Hungarian data association |
| `main.py` | Terminal interface with OpenCV display, keyboard controls, and optional video recording |
| `web_app.py` | Flask server with REST API, MJPEG streaming, video processing thread, and file upload handling |
| `templates/index.html` | Single-page web dashboard with Canvas charts, drag-and-drop upload, class filter, theme toggle |

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone or navigate to the project directory:**

```bash
cd "C:\Users\sai hp\Downloads\object detection and tracking"
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` — video capture, image processing, display
- `ultralytics` — YOLO object detection models
- `numpy` — numerical computation
- `scipy` — Hungarian algorithm for tracking association
- `flask` — web server

> **Note:** On first run, YOLO will automatically download the selected model file (e.g., `yolo11n.pt` ~6 MB). A cached copy is saved in the project directory for future use.

---

## 🖥️ Usage

### Option 1: Web Dashboard (Recommended)

Start the web server:

```bash
python web_app.py
```

Then open **http://127.0.0.1:5000** in your browser.

**Optional arguments:**

```bash
python web_app.py --host 0.0.0.0 --port 8080 --debug
python web_app.py --model yolov8s.pt --device cuda
python web_app.py --source "video.mp4" --conf 0.3
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `127.0.0.1` | Host address to bind to |
| `--port` | `5000` | Port number |
| `--debug` | `False` | Enable Flask debug mode |
| `--source` | `0` | Default video source (0 = webcam, or file path) |
| `--model` | `yolo11n.pt` | Default YOLO model |
| `--conf` | `0.5` | Default confidence threshold |
| `--device` | `cpu` | Inference device (`cpu`, `cuda`, `mps`) |

#### Using the Dashboard

1. **Select a video source:**
   - **Webcam** — select "📷 Webcam" from the Video Source dropdown
   - **Upload a file** — select "📁 Video File", then drag-and-drop or click to browse
2. **Configure settings:** Choose model, confidence threshold, and device
3. **Filter classes (optional):** Open the Class Filter card, search and check specific object classes, or use a preset
4. **Click "▶ Start Stream"** — the live video appears with detection boxes and tracking IDs
5. **Monitor performance:** Watch the FPS and Detection count charts update in real time
6. **Stop:** Click "■ Stop" or close the tab

### Option 2: Terminal / CLI

Run the terminal interface for a lightweight, no-browser experience:

```bash
python main.py
```

**With custom options:**

```bash
python main.py --source 0 --model yolo11n.pt --conf 0.5 --device cpu
python main.py --source "video.mp4" --output result.mp4 --resize 0.75
python main.py --source 0 --classes 0 2 3 --track-iou 0.4
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--source` | `-s` | `0` | Video source (`0` = webcam, or file path) |
| `--model` | `-m` | `yolo11n.pt` | YOLO model name or path |
| `--conf` | | `0.5` | Confidence threshold |
| `--iou` | | `0.45` | NMS IoU threshold |
| `--track-iou` | | `0.3` | IoU threshold for tracking association |
| `--max-age` | | `30` | Maximum frames to keep a track alive |
| `--device` | | `cpu` | Inference device (`cpu`, `cuda`, `mps`) |
| `--classes` | | `None` | Class IDs to detect (e.g. `0` for person) |
| `--output` | `-o` | `None` | Path to save output video |
| `--resize` | | `1.0` | Resize factor for display |

#### Keyboard Controls (Terminal)

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `P` | Pause / Resume |
| `R` | Reset tracker |

---

## 🌐 API Endpoints

The web dashboard is powered by a REST API. All endpoints return JSON.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Render the dashboard HTML page |
| `GET` | `/video_feed` | MJPEG video stream |
| `GET` | `/api/status` | Current stats (FPS, tracks, detections, frame count) |
| `POST` | `/api/start` | Start video processing with settings |
| `POST` | `/api/stop` | Stop video processing |
| `POST` | `/api/reset_tracker` | Reset the SORT tracker |
| `GET` | `/api/tracks` | Per-track details (ID, class, confidence) |
| `GET` | `/api/classes` | All 80 COCO class names |
| `GET` | `/api/models` | Available YOLO model options |
| `GET` | `/api/events` | Recent detection/track event log (last 100) |
| `POST` | `/api/upload` | Upload a video file |
| `POST` | `/api/update_settings` | Update runtime settings |

### Example: Start a stream via API

```bash
curl -X POST http://127.0.0.1:5000/api/start \
  -H "Content-Type: application/json" \
  -d '{"source": "0", "model": "yolo11n.pt", "conf": 0.5, "device": "cpu", "classes": [0, 2]}'
```

---

## 🔧 Configuration

### Available YOLO Models

| Model ID | Name | Speed | Accuracy | Size |
|----------|------|-------|----------|------|
| `yolo11n.pt` | YOLO11 Nano | ⚡ Fastest | Fair | ~5.4 MB |
| `yolo11s.pt` | YOLO11 Small | ⚡ Fast | Good | ~18.4 MB |
| `yolo11m.pt` | YOLO11 Medium | 🐢 Moderate | Better | ~53.9 MB |
| `yolov8n.pt` | YOLOv8 Nano | ⚡ Fastest | Fair | ~6.2 MB |
| `yolov8s.pt` | YOLOv8 Small | ⚡ Fast | Good | ~22.5 MB |

### Supported Video Formats (Upload)

`mp4`, `avi`, `mov`, `mkv`, `webm`, `flv`, `wmv`

### Supported Devices

- `cpu` — Works on any machine
- `cuda` — NVIDIA GPU (requires CUDA toolkit + cuDNN)
- `mps` — Apple Silicon GPU (macOS)

---

## 🧠 How It Works

### Processing Pipeline

```
Video Frame → YOLO Detection → SORT Tracking → Annotation → Display
```

1. **Capture** — Frame is read from webcam or video file
2. **Detection** — YOLO neural network detects objects and outputs bounding boxes with class labels and confidence scores
3. **Tracking** — SORT associates detections across frames using:
   - **Kalman Filter** — predicts each track's next position
   - **Hungarian Algorithm** — optimally matches detections to existing tracks
   - **IoU (Intersection over Union)** — measures overlap between predicted and detected boxes
4. **Annotation** — Bounding boxes, track IDs, and labels are drawn on the frame
5. **Display** — Frame is either shown in an OpenCV window (CLI) or streamed as MJPEG (web)

### SORT Tracker Details

The SORT implementation includes:
- **KalmanBoxTracker** — 7-dimensional state vector `[cx, cy, s, r, vx, vy, vs]` with constant velocity model
- **Data association** — Hungarian algorithm minimizing (1 - IoU) cost matrix
- **Track management** — new tracks created from unmatched detections, stale tracks removed after `max_age` frames
- **Track confirmation** — tracks require `min_hits` (3) consecutive matches before being reported

---

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| **Python 3** | Core language |
| **Ultralytics YOLO** | Object detection (v8/v9/v10/v11) |
| **OpenCV** | Video capture, image processing, annotation |
| **SciPy** | Hungarian algorithm (`linear_sum_assignment`) |
| **NumPy** | Numerical operations, matrix math |
| **Flask** | Web server, REST API |
| **HTML5 Canvas** | Real-time performance charts |
| **CSS3** | Dark/light theme, responsive layout |

---

## 📝 License

This project is for educational and research purposes. YOLO models are provided by Ultralytics under their respective licenses.

---

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — State-of-the-art object detection
- [SORT](https://github.com/abewley/sort) — Simple Online and Realtime Tracking by Alex Bewley
