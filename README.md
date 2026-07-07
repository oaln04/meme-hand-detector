# cr7 gesture detector

A lightweight Python computer-vision project that uses your webcam to detect hand gestures and display a matching image beside the camera feed.

## demo

![Meme Hand Detector demo](assets/readme/demo.gif)

## stack

**Core:** Python, OpenCV, MediaPipe
**Supporting:** NumPy, Pillow

## features

- Real-time webcam hand tracking
- Named gesture detection for bite, thumb, zero, one, two, three, four, and five
- Separate display windows for Output and Webcam
- Clean output image fitting that preserves each image's shape
- Confirmed-frame smoothing to reduce flickering between gestures
- Screenshot capture with one keypress
- Replaceable image assets for custom images

## setup

### 1. Clone the repo

```bash
git clone https://github.com/oaln04/meme-hand-detector.git
cd meme-hand-detector
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Controls:

```text
q or ESC  quit
s         save screenshot to screenshots/demo.png
```

## customizing images

Replace the files inside `assets/memes/` with your own images.

Supported names:

```text
no_hand.png
bite.png
thumb.png
zero.png
one.png
two.png
three.png
four.png
five.png
```

You can also use `.jpg`, `.jpeg`, or `.webp`.

## how it works

1. OpenCV reads frames from the webcam.
2. MediaPipe detects hand landmarks.
3. The app checks for special bite and thumb gestures first.
4. If neither special gesture is found, the raised-finger count becomes the gesture value, including a thumb only when it is clearly spread.
5. A small smoother confirms the same gesture across consecutive frames before switching the output.
6. The app displays the image mapped to that gesture.

## notes

This project demonstrates:

- Real-time computer vision
- Webcam input handling
- Landmark-based gesture recognition
- Simple UI composition with OpenCV
- Clean Python project structure
