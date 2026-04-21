# Scuba Cat Hand-Tracking App

This project shows a looping scuba-cat video on screen when a hand is detected by the webcam. When the thumb and index finger move close together, the app treats that as a pinch, freezes the cat video, and increases a counter.

The project is written in Python and combines webcam input, hand tracking, chroma key video processing, and a small Pygame UI.

## What the app does

1. Opens your webcam with OpenCV.
2. Detects one hand and tracks its landmarks with MediaPipe.
3. Measures the distance between the thumb tip and index finger tip.
4. Shows the cat video only when a hand is visible.
5. Freezes the current cat frame while you pinch.
6. Counts each new pinch as one "bite".

## Why these libraries are used

### `cv2` / OpenCV

We use OpenCV because it is very good for camera and video work.

It is used here to:

- open the webcam
- read frames from the `.webm` cat video
- flip the webcam image
- convert color formats between BGR, RGB, HSV, and RGBA
- remove the green screen from the cat video
- show a separate camera preview window

### `mediapipe`

We use MediaPipe for hand landmark detection.

It gives us 21 points on the hand, which is much better than just knowing that "a hand exists". We need those points so we can measure the thumb-to-index distance and detect a pinch gesture.

The code supports two MediaPipe styles:

- `mp.solutions` for older setups
- `mp.tasks` for newer setups like the Python 3.13 environment used here

This compatibility logic is important because different MediaPipe installs expose different APIs.

### `pygame`

We use Pygame to build the main app window and draw the UI.

It is used here to:

- create the cat display window
- draw the cat video with transparency
- draw the counter text
- draw debug text
- handle keyboard and window events

Pygame is a good fit because it is simple for real-time drawing and event handling.

### `numpy`

We use NumPy because OpenCV image frames are arrays, and NumPy makes it easy to define the green color range used for chroma key removal.

### `math`

We use `math.hypot()` to calculate the distance between two hand landmarks:

- thumb tip
- index finger tip

That distance is the core value used to decide whether the user is pinching.

### `time`

The newer MediaPipe Tasks API needs a timestamp for video-mode detection, so `time.perf_counter()` is used to generate that timestamp.

### `os` and `warnings`

These are used to reduce noisy startup messages from TensorFlow and Pygame so the terminal output is easier to read.

## Why the logic works this way

### Hand detection first

The cat video only appears when a hand is detected.

Why:

- it makes the effect feel interactive
- it avoids showing the cat when nobody is in front of the camera
- it gives a clear "you are controlling this" experience

### Pinch detection from landmark distance

The app does not guess the gesture from the whole hand shape. Instead, it uses a simple and reliable rule:

- if thumb tip and index tip are close enough, it is a pinch

Why:

- easy to understand
- fast to compute
- stable enough for a small interactive project
- easier to tune with `PINCH_THRESHOLD`

### State-based pinch counting

The app only counts when the pinch starts, not on every frame.

Why:

- the camera runs many frames per second
- without state tracking, one pinch would be counted many times

So the app uses this logic:

- open hand -> no count
- first pinch frame -> count `+1`
- keep pinching -> no extra count
- open again -> ready for the next count

### Freeze the cat while pinching

When the hand is pinching, the app stops advancing the cat video and keeps the last frame.

Why:

- it creates a visible reaction to the gesture
- it makes the cat feel like it is "biting" or pausing
- it is simpler than switching between separate open and closed images

### Green screen removal

The video has a green background, so the app removes green pixels and turns them into transparency.

Why:

- this lets the cat appear cleanly inside the Pygame window
- the background color of the app can still show behind the cat
- one video can behave like a sprite with alpha

The code does this by:

1. converting the frame to HSV color space
2. finding green pixels inside a chosen range
3. blurring the mask a little for softer edges
4. using the inverse of that mask as the alpha channel

## Files

- [cat.py](/c:/Users/ACER/Desktop/vattrend/cat.py) - main application
- [chroma-keyed-video.webm](/c:/Users/ACER/Desktop/vattrend/chroma-keyed-video.webm) - looping cat video with green background
- [hand_landmarker.task](/c:/Users/ACER/Desktop/vattrend/hand_landmarker.task) - MediaPipe model file used by newer MediaPipe installs

## How to run it

Make sure these Python packages are installed:

```bash
pip install opencv-python pygame mediapipe numpy
```

Then run:

```bash
python cat.py
```

## Controls

- `Q` or `Esc` = quit
- `R` = reset the bite counter
- pinching with thumb and index finger = freeze cat and count one bite

## Important settings

These values near the top of `cat.py` are the main things you may want to change:

- `PINCH_THRESHOLD`
  Used to decide how close the fingers must be to count as a pinch.
- `CAT_WINDOW_SIZE`
  Size of the app window.
- `CAT_SCALE`
  Size of the displayed cat video.
- `FPS`
  How fast the app updates.
- `GREEN_LOWER` and `GREEN_UPPER`
  Controls which green shades are removed from the video.

## Troubleshooting

### The cat does not appear

Check these things:

- your webcam is connected and allowed
- your hand is inside the camera frame
- `chroma-keyed-video.webm` is in the project folder
- `hand_landmarker.task` is in the project folder if your MediaPipe install uses the Tasks API

### The pinch counts too easily or not enough

Change:

```python
PINCH_THRESHOLD = 0.06
```

- smaller value = harder to trigger
- larger value = easier to trigger

### Green edges look bad

Adjust:

- `GREEN_LOWER`
- `GREEN_UPPER`

Those values decide which green shades are removed from the video.

### MediaPipe errors about `mp.solutions`

This project already includes compatibility code for newer MediaPipe versions. If `mp.solutions` is missing, it falls back to the Tasks API and uses `hand_landmarker.task`.

## Summary

This project uses:

- OpenCV for camera and video processing
- MediaPipe for hand landmarks
- Pygame for the visual window and UI
- a simple pinch-state machine for clean counting
- chroma key logic so a green-screen cat video can behave like a transparent animated character

That combination keeps the app simple, interactive, and easy to improve.
