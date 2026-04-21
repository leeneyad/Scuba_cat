from pathlib import Path
import math
import os
import sys
import time
import warnings

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

warnings.filterwarnings(
    "ignore",
    message="Protobuf gencode version .*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import cv2
import mediapipe as mp
import numpy as np
import pygame


WINDOW_TITLE = "Scuba Cat"
CAT_WINDOW_SIZE = (500, 500)
CAT_SCALE = (400, 400)
PINCH_THRESHOLD = 0.06
SHOW_DEBUG = True
BG_COLOR = (15, 15, 35)
COUNTER_COLOR = (0, 220, 255)
FPS = 30

VIDEO_PATH = Path(r"C:\Users\ACER\Desktop\vattrend\chroma-keyed-video.webm")
if not VIDEO_PATH.exists():
    VIDEO_PATH = Path(__file__).resolve().parent / "chroma-keyed-video.webm"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_MODEL_CANDIDATES = [
    Path(r"C:\Users\ACER\Desktop\vattrend\hand_landmarker.task"),
    Path(__file__).resolve().parent / "hand_landmarker.task",
]

GREEN_LOWER = np.array([35, 80, 80], dtype=np.uint8)
GREEN_UPPER = np.array([85, 255, 255], dtype=np.uint8)


def remove_green_screen(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    # Soften the edge before using it as alpha.
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    alpha = cv2.bitwise_not(mask)

    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame_rgba[:, :, 3] = alpha
    return frame_rgba


def cv2_to_pygame(frame_rgba, size):
    frame_rgba = cv2.resize(frame_rgba, size, interpolation=cv2.INTER_AREA)
    frame_rgba = cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2RGBA)
    return pygame.image.frombuffer(frame_rgba.tobytes(), size, "RGBA").convert_alpha()


def get_pinch_distance(landmarks):
    thumb = landmarks[4]
    index = landmarks[8]
    return math.hypot(thumb.x - index.x, thumb.y - index.y)


def draw_task_hand_landmarks(frame, landmarks, connections):
    height, width = frame.shape[:2]

    for connection in connections:
        start_point = landmarks[connection.start]
        end_point = landmarks[connection.end]
        start_xy = (int(start_point.x * width), int(start_point.y * height))
        end_xy = (int(end_point.x * width), int(end_point.y * height))
        cv2.line(frame, start_xy, end_xy, (0, 255, 0), 2)

    for point in landmarks:
        xy = (int(point.x * width), int(point.y * height))
        cv2.circle(frame, xy, 4, (0, 170, 255), -1)


def get_hand_tracker():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        mp_hands = mp.solutions.hands
        hand_tracker = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        return {
            "mode": "solutions",
            "tracker": hand_tracker,
            "drawer": mp.solutions.drawing_utils,
            "connections": mp_hands.HAND_CONNECTIONS,
        }

    hand_model_path = next((path for path in HAND_MODEL_CANDIDATES if path.exists()), None)
    if hand_model_path is None:
        print("Error: This MediaPipe install uses the newer Tasks API.")
        print("To enable hand tracking, download this file and place it next to cat.py:")
        print("hand_landmarker.task")
        print(HAND_MODEL_URL)
        sys.exit(1)

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_tracker = vision.HandLandmarker.create_from_options(options)
    return {
        "mode": "tasks",
        "tracker": hand_tracker,
        "drawer": None,
        "connections": vision.HandLandmarksConnections.HAND_CONNECTIONS,
    }


def detect_hand_landmarks(hand_setup, frame_rgb, frame_bgr):
    if hand_setup["mode"] == "solutions":
        results = hand_setup["tracker"].process(frame_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks_obj = results.multi_hand_landmarks[0]
            hand_setup["drawer"].draw_landmarks(
                frame_bgr,
                hand_landmarks_obj,
                hand_setup["connections"],
            )
            return hand_landmarks_obj.landmark
        return None

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(time.perf_counter() * 1000)
    results = hand_setup["tracker"].detect_for_video(mp_image, timestamp_ms)
    if results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]
        draw_task_hand_landmarks(frame_bgr, hand_landmarks, hand_setup["connections"])
        return hand_landmarks
    return None


hand_setup = get_hand_tracker()

pygame.init()
screen = pygame.display.set_mode(CAT_WINDOW_SIZE)
pygame.display.set_caption(WINDOW_TITLE)
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("Arial", 42, bold=True)
font_small = pygame.font.SysFont("Arial", 20)

cat_video = cv2.VideoCapture(str(VIDEO_PATH))
if not cat_video.isOpened():
    print(f"Error: Could not open video: {VIDEO_PATH}")
    hand_setup["tracker"].close()
    pygame.quit()
    sys.exit()

total_frames = max(int(cat_video.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
current_cat_surface = None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    cat_video.release()
    hand_setup["tracker"].close()
    pygame.quit()
    sys.exit()

pinch_count = 0
is_pinching = False
distance = 1.0
running = True

print("Scuba Cat is running.")
print("Q or Esc = quit | R = reset count")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False
            elif event.key == pygame.K_r:
                pinch_count = 0

    if not running:
        break

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_landmarks = detect_hand_landmarks(hand_setup, frame_rgb, frame)
    hand_detected = hand_landmarks is not None

    if hand_detected:
        distance = get_pinch_distance(hand_landmarks)

        if distance < PINCH_THRESHOLD and not is_pinching:
            is_pinching = True
            pinch_count += 1
        elif distance >= PINCH_THRESHOLD:
            is_pinching = False
    else:
        is_pinching = False
        distance = 1.0

    if hand_detected and not is_pinching:
        ret_vid, cat_frame = cat_video.read()
        if not ret_vid:
            cat_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_vid, cat_frame = cat_video.read()

        if ret_vid:
            cat_rgba = remove_green_screen(cat_frame)
            current_cat_surface = cv2_to_pygame(cat_rgba, CAT_SCALE)
    elif not hand_detected:
        current_cat_surface = None

    screen.fill(BG_COLOR)

    if hand_detected and current_cat_surface is not None:
        cat_x = (CAT_WINDOW_SIZE[0] - CAT_SCALE[0]) // 2
        cat_y = (CAT_WINDOW_SIZE[1] - CAT_SCALE[1]) // 2 - 20
        screen.blit(current_cat_surface, (cat_x, cat_y))

    count_text = font_big.render(f"Bites: {pinch_count}", True, COUNTER_COLOR)
    screen.blit(
        count_text,
        (CAT_WINDOW_SIZE[0] // 2 - count_text.get_width() // 2, CAT_WINDOW_SIZE[1] - 65),
    )

    if SHOW_DEBUG:
        if not hand_detected:
            status = "HIDDEN"
        elif is_pinching:
            status = "PINCH"
        else:
            status = "OPEN"
        hand_text = "Hand detected" if hand_detected else "No hand"
        current_frame = int(cat_video.get(cv2.CAP_PROP_POS_FRAMES))

        debug1 = font_small.render(
            f"Dist: {distance:.3f} | {status}", True, (160, 160, 160)
        )
        debug2 = font_small.render(
            f"{hand_text} | Frame: {current_frame}/{total_frames}",
            True,
            (160, 160, 160),
        )
        hint = font_small.render("R = Reset | Q = Quit", True, (70, 70, 90))

        screen.blit(debug1, (10, 10))
        screen.blit(debug2, (10, 32))
        screen.blit(hint, (10, CAT_WINDOW_SIZE[1] - 22))

    pygame.display.flip()
    clock.tick(FPS)

    cv2.imshow("Camera - Scuba Cat", frame)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        running = False

cap.release()
cat_video.release()
cv2.destroyAllWindows()
hand_setup["tracker"].close()
pygame.quit()
