"""
Meme Hand Gesture Detector
--------------------------
A simple computer-vision project that uses OpenCV + MediaPipe to detect a hand,
count raised fingers, and display a matching meme/image beside the webcam feed.

Controls:
- q / ESC : quit
- s       : save current screen to screenshots/demo.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


SUPPORTED_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp")
WINDOW_OUTPUT = "Output"
WINDOW_WEBCAM = "Webcam"
LABEL_FONT = cv2.FONT_HERSHEY_DUPLEX
LABEL_SCALE = 0.75
LABEL_THICKNESS = 1
HEADER_HEIGHT = 52
GESTURE_NAMES = {
    "no_hand": "No hand",
    "bite": "Bite",
    "thumb": "Thumb",
    "zero": "Fist / zero",
    "one": "1 finger",
    "two": "2 fingers",
    "three": "3 fingers",
    "four": "4 fingers",
    "five": "Open hand / 5 fingers",
}
FILE_ALIASES = {
    "no_hand": ("no_hand", "none", "idle"),
    "bite": ("bite", "pinch"),
    "thumb": ("thumb", "thumbs_up", "thumbsup"),
    "zero": ("zero", "0", "fist"),
    "one": ("one", "1"),
    "two": ("two", "2", "peace"),
    "three": ("three", "3"),
    "four": ("four", "4"),
    "five": ("five", "5", "open_hand", "open"),
}
COUNT_GESTURES = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
}
DEFAULT_CONFIRM_FRAMES = 3


class HandGestureDetector:
    """Detects one hand and estimates how many fingers are raised."""

    def __init__(self, min_detection_confidence: float = 0.7, min_tracking_confidence: float = 0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> Tuple[Optional[str], np.ndarray]:
        """Return gesture key and annotated frame. None means no hand detected."""
        annotated = frame_bgr.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None, annotated

        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label

        self.mp_draw.draw_landmarks(
            annotated,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
        )

        gesture = self._classify_gesture(hand_landmarks.landmark, handedness)
        return gesture, annotated

    @staticmethod
    def _classify_gesture(landmarks, handedness: str) -> str:
        """Classifies the hand into a named meme output case."""
        thumb_is_open, index_up, middle_up, ring_up, pinky_up = HandGestureDetector._finger_states(
            landmarks,
            handedness,
        )

        if HandGestureDetector._is_bite_gesture(landmarks, middle_up, ring_up, pinky_up):
            return "bite"

        if HandGestureDetector._is_thumb_gesture(landmarks, index_up, middle_up, ring_up, pinky_up):
            return "thumb"

        raised_fingers = sum((index_up, middle_up, ring_up, pinky_up)) + int(thumb_is_open)
        return COUNT_GESTURES.get(raised_fingers, "five")

    @staticmethod
    def _finger_states(landmarks, handedness: str) -> Tuple[bool, bool, bool, bool, bool]:
        """Returns thumb, index, middle, ring, and pinky open states."""
        # Thumb handling is intentionally conservative. Folded thumbs often look
        # "sideways open" in webcam landmarks, so normal 1-4 counts ignore the
        # thumb and only use it to separate four fingers from a true open hand.
        thumb_is_open = HandGestureDetector._is_thumb_spread(landmarks, handedness)

        # Index, middle, ring, pinky: tip above PIP joint means finger is raised.
        finger_tip_ids = [8, 12, 16, 20]
        finger_pip_ids = [6, 10, 14, 18]
        finger_states = tuple(
            landmarks[tip_id].y < landmarks[pip_id].y
            for tip_id, pip_id in zip(finger_tip_ids, finger_pip_ids)
        )

        return (thumb_is_open, *finger_states)

    @staticmethod
    def _is_bite_gesture(landmarks, middle_up: bool, ring_up: bool, pinky_up: bool) -> bool:
        """Detects a thumb/index pinch as the bite case."""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        index_mcp = landmarks[5]
        palm_scale = HandGestureDetector._landmark_distance(landmarks[0], landmarks[9])

        pinch_distance = HandGestureDetector._landmark_distance(thumb_tip, index_tip)
        thumb_to_palm = HandGestureDetector._landmark_distance(thumb_tip, index_mcp)
        pinch_is_close = pinch_distance < max(0.035, palm_scale * 0.30)
        pinch_is_outside_fist = thumb_to_palm > max(0.12, palm_scale * 0.45)
        other_fingers_down = not any((middle_up, ring_up, pinky_up))

        return pinch_is_close and pinch_is_outside_fist and other_fingers_down

    @staticmethod
    def _is_thumb_spread(landmarks, handedness: str) -> bool:
        """Detects a thumb spread away from the palm for the open-hand case."""
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        palm_scale = HandGestureDetector._landmark_distance(landmarks[0], landmarks[9])

        points_outward = (
            thumb_tip.x < thumb_ip.x < thumb_mcp.x
            if handedness == "Right"
            else thumb_tip.x > thumb_ip.x > thumb_mcp.x
        )
        thumb_distance = HandGestureDetector._landmark_distance(thumb_tip, index_mcp)
        thumb_joint_distance = HandGestureDetector._landmark_distance(thumb_ip, index_mcp)
        far_from_index = thumb_distance > max(0.12, palm_scale * 0.72)
        farther_than_joint = thumb_distance > thumb_joint_distance * 1.15

        return points_outward and far_from_index and farther_than_joint

    @staticmethod
    def _is_thumb_gesture(landmarks, index_up: bool, middle_up: bool, ring_up: bool, pinky_up: bool) -> bool:
        """Detects a clear thumbs-up before falling back to one-finger counting."""
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        index_tip = landmarks[8]
        index_mcp = landmarks[5]
        palm_scale = HandGestureDetector._landmark_distance(landmarks[0], landmarks[9])
        other_fingers_down = not any((index_up, middle_up, ring_up, pinky_up))
        thumb_points_up = thumb_tip.y < thumb_ip.y < thumb_mcp.y

        thumb_to_palm = HandGestureDetector._landmark_distance(thumb_tip, index_mcp)
        thumb_to_fingertip = HandGestureDetector._landmark_distance(thumb_tip, index_tip)
        thumb_is_extended = thumb_to_palm > max(0.14, palm_scale * 0.48)
        not_pinching = thumb_to_fingertip > max(0.08, palm_scale * 0.32)

        return thumb_points_up and thumb_is_extended and not_pinching and other_fingers_down

    @staticmethod
    def _landmark_distance(first, second) -> float:
        return float(((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5)

    def close(self) -> None:
        self.hands.close()


class GestureSmoother:
    """Keeps output stable while still allowing quick deliberate gesture changes."""

    def __init__(self, confirm_frames: int = DEFAULT_CONFIRM_FRAMES):
        self.confirm_frames = max(1, confirm_frames)
        self.current = "no_hand"
        self.candidate: Optional[str] = None
        self.candidate_frames = 0

    def update(self, gesture: str) -> str:
        if gesture == self.current:
            self.candidate = None
            self.candidate_frames = 0
            return self.current

        if gesture != self.candidate:
            self.candidate = gesture
            self.candidate_frames = 1
        else:
            self.candidate_frames += 1

        if self.candidate_frames >= self.confirm_frames:
            self.current = gesture
            self.candidate = None
            self.candidate_frames = 0

        return self.current


def load_meme_images(asset_dir: Path, target_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """Loads gesture images from assets/memes, with generated placeholders as fallback."""
    images: Dict[str, np.ndarray] = {}

    for gesture_value, aliases in FILE_ALIASES.items():
        image = None
        for alias in aliases:
            for ext in SUPPORTED_IMAGE_TYPES:
                candidate = asset_dir / f"{alias}{ext}"
                if candidate.exists():
                    image = cv2.imread(str(candidate))
                    break
            if image is not None:
                break

        if image is None:
            image = make_placeholder_image(GESTURE_NAMES[gesture_value], target_size)
        else:
            image = resize_to_fit(image, target_size)

        images[gesture_value] = image

    return images


def make_placeholder_image(text: str, size: Tuple[int, int]) -> np.ndarray:
    """Creates a clean fallback image when no meme asset exists."""
    width, height = size
    image = np.full((height, width, 3), 32, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (width - 1, height - 1), (90, 90, 90), 3)
    cv2.putText(image, "Replace this image", (35, height // 2 - 25), LABEL_FONT, 0.85, (235, 235, 235), 1)
    cv2.putText(image, text, (35, height // 2 + 25), LABEL_FONT, 0.8, (235, 235, 235), 1)
    return image


def resize_to_fit(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to fit fully inside target_size, with a clean background fill."""
    target_w, target_h = target_size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas = make_soft_background(image, target_size)
    x1 = (target_w - new_w) // 2
    y1 = (target_h - new_h) // 2
    canvas[y1 : y1 + new_h, x1 : x1 + new_w] = resized

    cv2.rectangle(canvas, (x1, y1), (x1 + new_w - 1, y1 + new_h - 1), (230, 230, 230), 1)
    return canvas


def make_soft_background(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Creates a subtle blurred fill so different aspect ratios still look polished."""
    target_w, target_h = target_size
    h, w = image.shape[:2]
    scale = max(target_w / w, target_h / h)
    fill_w, fill_h = max(1, int(w * scale)), max(1, int(h * scale))
    fill = cv2.resize(image, (fill_w, fill_h), interpolation=cv2.INTER_AREA)

    x1 = (fill_w - target_w) // 2
    y1 = (fill_h - target_h) // 2
    fill = fill[y1 : y1 + target_h, x1 : x1 + target_w]
    fill = cv2.GaussianBlur(fill, (41, 41), 0)
    return cv2.addWeighted(fill, 0.45, np.full((target_h, target_w, 3), 28, dtype=np.uint8), 0.55, 0)


def add_header(
    frame: np.ndarray,
    text: str,
    background_color: Tuple[int, int, int] = (22, 24, 28),
    accent_color: Tuple[int, int, int] = (92, 178, 255),
) -> np.ndarray:
    """Adds a clean header without covering the image content."""
    header = np.full((HEADER_HEIGHT, frame.shape[1], 3), background_color, dtype=np.uint8)
    cv2.rectangle(header, (0, 0), (6, HEADER_HEIGHT), accent_color, -1)
    cv2.putText(header, text, (20, 34), LABEL_FONT, LABEL_SCALE, (255, 255, 255), LABEL_THICKNESS, cv2.LINE_AA)
    return np.vstack((header, frame))


def build_display_frames(meme: np.ndarray, camera: np.ndarray, gesture_text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Creates the separate output and webcam views."""
    camera = cv2.resize(camera, (meme.shape[1], meme.shape[0]))
    output_labeled = add_header(meme.copy(), WINDOW_OUTPUT, accent_color=(64, 170, 255))
    camera_labeled = add_header(
        camera.copy(),
        f"{WINDOW_WEBCAM} - {gesture_text}",
        background_color=(34, 42, 36),
        accent_color=(92, 214, 142),
    )
    return output_labeled, camera_labeled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand gesture controlled meme/image selector.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index. Default: 0")
    parser.add_argument("--width", type=int, default=640, help="Single panel width. Default: 640")
    parser.add_argument("--height", type=int, default=480, help="Single panel height. Default: 480")
    parser.add_argument(
        "--history",
        type=int,
        default=DEFAULT_CONFIRM_FRAMES,
        help=f"Consecutive frames required before switching gestures. Default: {DEFAULT_CONFIRM_FRAMES}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parent
    asset_dir = root / "assets" / "memes"
    screenshot_dir = root / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)

    image_size = (args.width, max(1, args.height - HEADER_HEIGHT))
    meme_images = load_meme_images(asset_dir, image_size)
    detector = HandGestureDetector()
    smoother = GestureSmoother(confirm_frames=args.history)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try another camera index, for example: python app.py --camera 1")

    cv2.namedWindow(WINDOW_OUTPUT, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_WEBCAM, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_OUTPUT, 80, 80)
    cv2.moveWindow(WINDOW_WEBCAM, 80 + args.width + 30, 80)

    print("Running. Press q or ESC to quit. Press s to save a screenshot.")

    last_display = None
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            gesture, annotated_frame = detector.detect(frame)
            gesture_value = "no_hand" if gesture is None else gesture
            stable_gesture = smoother.update(gesture_value)

            gesture_text = GESTURE_NAMES.get(stable_gesture, f"{stable_gesture} fingers")
            output_frame, webcam_frame = build_display_frames(
                meme_images[stable_gesture],
                annotated_frame,
                f"Detected: {gesture_text}",
            )
            display = np.hstack((output_frame, webcam_frame))
            last_display = display

            cv2.imshow(WINDOW_OUTPUT, output_frame)
            cv2.imshow(WINDOW_WEBCAM, webcam_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("s") and last_display is not None:
                screenshot_path = screenshot_dir / "demo.png"
                cv2.imwrite(str(screenshot_path), last_display)
                print(f"Saved screenshot to {screenshot_path}")
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
