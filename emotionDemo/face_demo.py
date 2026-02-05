"""Live webcam demo that loads the Neconet `.pth` checkpoint and mirrors `show_predictions.py` preprocessing."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
from torch import nn

from Model_architecture_Code import Block, ResNet

EMOTION_LABELS = ["Horny", "Disgust", "Fear", "Gooning", "Sadness", "Lowkey sliming charlie kirk"]
CLASS_ORDER = ["angry", "disgusted", "fearful", "happy", "sad", "surprised"]
WEIGHTS_PATH = Path("C:\\Users\\Dell\\Desktop\\emotionDemo\\Neconet_Weights3.pth")
IMAGE_SIZE = (64, 64)
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def load_emotion_model(weights: Path, device: torch.device) -> nn.Module:
    if not weights.exists():
        raise FileNotFoundError(f"{weights} missing")
    state = torch.load(weights, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model = ResNet(Block, [2, 2, 2, 2], len(EMOTION_LABELS))
    model.to(device)
    model.load_state_dict(state)
    model.eval()
    LOGGER.info("Loaded pretrained weights with %d tensors", len(state) if isinstance(state, dict) else 0)
    return model


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_training_input(face: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if face.ndim == 3 else face
    if gray.shape != IMAGE_SIZE:
        gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return gray.astype(np.uint8)


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray, scale: float):
    h, w = frame.shape[:2]
    for (x, y) in landmarks:
        cv2.circle(frame, (int(x / scale), int(y / scale)), 2, (0, 255, 0), -1)


def preprocess_for_model(gray: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float()
    tensor = tensor / 255.0
    tensor = (tensor - NORMALIZE_MEAN) / NORMALIZE_STD
    return tensor


def draw_overlay(frame: np.ndarray, probs: np.ndarray, label: str, confidence: float, crop_box):
    start_x = frame.shape[1] - 210
    start_y = 20
    cv2.rectangle(frame, crop_box[0], crop_box[1], (255, 0, 0), 2)
    cv2.putText(
        frame,
        f"{label} ({confidence * 100:.1f}%)",
        (crop_box[0][0], max(crop_box[0][1] - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for idx, name in enumerate(EMOTION_LABELS):
        text = f"{name[:3]}: {probs[idx] * 100:4.1f}%"
        cv2.putText(
            frame,
            text,
            (start_x, start_y + idx * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def show_crop(gray: np.ndarray, label_text: str):
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(display, label_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Emotion Crop (64×64 grayscale)", display)


def main():
    device = pick_device()
    LOGGER.info("Using device %s", device)

    model = load_emotion_model(WEIGHTS_PATH, device)

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, flip_input=False, device="cuda" if device.type == "cuda" else "cpu"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera konnte nicht geöffnet werden")

    frame_idx = 0
    scale = 0.4
    SKIP_FRAMES = 2
    current_box: tuple[tuple[int, int], tuple[int, int]] | None = None
    last_landmarks: np.ndarray | None = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            detect_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            gray_detect = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)

            if frame_idx % SKIP_FRAMES == 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="No faces were detected\\.")
                    faces = fa.get_landmarks(gray_detect)
                if faces:
                    landmarks = faces[0]
                    last_landmarks = landmarks
                    min_x, min_y = np.min(landmarks[:, 0]) / scale, np.min(landmarks[:, 1]) / scale
                    max_x, max_y = np.max(landmarks[:, 0]) / scale, np.max(landmarks[:, 1]) / scale
                    x1, y1 = int(max(min_x, 0)), int(max(min_y, 0))
                    x2, y2 = int(min(max_x, frame.shape[1])), int(min(max_y, frame.shape[0]))
                    if x2 > x1 and y2 > y1:
                        current_box = ((x1, y1), (x2, y2))

            if last_landmarks is not None:
                draw_landmarks(frame, last_landmarks, scale)

            if current_box:
                (x1, y1), (x2, y2) = current_box
                crop = frame[y1:y2, x1:x2]
                gray = ensure_training_input(crop)
                tensor = preprocess_for_model(gray).to(device)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    pred_idx = int(np.argmax(probs))
                    label = EMOTION_LABELS[pred_idx]
                    confidence = float(probs[pred_idx])
                draw_overlay(frame, probs, label, confidence, ((x1, y1), (x2, y2)))
                show_crop(gray, f"{label} ({confidence * 100:.1f}%)")
            cv2.imshow("Emotion Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
