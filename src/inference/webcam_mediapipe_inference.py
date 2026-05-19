from pathlib import Path

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "mobilenet_best.pth"
HAND_LANDMARKER_PATH = PROJECT_ROOT / "models" / "mediapipe" / "hand_landmarker.task"

IMG_SIZE = 224

CLASS_NAMES = ["rock", "paper", "scissors"]
IDX_TO_CLASS = {0: "rock", 1: "paper", 2: "scissors"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes=3):
    model = models.mobilenet_v2(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, num_classes),
    )

    return model


def load_model(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def create_hand_landmarker(model_path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe hand landmarker model not found: {model_path}\n"
            "Download it to models/mediapipe/hand_landmarker.task"
        )

    base_options = python.BaseOptions(model_asset_path=str(model_path))

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    return vision.HandLandmarker.create_from_options(options)


def get_hand_bbox_from_landmarks(hand_landmarks, frame_width, frame_height, padding=50):
    x_coords = [landmark.x for landmark in hand_landmarks]
    y_coords = [landmark.y for landmark in hand_landmarks]

    x_min = int(min(x_coords) * frame_width) - padding
    x_max = int(max(x_coords) * frame_width) + padding
    y_min = int(min(y_coords) * frame_height) - padding
    y_max = int(max(y_coords) * frame_height) + padding

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame_width, x_max)
    y_max = min(frame_height, y_max)

    return x_min, y_min, x_max, y_max


def preprocess_crop(hand_crop, transform, device):
    rgb_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_crop)

    input_tensor = transform(pil_image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    return input_tensor


def predict_crop(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    predicted_class = IDX_TO_CLASS[prediction.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score


def draw_landmarks(frame, hand_landmarks):
    height, width, _ = frame.shape

    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)


def main():
    device = get_device()

    print(f"Using device: {device}")
    print(f"Loading classifier checkpoint from: {CHECKPOINT_PATH}")
    print(f"Loading MediaPipe model from: {HAND_LANDMARKER_PATH}")

    classifier = load_model(CHECKPOINT_PATH, device)
    hand_landmarker = create_hand_landmarker(HAND_LANDMARKER_PATH)

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not access webcam.")

    print("MediaPipe Tasks webcam inference started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame.")
                break

            frame_height, frame_width, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            detection_result = hand_landmarker.detect(mp_image)

            display_text = "No hand detected"

            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]

                x_min, y_min, x_max, y_max = get_hand_bbox_from_landmarks(
                    hand_landmarks=hand_landmarks,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    padding=50,
                )

                hand_crop = frame[y_min:y_max, x_min:x_max]

                if hand_crop.size > 0:
                    input_tensor = preprocess_crop(
                        hand_crop=hand_crop,
                        transform=transform,
                        device=device,
                    )

                    predicted_class, confidence_score = predict_crop(
                        classifier,
                        input_tensor,
                    )

                    display_text = f"{predicted_class}: {confidence_score:.2f}"

                    cv2.rectangle(
                        frame,
                        (x_min, y_min),
                        (x_max, y_max),
                        (0, 255, 0),
                        2,
                    )

                    draw_landmarks(frame, hand_landmarks)

            cv2.putText(
                frame,
                display_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("MediaPipe Hand-Cropped Gesture Classification", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        for _ in range(10):
            cv2.waitKey(1)


if __name__ == "__main__":
    main()