from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "mobilenet_best.pth"

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


def preprocess_frame(frame, transform, device):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    input_tensor = transform(pil_image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    return input_tensor


def predict_frame(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    predicted_class = IDX_TO_CLASS[prediction.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score


def main():
    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")

    model = load_model(CHECKPOINT_PATH, device)

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

    print("Webcam started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame.")
                break

            input_tensor = preprocess_frame(frame, transform, device)
            predicted_class, confidence_score = predict_frame(model, input_tensor)

            display_text = f"{predicted_class}: {confidence_score:.2f}"

            cv2.putText(
                frame,
                display_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Real-Time Gesture Classification", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        for _ in range(10):
            cv2.waitKey(1)


if __name__ == "__main__":
    main()