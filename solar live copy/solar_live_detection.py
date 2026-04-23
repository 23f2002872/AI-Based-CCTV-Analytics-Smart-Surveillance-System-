import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import time

# -----------------------------
# 1️⃣ Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2️⃣ Recreate Your ResNetCustom
# -----------------------------
class ResNetCustom(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        base_model = models.resnet50(weights=None)  # IMPORTANT: weights=None for loading state_dict

        # Freeze all layers
        for param in base_model.parameters():
            param.requires_grad = False

        for name, param in base_model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True

        self.features = nn.Sequential(*list(base_model.children())[:-2])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# -----------------------------
# 3️⃣ Load Model Weights
# -----------------------------
num_classes = 6

model = ResNetCustom(num_classes=num_classes)
model.load_state_dict(torch.load("resnet_custom_model.pt", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 4️⃣ Class Names (from your notebook)
# -----------------------------
class_names = [
    'Bird-drop',
    'Clean',
    'Dusty',
    'Electrical-damage',
    'Physical-Damage',
    'Snow-Covered'
]

# -----------------------------
# 5️⃣ Transform (MUST match training)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# 6️⃣ Start Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

prev_time = 0

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Convert OpenCV BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    label = class_names[predicted.item()]
    conf_score = confidence.item() * 100

    # Display Prediction
    cv2.putText(frame, f"{label} ({conf_score:.2f}%)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2)

    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2)

    cv2.imshow("Solar Panel Fault Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()