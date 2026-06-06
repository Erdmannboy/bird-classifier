# ./setup_check.py
from pathlib import Path
import importlib.util
import sys

import librosa
import matplotlib
import numpy
import pandas
import plotly
import sklearn
import soundfile
import streamlit
import torch
import torch.nn as nn


MODEL_PATH = Path("model_best.pth")
MODEL_CLASSES = ["Amsel", "Kohlmeise", "Rotkehlchen", "Background"]


class SpecAugment(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class BirdCNN(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.5):
        super().__init__()
        self.spec_aug = SpecAugment()

        def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def check_optional_package(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def check_model_file() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelldatei fehlt: {MODEL_PATH.resolve()}. "
            "Die App erwartet model_best.pth im Projektordner."
        )

    model = BirdCNN(num_classes=len(MODEL_CLASSES))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.zeros((1, 1, 128, 313), dtype=torch.float32)
    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (1, len(MODEL_CLASSES))
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Unerwartete Modell-Ausgabeform: {tuple(output.shape)}. "
            f"Erwartet: {expected_shape}."
        )


def main() -> None:
    print("ML4B Setup-Test")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"numpy: {numpy.__version__}")
    print(f"pandas: {pandas.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"plotly: {plotly.__version__}")
    print(f"streamlit: {streamlit.__version__}")
    print(f"torch: {torch.__version__}")
    print(f"librosa: {librosa.__version__}")
    print(f"soundfile: {soundfile.__version__}")

    birdnet_available = check_optional_package("birdnetlib")
    tensorflow_available = check_optional_package("tensorflow")
    tensorflow_hub_available = check_optional_package("tensorflow_hub")

    print(f"birdnetlib installiert: {birdnet_available}")
    print(f"tensorflow installiert: {tensorflow_available}")
    print(f"tensorflow_hub installiert: {tensorflow_hub_available}")

    check_model_file()

    print("Setup ist korrekt.")
    print("Modell konnte erfolgreich geladen werden.")


if __name__ == "__main__":
    main()