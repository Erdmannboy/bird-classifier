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


MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL = "birdcnn_release_mit_yamnet.pth"
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


def load_and_check(model_path: Path) -> None:
    """Lädt ein BirdCNN-State-Dict und prüft die Ausgabeform (1, 4)."""
    model = BirdCNN(num_classes=len(MODEL_CLASSES))
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.zeros((1, 1, 128, 313), dtype=torch.float32)
    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (1, len(MODEL_CLASSES))
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Unerwartete Modell-Ausgabeform ({model_path.name}): {tuple(output.shape)}. "
            f"Erwartet: {expected_shape}."
        )


def check_model_files() -> None:
    if not MODELS_DIR.is_dir():
        raise FileNotFoundError(
            f"Modell-Ordner fehlt: {MODELS_DIR}. "
            "Erwartet werden die Release-Modelle in models/."
        )

    models = sorted(MODELS_DIR.glob("*.pth"))
    if not models:
        raise FileNotFoundError(
            f"Keine Modelldatei in {MODELS_DIR} gefunden. Erwartet wird mindestens "
            f"{DEFAULT_MODEL}. Alternativ ein eigenes Modell über das Notebook trainieren."
        )

    default_path = MODELS_DIR / DEFAULT_MODEL
    if not default_path.exists():
        print(
            f"Hinweis: Standard-Modell {DEFAULT_MODEL} fehlt — die App nutzt dann das "
            f"erste verfügbare Modell oder BIRD_MODEL_PATH."
        )

    for model_path in models:
        load_and_check(model_path)
        print(f"Modell OK: {model_path.name}")


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

    check_model_files()

    print("Setup ist korrekt.")
    print("Alle gefundenen Modelle konnten erfolgreich geladen werden.")


if __name__ == "__main__":
    main()