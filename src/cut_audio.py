# cut_audio.py
# Schneidet lange MP3-Aufnahmen in 5-Sekunden-WAV-Clips.
# Bewertet jeden Clip auf Vogelgeraeusche via librosa-Spektral-Features
# (Ersatz fuer YAMNet ohne TensorFlow-Abhaengigkeit).

import os
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

# --- ANPASSEN , je nach Vogel--------------------------------------------------
BIRD_NAME = "Amsel"

BASE_DIR = Path("data")
INPUT_DIR  = BASE_DIR / "Amsel"      / "files"
OUTPUT_DIR = BASE_DIR / "Amsel"      / "clips"   # Hard Negatives (klar hoerbarer Vogel)

# so lassen: 
OUTPUT_BG  = BASE_DIR / "Background" / "clips"   # Background-Clips (schwacher Vogelanteil)

TARGET_LENGTH = 5      # Sekunden
SAMPLE_RATE = 16000

# Schwellen fuer den Vogel-Score (0-1).
# Tipp: falls zu viele oder zu wenige Clips gespeichert werden,
# BACKGROUND_THRESHOLD leicht anpassen (z. B. 0.10 oder 0.20).
HARD_NEGATIVE_THRESHOLD = 0.40
BACKGROUND_THRESHOLD    = 0.15
# -----------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_BG.mkdir(parents=True, exist_ok=True)


def bird_score(y: np.ndarray) -> float:
    """
    Librosa-basierter Vogelanteil-Score in [0, 1].

    Kombiniert zwei Merkmale:
      1. Energie im Vogelfrequenzbereich (> 1 kHz, Mel-Bins 45-127)
      2. Tonalitaet (niedrige spektrale Flachheit => tonales, vogelaehnliches Signal)

    Schwelle 0.40 trennt klar hoerbaren Vogel von schwachem Vogelanteil;
    0.15 trennt schwachen Vogelanteil von Stille / reinem Hintergrundrauschen.
    """
    if np.abs(y).max() < 1e-3:   # praktisch stilles Segment
        return 0.0

    # Mel-Spektrogramm (Energie pro Zeit-Frequenz-Zelle)
    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128)

    # Anteil der Energie oberhalb ~1 kHz (Vogelbereich)
    # Bei sr=16 kHz und n_mels=128 liegt Bin 45 bei ca. 1 kHz
    total = float(S.sum()) + 1e-12
    ratio = float(S[45:, :].sum()) / total

    # Spektrale Flachheit: nahe 0 => tonal (Vogelruf), nahe 1 => Rauschen
    flatness  = float(librosa.feature.spectral_flatness(y=y).mean())
    tonality  = float(np.exp(-8.0 * flatness))   # 1 bei Flachheit=0, ~0 ab 0.3

    return float(np.clip(0.6 * ratio + 0.4 * tonality, 0.0, 1.0))


hardneg_count  = 0
background_count = 0

files = sorted(f for f in os.listdir(INPUT_DIR)
               if f.startswith(BIRD_NAME) and f.endswith(".mp3"))
print(f"Gefunden: {len(files)} MP3-Dateien")

for file in files:
    path = INPUT_DIR / file
    print(f"Verarbeite {file}")

    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        step = TARGET_LENGTH * sr

        for i in range(0, len(y), step):
            segment = y[i : i + step]
            if len(segment) < step:
                continue

            score = bird_score(segment)
            out_name = None

            if score >= HARD_NEGATIVE_THRESHOLD:
                out_name = f"{BIRD_NAME}_{file[:-4]}_{i}_hardneg.wav"
                hardneg_count += 1
            elif score >= BACKGROUND_THRESHOLD:
                out_name = f"{BIRD_NAME}_{file[:-4]}_{i}_bg.wav"
                background_count += 1

            if out_name:
                dest = OUTPUT_BG if out_name.endswith("_bg.wav") else OUTPUT_DIR
                sf.write(dest / out_name, segment, SAMPLE_RATE)

    except Exception as e:
        print(f"Fehler bei {file}: {e}")

print(f"Fertig. Hard Negatives: {hardneg_count}, Background: {background_count}")
