# cut_audio.py
# Schneidet lange MP3-Aufnahmen in 5-Sekunden-WAV-Clips.
# YAMNet (vortrainiertes Google-Modell) bewertet pro Clip, wie stark Vogel-
# geraeusche vorkommen. Damit trennen wir "Hard Negatives" (fremder Vogel klar
# hoerbar) von normalem Hintergrund. YAMNet bestimmt NICHT die Vogelart.

import os
from pathlib import Path

import librosa
import soundfile as sf
import tensorflow_hub as hub

# --- ANPASSEN ----------------------------------------------------------------
# Welche Art wird gerade verarbeitet? (steuert nur den Dateinamen-Filter unten)
BIRD_NAME = "Taube"

# Basisordner der Rohdaten und Zielordner fuer die Clips (relativ zum Projekt)
BASE_DIR = Path("data")
INPUT_DIR = BASE_DIR / "Background" / "files"
OUTPUT_DIR = BASE_DIR / "Background" / "clips"

# Clip-Laenge und Sampling-Rate
TARGET_LENGTH = 5      # Sekunden
SAMPLE_RATE = 16000

# Schwellen fuer den YAMNet-Vogel-Score:
#   >= HARD_NEGATIVE_THRESHOLD -> klar hoerbarer (fremder) Vogel
#   >= BACKGROUND_THRESHOLD     -> schwacher/diffuser Hintergrund
HARD_NEGATIVE_THRESHOLD = 0.40
BACKGROUND_THRESHOLD = 0.15
# -----------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# YAMNet einmalig laden.
print("Lade YAMNet ...")
model = hub.load("https://tfhub.dev/google/yamnet/1")

class_map_path = model.class_map_path().numpy().decode("utf-8")
with open(class_map_path) as f:
    class_names = [line.strip().split(",")[2] for line in f.readlines()[1:]]
print("YAMNet geladen.")


def yamnet_bird_score(y):
    # Hoechster Score ueber alle vogel-bezogenen YAMNet-Klassen.
    scores, _, _ = model(y)
    mean_scores = scores.numpy().mean(axis=0)

    bird_score = 0.0
    for i, name in enumerate(class_names):
        name_lower = name.lower()
        if any(k in name_lower for k in ("bird", "chirp", "tweet")):
            bird_score = max(bird_score, mean_scores[i])
    return bird_score


hardneg_count = 0
background_count = 0

for file in os.listdir(INPUT_DIR):

    # Nur Dateien der aktuell gewaehlten Art und nur MP3s.
    if not file.startswith(BIRD_NAME) or not file.endswith(".mp3"):
        continue

    path = INPUT_DIR / file
    print(f"Verarbeite {file}")

    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        step = TARGET_LENGTH * sr

        # In 5-Sekunden-Segmente schneiden.
        for i in range(0, len(y), step):
            segment = y[i:i + step]
            if len(segment) < step:    # zu kurzes Rest-Segment ueberspringen
                continue

            bird_score = yamnet_bird_score(segment)
            out_name = None

            if bird_score >= HARD_NEGATIVE_THRESHOLD:
                out_name = f"{BIRD_NAME}_{file[:-4]}_{i}_hardneg.wav"
                hardneg_count += 1
            elif bird_score >= BACKGROUND_THRESHOLD:
                out_name = f"{BIRD_NAME}_{file[:-4]}_{i}_bg.wav"
                background_count += 1

            if out_name:
                sf.write(OUTPUT_DIR / out_name, segment, SAMPLE_RATE)

    except Exception as e:
        print("Fehler bei", file, e)

print(f"Fertig. Hard Negatives: {hardneg_count}, Background: {background_count}")