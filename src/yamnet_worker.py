"""
YAMNet-Worker: laeuft als Subprocess, isoliert von librosa/pandas.

Aufruf: python yamnet_worker.py <temp_dir>
Eingabe: Verzeichnis mit .wav-Dateien (16 kHz, mono, float32)
Ausgabe: JSON-Objekt {"dateiname.wav": bird_score, ...} auf stdout
"""

import json
import os
import sys
from pathlib import Path

# Persistentes Cache-Verzeichnis fuer TF-Hub-Modelle (verhindert Re-Download).
_CACHE = Path.home() / ".cache" / "tfhub_modules"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TFHUB_CACHE_DIR", str(_CACHE))

# WICHTIG: pyarrow NICHT importieren.
# Frueher stand hier `import pyarrow` als angeblicher Workaround gegen einen
# Arrow/TF-dylib-Crash. Auf macOS 26 + TensorFlow 2.21 fuehrt das pyarrow-vor-TF
# importieren stattdessen zu einem DEADLOCK beim ersten TF-Op (model(...) bzw.
# class_map_path() haengen endlos). Diagnose: ohne pyarrow laeuft die Inferenz in
# ~0,1 s; mit pyarrow haengt sie unbegrenzt. Der Worker nutzt Arrow ohnehin nicht.
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Sofortige Lebenszeichen-Meldung: der folgende TensorFlow-Import dauert auf macOS
# einmalig ~15-30 s. Ohne diese Zeile sieht man so lange gar nichts und denkt,
# der Prozess haengt.
print("[YAMNet] Importiere TensorFlow (einmalig, kann ~15-30 s dauern) ...",
      file=sys.stderr, flush=True)
import time
_t0 = time.perf_counter()
import tensorflow_hub as hub
print(f"[YAMNet] TensorFlow geladen ({time.perf_counter() - _t0:.1f} s).",
      file=sys.stderr, flush=True)

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"


def _cached_savedmodel() -> Path | None:
    """YAMNet-SavedModel im TF-Hub-Cache finden.

    Eindeutig ueber das YAMNet-spezifische Asset yamnet_class_map.csv — so wird
    nicht versehentlich ein anderes gecachtes TF-Hub-Modell geladen.
    """
    for sub in _CACHE.glob("*"):
        if (sub / "saved_model.pb").exists() and (sub / "assets" / "yamnet_class_map.csv").exists():
            return sub
    return None


def load_yamnet():
    """YAMNet laden — bevorzugt direkt aus dem lokalen Cache (offline, schnell).

    hub.load(URL) loest die tfhub.dev-Adresse erst uebers Netz auf; seit der
    Migration zu Kaggle (2024) haengt/bremst das auch bei vorhandenem Cache.
    Liegt das Modell bereits entpackt im Cache, laden wir es direkt vom Pfad.
    """
    t0 = time.perf_counter()
    cached = _cached_savedmodel()
    if cached is not None:
        print(f"[YAMNet] Lade Modell aus Cache (offline): {cached}", file=sys.stderr, flush=True)
        model = hub.load(str(cached))
    else:
        print(f"[YAMNet] Kein Cache gefunden — lade einmalig von {YAMNET_URL} ...",
              file=sys.stderr, flush=True)
        model = hub.load(YAMNET_URL)
    print(f"[YAMNet] Modell geladen ({time.perf_counter() - t0:.1f} s).",
          file=sys.stderr, flush=True)
    return model


def main():
    if len(sys.argv) != 2:
        print("Usage: yamnet_worker.py <temp_dir>", file=sys.stderr)
        sys.exit(1)

    temp_dir = Path(sys.argv[1])

    model = load_yamnet()

    # --- Schritt-fuer-Schritt-Logging, um Haenger exakt zu lokalisieren --------
    print("[YAMNet] Lese Klassen-Map (class_map_path) ...", file=sys.stderr, flush=True)
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path) as f:
        class_names = [line.strip().split(",")[2] for line in f.readlines()[1:]]

    # Vogelrelevante AudioSet-Klassen. Fuer unsere Singvoegel (Amsel/Kohlmeise/
    # Rotkehlchen) decken diese Stichworte die passenden Klassen ab, u. a.
    # "Bird", "Bird vocalization, bird call, bird song" und "Chirp, tweet".
    bird_kws = ("bird", "chirp", "tweet")
    bird_idx = [i for i, name in enumerate(class_names)
                if any(k in name.lower() for k in bird_kws)]
    print(f"[YAMNet] {len(class_names)} Klassen, davon {len(bird_idx)} Vogel-Klassen.",
          file=sys.stderr, flush=True)

    wav_files = sorted(temp_dir.glob("*.wav"))
    total = len(wav_files)
    print(f"[YAMNet] {total} Segmente gefunden.", file=sys.stderr, flush=True)

    # Warm-up: der ERSTE Inferenz-Aufruf baut den TF-Graph auf (einmalig, kann
    # dauern). Separat geloggt, damit ein Haenger hier eindeutig erkennbar ist.
    if wav_files:
        print("[YAMNet] Erster Inferenz-Aufruf (TF-Graph-Aufbau, einmalig) ...",
              file=sys.stderr, flush=True)
        tw = time.perf_counter()
        y0, _ = sf.read(str(wav_files[0]), dtype="float32")
        if y0.ndim > 1:
            y0 = y0.mean(axis=1)
        _ = model(y0)
        print(f"[YAMNet] Erster Aufruf OK ({time.perf_counter() - tw:.1f} s). "
              "Starte Scoring ...", file=sys.stderr, flush=True)

    # tqdm-Leiste auf stderr (stdout bleibt fuer das JSON-Ergebnis reserviert).
    # Zeigt Fortschritt, Rate (Clips/s) und ETA live im Terminal.
    results = {}
    for wav_file in tqdm(wav_files, total=total, desc="[YAMNet] Scoring",
                         unit="clip", file=sys.stderr, dynamic_ncols=True):
        y, _ = sf.read(str(wav_file), dtype="float32")
        if y.ndim > 1:                       # Sicherheitsnetz: zu Mono mitteln
            y = y.mean(axis=1)

        # YAMNet liefert pro ~0,48-s-Frame Scores fuer alle 521 Klassen.
        scores, _, _ = model(y)
        s = scores.numpy()                   # Shape (Frames, 521)

        if bird_idx:
            # Pro Frame die staerkste Vogelklasse, dann das MAXIMUM ueber die Zeit:
            # "Ist in IRGENDEINEM Zeitfenster ein Vogel hoerbar?". Praesenz-Erkennung
            # statt Mittelwert ueber den ganzen Clip — so gehen kurze Gesangsphasen
            # (z. B. ein einzelner Ruf in 5 s) nicht im Durchschnitt unter.
            bird_score = float(s[:, bird_idx].max())
        else:
            bird_score = 0.0
        results[wav_file.name] = bird_score

    print(json.dumps(results))


if __name__ == "__main__":
    main()
