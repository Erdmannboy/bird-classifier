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

# pyarrow MUSS vor tensorflow importiert werden: Arrow's dylib-Initialisierer
# laeuft dann sauber durch, bevor TF seinerseits Arrow als transitive Abhaengigkeit
# laedt. Wird Arrow danach nochmals per dlopen angefordert, gibt der Linker den
# bereits initialisierten Handle zurueck und ueberspring den Initialisierer –
# vermeidet den pthread_mutex EINVAL-Absturz auf macOS 26+.
import pyarrow  # noqa: F401
import numpy as np
import soundfile as sf
import tensorflow_hub as hub


def main():
    if len(sys.argv) != 2:
        print("Usage: yamnet_worker.py <temp_dir>", file=sys.stderr)
        sys.exit(1)

    temp_dir = Path(sys.argv[1])

    print(f"[YAMNet] Lade Modell (Cache: {_CACHE}) ...", file=sys.stderr, flush=True)
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path) as f:
        class_names = [line.strip().split(",")[2] for line in f.readlines()[1:]]

    bird_kws = ("bird", "chirp", "tweet")

    wav_files = sorted(temp_dir.glob("*.wav"))
    total = len(wav_files)
    print(f"[YAMNet] Modell geladen, starte Scoring von {total} Segmenten ...", file=sys.stderr)

    results = {}
    for idx, wav_file in enumerate(wav_files, start=1):
        y, _ = sf.read(str(wav_file), dtype="float32")
        scores, _, _ = model(y)
        mean_scores = scores.numpy().mean(axis=0)
        bird_score = max(
            (float(mean_scores[i]) for i, name in enumerate(class_names)
             if any(k in name.lower() for k in bird_kws)),
            default=0.0,
        )
        results[wav_file.name] = bird_score

        if idx % 100 == 0 or idx == total:
            print(f"[YAMNet] {idx}/{total} Segmente gescort ...", file=sys.stderr, flush=True)

    print(json.dumps(results))


if __name__ == "__main__":
    main()
