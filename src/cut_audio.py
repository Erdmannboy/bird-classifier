# cut_audio.py
# Schneidet alle konfigurierten Vogelaufnahmen parallel in 5-Sekunden-WAV-Clips.
# Ziel-Arten: Score-basiertes Routing (Clips vs. Background).
# Background-Arten: alle Clips direkt → data/Background/clips/.

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# --- ANPASSEN ----------------------------------------------------------------
# Ziel-Arten: Clips mit hohem Bird-Score → data/<Art>/clips/
#             Clips mit mittlerem Score  → data/Background/clips/
PROCESS_SPECIES = ["Amsel", "Kohlmeise", "Rotkehlchen"]

# Background-Arten: Dateien liegen in data/Background/files/
#                   → ALLE Clips (inkl. Stille) → data/Background/clips/
BACKGROUND_SPECIES = ["Spatz", "Taube", "Krähe"]

# Basisordner für die Daten (immer relativ zum Projektordner, egal von wo aufgerufen)
BASE_DIR = Path(__file__).parent.parent / "data"

TARGET_LENGTH = 5      # Sekunden
# 32 kHz: identisch zu Notebook (audio_to_mel) und app.py. Vorher 16 kHz —
# das führte zu leeren oberen Mel-Bändern und Train/Inferenz-Mismatch.
SAMPLE_RATE   = 32000

# Schwellen für den Vogel-Score (0–1). Nur für PROCESS_SPECIES relevant.
# BACKGROUND_THRESHOLD leicht anpassen falls zu viele/wenige BG-Clips entstehen.
HARD_NEGATIVE_THRESHOLD = 0.40
BACKGROUND_THRESHOLD    = 0.15

# Frequenzband für den Heuristik-Score (vogelrelevant, unabhängig von SAMPLE_RATE).
# Der Score misst den Energieanteil oberhalb 1 kHz; SCORE_FMAX begrenzt den
# betrachteten Bereich, damit die Schwellen bei jeder Sample-Rate gleich wirken.
SCORE_FMAX = 8000
# -----------------------------------------------------------------------------

OUTPUT_BG = BASE_DIR / "Background" / "clips"

# Erster Mel-Bin oberhalb 1 kHz (für bird_score). Einmal berechnet statt hart "45".
_MEL_FREQS = librosa.mel_frequencies(n_mels=128, fmax=SCORE_FMAX)
_BIRD_BIN = int(np.argmax(_MEL_FREQS >= 1000))

for _name in BACKGROUND_SPECIES:
    if _name in PROCESS_SPECIES:
        raise SystemExit(
            f"'{_name}' steht sowohl in PROCESS_SPECIES als auch BACKGROUND_SPECIES. "
            "Bitte nur einer Liste zuordnen."
        )


def bird_score(y: np.ndarray) -> float:
    """
    Librosa-basierter Vogelanteil-Score in [0, 1].

    Kombiniert zwei Merkmale:
      1. Energie im Vogelfrequenzbereich (> 1 kHz)
      2. Tonalität (niedrige spektrale Flachheit => tonales, vogelähnliches Signal)
    """
    if np.abs(y).max() < 1e-3:
        return 0.0
    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128, fmax=SCORE_FMAX)
    total = float(S.sum()) + 1e-12
    ratio = float(S[_BIRD_BIN:, :].sum()) / total
    flatness = float(librosa.feature.spectral_flatness(y=y).mean())
    tonality = float(np.exp(-8.0 * flatness))
    return float(np.clip(0.6 * ratio + 0.4 * tonality, 0.0, 1.0))


def process_species(name: str, is_background: bool, position: int) -> tuple[int, int]:
    """Verarbeitet alle MP3s einer Art. Gibt (clip_count, bg_count) zurück."""
    if is_background:
        input_dir = BASE_DIR / "Background" / "files"
        clip_dir = OUTPUT_BG
    else:
        input_dir = BASE_DIR / name / "files"
        clip_dir = BASE_DIR / name / "clips"
        clip_dir.mkdir(parents=True, exist_ok=True)

    OUTPUT_BG.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in input_dir.iterdir()
                   if p.name.startswith(name) and p.suffix == ".mp3")

    if not files:
        tqdm.write(f"[{name}] Keine MP3-Dateien in {input_dir}")
        return 0, 0

    step = TARGET_LENGTH * SAMPLE_RATE
    clip_count = 0
    bg_count = 0

    bar = tqdm(
        total=len(files),
        desc=f"{name}",
        unit="file",
        position=position,
        leave=True,
        dynamic_ncols=True,
    )

    for path in files:
        try:
            y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

            for i in range(0, len(y), step):
                segment = y[i : i + step]
                if len(segment) < step:
                    continue

                if is_background:
                    out_name = f"{name}_{path.stem}_{i}_bg.wav"
                    sf.write(OUTPUT_BG / out_name, segment, SAMPLE_RATE)
                    bg_count += 1
                else:
                    score = bird_score(segment)
                    if score >= HARD_NEGATIVE_THRESHOLD:
                        out_name = f"{name}_{path.stem}_{i}_hardneg.wav"
                        sf.write(clip_dir / out_name, segment, SAMPLE_RATE)
                        clip_count += 1
                    elif score >= BACKGROUND_THRESHOLD:
                        out_name = f"{name}_{path.stem}_{i}_bg.wav"
                        sf.write(OUTPUT_BG / out_name, segment, SAMPLE_RATE)
                        bg_count += 1

        except Exception as e:
            tqdm.write(f"[{name}] Fehler bei {path.name}: {e}")

        bar.update(1)

    bar.close()
    label = "Clips" if not is_background else "BG-Clips"
    count = clip_count if not is_background else bg_count
    tqdm.write(f"[{name}] Fertig — {count} {label}, {bg_count if not is_background else 0} Background-Clips")
    return clip_count, bg_count


if __name__ == "__main__":
    all_species = [(name, False) for name in PROCESS_SPECIES] + \
                  [(name, True)  for name in BACKGROUND_SPECIES]

    if not all_species:
        raise SystemExit("Keine Arten konfiguriert. Bitte PROCESS_SPECIES oder BACKGROUND_SPECIES befüllen.")

    tqdm.write(f"Ziel-Arten:       {', '.join(PROCESS_SPECIES) if PROCESS_SPECIES else '(keine)'}")
    tqdm.write(f"Background-Arten: {', '.join(BACKGROUND_SPECIES) if BACKGROUND_SPECIES else '(keine)'}\n")

    with ThreadPoolExecutor(max_workers=len(all_species)) as executor:
        futures = {
            executor.submit(process_species, name, is_bg, i): name
            for i, (name, is_bg) in enumerate(all_species)
        }
        total_clips = 0
        total_bg = 0
        for future in as_completed(futures):
            name = futures[future]
            try:
                clips, bg = future.result()
                total_clips += clips
                total_bg += bg
            except Exception as e:
                tqdm.write(f"[{name}] Fehler: {e}")

    tqdm.write(f"\nAlle Arten verarbeitet. Gesamt: {total_clips} Clips, {total_bg} Background-Clips.")
