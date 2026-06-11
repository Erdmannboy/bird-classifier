# cut_audio.py
# Schneidet Vogelaufnahmen in 5-Sekunden-WAV-Clips und sortiert sie per YAMNet.
#
# Ablauf:
#   1. Alle Aufnahmen werden in 5-s-Segmente geschnitten (librosa, 32 kHz —
#      identisch zu Notebook/app.py). Pro Segment wird zusaetzlich eine 16-kHz-
#      Kopie fuer YAMNet abgelegt.
#   2. YAMNet (isolierter Subprocess, src/yamnet_worker.py) bewertet je Segment
#      den Vogel-Anteil [0, 1].
#   3. Routing:
#        - PROCESS_SPECIES: Segment MIT Vogel        -> data/<Art>/clips/
#                           Segment OHNE Vogel        -> data/Background/clips/
#          (Stille, Schweigen, Rauschen aus Zielart-Aufnahmen MUESSEN in den
#           Background — das ist beabsichtigt und wichtig fuer die Klasse.)
#        - BACKGROUND_SPECIES: ALLE Segmente          -> data/Background/clips/
#          (Spatz/Taube/Kraehe sind selbst Voegel, nur eben nicht unsere Zielarten
#           — deshalb hier bewusst KEIN YAMNet-Filter.)
#
# YAMNet laeuft als Subprocess, weil TensorFlow und librosa/PyTorch sich im selben
# Prozess auf macOS nicht zuverlaessig vertragen (vgl. app.py / yamnet_worker.py).

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm

# --- ANPASSEN ----------------------------------------------------------------
# Ziel-Arten: Clips mit Vogel -> data/<Art>/clips/, ohne Vogel -> Background.
PROCESS_SPECIES = ["Amsel", "Kohlmeise", "Rotkehlchen"]

# Background-Arten: ALLE Clips -> data/Background/clips/ (kein YAMNet-Filter).
BACKGROUND_SPECIES = ["Spatz", "Taube", "Krähe"]

# Basisordner fuer die Daten (immer relativ zum Projektordner, egal von wo aufgerufen).
BASE_DIR = Path(__file__).parent.parent / "data"

TARGET_LENGTH = 5      # Sekunden
# 32 kHz: identisch zu Notebook (audio_to_mel) und app.py. Das ist die Sample-Rate
# der finalen Clips, mit denen trainiert/inferiert wird.
SAMPLE_RATE   = 32000
# YAMNet erwartet exakt 16 kHz mono — dafuer wird pro Segment eine Kopie erzeugt.
YAMNET_SR     = 16000

# YAMNet-Vogel-Score [0, 1], ab dem ein Zielart-Segment als "Vogel vorhanden" gilt.
# Darunter -> Background (Stille/Rauschen aus Zielart-Aufnahmen landen hier).
# Hoeher  = strengere Zielklasse (mehr Segmente wandern in Background),
# niedriger = mehr (auch leise/zweifelhafte) Segmente bleiben in der Zielklasse.
BIRD_PRESENCE_THRESHOLD = 0.20
# -----------------------------------------------------------------------------

OUTPUT_BG = BASE_DIR / "Background" / "clips"
YAMNET_WORKER = Path(__file__).parent / "yamnet_worker.py"

for _name in BACKGROUND_SPECIES:
    if _name in PROCESS_SPECIES:
        raise SystemExit(
            f"'{_name}' steht sowohl in PROCESS_SPECIES als auch BACKGROUND_SPECIES. "
            "Bitte nur einer Liste zuordnen."
        )


def _mp3_files(input_dir: Path, name: str) -> list[Path]:
    """MP3s in input_dir, deren Name mit der Art beginnt. Fehlt der Ordner → []."""
    if not input_dir.is_dir():
        tqdm.write(f"[{name}] Ordner fehlt: {input_dir}")
        return []
    return sorted(p for p in input_dir.iterdir()
                  if p.name.startswith(name) and p.suffix == ".mp3")


def segment_process_species(name: str, stage32: Path, stage16: Path,
                            position: int) -> dict[str, str]:
    """Schneidet eine Ziel-Art und stagt 32-kHz- + 16-kHz-Segmente.

    Gibt ein Manifest {stem: art} zurueck; das Routing entscheidet spaeter
    anhand der YAMNet-Scores, wohin jedes Segment wandert.
    """
    files = _mp3_files(BASE_DIR / name / "files", name)
    manifest: dict[str, str] = {}
    if not files:
        return manifest

    step = TARGET_LENGTH * SAMPLE_RATE
    bar = tqdm(total=len(files), desc=f"{name} schneiden", unit="file",
               position=position, leave=True, dynamic_ncols=True)

    for path in files:
        try:
            y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            for i in range(0, len(y), step):
                segment = y[i:i + step]
                if len(segment) < step:
                    continue
                # stem ist global eindeutig (Art-Praefix + Quelldatei + Offset).
                stem = f"{name}_{path.stem}_{i}"
                sf.write(stage32 / f"{stem}.wav", segment, SAMPLE_RATE)
                seg16 = librosa.resample(segment, orig_sr=SAMPLE_RATE, target_sr=YAMNET_SR)
                sf.write(stage16 / f"{stem}.wav", seg16, YAMNET_SR)
                manifest[stem] = name
        except Exception as e:
            tqdm.write(f"[{name}] Fehler bei {path.name}: {e}")
        bar.update(1)

    bar.close()
    return manifest


def process_background_species(name: str, position: int) -> int:
    """Schneidet eine Background-Art; ALLE Segmente -> data/Background/clips/.

    Background-Aufnahmen liegen gemeinsam in data/Background/files/ und werden
    nur nach Namens-Praefix der Art gefiltert. Kein YAMNet — alles ist Background.
    """
    files = _mp3_files(BASE_DIR / "Background" / "files", name)
    if not files:
        return 0

    step = TARGET_LENGTH * SAMPLE_RATE
    count = 0
    bar = tqdm(total=len(files), desc=f"{name} (BG)", unit="file",
               position=position, leave=True, dynamic_ncols=True)

    for path in files:
        try:
            y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            for i in range(0, len(y), step):
                segment = y[i:i + step]
                if len(segment) < step:
                    continue
                out_name = f"{name}_{path.stem}_{i}_bg.wav"
                sf.write(OUTPUT_BG / out_name, segment, SAMPLE_RATE)
                count += 1
        except Exception as e:
            tqdm.write(f"[{name}] Fehler bei {path.name}: {e}")
        bar.update(1)

    bar.close()
    tqdm.write(f"[{name}] Fertig — {count} Background-Clips")
    return count


def run_yamnet(yamnet_dir: Path) -> dict[str, float]:
    """Ruft den YAMNet-Worker als Subprocess auf -> {stem: bird_score}."""
    n = len(list(yamnet_dir.glob("*.wav")))
    if n == 0:
        return {}

    tqdm.write(f"\n[YAMNet] Bewerte {n} Zielart-Segmente im Subprocess ...")
    # stderr (Modell-Laden + Fortschritt) erbt die Konsole; nur stdout = JSON.
    result = subprocess.run(
        [sys.executable, str(YAMNET_WORKER), str(yamnet_dir)],
        stdout=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"[YAMNet] Subprocess fehlgeschlagen (Code {result.returncode}). "
            "Siehe stderr oben."
        )
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise SystemExit(
            f"[YAMNet] Konnte Worker-Output nicht parsen: {result.stdout[:300]!r}"
        )
    # Worker liefert {dateiname.wav: score} -> auf stem reduzieren.
    return {Path(k).stem: float(v) for k, v in raw.items()}


def route_by_score(manifest: dict[str, str], scores: dict[str, float],
                   stage32: Path) -> tuple[int, int]:
    """Verschiebt gestagte 32-kHz-Clips anhand der YAMNet-Scores an ihr Ziel."""
    clips = 0
    bg = 0
    missing = 0

    for stem, species in tqdm(manifest.items(), desc="Routing", unit="clip",
                              dynamic_ncols=True):
        src = stage32 / f"{stem}.wav"
        if not src.exists():
            continue

        score = scores.get(stem)
        if score is None:
            # Kein Score (Worker hat Segment nicht bewertet) -> sicherheitshalber
            # als "kein Vogel" behandeln und in den Background legen.
            missing += 1
            score = 0.0

        if score >= BIRD_PRESENCE_THRESHOLD:
            dest = BASE_DIR / species / "clips" / f"{stem}.wav"
            clips += 1
        else:
            # Stille / Rauschen / kein Zielvogel -> Background (beabsichtigt!).
            dest = OUTPUT_BG / f"{stem}_nobird_bg.wav"
            bg += 1

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))

    if missing:
        tqdm.write(f"[Routing] ⚠️ {missing} Segmente ohne YAMNet-Score → als Background behandelt.")
    return clips, bg


if __name__ == "__main__":
    if not (PROCESS_SPECIES or BACKGROUND_SPECIES):
        raise SystemExit("Keine Arten konfiguriert. Bitte PROCESS_SPECIES oder BACKGROUND_SPECIES befuellen.")
    if PROCESS_SPECIES and not YAMNET_WORKER.exists():
        raise SystemExit(f"YAMNet-Worker nicht gefunden: {YAMNET_WORKER}")

    OUTPUT_BG.mkdir(parents=True, exist_ok=True)
    for _name in PROCESS_SPECIES:
        (BASE_DIR / _name / "clips").mkdir(parents=True, exist_ok=True)

    tqdm.write(f"Ziel-Arten:       {', '.join(PROCESS_SPECIES) if PROCESS_SPECIES else '(keine)'}")
    tqdm.write(f"Background-Arten: {', '.join(BACKGROUND_SPECIES) if BACKGROUND_SPECIES else '(keine)'}")
    tqdm.write(f"YAMNet-Schwelle:  {BIRD_PRESENCE_THRESHOLD}\n")

    with tempfile.TemporaryDirectory(prefix="cutaudio_") as _tmp:
        tmp = Path(_tmp)
        stage32 = tmp / "clips32"   # finale 32-kHz-Clips (werden spaeter verschoben)
        stage16 = tmp / "yamnet16"  # 16-kHz-Kopien nur fuer YAMNet
        stage32.mkdir()
        stage16.mkdir()

        # --- Phase 1: Segmentieren (parallel pro Art) -----------------------
        jobs = [(n, False) for n in PROCESS_SPECIES] + \
               [(n, True) for n in BACKGROUND_SPECIES]

        manifest: dict[str, str] = {}
        bg_from_species = 0

        with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as executor:
            futures = {}
            for pos, (name, is_bg) in enumerate(jobs):
                if is_bg:
                    fut = executor.submit(process_background_species, name, pos)
                    futures[fut] = ("bg", name)
                else:
                    fut = executor.submit(segment_process_species, name, stage32, stage16, pos)
                    futures[fut] = ("proc", name)

            for future in as_completed(futures):
                kind, name = futures[future]
                try:
                    res = future.result()
                    if kind == "bg":
                        bg_from_species += res
                    else:
                        manifest.update(res)
                except Exception as e:
                    tqdm.write(f"[{name}] Fehler: {e}")

        # --- Phase 2: YAMNet bewertet die Zielart-Segmente ------------------
        scores = run_yamnet(stage16)

        # --- Phase 3: Routing anhand der Scores -----------------------------
        clips, bg_from_targets = route_by_score(manifest, scores, stage32)

    tqdm.write(
        f"\nFertig.\n"
        f"  Ziel-Clips (mit Vogel):              {clips}\n"
        f"  Background aus Zielarten (kein Vogel): {bg_from_targets}\n"
        f"  Background aus Background-Arten:       {bg_from_species}\n"
        f"  Background gesamt:                     {bg_from_targets + bg_from_species}"
    )
