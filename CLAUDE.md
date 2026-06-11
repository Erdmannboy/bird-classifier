# CLAUDE.md — Bird Species Classifier

## Projektbeschreibung

Ein Proof-of-Concept-System, das Vogelgesang aus kurzen Audioaufnahmen (5 s)
klassifiziert. Erkannt werden drei heimische Arten — Amsel, Kohlmeise, Rotkehlchen —
sowie eine vierte Klasse „Background" (kein Zielvogel hörbar). Eine Streamlit-App
vergleicht das eigene BirdCNN-Modell live mit dem bekannten System BirdNET.

---

## Tech-Stack

| Komponente | Bibliothek / Werkzeug |
|---|---|
| Deep Learning | PyTorch |
| Audio-Features | librosa, soundfile |
| Daten-Split / Metriken | scikit-learn |
| Daten-Bereinigung | TensorFlow Hub / YAMNet |
| BirdNET-Vergleich | birdnetlib |
| Web-App | Streamlit |
| Daten-Download | requests (Xeno-Canto API) |
| Notebooks | JupyterLab / notebook / ipykernel |
| Visualisierung | matplotlib, plotly |

---

## Wichtige Befehle

### Setup

```bash
# Abhängigkeiten installieren (erzeugt .venv + uv.lock)
# Enthält auch birdnetlib für den BirdNET-Vergleich.
uv sync

# Umgebung testen
uv run python setup_check.py
```

### Datenbeschaffung & Vorverarbeitung

```bash
# 1. MP3-Aufnahmen von Xeno-Canto herunterladen
#    Vorher: API_KEY in src/bird_data.py eintragen
uv run python src/bird_data.py

# 2. Lange Aufnahmen in 5-s-WAV-Clips schneiden + YAMNet-Filterung
uv run python src/cut_audio.py

# 3. Train / Val / Test-Splits als CSV erzeugen (data_splits/)
uv run python src/build_dataset.py
```

### Training

```bash
# Notebook im Browser öffnen (Projekt-Root als Arbeitsverzeichnis)
uv run jupyter lab notebooks/bird_training.ipynb
# Ergebnis: models/birdcnn_<timestamp>_best.pth (eindeutiger Name pro Lauf,
# überschreibt nie das mitgelieferte models/birdcnn_release.pth)
```

### Inferenz / App starten

```bash
# Lädt standardmäßig models/birdcnn_release.pth; per Dropdown (Sidebar) kann
# jedes weitere models/*.pth ausgewählt werden.
uv run streamlit run app.py

# Festen Modell-Pfad erzwingen (hat Vorrang vor dem Dropdown)
BIRD_MODEL_PATH=/pfad/zu/modell.pth uv run streamlit run app.py
```

### Tests

> TODO: Keine automatisierte Test-Suite vorhanden.
> Funktionscheck der Umgebung via `uv run python setup_check.py`.

### Lint

> TODO: Kein Linter konfiguriert.

---

## Repo-Konventionen

### Ordnerstruktur

```
bird-classifier/
├── src/            # Datenpipeline-Skripte (Download, Schneiden, Split)
├── notebooks/      # Trainings-Notebook
├── data/           # Rohdaten (MP3s + WAV-Clips) — nicht committed
├── data_splits/    # CSV-Splits (train/val/test) — nicht committed
├── docs/           # Projektdokumentation (structure.md, crisp-dm.md)
├── app.py          # Streamlit-Anwendung (Einstiegspunkt)
├── models/         # Modell-Artefakte
│   ├── birdcnn_release.pth          # mitgeliefertes Modell (committed)
│   └── birdcnn_<timestamp>_best.pth # selbst trainiert (gitignored)
├── pyproject.toml      # Abhängigkeitsdefinition (UV)
├── uv.lock             # Reproduzierbares Lockfile
├── setup_check.py
├── project.md      # Ausführliche ML4B-Projektdoku (DE)
└── CLAUDE.md       # Diese Datei
```

### Namensgebung

- Python-Skripte: `snake_case.py`
- Notebooks: beschreibender Name (`bird_training.ipynb`)
- Modell-Artefakte (in `models/`): `birdcnn_release.pth` (mitgeliefert, committed),
  `birdcnn_<timestamp>_best.pth` (selbst trainiert, gitignored). Die App lädt
  standardmäßig das Release-Modell; weitere `models/*.pth` sind im Dropdown wählbar.
- CSV-Splits: `train.csv`, `val.csv`, `test.csv` → liegen in `data_splits/`
- Klassen-Reihenfolge (fest): `Amsel=0, Kohlmeise=1, Rotkehlchen=2, Background=3`

### Dokumentationssprache

Alle Dokumente (CLAUDE.md, README.md, docs/) werden auf **Deutsch** geschrieben.
Code-Kommentare dürfen Deutsch oder Englisch sein.

---

## Dokumentations-Workflow

Für **jeden** Dokumentationsschritt wird ein eigener Feature-Branch erstellt.
Kein direkter Commit oder Push auf `main`.

```bash
# Vor jedem Schritt: zurück auf main und aktualisieren
git checkout main && git pull origin main

# Neuen Branch anlegen
git checkout -b docs/<name>   # z. B. docs/readme, docs/structure, docs/crisp-dm

# ... Änderungen vornehmen ...

# Committen und pushen
git add <dateien>
git commit -m "docs: <kurze Beschreibung>"
git push -u origin docs/<name>
```

Conventional Commits verwenden: `docs:`, `fix:`, `feat:` etc.
Danach Pull Request öffnen und reviewen lassen — kein direktes Merge in `main`.

---

## Hinweise für zukünftige Claude-Code-Sessions

### Do's

- Metriken, Dateianzahlen und Ergebnisse ausschließlich aus `project.md` oder
  dem Notebook `notebooks/bird_training.ipynb` belegen.
- Vor Änderungen an der Vorverarbeitung sowohl `app.py` als auch das Notebook
  prüfen — Parameter müssen übereinstimmen (32 kHz, 128 Mel-Bänder, 313 Frames,
  Normalisierung auf mean=0, std=1).
- BirdNET läuft absichtlich in einem Subprocess (`app.py`, Zeile ~138).
  Grund: PyTorch und TensorFlow-Lite vertragen sich im selben Prozess nicht.
- `data/` und `data_splits/` sind gitignored — nur `.gitkeep` ist committed.
  Keine Binärdaten oder großen Audiodateien committen.

### Don'ts

- Keine Metriken, Dateianzahlen oder Ergebnisse erfinden oder schätzen.
  Unbekanntes als `> TODO:` markieren.
- Nicht direkt auf `main` committen oder pushen.
- Die Klassen-Reihenfolge `[Amsel=0, Kohlmeise=1, Rotkehlchen=2, Background=3]`
  nicht ohne Anpassung von `app.py` und Notebook ändern.
- Keine neuen Abhängigkeiten ohne Eintrag in `pyproject.toml` einführen.
  Danach `uv sync` ausführen, damit `uv.lock` aktualisiert wird.
- `models/birdcnn_release.pth` ist ein committetes Binär-Artefakt — nicht in
  Textdokumenten als Quellcode behandeln oder verändern. Training überschreibt es
  nie (eindeutige Zeitstempel-Namen); ein neues Release wird bewusst durch
  Umbenennen nach `birdcnn_release.pth` gesetzt.
