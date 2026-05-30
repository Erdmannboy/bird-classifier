# Projektstruktur — Bird Species Classifier

## Verzeichnisbaum

```
bird-classifier/
├── app.py
├── model_best.pth
├── model.pth
├── project.md
├── requirements.txt
├── setup_check.py
├── CLAUDE.md
│
├── data/
│   └── .gitkeep
│
├── data_splits/
│   └── .gitkeep
│
├── docs/
│   ├── structure.md        ← diese Datei
│   └── crisp-dm.md
│
├── notebooks/
│   └── bird_training.ipynb
│
└── src/
    ├── bird_data.py
    ├── cut_audio.py
    └── build_dataset.py
```

> `data/` und `data_splits/` sind gitignored und enthalten lokal die Rohdaten
> bzw. die CSV-Splits. Im Repository liegt jeweils nur ein `.gitkeep`.
> `model_best.pth` und `model.pth` sind Binär-Artefakte und werden direkt
> im Projektordner abgelegt (nicht in einem Unterordner).

---

## Dateien und Ordner im Detail

### Einstiegspunkte

| Pfad | Zweck |
| --- | --- |
| `app.py` | Streamlit-Web-App. Lädt `model_best.pth`, nimmt eine WAV-Datei entgegen (Upload oder Live-Aufnahme), zeigt Mel-Spektrogramm mit wählbarem 5-s-Ausschnitt, gibt CNN-Vorhersage und optionalen BirdNET-Vergleich aus. Einstiegspunkt: `streamlit run app.py`. |
| `setup_check.py` | Einfacher Umgebungstest: gibt Python-Version und Versionsnummern der wichtigsten Bibliotheken aus. Kein Unittest-Framework, nur manueller Smoke-Test. |

### Modell-Artefakte

| Pfad | Zweck |
| --- | --- |
| `model_best.pth` | PyTorch State-Dict des BirdCNN mit der besten Validation-Accuracy (87,32 %). Wird von `app.py` für Inferenz geladen. |
| `model.pth` | PyTorch State-Dict der letzten Trainingsepoche (Epoche 20). Als Referenz gespeichert; für die App wird `model_best.pth` bevorzugt. |

### Konfiguration & Doku

| Pfad | Zweck |
| --- | --- |
| `requirements.txt` | Python-Abhängigkeiten ohne Versionspinning. Sektionen: Grundlagen, Visualisierung, Audio, ML, Streamlit, Xeno-Canto, YAMNet, BirdNET, Notebook, Export. |
| `CLAUDE.md` | Schnellreferenz für Claude-Code-Sessions: Befehle, Konventionen, Do's & Don'ts. |
| `project.md` | Ausführliche ML4B-Projektdokumentation auf Deutsch: Idee, Business Understanding, Daten, Modell, Ergebnisse, Reflexion. |

### Datenpipeline — `src/`

Die drei Skripte müssen in dieser Reihenfolge ausgeführt werden:

| Pfad | Schritt | Zweck |
| --- | --- | --- |
| `src/bird_data.py` | 1 | Lädt MP3-Aufnahmen von der Xeno-Canto API herunter. Konfigurierbar: Art (`SEARCH_BIRD`), Zielordner (`TARGET_CLASS`), maximale Dateianzahl (`MAX_FILES`), Mindestlänge in Sekunden. Speichert Dateien nach `data/<Klasse>/files/`. |
| `src/cut_audio.py` | 2 | Schneidet lange MP3s in 5-Sekunden-WAV-Clips. Nutzt YAMNet (TensorFlow Hub) zur Klassifikation: Clips mit Vogel-Score ≥ 0,40 werden als „Hard Negative" markiert, Score ≥ 0,15 als normaler Hintergrund. Ausgabe: `data/<Klasse>/clips/`. |
| `src/build_dataset.py` | 3 | Sammelt alle WAV-Clips, vergibt numerische Labels und erzeugt Train/Val/Test-Splits (70 / 15 / 15, Seed=42). Split erfolgt nach `recording_id`, nicht nach einzelnen Clips — verhindert Data Leakage. Ausgabe: `data_splits/train.csv`, `val.csv`, `test.csv`. |

Jedes Skript enthält einen konfigurierbaren Block am Anfang (`# --- ANPASSEN ---`),
in dem Art, Pfade und Schwellwerte angepasst werden.

### Training — `notebooks/`

| Pfad | Zweck |
| --- | --- |
| `notebooks/bird_training.ipynb` | Vollständiges Trainings-Notebook. Liest die CSV-Splits aus `data_splits/`, berechnet Mel-Spektrogramme, definiert BirdCNN (V3), trainiert 20 Epochen mit AdamW + CosineAnnealing, speichert `model_best.pth` und `model.pth` im Projektordner, gibt Classification Report und Confusion Matrix aus. |

### Daten — `data/`

| Pfad | Zweck |
| --- | --- |
| `data/` | Basisordner für alle Rohdaten. Nicht committed (nur `.gitkeep`). |
| `data/<Klasse>/files/` | MP3-Dateien direkt von Xeno-Canto, erzeugt von `src/bird_data.py`. Klassen: `Amsel`, `Kohlmeise`, `Rotkehlchen`, `Background`. |
| `data/<Klasse>/clips/` | 5-s-WAV-Clips, erzeugt von `src/cut_audio.py`. Dateiname-Schema: `<Art>_<AufnahmeID>_<Offset>_<bg|hardneg>.wav`. |

### Splits — `data_splits/`

| Pfad | Zweck |
| --- | --- |
| `data_splits/` | CSV-Dateien mit Pfad, Klasse, Label und Aufnahme-ID pro Clip. Nicht committed (nur `.gitkeep`). |
| `data_splits/train.csv` | Trainingsdaten (≈ 70 % der Aufnahmen, 8.760 Clips). |
| `data_splits/val.csv` | Validierungsdaten (≈ 15 % der Aufnahmen, 1.783 Clips). |
| `data_splits/test.csv` | Testdaten (≈ 15 % der Aufnahmen, 2.086 Clips). |

Spalten je CSV: `path`, `label`, `class_name`, `recording_id`.

### Dokumentation — `docs/`

| Pfad | Zweck |
| --- | --- |
| `docs/structure.md` | Diese Datei — Verzeichnisbaum und Datei-Erklärungen. |
| `docs/crisp-dm.md` | CRISP-DM-Dokumentation der sechs Projektphasen mit Pfadverweisen. |

---

## Ablage-Konventionen

| Was | Wo |
| --- | --- |
| Python-Skripte der Datenpipeline | `src/` |
| Jupyter-Notebooks | `notebooks/` |
| Rohdaten (MP3, WAV) | `data/<Klasse>/files/` bzw. `data/<Klasse>/clips/` — **nicht committen** |
| CSV-Splits | `data_splits/` — **nicht committen** |
| Modell-Checkpoints (`.pth`) | Projektordner (`/`) — direkt im Root |
| Projektdokumentation (Markdown) | `docs/` |
| Konfiguration für Claude Code | `CLAUDE.md` im Root |

> Große Binärdateien (Audiodaten, Modellgewichte > wenige MB) gehören nicht
> ins Repository. Für Modell-Artefakte sollte langfristig ein Artefakt-Store
> (z. B. DVC, MLflow) in Betracht gezogen werden.
