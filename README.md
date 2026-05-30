# Bird Species Classifier

Ein Proof-of-Concept-System zur automatischen Erkennung von Vogelgesang aus kurzen
Audioaufnahmen (5 Sekunden). Das Modell unterscheidet drei heimische Vogelarten —
**Amsel**, **Kohlmeise** und **Rotkehlchen** — sowie eine vierte Klasse
**Background** (kein Zielvogel hörbar). Eine Streamlit-App stellt die eigene
CNN-Vorhersage dem bekannten System BirdNET direkt gegenüber.

---

## Features

- Klassifikation von 5-Sekunden-Audioausschnitten in 4 Klassen
- Erkennung von „kein Vogel" durch explizite Background-Klasse
- Interaktive Streamlit-App: WAV-Datei hochladen oder live aufnehmen
- Mel-Spektrogramm-Visualisierung mit wählbarem Analysefenster
- Direktvergleich mit BirdNET (optional, via `birdnetlib`)
- Reproduzierbare Datenpipeline: Download → Zuschneiden → Split → Training
- Aufnahme-basierter Train/Val/Test-Split verhindert Data Leakage

---

## Tech-Stack

| Bereich | Werkzeuge |
|---|---|
| Deep Learning | PyTorch |
| Audio-Features | librosa, soundfile |
| Daten-Bereinigung | TensorFlow Hub, YAMNet |
| Daten-Split / Metriken | scikit-learn |
| Web-App | Streamlit |
| BirdNET-Vergleich | birdnetlib (optional) |
| Daten-Download | requests (Xeno-Canto API) |
| Visualisierung | matplotlib, plotly |
| Notebooks | JupyterLab |

---

## Voraussetzungen (Prerequisites)

- Python 3.9 oder neuer
- `model_best.pth` im Projektordner (enthalten im Repository oder selbst trainiert)
- Für den Xeno-Canto-Download: kostenloser API-Key von [xeno-canto.org](https://xeno-canto.org)
- Für BirdNET-Vergleich: `pip install birdnetlib` (optional)

---

## Installation

```bash
# 1. Repository klonen
git clone https://github.com/sommedav/bird-classifier.git
cd bird-classifier

# 2. Abhängigkeiten installieren
pip install -r requirements.txt

# 3. Umgebung prüfen
python setup_check.py
```

---

## Nutzung

### App starten (Schnellstart)

`model_best.pth` liegt bereits im Repo — die App ist sofort nutzbar:

```bash
streamlit run app.py
```

Der Browser öffnet sich automatisch. Du kannst eine WAV-Datei hochladen oder
direkt aufnehmen. Mit dem Slider wählst du den 5-Sekunden-Ausschnitt;
die App zeigt Mel-Spektrogramm, CNN-Vorhersage und (optional) BirdNET-Vergleich.

Alternativer Modell-Pfad via Umgebungsvariable:

```bash
BIRD_MODEL_PATH=/pfad/zu/modell.pth streamlit run app.py
```

### Modell selbst trainieren

Führe die folgenden Schritte der Reihe nach aus:

```bash
# Schritt 1 — Rohdaten herunterladen (API-Key in src/bird_data.py eintragen)
python src/bird_data.py

# Schritt 2 — Aufnahmen in 5-s-WAV-Clips schneiden + YAMNet-Filterung
python src/cut_audio.py

# Schritt 3 — Train/Val/Test-Splits erzeugen
python src/build_dataset.py

# Schritt 4 — Notebook öffnen und alle Zellen ausführen
jupyter lab notebooks/bird_training.ipynb
```

Das Training speichert `model_best.pth` (bester Val-Checkpoint) und `model.pth`
(finale Epoche) im Projektordner.

---

## Projektstruktur

Eine vollständige Übersicht aller Ordner und Dateien mit Erklärungen:
→ [docs/structure.md](docs/structure.md)

```
bird-classifier/
├── src/                    # Datenpipeline-Skripte
├── notebooks/              # Trainings-Notebook
├── data/                   # Rohdaten (nicht committed)
├── data_splits/            # CSV-Splits (nicht committed)
├── docs/                   # Projektdokumentation
├── app.py                  # Streamlit-App
├── model_best.pth          # Bestes Modell-Checkpoint
├── requirements.txt
└── ...
```

---

## Modell & Daten

### Architektur — BirdCNN

Eigenes Convolutional Neural Network in PyTorch:

- **Input:** Mel-Spektrogramm `(1, 128, 313)` — 32 kHz, 128 Mel-Bänder, 313 Frames
- **Feature-Extraktion:** 4 Conv-Blöcke (je 2× Conv3×3 → BatchNorm → ReLU → MaxPool),
  Kanaltiefe 1 → 32 → 64 → 128 → 256
- **Klassifikationskopf:** Global Average Pooling → Dropout(0,5) → Linear(256 → 4)
- **Augmentierung:** SpecAugment (Frequenz- und Zeit-Masking, nur beim Training)
- **Training:** 20 Epochen, AdamW (lr=1e-3, weight decay=1e-4),
  CosineAnnealingLR, CrossEntropyLoss (label smoothing=0,1)

### Datensatz

- **Quelle:** [Xeno-Canto](https://xeno-canto.org) — freie Vogelgesang-Aufnahmen
- **Arten:** Amsel, Kohlmeise, Rotkehlchen (Zielklassen) + Taube, Krähe, Spatz (Background)
- **Clip-Länge:** 5 Sekunden WAV, 32 kHz / 16 kHz (Schnitt-Schritt)
- **Datensatzgröße:** 12.629 Clips (Train 8.760 / Val 1.783 / Test 2.086)
- **Split-Strategie:** Aufnahme-basiert (70 % / 15 % / 15 %, Seed=42) → kein Data Leakage
- **Background-Erzeugung:** YAMNet bewertet jeden Clip; Hard Negatives (Score ≥ 0,40)
  und normaler Hintergrund (Score ≥ 0,15) werden getrennt erfasst

Vollständige CRISP-DM-Dokumentation: → [docs/crisp-dm.md](docs/crisp-dm.md)

---

## Evaluationsergebnisse

Bewertet auf dem unberührten Test-Set (2.086 Clips):

| Klasse | Precision | Recall | F1 | Anzahl |
|---|---|---|---|---|
| Amsel | 0,949 | 0,936 | 0,943 | 580 |
| Kohlmeise | 0,922 | 0,866 | 0,893 | 673 |
| Rotkehlchen | 0,732 | 0,884 | 0,801 | 327 |
| Background | 0,858 | 0,826 | 0,842 | 506 |
| **Gesamt** | **0,884** | **0,879** | **0,880** | **2.086** |

**Test-Accuracy: 87,87 %** (Best-Val-Accuracy: 87,32 %)

---

## Tests

> TODO: Keine automatisierte Test-Suite vorhanden.

Umgebungscheck:

```bash
python setup_check.py
```

---

## Bekannte Einschränkungen

- Nur drei Vogelarten erkennbar; alle anderen Arten landen in „Background"
- Trainiert auf Studioqualität-Aufnahmen von Xeno-Canto — Feldaufnahmen können abweichen
- Rotkehlchen hat die niedrigste Precision (0,732) — häufige Verwechslung mit Kohlmeise
- BirdNET und PyTorch laufen aus Kompatibilitätsgründen in getrennten Prozessen
- Kein Docker-Image / kein Cloud-Deployment vorhanden

---

## Lizenz

> TODO: Keine Lizenz-Datei vorhanden. Bitte `LICENSE` ergänzen.

## Quellen / Danksagungen

- [Xeno-Canto](https://xeno-canto.org) — Audiodaten
- [YAMNet](https://tfhub.dev/google/yamnet/1) (Google) — Qualitätskontrolle der Background-Clips
- [BirdNET](https://github.com/kahst/BirdNET-Analyzer) / `birdnetlib` — Vergleichssystem
- [librosa](https://librosa.org) — Audio-Analyse und Mel-Spektrogramme
- [PyTorch](https://pytorch.org) — Modelltraining und Inferenz
- [Streamlit](https://streamlit.io) — Web-App
