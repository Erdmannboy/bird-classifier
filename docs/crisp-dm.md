# CRISP-DM Dokumentation — Bird Species Classifier

Dieses Dokument beschreibt den ML-Entwicklungsprozess entlang der sechs
CRISP-DM-Phasen. Alle Aussagen beziehen sich auf konkrete Dateien im Repository.

---

## 1. Business Understanding

### Problemstellung

Vögel am Gesang zu bestimmen erfordert Vorwissen und ist für Laien schwierig.

### Ziel

Proof-of-Concept-Klassifikator, der aus einer kurzen Audioaufnahme (5 Sekunden)
automatisch eine Einschätzung der Vogelart liefert — ohne Expertenwissen.
Das System erkennt drei häufige Gartenvögel und sagt explizit, wenn kein Zielvogel
zu hören ist (statt eine falsche Art zu erzwingen).

### Zielklassen

| Label | Art | Wissenschaftlicher Name |
| --- | --- | --- |
| 0 | Amsel | Turdus merula |
| 1 | Kohlmeise | Parus major |
| 2 | Rotkehlchen | Erithacus rubecula |
| 3 | Background | (kein Zielvogel) |

### Zielgruppe

Hobby-Ornithologen, Naturinteressierte, Gartenbesitzer — keine Fachkenntnisse
erforderlich.

### Bewertungskriterium

Die Gesamtaccuracy auf dem unberührten Test-Set sowie die Per-Class-Accuracy
entscheiden über die Modellqualität. Die Klasse „Background" ist besonders wichtig,
damit das System nicht immer eine Vogelart ausgibt.

### Abgrenzung

Kein Produktivsystem — bewusster Machbarkeitsnachweis für drei Arten.

**Hauptdatei:** `project.md` (Abschnitte 1–2)

---

## 2. Data Understanding

### Datenquelle

Alle Aufnahmen stammen von [Xeno-Canto](https://xeno-canto.org) —
einer öffentlichen Plattform mit tausenden frei verfügbaren Vogelaufnahmen.
Der Download erfolgt über die Xeno-Canto API v3.

**Skript:** `src/bird_data.py`

```
API-Endpunkt: https://xeno-canto.org/api/3/recordings
Filter: type:song q:A (Gesang, beste Qualität), Mindestlänge 20 s
```

### Heruntergeladene Arten

| Verwendung | Art | Xeno-Canto-Suchbegriff |
| --- | --- | --- |
| Zielklasse | Amsel | `en:"Common Blackbird"` |
| Zielklasse | Kohlmeise | `en:"Great Tit"` |
| Zielklasse | Rotkehlchen | `en:"European Robin"` |
| Background (Hard Negatives) | Krähe | `en:"Carrion Crow"` |
| Background (Hard Negatives) | Taube | `en:"Wood Pigeon"` |
| Background (Hard Negatives) | Spatz | `en:"House Sparrow"` |

Konfiguration je Art in `src/bird_data.py`, dict `BIRD_CONFIG`.

### Rohformat

MP3-Dateien, teils mehrere Minuten lang, Qualitätsstufe A.
Abgelegt nach dem Download unter `data/<Klasse>/files/`.

### Features

Aus dem Audiosignal werden **Mel-Spektrogramme** berechnet:

| Parameter | Wert |
| --- | --- |
| Sampling-Rate | 32.000 Hz (Zielklassen) / 16.000 Hz (Background-Schnitt) |
| Mel-Bänder | 128 |
| fmax | 16.000 Hz |
| Zeitfenster | feste Breite von 313 Frames (Padding / Crop) |
| Normalisierung | mean=0, std=1 pro Clip |
| Eingabeform CNN | `(1, 128, 313)` |

Feature-Berechnung: `notebooks/bird_training.ipynb` (Funktion `audio_to_mel`),
Inferenz: `app.py` (Funktion `array_to_mel_for_model`)

### Labels

- **Zielarten:** Klasse ergibt sich direkt aus dem Download (was als Amsel gesucht
  wird, erhält das Label Amsel).
- **Background:** Kein manuelles Labeling. Das vortrainierte Google-Modell
  **YAMNet** bewertet pro Clip den Vogel-Score (Skript: `src/cut_audio.py`).

### Datensatzgröße (nach Vorverarbeitung)

| Split | Clips |
| --- | --- |
| Train | 8.760 |
| Validation | 1.783 |
| Test | 2.086 |
| **Gesamt** | **12.629** |

Klassenverteilung im Trainingsset: ≈ 1.888 Amsel, 1.939 Kohlmeise,
2.203 Rotkehlchen, 2.730 Background.

**Hauptdateien:** `src/bird_data.py`, `project.md` (Abschnitt 3)

### Identifizierte Risiken

- Verrauschte Clips trotz Qualitätsstufe A → gelöst durch YAMNet-Filterung
- Label-Rauschen durch selbst gesetzte YAMNet-Schwellen
- Generalisierungslücke: Daten aus relativ sauberen Einzelaufnahmen,
  keine Feldaufnahmen
- Data Leakage (wenn Clips derselben Aufnahme in Train und Test landen)
  → gelöst durch aufnahmebasierten Split (siehe Phase 3)

---

## 3. Data Preparation

### Schritt 1 — Zuschneiden in Clips

**Skript:** `src/cut_audio.py`

Lange MP3-Aufnahmen werden in **5-Sekunden-WAV-Segmente** zerschnitten.
Kurze Rest-Segmente am Ende werden verworfen.

YAMNet (geladen via `tensorflow_hub`) klassifiziert jeden Clip:

| YAMNet Vogel-Score | Zuweisung | Dateiname-Suffix |
| --- | --- | --- |
| ≥ 0,40 | Hard Negative (fremder Vogel klar hörbar) | `_hardneg.wav` |
| ≥ 0,15 | Background (schwacher/diffuser Vogel) | `_bg.wav` |
| < 0,15 | Verworfen | — |

Ausgabe: `data/<Klasse>/clips/`

YAMNet bestimmt **nicht** die Vogelart, sondern dient nur der Qualitätskontrolle.

### Schritt 2 — Feature-Berechnung

Mel-Spektrogramme werden zur Ladezeit im Notebook berechnet (kein separates
Vorverarbeitungs-Skript). Funktion `audio_to_mel` in
`notebooks/bird_training.ipynb`:

1. Laden mit `librosa.load` (sr=32.000 Hz)
2. `librosa.feature.melspectrogram` (n_mels=128, fmax=16.000)
3. Konvertierung zu Dezibel: `librosa.power_to_db`
4. Auf 313 Frames bringen (Padding mit Nullen oder Abschneiden)
5. Z-Normalisierung: `(x - mean) / (std + 1e-6)`

Dieselbe Pipeline ist identisch in `app.py` (`array_to_mel_for_model`)
implementiert — wichtig für konsistente Inferenz.

### Schritt 3 — Augmentierung

**SpecAugment** (Frequenz- und Zeit-Masking) wird im Training angewendet:

| Parameter | Wert |
| --- | --- |
| Frequenz-Masken | 2 Masken, max. 24 Bänder |
| Zeit-Masken | 2 Masken, max. 40 Frames |

Implementiert als `SpecAugment`-Klasse in `notebooks/bird_training.ipynb`
(und als Stub in `app.py`, da beim Inference `model.eval()` aktiv ist).

### Schritt 4 — Train / Val / Test-Split

**Skript:** `src/build_dataset.py`

- Alle WAV-Clips werden gesammelt, Labels vergeben (`LABEL_MAP`)
- Clips werden nach `recording_id` (erste Ziffernfolge im Dateinamen) gruppiert
- Split der **Aufnahmen** (nicht der Clips): 70 % / 15 % / 15 %, Seed=42
- So landen Clips derselben Originalaufnahme nie gleichzeitig in Train und Test

Ausgabe: `data_splits/train.csv`, `data_splits/val.csv`, `data_splits/test.csv`

Spalten: `path`, `label`, `class_name`, `recording_id`

**Hauptdateien:** `src/cut_audio.py`, `src/build_dataset.py`,
`notebooks/bird_training.ipynb`

---

## 4. Modeling

### Architektur — BirdCNN (V3)

Eigenes Convolutional Neural Network in PyTorch.
Definiert in `notebooks/bird_training.ipynb` und `app.py`.

```
Input: (B, 1, 128, 313)  ← Mel-Spektrogramm, Batch × Kanal × Freq × Zeit
│
├── SpecAugment (nur Training)
│
├── Conv-Block 1: Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU → MaxPool2d
│   Kanäle: 1 → 32
├── Conv-Block 2: wie oben
│   Kanäle: 32 → 64
├── Conv-Block 3: wie oben
│   Kanäle: 64 → 128
├── Conv-Block 4: wie oben
│   Kanäle: 128 → 256
│
├── AdaptiveAvgPool2d(1)   ← Global Average Pooling
├── Flatten
├── Dropout(0.5)
└── Linear(256 → 4)        ← 4 Klassen
```

### Hyperparameter

| Parameter | Wert |
| --- | --- |
| Epochen | 20 |
| Batch-Größe | 32 |
| Optimizer | AdamW |
| Lernrate | 1e-3 |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingLR (T_max=20) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Dropout | 0,5 |
| Device | MPS (Apple Silicon) / CPU |

### Modell-Checkpointing

Nach jeder Epoche wird auf der Validierung evaluiert. Wenn `val_acc` den
bisherigen Bestwert übertrifft, wird der State-Dict als `model_best.pth`
gespeichert. Am Ende wird das finale Modell zusätzlich als `model.pth`
gespeichert.

### Erste Modellversion (Baseline)

Das erste Modell (referenziert in `project.md` als `notebook.ipynb`) hatte
ein kritisches Problem: Viele Clips enthielten Stille oder Rauschen und waren
trotzdem als Vogel gelabelt. Das Modell lernte Rauschen statt Vogelgesang.
Daraus entstand die Idee zur YAMNet-Bereinigung und der Background-Klasse.

**Hauptdatei:** `notebooks/bird_training.ipynb`, `app.py`

---

## 5. Evaluation

### Methode

Bewertet wird auf dem zuvor unberührten Test-Set (2.086 Clips) mit dem
`model_best.pth`-Checkpoint (bestes Val-Ergebnis aus Epoche 2: 87,32 %).

Metriken berechnet mit `sklearn.metrics` in `notebooks/bird_training.ipynb`.

### Ergebnisse (aus Notebook-Output)

**Test-Accuracy: 87,87 %**

#### Per-Class-Accuracy

| Klasse | Accuracy | Anzahl Clips |
| --- | --- | --- |
| Amsel | 93,62 % | 580 |
| Kohlmeise | 86,63 % | 673 |
| Rotkehlchen | 88,38 % | 327 |
| Background | 82,61 % | 506 |

#### Classification Report

| Klasse | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| Amsel | 0,949 | 0,936 | 0,943 | 580 |
| Kohlmeise | 0,922 | 0,866 | 0,893 | 673 |
| Rotkehlchen | 0,732 | 0,884 | 0,801 | 327 |
| Background | 0,858 | 0,826 | 0,842 | 506 |
| **accuracy** | | | **0,879** | **2.086** |
| macro avg | 0,865 | 0,878 | 0,870 | 2.086 |
| weighted avg | 0,884 | 0,879 | 0,880 | 2.086 |

#### Confusion Matrix (Zeilen = wahr, Spalten = vorhergesagt)

|  | Amsel | Kohlmeise | Rotkehlchen | Background |
| --- | --- | --- | --- | --- |
| **Amsel** | 543 | 0 | 10 | 27 |
| **Kohlmeise** | 6 | 583 | 63 | 21 |
| **Rotkehlchen** | 1 | 16 | 289 | 21 |
| **Background** | 22 | 33 | 33 | 418 |

### Interpretation

- Amsel wird am zuverlässigsten erkannt (F1 = 0,943)
- Häufigste Verwechslung: Kohlmeise → Rotkehlchen (63 Fälle) → niedrigste
  Precision bei Rotkehlchen (0,732)
- Background ist die schwierigste Klasse (breiteste Restklasse)
- Für ein selbst trainiertes CNN auf 4 Klassen ist das Ergebnis solide

**Hauptdatei:** `notebooks/bird_training.ipynb` (Evaluation-Zelle am Ende),
`project.md` (Abschnitt 7)

---

## 6. Deployment

### Anwendung

Die Anwendung ist eine **Streamlit-App** (`app.py`), die lokal gestartet wird.

```bash
streamlit run app.py
```

Kein Docker-Image, kein Cloud-Deployment vorhanden.

### Funktionsumfang der App

| Feature | Details |
| --- | --- |
| Audio-Eingabe | WAV-Datei hochladen oder Live-Aufnahme via Browser |
| Mel-Spektrogramm | Visualisierung der gesamten Aufnahme mit farbig markiertem Analysefenster |
| Fenster-Auswahl | Slider zur Wahl des 5-s-Ausschnitts (bei Aufnahmen > 5 s) |
| CNN-Vorhersage | Konfidenz-Wahrscheinlichkeiten für alle 4 Klassen; Background → „Kein Vogel" |
| BirdNET-Vergleich | Optional via `birdnetlib`; läuft in separatem Subprocess |
| Arten-Info | Karten mit Emoji, deutschem und wissenschaftlichem Name |

### Modell-Pfad

Standardmäßig `model_best.pth` im Projektordner, überschreibbar via:

```bash
BIRD_MODEL_PATH=/pfad/zu/modell.pth streamlit run app.py
```

### Konsistenz Training ↔ Inferenz

Die Vorverarbeitungs-Parameter sind in `app.py` und `notebooks/bird_training.ipynb`
identisch implementiert:

| Parameter | Wert |
| --- | --- |
| Sampling-Rate | 32.000 Hz |
| Mel-Bänder | 128 |
| fmax | 16.000 Hz |
| Frames | 313 (Padding / Crop) |
| Normalisierung | mean=0, std=1 |

### BirdNET-Subprocess

BirdNET (`birdnetlib`) nutzt TensorFlow-Lite, das sich mit PyTorch im selben
Prozess nicht zuverlässig verträgt. Lösung: BirdNET wird in einem separaten
Python-Subprocess aufgerufen (`subprocess.run`). Das JSON-Ergebnis wird über
stdout zurückgegeben. Implementiert in `app.py` als `_BIRDNET_SUBPROCESS_SCRIPT`.

> TODO: Docker-Image für reproduzierbares Deployment erstellen.
> TODO: Cloud-Deployment (z. B. Streamlit Cloud, Hugging Face Spaces) evaluieren.

**Hauptdatei:** `app.py`, `project.md` (Abschnitt 8)
