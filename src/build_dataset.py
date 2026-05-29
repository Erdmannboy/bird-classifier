# build_dataset.py
# Sammelt alle WAV-Clips, vergibt Labels und teilt sie in Train/Val/Test.
# Wichtig: gesplittet wird nach Original-Aufnahme (recording_id), damit Clips
# derselben Aufnahme nicht gleichzeitig in Train und Test landen (Data Leakage).

import os
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# --- ANPASSEN ----------------------------------------------------------------
# Ordner mit den Clips und Zielordner fuer die CSV-Splits (relativ zum Projekt)
BASE_DIR = Path("data")
OUTPUT_DIR = Path("data_splits")

# Wie viele Clips pro Klasse maximal verwenden? (Rest wird zufaellig verworfen)
TARGET_COUNTS = {
    "Amsel": 3000,
    "Kohlmeise": 3000,
    "Rotkehlchen": 3000,
    "Background": 5000,
}

# Numerische Labels je Klasse (Reihenfolge muss zum Modell passen)
LABEL_MAP = {
    "Amsel": 0,
    "Kohlmeise": 1,
    "Rotkehlchen": 2,
    "Background": 3,
}

# Aufteilung und fester Seed fuer reproduzierbare Splits
TEST_AND_VAL_SHARE = 0.30   # davon je zur Haelfte Val und Test -> 70/15/15
RANDOM_STATE = 42
# -----------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_recording_id(filename):
    # Erste Zahlenfolge im Dateinamen ist die Aufnahme-ID, sonst "unknown".
    for part in filename.split("_"):
        if part.isdigit():
            return part
    return "unknown"


# Alle Clips einsammeln und als Datensaetze aufbauen.
all_samples = []

for class_name in TARGET_COUNTS:
    clips_dir = BASE_DIR / class_name / "clips"
    files = [f for f in os.listdir(clips_dir) if f.endswith(".wav")]
    print(f"{class_name}: {len(files)} Dateien gefunden")

    # Bei zu vielen Dateien zufaellig auf die Zielmenge reduzieren.
    target_count = TARGET_COUNTS[class_name]
    if len(files) > target_count:
        files = random.sample(files, target_count)

    for file in files:
        recording_id = extract_recording_id(file)
        all_samples.append({
            "path": str(clips_dir / file),
            "label": LABEL_MAP[class_name],
            "class_name": class_name,
            "recording_id": f"{class_name}_{recording_id}",
        })

print(f"Samples gesamt: {len(all_samples)}")

# Nach Aufnahme gruppieren, dann die Aufnahmen (nicht die einzelnen Clips) splitten.
grouped = defaultdict(list)
for sample in all_samples:
    grouped[sample["recording_id"]].append(sample)

recording_ids = list(grouped.keys())

train_ids, temp_ids = train_test_split(
    recording_ids, test_size=TEST_AND_VAL_SHARE, random_state=RANDOM_STATE
)
val_ids, test_ids = train_test_split(
    temp_ids, test_size=0.50, random_state=RANDOM_STATE
)


def samples_for(ids):
    out = []
    for rid in ids:
        out.extend(grouped[rid])
    return out


train_df = pd.DataFrame(samples_for(train_ids))
val_df = pd.DataFrame(samples_for(val_ids))
test_df = pd.DataFrame(samples_for(test_ids))

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
print("Verteilung im Trainingsset:")
print(train_df["class_name"].value_counts())