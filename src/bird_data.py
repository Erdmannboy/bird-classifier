# bird_data.py
# Lädt Vogelaufnahmen von Xeno-Canto herunter und speichert sie als MP3.
# Pro Aufruf wird eine Vogelart (SEARCH_BIRD) geladen und in einen
# Klassen-Ordner (TARGET_CLASS) einsortiert.

import os
import time
from pathlib import Path

import requests

# --- ANPASSEN ----------------------------------------------------------------
# Euren eigenen Xeno-Canto-API-Key holen

# Einen kostenlosen Key bekommt ihr nach Registrierung auf https://xeno-canto.org
API_KEY = "f4eb2feb67656eda01ca89ef084aa49c0e7d3ee7"

# Welche Art soll geladen werden? (einer der Schlüssel aus BIRD_CONFIG unten)
SEARCH_BIRD = "Kohlmeise"

# In welchen Klassen-Ordner kommen die Dateien? (z.B. "Amsel" oder "Background")
TARGET_CLASS = "Kohlmeise"

# Basisordner für die Rohdaten (relativ zum Projekt)
BASE_DIR = Path("data")

# Wie viele Dateien maximal, und welche Mindestlänge (Sekunden)?
MAX_FILES = 60
MIN_SECONDS = 20
# -----------------------------------------------------------------------------

PER_PAGE = 100
BASE_URL = "https://xeno-canto.org/api/3/recordings"

# Suchanfragen je Art: en = englischer Name, type:song, q:A = beste Tonqualitaet
BIRD_CONFIG = {
    "Amsel":       {"query": 'en:"Common Blackbird" type:song q:A'},
    "Kohlmeise":   {"query": 'en:"Great Tit" type:song q:A'},
    "Rotkehlchen": {"query": 'en:"European Robin" type:song q:A'},
    "Krähe":       {"query": 'en:"Carrion Crow" type:song q:A'},
    "Taube":       {"query": 'en:"Wood Pigeon" type:song q:A'},
    "Spatz":       {"query": 'en:"House Sparrow" type:song q:A'},
}

if not API_KEY:
    raise SystemExit("Kein API-Key gefunden. Bitte XENO_CANTO_API_KEY setzen.")

QUERY = BIRD_CONFIG[SEARCH_BIRD]["query"]

SAVE_DIR = BASE_DIR / TARGET_CLASS / "files"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def length_to_seconds(length_str):
    # Wandelt "m:ss" oder "h:mm:ss" in Sekunden um, sonst None.
    parts = list(map(int, length_str.split(":")))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


downloaded = 0
page = 1
num_pages = None

print(f"Starte Download fuer '{SEARCH_BIRD}' -> Ordner '{TARGET_CLASS}'")

# Seitenweise durch die API gehen, bis genug Dateien geladen sind.
while downloaded < MAX_FILES:

    if num_pages is not None and page > num_pages:
        print("Letzte Seite erreicht.")
        break

    params = {"query": QUERY, "key": API_KEY, "per_page": PER_PAGE, "page": page}

    try:
        response = requests.get(BASE_URL, params=params, timeout=20)
    except Exception as e:
        print("Request-Fehler:", e)
        break

    if response.status_code != 200:
        print("API-Fehler:", response.text)
        break

    data = response.json()

    if page == 1:
        if "numPages" not in data:
            print("Unerwartete API-Antwort:", data)
            break
        num_pages = int(data["numPages"])
        print(f"Treffer gesamt: {data['numRecordings']} ({num_pages} Seiten)")

    recordings = data.get("recordings", [])
    if not recordings:
        print("Keine Aufnahmen gefunden.")
        break

    for rec in recordings:
        if downloaded >= MAX_FILES:
            break

        rec_id = rec.get("id", "unknown")

        # Zu kurze Aufnahmen ueberspringen.
        seconds = length_to_seconds(rec.get("length", "0:00"))
        if seconds is None or seconds < MIN_SECONDS:
            continue

        file_url = rec.get("file")
        if not file_url:
            continue

        file_path = SAVE_DIR / f"{SEARCH_BIRD}_{rec_id}.mp3"
        if file_path.exists():
            continue

        try:
            audio = requests.get(file_url, timeout=20)
            if audio.status_code != 200:
                print("Download fehlgeschlagen:", rec_id)
                continue
            file_path.write_bytes(audio.content)
            downloaded += 1
            print(f"[{downloaded}] {file_path.name} ({seconds}s)")
            time.sleep(0.2)
        except Exception as e:
            print("Download-Fehler:", e)

    page += 1

print(f"Fertig. {downloaded} Dateien geladen.")