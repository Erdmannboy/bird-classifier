# bird_data.py
# Lädt Vogelaufnahmen von Xeno-Canto herunter und speichert sie als MP3.
# Alle konfigurierten Arten werden gleichzeitig (parallel) geladen.

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# --- ANPASSEN ----------------------------------------------------------------
# Xeno-Canto API-Key (kostenlos nach Registrierung auf https://xeno-canto.org)
API_KEY = "f4eb2feb67656eda01ca89ef084aa49c0e7d3ee7"

# Arten, die in ihren eigenen Ordner kommen (data/<Art>/files/)
DOWNLOAD_SPECIES = ["Amsel", "Rotkehlchen", "Kohlmeise"]

# Arten, die in die Background-Klasse kommen (data/Background/files/)
# Müssen NICHT zusätzlich in DOWNLOAD_SPECIES stehen.
BACKGROUND_SPECIES = ["Spatz", "Taube", "Krähe"]

# Maximale Dateien pro Art
MAX_FILES = 60

# Mindestlänge einer Aufnahme in Sekunden
MIN_SECONDS = 20

# Basisordner für die Rohdaten (immer relativ zum Projektordner, egal von wo aufgerufen)
BASE_DIR = Path(__file__).parent.parent / "data"
# -----------------------------------------------------------------------------

PER_PAGE = 100
BASE_URL = "https://xeno-canto.org/api/3/recordings"

# Suchanfragen je Art: en = englischer Name, type:song, q:A = beste Tonqualität
BIRD_CONFIG = {
    "Amsel":       {"query": 'en:"Common Blackbird" type:song q:A'},
    "Kohlmeise":   {"query": 'en:"Great Tit" type:song q:A'},
    "Rotkehlchen": {"query": 'en:"European Robin" type:song q:A'},
    "Krähe":       {"query": 'en:"Carrion Crow" type:song q:A'},
    "Taube":       {"query": 'en:"Wood Pigeon" type:song q:A'},
    "Spatz":       {"query": 'en:"House Sparrow" type:song q:A'},
}

if not API_KEY:
    raise SystemExit("Kein API-Key gefunden. Bitte API_KEY eintragen.")

for _name in DOWNLOAD_SPECIES + BACKGROUND_SPECIES:
    if _name not in BIRD_CONFIG:
        raise SystemExit(f"Unbekannte Art '{_name}'. Verfügbar: {list(BIRD_CONFIG)}")


def length_to_seconds(length_str: str) -> int | None:
    # Wandelt "m:ss" oder "h:mm:ss" in Sekunden um, sonst None.
    parts = list(map(int, length_str.split(":")))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def download_species(name: str, position: int) -> int:
    """Lädt bis zu MAX_FILES Aufnahmen für eine Art herunter. Gibt Anzahl zurück."""
    target_class = "Background" if name in BACKGROUND_SPECIES else name
    save_dir = BASE_DIR / target_class / "files"
    save_dir.mkdir(parents=True, exist_ok=True)

    already = len(list(save_dir.glob(f"{name}_*.mp3")))
    need = MAX_FILES - already

    if need <= 0:
        tqdm.write(f"[{name}] Bereits {already} Dateien vorhanden — überspringe.")
        return 0

    query = BIRD_CONFIG[name]["query"]
    downloaded = 0
    page = 1
    num_pages = None

    bar = tqdm(
        total=MAX_FILES,
        initial=already,
        desc=f"{name} → {target_class}",
        unit="file",
        position=position,
        leave=True,
        dynamic_ncols=True,
    )

    while downloaded < need:
        if num_pages is not None and page > num_pages:
            tqdm.write(f"[{name}] Letzte Seite erreicht.")
            break

        params = {"query": query, "key": API_KEY, "per_page": PER_PAGE, "page": page}

        try:
            response = requests.get(BASE_URL, params=params, timeout=20)
        except Exception as e:
            tqdm.write(f"[{name}] Request-Fehler: {e}")
            break

        if response.status_code != 200:
            tqdm.write(f"[{name}] API-Fehler: {response.text}")
            break

        data = response.json()

        if page == 1:
            if "numPages" not in data:
                tqdm.write(f"[{name}] Unerwartete API-Antwort: {data}")
                break
            num_pages = int(data["numPages"])
            tqdm.write(f"[{name}] Treffer gesamt: {data['numRecordings']} ({num_pages} Seiten)")

        recordings = data.get("recordings", [])
        if not recordings:
            tqdm.write(f"[{name}] Keine Aufnahmen gefunden.")
            break

        for rec in recordings:
            if downloaded >= need:
                break

            rec_id = rec.get("id", "unknown")
            seconds = length_to_seconds(rec.get("length", "0:00"))
            if seconds is None or seconds < MIN_SECONDS:
                continue

            file_url = rec.get("file")
            if not file_url:
                continue

            file_path = save_dir / f"{name}_{rec_id}.mp3"
            if file_path.exists():
                continue

            try:
                audio = requests.get(file_url, timeout=20)
                if audio.status_code != 200:
                    tqdm.write(f"[{name}] Download fehlgeschlagen: {rec_id}")
                    continue
                file_path.write_bytes(audio.content)
                downloaded += 1
                bar.update(1)
                time.sleep(0.2)
            except Exception as e:
                tqdm.write(f"[{name}] Download-Fehler: {e}")

        page += 1

    bar.close()
    return downloaded


if __name__ == "__main__":
    all_species = DOWNLOAD_SPECIES + BACKGROUND_SPECIES
    tqdm.write(f"Ziel-Arten:       {', '.join(DOWNLOAD_SPECIES) if DOWNLOAD_SPECIES else '(keine)'}")
    tqdm.write(f"Background-Arten: {', '.join(BACKGROUND_SPECIES) if BACKGROUND_SPECIES else '(keine)'}\n")

    with ThreadPoolExecutor(max_workers=len(all_species)) as executor:
        futures = {
            executor.submit(download_species, name, i): name
            for i, name in enumerate(all_species)
        }
        total = 0
        for future in as_completed(futures):
            name = futures[future]
            try:
                total += future.result()
            except Exception as e:
                tqdm.write(f"[{name}] Fehler: {e}")

    tqdm.write(f"\nAlle Downloads abgeschlossen. Gesamt: {total} Dateien.")
