# Testdaten

Fünf Beispiel-Clips (5 s, WAV) zum manuellen Ausprobieren der App und zum
Abgleich der Vorhersage mit der tatsächlichen Art.

## Wahrheitstabelle

Die Dateinamen sind bewusst **neutral durchnummeriert** — die tatsächliche Art
(„Ground Truth") steht ausschließlich in dieser Tabelle. So lässt sich blind
testen und danach vergleichen, was das Modell vorhergesagt hat.

| Datei         | Tatsächliche Art (Ground Truth) | Quelle / Notiz                        |
|---------------|---------------------------------|---------------------------------------|
| `test_01.wav` | `Background`                    | Selbst gepfiffen (Mensch, kein Vogel) |
| `test_02.wav` | `Background`                    | Rabe (Vogel, aber keine Zielart)      |
| `test_03.wav` | `Amsel`                         |                                       |
| `test_04.wav` | `Kohlmeise`                     |                                       |
| `test_05.wav` | `Rotkehlchen`                   |                                       |

Erlaubte Werte für die Art: `Amsel`, `Kohlmeise`, `Rotkehlchen`, `Background`.

## So testen

1. Eigene 5 WAV-Clips in diesen Ordner kopieren und exakt
   `test_01.wav` … `test_05.wav` benennen.
2. Oben in der Tabelle bei jeder Datei die tatsächliche Art eintragen
   (die `_???_` ersetzen).
3. App starten: `uv run streamlit run app.py`
4. Jede Datei hochladen, Vorhersage notieren und mit der Tabelle vergleichen.
