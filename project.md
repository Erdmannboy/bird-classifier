# ML4B Project Documentation – Bird Species Classifier

## 1. Project Idea

Wir haben ein eigenes Machine-Learning-System gebaut, das Vogelarten anhand von
Audioaufnahmen erkennt. Unterschieden werden drei heimische Arten – Amsel, Kohlmeise
und Rotkehlchen – plus eine vierte Klasse "Background", die greift, wenn gar kein
relevanter Vogelgesang zu hören ist (Stille, Wind, Rauschen oder andere Vögel).
Dazu kommt eine kleine Streamlit-App, in der man eine Aufnahme hochladen oder direkt
aufnehmen kann; sie zeigt das Mel-Spektrogramm und gibt die wahrscheinliche Vogelart
aus und vergleicht sie mit dem bekannten System BirdNET.

Ehrlich gesagt: Den eigentlichen ML-Teil – also das Netz aufbauen und trainieren –
haben wir größtenteils mit Hilfe von KI gelöst. Was wir dagegen wirklich selbst
erarbeitet haben, war der ganze Weg davor: woher man überhaupt brauchbare Daten
bekommt, wie man die langen Aufnahmen sinnvoll zuschneidet, und vor allem die
Probleme, auf die wir dabei gestoßen sind (z. B. dass das erste Modell auf Stille
trainiert hat). Genau an diesen Stellen haben wir am meisten verstanden – wie wichtig
saubere Daten sind, warum man den Datensatz richtig aufteilen muss und wie man ein
Modell überhaupt sinnvoll bewertet.

## 2. Business Understanding

Das ist kein fertiges Produkt, sondern ein Proof of Concept. Wir wollten vor allem
zeigen, dass so etwas grundsätzlich funktioniert: Aus einer kurzen Aufnahme lässt
sich für ein paar häufige Gartenvögel automatisch eine Einschätzung gewinnen.

### Problem

Vögel am Gesang zu bestimmen, ist ohne Vorwissen schwierig und mühsam.

### Target Group

Für alle, die so etwas interessant finden – Hobby-Ornithologen, Naturinteressierte,
Gartenbesitzer – und die ohne Expertenwissen schnell eine grobe Einordnung bekommen
wollen.

### Expected Value

Eine Aufnahme wird in Sekunden grob eingeordnet, und durch die Klasse "Background"
sagt das System auch, wenn gar kein Zielvogel zu hören ist, statt einen Vogel zu
erzwingen. Mehr als ein Machbarkeitsnachweis für drei Arten ist es bewusst nicht.

## 3. Data Understanding

### Data Source

Als Datenquelle haben wir die öffentliche Plattform **Xeno-Canto** genutzt, die
tausende frei verfügbare Vogelaufnahmen enthält. Den Download haben wir über die
Xeno-Canto-API mit dem Skript `bird_data.py` automatisiert und dabei gezielt nach
Gesang in guter Qualität (`type:song q:A`) mit mindestens 20 Sekunden Länge gesucht.

Geladen haben wir die drei Zielarten – Amsel (Common Blackbird), Kohlmeise (Great Tit)
und Rotkehlchen (European Robin) – sowie zusätzlich Taube, Krähe und Spatz, aus denen
wir später schwierige Negativbeispiele für die Klasse "Background" gemacht haben. Die
Rohdaten lagen als MP3-Dateien vor und waren oft mehrere Minuten lang.

### Sensors or Features

Wir arbeiten mit reinem Audio. Aus dem Signal berechnen wir mit `librosa`
**Mel-Spektrogramme** – eine bildähnliche Darstellung, die zeigt, welche Frequenzen
wann und wie stark vorkommen. Dieses "Bild" ist das eigentliche Feature für unser CNN.
Einstellungen: 32 kHz Sampling-Rate, 128 Mel-Bänder, `fmax` = 16 kHz, feste Breite von
313 Zeit-Frames.

### Labels

Die Labels für die drei Vogelarten ergeben sich direkt aus dem Download – was wir als
Amsel suchen, bekommt das Label Amsel usw. (numerisch: Amsel = 0, Kohlmeise = 1,
Rotkehlchen = 2, Background = 3). Für "Background" haben wir nicht von Hand gelabelt,
sondern das vortrainierte Google-Modell **YAMNet** als Qualitätskontrolle eingesetzt
(`cut_audio.py`). YAMNet schätzt pro Clip, wie stark Vogelgeräusche vorkommen. Daraus
entstanden zwei Arten von Background: "Hard Negatives" (klar hörbarer, aber fremder
Vogel wie Taube oder Krähe, YAMNet-Score ≥ 0.40) und normaler Hintergrund (schwache
oder diffuse Geräusche, Score ≥ 0.15). Wichtig: YAMNet bestimmt nicht die Vogelart,
sondern dient nur dem Aufräumen der Daten.

### Risks

Die größten Risiken waren: verrauschte Clips, die trotzdem als Vogel gelabelt sind
(genau das ist uns am Anfang passiert), Label-Rauschen durch die selbst gesetzten
YAMNet-Schwellen, ein anfängliches Ungleichgewicht zugunsten des Hintergrunds, und die
begrenzte Generalisierung – nur drei Arten, Daten überwiegend aus relativ sauberen
Einzelaufnahmen. Außerdem das Risiko von Data Leakage, wenn Clips derselben
Originalaufnahme in Train und Test landen (siehe Abschnitt 5).

## 4. Related Work

Statt eigener Forschung haben wir bestehende Werkzeuge kombiniert. Die wichtigsten:

- **Xeno-Canto** (Plattform/API) – Quelle aller Aufnahmen.
- **YAMNet** von Google (vortrainiertes Modell) – Qualitätskontrolle und Erzeugung der
  Background-Clips.
- **BirdNET** über `birdnetlib` (Referenzsystem) – Vergleichsmaßstab in der App.
- **librosa** – Laden der Audiodaten und Berechnung der Mel-Spektrogramme.
- **PyTorch** – Aufbau und Training des eigenen CNN.
- **scikit-learn** – Daten-Split und Evaluations-Metriken.
- **Streamlit** – die Benutzeroberfläche der App.

## 5. Data Preparation

**Schneiden in Clips:** Die langen MP3s sind als Ganzes nicht zum Trainieren geeignet.
Mit `cut_audio.py` schneiden wir sie in 5-Sekunden-Clips und speichern sie als WAV.
Kurze Segmente sind ein gängiger Standard für Audio-Klassifikation und reduzieren
unnötigen Hintergrund. Im selben Schritt sortiert YAMNet jeden Clip in Background bzw.
Hard Negatives ein.

**Feature-Berechnung:** Jeder Clip wird in ein Mel-Spektrogramm umgewandelt (32 kHz,
128 Mel-Bänder, `fmax` = 16 kHz). Die Breite wird fest auf 313 Frames gebracht (zu
kurze Clips werden mit Nullen aufgefüllt, zu lange abgeschnitten), danach normalisieren
wir auf Mittelwert 0 und Standardabweichung 1. Die Eingabeform pro Clip ist
`(1, 128, 313)`.

**Splitting (file-based):** Mit `build_dataset.py` teilen wir in Train/Validation/Test.
Wichtig: Wir gruppieren zuerst nach der ursprünglichen Aufnahme-ID und teilen dann
diese Aufnahmen auf (70 % / 15 % / 15 %, fester Seed). So landen Clips derselben
Originalaufnahme nie gleichzeitig in Train und Test – das verhindert Data Leakage. Das
Ergebnis liegt als `train.csv`, `val.csv` und `test.csv` vor (mit Pfad, Klasse,
Label und Aufnahme-ID).

Am Ende hatten wir 8.760 Clips zum Trainieren, 1.783 für die Validierung und 2.086 für
den Test (zusammen 12.629). Im Trainingsset verteilt sich das auf rund 1.888 Amsel,
1.939 Kohlmeise, 2.203 Rotkehlchen und 2.730 Background.

## 6. Modeling

**Erstes Modellproblem (Baseline):** Die erste Version (`notebook.ipynb`) lief zwar,
hatte aber ein klares Problem: Weil viele Clips Stille oder Rauschen enthielten und
trotzdem als Vogel gelabelt waren, hat das Modell falsche Muster gelernt – selbst bei
stillen Aufnahmen wurde mit hoher Sicherheit ein Vogel vorhergesagt. Aus diesem Fehler
ist die Idee entstanden, die Daten mit YAMNet zu bereinigen und "Background" als vierte
Klasse einzuführen.

**Finales Modell:** Das finale Modell (`bird_training.ipynb`) ist ein eigenes
Convolutional Neural Network (BirdCNN) in PyTorch. Es bekommt ein Mel-Spektrogramm und
gibt Wahrscheinlichkeiten für die vier Klassen aus. Es besteht aus vier
Convolution-Blöcken (jeweils zweimal Conv 3×3 → BatchNorm → ReLU, dann MaxPooling;
Kanäle 1 → 32 → 64 → 128 → 256), einem Global Average Pooling und einem Klassifikations-
kopf aus Dropout (0.5) und einer Linear-Schicht (256 → 4). Beim Training kommt zusätzlich
SpecAugment (Frequenz- und Zeit-Masking) zum Einsatz, um Overfitting zu reduzieren.

Trainiert wurde über 20 Epochen mit Batch-Größe 32, CrossEntropy-Loss (Label Smoothing
0.1), dem AdamW-Optimizer (lr = 1e-3, weight decay = 1e-4) und einem CosineAnnealing-
Scheduler. Nach jeder Epoche haben wir auf der Validierung gemessen und jeweils das beste
Modell als `model_best.pth` gespeichert; das finale Modell zusätzlich als `model.pth`.

## 7. Evaluation

Bewertet wird auf dem zuvor unberührten Test-Set (2.086 Clips). Ausgewählt wurde das
Modell mit der besten Validation-Accuracy (87,32 %). Auf dem Test-Set erreicht es eine
**Gesamt-Accuracy von 87,87 %**.

Pro Klasse sieht das so aus – Amsel wird mit Abstand am zuverlässigsten erkannt,
Background ist am schwierigsten:

```
Klasse         Accuracy   Anzahl
Amsel           93.62 %     580
Kohlmeise       86.63 %     673
Rotkehlchen     88.38 %     327
Background      82.61 %     506
```

Der Classification Report (Precision / Recall / F1):

```
              precision   recall   f1-score   support
Amsel            0.949    0.936     0.943       580
Kohlmeise        0.922    0.866     0.893       673
Rotkehlchen      0.732    0.884     0.801       327
Background       0.858    0.826     0.842       506

accuracy                            0.879      2086
macro avg        0.865    0.878     0.870      2086
weighted avg     0.884    0.879     0.880      2086
```

Confusion Matrix (Zeilen = wahr, Spalten = vorhergesagt):

```
              Amsel  Kohlmeise  Rotkehlchen  Background
Amsel          543       0          10           27
Kohlmeise        6     583          63           21
Rotkehlchen      1      16         289           21
Background      22      33          33          418
```

**Interpretation:** Amsel läuft am besten. Die häufigste Verwechslung ist Kohlmeise →
Rotkehlchen (63 Fälle), weshalb Rotkehlchen die niedrigste Precision (0,732) hat – das
Modell sagt also "Rotkehlchen" tendenziell zu oft. Background ist die schwierigste
Klasse und wird mit allen drei Vogelarten verwechselt, was bei einer so breiten
"Restklasse" plausibel ist. Für ein selbst gebautes CNN auf vier Klassen ist das ein
solides Ergebnis.

## 8. Deployment

Die Anwendung ist eine Streamlit-App (`app.py`) und wird lokal gestartet mit:

```bash
streamlit run app.py
```

In der App kann man eine WAV-Datei hochladen oder live aufnehmen, sieht das
Mel-Spektrogramm, wählt einen 5-Sekunden-Ausschnitt und bekommt die Vorhersage unseres
eigenen CNN (`model_best.pth`) – die Klasse "Background" wird dabei als "Kein Vogel"
angezeigt. Zusätzlich wird dieselbe Aufnahme von BirdNET analysiert und das Ergebnis
gegenübergestellt. BirdNET läuft bewusst in einem eigenen Subprozess, weil sich PyTorch
und TensorFlow-Lite sonst in die Quere kommen.

Damit Training und App zusammenpassen, ist die Vorverarbeitung in der App identisch zum
Training (32 kHz, 128 Mel-Bänder, 313 Frames, gleiche Normalisierung). Die benötigten
Bibliotheken stehen in `requirements.txt`.

## 9. Reflection

Die wichtigste Lektion war, dass saubere Daten mehr bringen als ein komplexeres Netz:
Der eigentliche Fortschritt kam nicht durch ein größeres Modell, sondern durch das
Aufräumen der Daten mit YAMNet und die Einführung der Background-Klasse – die erste
Version hatte vor allem Rauschen "gelernt". Auch die bewussten Hard Negatives (Taube,
Krähe, Spatz) haben geholfen, Zielvögel von anderen Vögeln zu trennen, und das
file-based Splitting nach Aufnahme-ID war nötig, um zu optimistische Ergebnisse durch
Data Leakage zu vermeiden.

Offen geblieben sind ein paar Schwächen: Rotkehlchen wird zu oft vorhergesagt, und
Background ist die schwierigste Klasse – mehr und vielfältigere Daten würden hier
wahrscheinlich am meisten bringen. Technisch hat uns BirdNET zusammen mit PyTorch
zunächst Konflikte beschert, die wir über einen Subprozess gelöst haben. Und auch wenn
wir den ML-Teil stark mit KI erarbeitet haben, haben wir gerade an den Daten- und
Problemstellen viel darüber gelernt, worauf es bei so einem Projekt wirklich ankommt.
