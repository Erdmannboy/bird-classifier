## Quick Start

### Modell testen

pip install -r requirements.txt
streamlit run app.py

Voraussetzung:
- model_best.pth liegt im Projektordner (das ist schon das fertige Modell, von uns trainiert)


-> Trainiert euch am besten ein Modell selber noch mal, dafür müsst ihr aber folgende Schritte beachten


### Modell neu trainieren

1. Xeno-Canto API Key erstellen
2. bird_data.py ausführen
3. cut_audio.py ausführen
4. build_dataset.py ausführen
5. bird_training.ipynb trainieren