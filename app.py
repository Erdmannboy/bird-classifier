import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import librosa
import librosa.display
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import plotly.graph_objects as go
import torch.nn as nn

# ======================================
# CONFIG
# ======================================

# Modell-Output-Reihenfolge (passend zu data_splits/train.csv)
MODEL_CLASSES = ["Amsel", "Kohlmeise", "Rotkehlchen", "Background"]
BIRD_CLASSES = ["Amsel", "Kohlmeise", "Rotkehlchen"]
NO_BIRD_LABEL = "Kein Vogel"

CLASS_INFO = {
    "Amsel":       {"emoji": "🐦‍⬛", "color": "#2C3E50", "scientific": "Turdus merula"},
    "Kohlmeise":   {"emoji": "🐤",   "color": "#F39C12", "scientific": "Parus major"},
    "Rotkehlchen": {"emoji": "🐦",   "color": "#E74C3C", "scientific": "Erithacus rubecula"},
    NO_BIRD_LABEL: {"emoji": "🤷",   "color": "#95A5A6", "scientific": "Keiner der drei Vögel"},
}

BIRD_FACTS = {
    "Amsel": {
        "lifespan": "2–3 Jahre (Rekord: ca. 18 Jahre)",
        "weight": "80–125 g",
        "wingspan": "34–38 cm",
        "habitat": "Wälder, Parks, Gärten, Gebüsche",
        "diet": "Würmer, Insekten, Beeren, Früchte",
        "nesting": "März–Juli · 2–3 Bruten · 4–5 Eier",
        "distribution_text": "Heimisch in Europa, Nordafrika und Teilen Asiens; eingeführt in Australien und Neuseeland.",
        "distribution_countries": [
            "DEU","AUT","CHE","FRA","GBR","IRL","ITA","ESP","PRT","NLD","BEL","LUX",
            "DNK","SWE","NOR","FIN","POL","CZE","SVK","HUN","ROU","BGR","GRC","HRV",
            "SVN","SRB","BIH","MKD","ALB","MNE","TUR","GEO","ARM","AZE","UKR","BLR",
            "MDA","LTU","LVA","EST","ISL","CYP","MLT","DZA","MAR","TUN","LBY","EGY",
            "IRN","IRQ","SYR","LBN","ISR","JOR","AFG","PAK","IND","NPL","CHN","AUS","NZL",
        ],
        "fun_facts": [
            "Männchen sind tiefschwarz mit leuchtend gelbem Schnabel – Weibchen und Jungtiere hingegen braun.",
            "Ursprünglich ein scheuer Waldvogel, erst seit dem 19. Jh. typischer Stadt- und Gartenbewohner.",
            "Ihr Gesang gilt als einer der melodischsten Europas – am lautesten in der Morgen- und Abenddämmerung.",
        ],
    },
    "Kohlmeise": {
        "lifespan": "2–3 Jahre (Rekord: ~13 Jahre)",
        "weight": "14–22 g",
        "wingspan": "22–25 cm",
        "habitat": "Laub- und Mischwälder, Parks, Gärten",
        "diet": "Insekten, Samen, Nüsse, Beeren",
        "nesting": "April–Juni · 1–2 Bruten · 6–12 Eier",
        "distribution_text": "Von Westeuropa bis Japan, von Skandinavien bis Nordafrika und Südostasien.",
        "distribution_countries": [
            "DEU","AUT","CHE","FRA","GBR","IRL","ITA","ESP","PRT","NLD","BEL","LUX",
            "DNK","SWE","NOR","FIN","POL","CZE","SVK","HUN","ROU","BGR","GRC","HRV",
            "SVN","SRB","BIH","MKD","ALB","MNE","TUR","GEO","ARM","AZE","UKR","BLR",
            "RUS","KAZ","UZB","TKM","IRN","AFG","PAK","IND","NPL","CHN","JPN","KOR",
            "PRK","MNG","VNM","THA","MYS","IDN","DZA","MAR","TUN","EGY","ISR","LBN",
            "JOR","LTU","LVA","EST","ISL","CYP","MLT","MDA",
        ],
        "fun_facts": [
            "Eine der häufigsten Vogelarten Europas – an fast jedem Gartenvogelhäuschen anzutreffen.",
            "Bekannt für das Öffnen von Milchflaschen-Aludeckeln – eine Fähigkeit, die sich unter Artgenossen verbreitete.",
            "Kann bis zu 12 Eier pro Gelege legen – für einen so kleinen Vogel eine außergewöhnlich große Anzahl.",
        ],
    },
    "Rotkehlchen": {
        "lifespan": "1–3 Jahre (Rekord: ~11 Jahre)",
        "weight": "14–21 g",
        "wingspan": "20–22 cm",
        "habitat": "Wälder, Hecken, Parks, Gärten – bevorzugt feuchte Lagen",
        "diet": "Insekten, Würmer, Beeren, Samen",
        "nesting": "März–August · 2 Bruten · 4–6 Eier",
        "distribution_text": "Europa, Westasien und Nordafrika; viele Populationen ziehen im Winter in den Süden.",
        "distribution_countries": [
            "DEU","AUT","CHE","FRA","GBR","IRL","ITA","ESP","PRT","NLD","BEL","LUX",
            "DNK","SWE","NOR","FIN","POL","CZE","SVK","HUN","ROU","BGR","GRC","HRV",
            "SVN","SRB","BIH","MKD","ALB","MNE","TUR","GEO","ARM","AZE","UKR","BLR",
            "MDA","LTU","LVA","EST","ISL","CYP","MLT","DZA","MAR","TUN","LBY","EGY",
            "IRN","SYR","LBN","ISR","JOR","RUS","KAZ",
        ],
        "fun_facts": [
            "In Großbritannien gilt es als inoffizieller Nationalvogel und ist das beliebteste Gartenvogel-Symbol.",
            "Trotz seiner geringen Größe sehr territorial – verteidigt sein Revier auch im Winter aggressiv.",
            "Folgt oft Gärtnern und wühlenden Tieren – wartet geduldig auf freigelegte Würmer und Insekten.",
        ],
    },
}

# BirdNET → unsere lokalen Namen (per scientific name)
BIRDNET_SCIENTIFIC_TO_LOCAL = {
    "Turdus merula":      "Amsel",
    "Parus major":        "Kohlmeise",
    "Erithacus rubecula": "Rotkehlchen",
}

# ANPASSEN: Pfad zum trainierten Modell. Standard = "model_best.pth" im selben
# Ordner; alternativ per Umgebungsvariable BIRD_MODEL_PATH ueberschreiben.
MODEL_PATH = os.environ.get("BIRD_MODEL_PATH", "model_best.pth")

TARGET_SR = 32000
TARGET_DURATION = 5.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
TARGET_FRAMES = 313
HOP_LENGTH = 512

# ======================================
# CNN MODEL  (V3 — passend zu bird_training.ipynb)
# ======================================

class SpecAugment(nn.Module):
    """Im Training aktiv — in der App nur Stub, da model.eval(). Keine Parameter."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class BirdCNN(nn.Module):
    def __init__(self, num_classes=4, dropout=0.5):
        super().__init__()
        self.spec_aug = SpecAugment()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

# ======================================
# CACHED LOADERS
# ======================================

@st.cache_resource
def load_model():
    m = BirdCNN(num_classes=len(MODEL_CLASSES))
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    return m

def birdnetlib_installed() -> bool:
    """Leichter Check — initialisiert nichts."""
    return importlib.util.find_spec("birdnetlib") is not None

# BirdNET laeuft in einem eigenen Subprozess. Grund: PyTorch und das von BirdNET
# genutzte TensorFlow-Lite vertragen sich im selben Prozess nicht zuverlaessig.
_BIRDNET_SUBPROCESS_SCRIPT = textwrap.dedent("""
    import sys, json, contextlib
    # Alle Logs/Prints des Analyzers nach stderr — stdout bleibt sauber fürs JSON.
    with contextlib.redirect_stdout(sys.stderr):
        from birdnetlib import Recording
        from birdnetlib.analyzer import Analyzer
        analyzer = Analyzer()
        rec = Recording(analyzer, sys.argv[1], min_conf=0.05)
        rec.analyze()
        detections = rec.detections
    sys.stdout.write(json.dumps(detections))
""")

@st.cache_data(show_spinner=False)
def load_full_audio(audio_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    y, sr = librosa.load(path, sr=TARGET_SR)
    return y, sr, path

@st.cache_data(show_spinner=False)
def compute_mel(audio_bytes: bytes):
    y, sr, _ = load_full_audio(audio_bytes)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=16000, hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(mel, ref=np.max)

@st.cache_data(show_spinner="🦜 BirdNET analysiert (Subprocess — kann beim ersten Mal etwas dauern)...")
def run_birdnet_full(audio_bytes: bytes):
    """BirdNET einmal auf die volle Aufnahme — als Subprocess (s. _BIRDNET_SUBPROCESS_SCRIPT)."""
    if not birdnetlib_installed():
        return None

    _, _, full_path = load_full_audio(audio_bytes)

    try:
        result = subprocess.run(
            [sys.executable, "-c", _BIRDNET_SUBPROCESS_SCRIPT, full_path],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        st.warning("⏱️ BirdNET-Subprocess hat das Zeitlimit überschritten.")
        return None

    if result.returncode != 0:
        st.warning(
            "BirdNET-Subprocess ist fehlgeschlagen. "
            f"stderr (gekürzt): `{result.stderr.strip()[-400:]}`"
        )
        return None

    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        st.warning(
            "BirdNET-Output konnte nicht geparst werden. "
            f"stdout: `{result.stdout[:300]}`"
        )
        return None

def detections_in_window(detections, start_sec: float, end_sec: float):
    """BirdNET chunked in 3s; behalte alles was sich mit [start, end] überschneidet."""
    if not detections:
        return []
    return [
        d for d in detections
        if d.get("start_time", 0.0) < end_sec
        and d.get("end_time", 0.0) > start_sec
    ]

# ======================================
# AUDIO HELPERS
# ======================================

def crop_audio(y: np.ndarray, sr: int, start_sec: float) -> np.ndarray:
    start_sample = int(start_sec * sr)
    end_sample = start_sample + TARGET_SAMPLES
    cropped = y[start_sample:end_sample]
    if len(cropped) < TARGET_SAMPLES:
        cropped = np.pad(cropped, (0, TARGET_SAMPLES - len(cropped)), mode="constant")
    return cropped

def audio_to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    return buf.read()

def array_to_mel_for_model(y: np.ndarray, sr: int) -> np.ndarray:
    """Genau wie im Notebook: 313 Frames fixed + (mean=0, std=1) Normalisierung."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=16000, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    n = mel_db.shape[1]
    if n < TARGET_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, TARGET_FRAMES - n)), mode="constant")
    elif n > TARGET_FRAMES:
        mel_db = mel_db[:, :TARGET_FRAMES]
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db

# ======================================
# PREDICTION HELPERS
# ======================================

def predict_my_model(model, y_cropped: np.ndarray, sr: int) -> np.ndarray:
    mel = array_to_mel_for_model(y_cropped, sr)
    t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1).numpy()[0]
    return probs  # Reihenfolge: MODEL_CLASSES

def my_model_display_probs(probs: np.ndarray) -> dict:
    """MODEL_CLASSES-Probs → dict mit Anzeige-Labels (Background → Kein Vogel)."""
    p = dict(zip(MODEL_CLASSES, probs.tolist()))
    p[NO_BIRD_LABEL] = p.pop("Background")
    return p

def birdnet_display_probs(detections) -> dict:
    """Max-Confidence pro Spezies + Rest als 'Kein Vogel'."""
    bird_max = {b: 0.0 for b in BIRD_CLASSES}
    if detections:
        for d in detections:
            sci = d.get("scientific_name", "")
            local = BIRDNET_SCIENTIFIC_TO_LOCAL.get(sci)
            if local:
                bird_max[local] = max(bird_max[local], float(d.get("confidence", 0.0)))
    max_bird_conf = max(bird_max.values()) if bird_max else 0.0
    return {**bird_max, NO_BIRD_LABEL: max(0.0, 1.0 - max_bird_conf)}

def birdnet_top_detection(detections):
    """Top-Detection (egal welche Spezies) → (display_name, scientific, conf, is_in_our_3) oder None."""
    if not detections:
        return None
    top = max(detections, key=lambda d: d.get("confidence", 0.0))
    sci = top.get("scientific_name", "")
    common = top.get("common_name", sci)
    conf = float(top.get("confidence", 0.0))
    if sci in BIRDNET_SCIENTIFIC_TO_LOCAL:
        return (BIRDNET_SCIENTIFIC_TO_LOCAL[sci], sci, conf, True)
    return (common, sci, conf, False)

# ======================================
# BIRD FACTS RENDERER
# ======================================

def _is_dark_theme() -> bool:
    """True, wenn der Browser/das aktive Streamlit-Theme auf Dunkel steht.

    CSS-Variablen regeln das HTML automatisch per prefers-color-scheme; Plotly
    kennt diese Variablen aber nicht, daher hier die serverseitige Erkennung.
    """
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def render_bird_facts(bird_name: str):
    if bird_name not in BIRD_FACTS:
        return

    facts = BIRD_FACTS[bird_name]
    info = CLASS_INFO[bird_name]
    color = info["color"]

    st.markdown(
        f'<div class="section-title">📚 Wissenswertes: {info["emoji"]} {bird_name}</div>',
        unsafe_allow_html=True,
    )

    col_facts, col_map = st.columns([1, 1], gap="large")

    with col_facts:
        fun_facts_html = "".join(
            f'<div style="padding:6px 0; border-bottom:1px solid var(--divider-soft); font-size:0.88rem; color:var(--text-body);">'
            f'💡 {f}</div>'
            for f in facts["fun_facts"]
        )
        st.markdown(f"""
        <div class="card">
            <div style="font-size:0.72rem; font-weight:700; letter-spacing:1.5px;
                        text-transform:uppercase; color:{color}; margin-bottom:0.9rem;">
                Steckbrief
            </div>
            <table style="width:100%; border-collapse:collapse;">
                <tr>
                    <td style="padding:5px 12px 5px 0; color:var(--text-muted); font-size:0.85rem; white-space:nowrap;">🔬 Wiss. Name</td>
                    <td style="padding:5px 0; font-weight:600; font-style:italic; font-size:0.85rem;">{info['scientific']}</td>
                </tr>
                <tr>
                    <td style="padding:5px 12px 5px 0; color:var(--text-muted); font-size:0.85rem;">⏳ Lebenserwartung</td>
                    <td style="padding:5px 0; font-weight:600; font-size:0.85rem;">{facts['lifespan']}</td>
                </tr>
                <tr>
                    <td style="padding:5px 12px 5px 0; color:var(--text-muted); font-size:0.85rem;">⚖️ Gewicht</td>
                    <td style="padding:5px 0; font-weight:600; font-size:0.85rem;">{facts['weight']}</td>
                </tr>
                <tr>
                    <td style="padding:5px 12px 5px 0; color:var(--text-muted); font-size:0.85rem;">🦅 Spannweite</td>
                    <td style="padding:5px 0; font-weight:600; font-size:0.85rem;">{facts['wingspan']}</td>
                </tr>
                <tr>
                    <td style="padding:5px 12px 5px 0; color:var(--text-muted); font-size:0.85rem;">🌿 Lebensraum</td>
                    <td style="padding:5px 0; font-weight:600; font-size:0.85rem;">{facts['habitat']}</td>
                </tr>
                <tr>
                    <td style="padding:5px 12px 5px 0; color:var(--text-muted); font-size:0.85rem;">🍎 Ernährung</td>
                    <td style="padding:5px 0; font-weight:600; font-size:0.85rem;">{facts['diet']}</td>
                </tr>
                <tr>
                    <td style="padding:5px 12px 5px 0; color:var(--text-muted); font-size:0.85rem;">🥚 Brutzeit & Gelege</td>
                    <td style="padding:5px 0; font-weight:600; font-size:0.85rem;">{facts['nesting']}</td>
                </tr>
            </table>
            <div style="margin-top:1.1rem; font-size:0.72rem; font-weight:700; letter-spacing:1.5px;
                        text-transform:uppercase; color:{color}; margin-bottom:0.5rem;">
                Interessantes
            </div>
            {fun_facts_html}
        </div>
        """, unsafe_allow_html=True)

    with col_map:
        n = len(facts["distribution_countries"])
        dark = _is_dark_theme()

        # Amsel-Akzent (#2C3E50) ist auf dunklem Grund praktisch unsichtbar —
        # im Dunkelmodus daher zu einem helleren Slate-Blau wechseln.
        fill_color = "#5b7da6" if (dark and color.upper() == "#2C3E50") else color
        title_color = "#e9eef5" if dark else "#2c3e50"
        land_color = "#2a2f3a" if dark else "#f0f0f0"
        ocean_color = "rgba(70,100,135,0.30)" if dark else "rgba(173,216,230,0.4)"
        coast_color = "#3a4150" if dark else "#cccccc"
        country_color = "#333a47" if dark else "#dddddd"
        line_color = "#1a1f29" if dark else "white"

        fig = go.Figure(go.Choropleth(
            locations=facts["distribution_countries"],
            z=[1] * n,
            colorscale=[[0, fill_color], [1, fill_color]],
            showscale=False,
            marker_line_color=line_color,
            marker_line_width=0.5,
            zmin=0,
            zmax=1,
        ))
        fig.update_layout(
            title=dict(
                text=f"Verbreitung der {bird_name}",
                x=0.5,
                xref="paper",
                xanchor="center",
                font=dict(size=14, color=title_color),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            height=310,
            geo=dict(
                showframe=False,
                showcoastlines=True,
                coastlinecolor=coast_color,
                showcountries=True,
                countrycolor=country_color,
                bgcolor="rgba(0,0,0,0)",
                showland=True,
                landcolor=land_color,
                showocean=True,
                oceancolor=ocean_color,
                projection_type="natural earth",
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"🌍 {facts['distribution_text']}")


# ======================================
# PAGE CONFIG + CSS
# ======================================

st.set_page_config(
    page_title="Bird Classifier",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* ===== THEME-VARIABLEN — folgen automatisch der Hell-/Dunkel- =====
       ===== Einstellung des Browsers via prefers-color-scheme.      ===== */
    :root {
        --card-bg: #ffffff;
        --card-border: #eef0f3;
        --text-strong: #2c3e50;
        --text-body: #34495e;
        --text-muted: #7f8c8d;
        --text-faint: #95a5a6;
        --divider: #ecf0f1;
        --divider-soft: #f5f5f5;
        --result-bg: linear-gradient(135deg, #f6f9fc 0%, #eef2f7 100%);
        --result-border: #e6ebf1;
        --bar-bg: #ecf0f1;
        --accent-mine: #4754c1;
        --accent-birdnet: #16a085;
        --tag-mine-bg: #eef0fb;
        --tag-birdnet-bg: #e8f7f2;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --card-bg: #1a1f29;
            --card-border: #2b323e;
            --text-strong: #e9eef5;
            --text-body: #c7d0db;
            --text-muted: #94a1b0;
            --text-faint: #8794a3;
            --divider: #2b323e;
            --divider-soft: #262c36;
            --result-bg: linear-gradient(135deg, #1c2230 0%, #181d27 100%);
            --result-border: #2b323e;
            --bar-bg: #2b323e;
            --accent-mine: #aab4ff;
            --accent-birdnet: #4fd1b0;
            --tag-mine-bg: #232a47;
            --tag-birdnet-bg: #15352d;
        }
    }

    #MainMenu, footer, header {visibility: hidden;}
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem; border-radius: 20px; color: white;
        text-align: center; margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.25);
    }
    .hero h1 { font-size: 2.6rem; margin: 0; font-weight: 700; letter-spacing: -0.5px; }
    .hero p  { font-size: 1.1rem; opacity: 0.95; margin: 0.5rem 0 0 0; }

    .card {
        background: var(--card-bg); padding: 1.5rem; border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05); border: 1px solid var(--card-border);
        color: var(--text-body); margin-bottom: 1rem;
    }

    .result-box {
        background: var(--result-bg);
        padding: 1.6rem 1.2rem; border-radius: 16px;
        text-align: center; border: 2px solid var(--result-border);
        margin-bottom: 1rem;
    }
    .result-tag        { font-size: 0.75rem; font-weight: 700; letter-spacing: 1.5px;
                         text-transform: uppercase; color: var(--text-faint); margin-bottom: 0.4rem; }
    .result-emoji      { font-size: 3.2rem; line-height: 1; margin-bottom: 0.3rem; }
    .result-name       { font-size: 1.6rem; font-weight: 700; color: var(--text-strong); margin: 0; }
    .result-scientific { font-style: italic; color: var(--text-muted); margin: 0.2rem 0 0.8rem 0; font-size: 0.9rem; }
    .result-conf-label { font-size: 0.75rem; color: var(--text-faint); text-transform: uppercase; letter-spacing: 1px; }
    .result-conf-val   { font-size: 1.5rem; font-weight: 700; color: #27ae60; }

    .result-tag.mine    { color: var(--accent-mine); }
    .result-tag.birdnet { color: var(--accent-birdnet); }

    .prob-row { display: flex; align-items: center; gap: 12px; padding: 8px 0; }
    .prob-label   { flex: 0 0 145px; font-weight: 600; color: var(--text-body); font-size: 0.95rem; }
    .prob-bar-bg  { flex: 1; background: var(--bar-bg); height: 20px; border-radius: 10px; overflow: hidden; }
    .prob-bar-fill{ height: 100%; border-radius: 10px; transition: width 0.4s ease; }
    .prob-value   { flex: 0 0 60px; text-align: right; font-weight: 700;
                    font-variant-numeric: tabular-nums; color: var(--text-strong); }

    .section-title {
        font-size: 1.1rem; font-weight: 700; color: var(--text-strong);
        margin: 1.5rem 0 0.8rem 0; padding-bottom: 0.4rem;
        border-bottom: 2px solid var(--divider);
    }

    .compare-col-title {
        font-size: 1rem; font-weight: 700;
        padding: 0.5rem 0.8rem; border-radius: 8px;
        margin-bottom: 0.6rem; text-align: center;
    }
    .compare-col-title.mine    { background: var(--tag-mine-bg); color: var(--accent-mine); }
    .compare-col-title.birdnet { background: var(--tag-birdnet-bg); color: var(--accent-birdnet); }
</style>
""", unsafe_allow_html=True)

# ======================================
# HEADER
# ======================================

st.markdown("""
<div class="hero">
    <h1>🐦 Bird Species Classifier</h1>
    <p>Mein CNN gegen BirdNET — wer erkennt den Vogel besser?</p>
</div>
""", unsafe_allow_html=True)

# ======================================
# MODELL LADEN (mit Fehlerbehandlung)
# ======================================

try:
    model = load_model()
    model_error = None
except Exception as e:
    model = None
    model_error = str(e)

if model is None:
    st.error(
        f"❌ Mein Modell konnte nicht geladen werden.\n\n"
        f"Pfad: `{MODEL_PATH}`\n\n"
        f"Fehler: `{model_error}`\n\n"
        f"Trainiere zuerst das Notebook `notebooks/bird_training.ipynb` "
        f"durch — das speichert `model_best.pth`."
    )
    st.stop()

# ======================================
# INPUT
# ======================================

tab_upload, tab_record = st.tabs(["📁 Datei hochladen", "🎙️ Live aufnehmen"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "WAV-Datei", type=["wav"], label_visibility="collapsed"
    )

with tab_record:
    recorded = st.audio_input("Drück den Knopf und nimm den Vogel auf")

audio_source = uploaded_file if uploaded_file is not None else recorded

# ---- BirdNET-Vergleich Toggle ----
if birdnetlib_installed():
    birdnet_enabled = st.checkbox(
        "🦜 BirdNET-Vergleich aktiv",
        value=True,
        help="Beim ersten Mal pro Datei dauert die Analyse ~5–30s. "
             "Danach reagiert der Slider sofort (Detektionen werden nur noch gefiltert).",
    )
else:
    birdnet_enabled = False

# ======================================
# KEIN INPUT → SPECIES CARDS
# ======================================

if audio_source is None:
    st.markdown('<div class="section-title">Erkennbare Arten</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, name in zip(cols, BIRD_CLASSES):
        info = CLASS_INFO[name]
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div style="font-size:3.5rem; line-height:1;">{info['emoji']}</div>
                <div style="font-weight:700; font-size:1.2rem; color:var(--text-strong); margin-top:0.5rem;">{name}</div>
                <div style="font-style:italic; color:var(--text-muted); font-size:0.9rem;">{info['scientific']}</div>
            </div>
            """, unsafe_allow_html=True)

    if not birdnetlib_installed():
        st.info(
            "ℹ️ **BirdNET nicht aktiv.** `birdnetlib` ist nicht installiert. "
            "Abhängigkeiten synchronisieren, damit die App beide Modelle "
            "nebeneinander vergleichen kann:  \n"
            "```bash\nuv sync\n```"
        )

# ======================================
# AUDIO INPUT → ANALYSE
# ======================================

if audio_source is not None:

    audio_bytes = audio_source.getvalue() if hasattr(audio_source, "getvalue") else audio_source.read()

    y_full, sr, _ = load_full_audio(audio_bytes)
    full_duration = len(y_full) / sr
    mel_full = compute_mel(audio_bytes)

    # ---- Volle Aufnahme ----
    st.markdown('<div class="section-title">🎵 Aufnahme</div>', unsafe_allow_html=True)
    st.audio(audio_bytes)
    st.caption(f"Länge: {full_duration:.2f}s")

    # ---- Slider ----
    if full_duration > TARGET_DURATION:
        st.markdown(
            '<div class="section-title">🎚️ Wähle den 5-Sekunden-Ausschnitt</div>',
            unsafe_allow_html=True,
        )
        start_sec = st.slider(
            "Startzeit",
            min_value=0.0,
            max_value=float(full_duration - TARGET_DURATION),
            value=0.0,
            step=0.1,
            format="%.1f s",
            label_visibility="collapsed",
        )
        st.caption(f"Analysefenster: **{start_sec:.1f}s – {start_sec + TARGET_DURATION:.1f}s**")
    else:
        start_sec = 0.0
        if full_duration < TARGET_DURATION:
            st.info(f"ℹ️ Aufnahme ist nur {full_duration:.2f}s — wird auf 5s mit Stille aufgefüllt.")

    # ---- Crop ----
    y_cropped = crop_audio(y_full, sr, start_sec)
    cropped_wav = audio_to_wav_bytes(y_cropped, sr)

    # ---- Layout ----
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-title">📊 Mel-Spektrogramm</div>', unsafe_allow_html=True)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#1e1e2e")
        ax.set_facecolor("#1e1e2e")

        librosa.display.specshow(
            mel_full, sr=sr, hop_length=HOP_LENGTH,
            x_axis="time", y_axis="mel", cmap="magma", ax=ax, fmax=16000
        )

        rect = patches.Rectangle(
            (start_sec, 0), TARGET_DURATION, sr // 2,
            linewidth=2.5, edgecolor="#2ecc71", facecolor="#2ecc71",
            alpha=0.25, zorder=10,
        )
        ax.add_patch(rect)
        ax.axvline(start_sec, color="#2ecc71", linewidth=2, zorder=11)
        ax.axvline(start_sec + TARGET_DURATION, color="#2ecc71", linewidth=2, zorder=11)

        ax.set_xlabel("Zeit (s)", color="white")
        ax.set_ylabel("Mel-Frequenz", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown('<div class="section-title">🔊 Nur der Ausschnitt</div>', unsafe_allow_html=True)
        st.audio(cropped_wav, format="audio/wav")

    # ---- Mein Modell ----
    my_raw_probs = predict_my_model(model, y_cropped, sr)
    my_probs = my_model_display_probs(my_raw_probs)
    my_top_label = max(my_probs, key=my_probs.get)
    my_top_conf = my_probs[my_top_label] * 100
    my_top_info = CLASS_INFO[my_top_label]

    # ---- BirdNET (volle Aufnahme einmal analysieren, dann pro Fenster filtern) ----
    if birdnet_enabled and birdnetlib_installed():
        all_detections = run_birdnet_full(audio_bytes)
        birdnet_available = all_detections is not None
        birdnet_detections = (
            detections_in_window(all_detections, start_sec, start_sec + TARGET_DURATION)
            if birdnet_available else None
        )
    else:
        birdnet_available = False
        birdnet_detections = None

    bn_probs = birdnet_display_probs(birdnet_detections) if birdnet_available else None
    bn_top = birdnet_top_detection(birdnet_detections) if birdnet_available else None

    # ---- Result Cards (rechts) ----
    with col_right:
        st.markdown('<div class="section-title">🎯 Vergleich</div>', unsafe_allow_html=True)

        # ---- MEIN MODELL ----
        st.markdown(f"""
        <div class="result-box">
            <div class="result-tag mine">Mein Modell</div>
            <div class="result-emoji">{my_top_info['emoji']}</div>
            <div class="result-name">{my_top_label}</div>
            <div class="result-scientific">{my_top_info['scientific']}</div>
            <div class="result-conf-label">Konfidenz</div>
            <div class="result-conf-val">{my_top_conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # ---- BIRDNET ----
        if not birdnet_available:
            st.markdown(f"""
            <div class="result-box">
                <div class="result-tag birdnet">BirdNET</div>
                <div class="result-emoji">📦</div>
                <div class="result-name" style="font-size:1.1rem;">birdnetlib nicht installiert</div>
                <div class="result-scientific">uv sync</div>
            </div>
            """, unsafe_allow_html=True)
        elif bn_top is None:
            st.markdown(f"""
            <div class="result-box">
                <div class="result-tag birdnet">BirdNET</div>
                <div class="result-emoji">{CLASS_INFO[NO_BIRD_LABEL]['emoji']}</div>
                <div class="result-name">{NO_BIRD_LABEL}</div>
                <div class="result-scientific">{CLASS_INFO[NO_BIRD_LABEL]['scientific']}</div>
                <div class="result-conf-label">Keine Detektion</div>
                <div class="result-conf-val" style="color:var(--text-faint);">—</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            bn_display_name, bn_sci, bn_conf, is_ours = bn_top
            if is_ours:
                bn_info = CLASS_INFO[bn_display_name]
                emoji = bn_info["emoji"]
                sci = bn_info["scientific"]
            else:
                emoji = "🦜"
                sci = bn_sci or "andere Art"
            st.markdown(f"""
            <div class="result-box">
                <div class="result-tag birdnet">BirdNET</div>
                <div class="result-emoji">{emoji}</div>
                <div class="result-name">{bn_display_name}</div>
                <div class="result-scientific">{sci}</div>
                <div class="result-conf-label">Konfidenz</div>
                <div class="result-conf-val">{bn_conf * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Wahrscheinlichkeiten im Vergleich (volle Breite) ----
    st.markdown(
        '<div class="section-title">📊 Wahrscheinlichkeiten im Vergleich</div>',
        unsafe_allow_html=True,
    )

    def render_prob_bars(probs_dict):
        items = sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)
        rows = []
        for name, p in items:
            info = CLASS_INFO[name]
            pct = p * 100
            rows.append(f"""
            <div class="prob-row">
                <div class="prob-label">{info['emoji']} {name}</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct}%; background:{info['color']};"></div>
                </div>
                <div class="prob-value">{pct:.1f}%</div>
            </div>
            """)
        return "".join(rows)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(
            '<div class="compare-col-title mine">Mein Modell</div>',
            unsafe_allow_html=True,
        )
        st.markdown(render_prob_bars(my_probs), unsafe_allow_html=True)

    with col_b:
        st.markdown(
            '<div class="compare-col-title birdnet">BirdNET</div>',
            unsafe_allow_html=True,
        )
        if not birdnet_available:
            st.info("`uv sync` für den Vergleich.")
        else:
            st.markdown(render_prob_bars(bn_probs), unsafe_allow_html=True)
            if birdnet_detections:
                with st.expander("🔍 Alle BirdNET-Detektionen"):
                    for d in sorted(birdnet_detections, key=lambda x: -x.get("confidence", 0.0)):
                        st.write(
                            f"**{d.get('common_name','?')}** "
                            f"_({d.get('scientific_name','?')})_ — "
                            f"{d.get('confidence',0)*100:.1f}% "
                            f"@ {d.get('start_time',0):.1f}–{d.get('end_time',0):.1f}s"
                        )

    # ---- Vogel-Fakten (nur wenn eine der drei Arten erkannt wurde) ----
    if my_top_label in BIRD_CLASSES:
        render_bird_facts(my_top_label)