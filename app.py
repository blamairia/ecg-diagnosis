import os
import tempfile

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wfdb
import torch

from resnet import resnet34  # Make sure resnet34.py is in the same folder

# -----------------------------
# 1) MODEL LOADING (cached)
# -----------------------------
MODEL_PATH = "resnet34_model.pth"

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Instantiate ResNet34(input_channels=12, num_classes=9), load weights,
    send to CPU or GPU, and return (model, device).
    """
    net = resnet34(input_channels=12, num_classes=9)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    return net, device

model, device = load_model()

# -----------------------------
# 2) TRANSLATIONS / TEXT
# -----------------------------
LANGUAGES = {"English": "en", "Français": "fr"}
lang_choice = st.sidebar.selectbox("Language / Langue", ["English", "Français"])
lang = LANGUAGES[lang_choice]

# Text dictionaries
TEXT = {
    "title": {
        "en": "ECG Arrhythmia Detection (PyTorch ResNet)",
        "fr": "Détection d’arythmie ECG (PyTorch ResNet)"
    },
    "description": {
        "en": (
            "Upload exactly two files (same basename): a `.hea` file and its matching `.mat` file.  \n"
            "We’ll read all 12 leads from the MATLAB‐formatted record, let you choose which lead to visualize, "
            "and then classify the recording into 9 arrhythmia types."
        ),
        "fr": (
            "Uploadez exactement deux fichiers (même nom de base) : un fichier `.hea` et son fichier `.mat` correspondant.  \n"
            "Nous lirons les 12 dérivations du fichier MATLAB, vous laisserez choisir la dérivation à visualiser, "
            "puis classifierons l’enregistrement en 9 types d’arythmie."
        )
    },
    "upload_hint": {
        "en": "Please upload exactly two files (`.hea` + `.mat`) with the same basename.",
        "fr": "Veuillez uploader exactement deux fichiers (`.hea` + `.mat`) portant le même nom de base."
    },
    "lead_selector": {
        "en": "Select lead to display:",
        "fr": "Choisir la dérivation à afficher :"
    },
    "prob_title": {
        "en": "Predicted Class Probabilities",
        "fr": "Probabilités prédites par classe"
    },
    "prob_explain": {
        "en": (
            "Below are the nine arrhythmia classes with their respective probabilities (from 0 to 1).  \n"
            "A higher probability means the model is more confident that the rhythm is present."
        ),
        "fr": (
            "Voici les neuf classes d’arythmie avec leurs probabilités respectives (de 0 à 1).  \n"
            "Une probabilité élevée signifie que le modèle est plus confiant que le rythme est présent."
        )
    },
    "abnormal_hint": {
        "en": (
            "If the model or a future explainer provides time ranges of abnormality, they will be overlaid in red."
        ),
        "fr": (
            "Si le modèle ou un futur module d’explication fournit des plages horaires d’anomalie, elles seront surlignées en rouge."
        )
    }
}

st.set_page_config(page_title=TEXT["title"][lang], layout="wide")
st.title(TEXT["title"][lang])
st.write(TEXT["description"][lang])

# -----------------------------
# 3) FILE-UPLOADER
# -----------------------------
uploaded = st.file_uploader(
    TEXT["upload_hint"][lang],
    type=["hea", "mat"],
    accept_multiple_files=True
)

if not (uploaded and len(uploaded) == 2):
    st.info(TEXT["upload_hint"][lang])
    st.stop()

# -----------------------------
# 4) SAVE UPLOADS TO TMP DIR
# -----------------------------
tmpdir = tempfile.mkdtemp()
for f in uploaded:
    path = os.path.join(tmpdir, f.name)
    with open(path, "wb") as out:
        out.write(f.getbuffer())

# Infer basename (must match between .hea and .mat)
base1 = os.path.splitext(uploaded[0].name)[0]
base2 = os.path.splitext(uploaded[1].name)[0]
if base1 != base2:
    st.error("❌ " + {
        "en": "The two files must share the exact same basename (e.g. both start with '100').",
        "fr": "Les deux fichiers doivent partager exactement le même nom de base (par ex. tous deux commencent par '100')."
    }[lang])
    st.stop()
record_name = base1
recpath = os.path.join(tmpdir, record_name)

# -----------------------------
# 5) READ WFDB RECORD (MATLAB)
# -----------------------------
try:
    record = wfdb.rdrecord(recpath)
    sig_all = record.p_signal  # shape = [n_samples, n_leads]
    lead_names = record.sig_name  # list of channel labels, e.g. ['I','II','III',…,'V6']
except Exception as e:
    st.error("❌ " + {
        "en": f"Could not read WFDB record `{record_name}`: {e}",
        "fr": f"Impossible de lire l’enregistrement WFDB `{record_name}` : {e}"
    }[lang])
    st.stop()

# Ensure we have at least one channel
if sig_all.ndim != 2 or len(lead_names) == 0:
    st.error("❌ " + {
        "en": "Loaded signal has unexpected shape; cannot proceed.",
        "fr": "Le signal chargé a une forme inattendue ; impossible de continuer."
    }[lang])
    st.stop()

nsteps, nleads = sig_all.shape  # usually [≥15000, 12]

# -----------------------------
# 6) LEAD SELECTION & INTERACTIVE PLOT
# -----------------------------
st.markdown(f"**{TEXT['lead_selector'][lang]}**")
selected_lead = st.selectbox(
    label="",
    options=lead_names,     # e.g. ["I","II","III",…,"V6"]
    index=lead_names.index("II") if "II" in lead_names else 0
)

# Extract that channel’s data
lead_idx = lead_names.index(selected_lead)
lead_signal = sig_all[:, lead_idx]  # shape = [nsteps]

# Build a Pandas Series with a time index (in seconds):
# Assume sample rate = record.fs (e.g. 500 Hz)
fs = record.fs if hasattr(record, "fs") else 500
times = np.arange(len(lead_signal)) / fs
df_lead = pd.DataFrame({
    "Time (s)": times,
    f"Lead {selected_lead} (µV)": lead_signal
})

st.markdown("**" + {
    "en": "Interactive ECG Plot (zoom/pan with your mouse):",
    "fr": "Graphique ECG interactif (zoom/défilement avec la souris) :"
}[lang])

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_lead["Time (s)"],
    y=df_lead[f"Lead {selected_lead} (µV)"],
    mode="lines",
    line=dict(color="blue"),
    name=f"Lead {selected_lead}"
))

# Placeholder: if you have a list of abnormality ranges, overlay them as red rectangles.
# Example format for abnormal_ranges: [(start_time_s, end_time_s), …]
abnormal_ranges = []  # ← in the future, replace with real output from a localization module

for (t0, t1) in abnormal_ranges:
    fig.add_vrect(
        x0=t0, x1=t1,
        fillcolor="red",
        opacity=0.25,
        line_width=0,
        layer="below",
        annotation_text=""
    )

fig.update_layout(
    xaxis_title="Time (s)",
    yaxis_title=f"Lead {selected_lead} amplitude (µV)",
    margin=dict(l=40, r=20, t=30, b=40),
    height=300
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(f"_{TEXT['abnormal_hint'][lang]}_", unsafe_allow_html=True)

# -----------------------------
# 7) PREPROCESS FULL 12-LEAD SIGNAL
# -----------------------------
# We want exactly 15 000 samples per lead (30 s at 500 Hz). If shorter, pad at top; if longer, take the last 15 000.
if nsteps >= 15000:
    clipped = sig_all[-15000:, :]  # last 15 000 rows
else:
    clipped = sig_all          # fewer rows, will zero‐pad next

buffered = np.zeros((15000, nleads), dtype=np.float32)
buffered[-clipped.shape[0]:, :] = clipped
# Convert to [n_leads, 15000], then [1, n_leads, 15000]
x_np = buffered.transpose()           # shape = [nleads, 15000]
x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(device).float()  # [1, nleads, 15000]

# -----------------------------
# 8) MODEL INFERENCE → 9 PROBS
# -----------------------------
with torch.no_grad():
    logits = model(x_tensor)                    # [1, 9]
    probs = torch.sigmoid(logits)[0].cpu().numpy()  # [9]

classes = ['SNR','AF','IAVB','LBBB','RBBB','PAC','PVC','STD','STE']
prob_dict = {cls: float(probs[i]) for i, cls in enumerate(classes)}

st.markdown(f"## {TEXT['prob_title'][lang]}")
st.markdown(TEXT["prob_explain"][lang])

# If pandas ≥ 1.0 is installed, we can do:
try:
    df_probs = pd.DataFrame({
        "Class": classes,
        "Probability": [f"{p:.3f}" for p in probs]
    })
    st.table(df_probs)
except Exception:
    # Fallback if pandas < 1.0 and pyarrow is missing:
    for cls, p in prob_dict.items():
        st.markdown(f"- **{cls}**: {p:.3f}")

# -----------------------------
# 9) BAR CHART OF PROBABILITIES
# -----------------------------
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=classes,
    y=probs,
    marker_color="navy",
    text=[f"{p:.2f}" for p in probs],
    textposition="auto"
))
fig2.update_layout(
    yaxis=dict(title="Probability", range=[0, 1]),
    xaxis=dict(title="Class"),
    margin=dict(l=40, r=20, t=30, b=40),
    height=300
)
st.plotly_chart(fig2, use_container_width=True)
