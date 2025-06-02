import os
import tempfile

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import torch

from resnet import resnet34  # Import your ResNet definition
# Note: resnet34 must be in the same folder or on PYTHONPATH

st.set_page_config(page_title="ECG Arrhythmia Classifier", layout="wide")

# ------------------------------------------------------------
# 1) Specify where your PyTorch‐trained weights live:
#
#    Make sure you have a file named exactly "resnet34_model.pth"
#    (or edit this path) in the same directory as app.py.
# ------------------------------------------------------------
MODEL_PATH = "models/resnet34_CPSC_all_42.pth"

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Instantiate a ResNet34( input_channels=12 ), load weights,
    set to eval(), and return it (on CPU or GPU if available).
    """
    # 1. Build the model with 12 input channels (12 leads) and 9 outputs
    net = resnet34(input_channels=12, num_classes=9)
    # 2. Load state_dict
    #    We assume it was saved via `torch.save(net.state_dict(), MODEL_PATH)` in main.py
    #
    #    If you need GPU inference, uncomment the GPU lines and ensure you have a CUDA device.
    #
    device = torch.device("cuda:0") if (torch.cuda.is_available()) else torch.device("cpu")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    return net, device

# Load once and cache
model, device = load_model()

st.title("ECG Arrhythmia Detection (PyTorch ResNet)")
st.write("Upload a MIT-BIH/ CPSC `.mat` + `.hea` pair. We’ll read all 12 leads, "
         "show lead II, then classify into 9 arrhythmia classes.")

# ------------------------------------------------------------
# 2) File uploader: accept exactly two files (same basename)
# ------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload exactly two files (same basename): `<name>.mat` and `<name>.hea`",
    type=["mat", "hea"],
    accept_multiple_files=True
)

if uploaded and len(uploaded) == 2:
    # 2a) Write them into a temp dir so wfdb can read by basename
    tmpdir = tempfile.mkdtemp()
    for f in uploaded:
        with open(os.path.join(tmpdir, f.name), "wb") as out:
            out.write(f.getbuffer())

    # 2b) Infer record basename (strip extension)
    #     We assume both files have the same prefix, e.g. "100_sdnn.dat" + "100_sdnn.hea"
    base = os.path.splitext(uploaded[0].name)[0]
    recpath = os.path.join(tmpdir, base)  # wfdb expects no extension here

    # --------------------------------------------------------
    # 3) Read the record using wfdb (all leads, all samples)
    # --------------------------------------------------------
    try:
        # Read every available lead; we expect 12 leads if CPSC or MIT-BIH
        record = wfdb.rdrecord(recpath)
        sig_all = record.p_signal  # NumPy shape = [n_samples, n_leads]
        # If your record has >12 leads, you can select exactly the first 12 columns:
        # sig_all = sig_all[:, :12]
    except Exception as e:
        st.error(f"Could not read WFDB record: {e}")
    else:
        # ------------------------------------------------------------
        # 4) Plot a quick preview of lead II (channel index 1) for first 1000 samples
        # ------------------------------------------------------------
        sig_lead2 = sig_all[:, 1] if sig_all.shape[1] >= 2 else sig_all[:, 0]
        npts = sig_lead2.shape[0]
        if npts < 1000:
            # pad if shorter
            sig2_1000 = np.zeros(1000, dtype=np.float32)
            sig2_1000[:npts] = sig_lead2
        else:
            sig2_1000 = sig_lead2[:1000]

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(np.arange(1000) / 100.0, sig2_1000, linewidth=1)
        ax.set_title(f"Lead II (first 1000 samples) — {base}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("µV")
        st.pyplot(fig)

        # ------------------------------------------------------------
        # 5) Prepare the full 12-lead input just as `ECGDataset` does:
        #    - We want exactly 15 000 samples (30 s @ 500 Hz)
        #    - If record is shorter, zero-pad at the _top_ (front); if longer, take last 15 000
        # ------------------------------------------------------------
        nsteps, nleads = sig_all.shape  # normally [nsteps, 12]
        # Clip to last 15 000 rows, then pad if needed
        clipped = sig_all[-15000:, :] if nsteps >= 15000 else sig_all
        result = np.zeros((15000, nleads), dtype=np.float32)
        result[-clipped.shape[0] :, :] = clipped  # zero-pad the “front” if needed

        # Now transpose → shape [n_leads, 15000], then add batch dim → [1, n_leads, 15000]
        x_np = result.transpose()  # [12, 15000]
        x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(device)  # [1, 12, 15000]
        x_tensor = x_tensor.float()

        # ------------------------------------------------------------
        # 6) Run the ResNet inference (no grad → just feed-forward)
        # ------------------------------------------------------------
        with torch.no_grad():
            logits = model(x_tensor)                  # shape [1,9]
            probs = torch.sigmoid(logits)[0].cpu().numpy()  # [9] as float32

        # ------------------------------------------------------------
        # 7) Display the nine class probabilities as a simple DataFrame
        # ------------------------------------------------------------
        classes = ['SNR','AF','IAVB','LBBB','RBBB','PAC','PVC','STD','STE']
        # Build a table: class name  |  probability
        prob_table = np.vstack((classes, [f"{p:.3f}" for p in probs])).T
        st.markdown("## Predicted Class Probabilities")
        st.table(prob_table)

        # ------------------------------------------------------------
        # 8) (Optional) Bar chart of the nine probabilities
        # ------------------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.bar(classes, probs, color="tab:blue")
        ax2.set_ylabel("Probability")
        ax2.set_ylim(0, 1)
        ax2.set_title("Confidence for Each Class")
        ax2.set_xticklabels(classes, rotation=45, ha="right")
        st.pyplot(fig2)

else:
    st.info("Please upload exactly two files (`.dat` + `.hea`) with identical basenames.")
