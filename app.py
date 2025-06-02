import os
import tempfile

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import wfdb
import torch

from resnet import resnet34  # Ensure resnet34.py is in the same folder

# ------------------------------------------------------------
# 1) PAGE CONFIG (MUST BE FIRST Streamlit CALL)
# ------------------------------------------------------------
st.set_page_config(
    page_title="ECG Arrhythmia Detection (PyTorch ResNet)",
    layout="wide",
)

# ------------------------------------------------------------
# 2) MODEL LOADING (cached)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 3) TRANSLATIONS / TEXT
# ------------------------------------------------------------
LANGUAGES = {
    "English": "en",
    "Français": "fr",
    "العربية": "ar"
}
lang_choice = st.sidebar.selectbox("Language / Langue / اللغة", ["English", "Français", "العربية"])
lang = LANGUAGES[lang_choice]

# Each TEXT[key] is a dict with "en", "fr", "ar"
TEXT = {
    "title": {
        "en": "ECG Arrhythmia Detection (PyTorch ResNet)",
        "fr": "Détection d’arythmie ECG (PyTorch ResNet)",
        "ar": "كشف اضطراب النظم الكهربائي للقلب (PyTorch ResNet)"
    },
    "description": {
        "en": (
            "Upload exactly two files (same basename): a `.hea` file and its matching `.mat` file.  \n"
            "We’ll read all 12 leads from the MATLAB‐formatted record, let you choose which lead(s) to visualize, "
            "and then classify the recording into 9 arrhythmia types."
        ),
        "fr": (
            "Uploadez exactement deux fichiers (même nom de base) : un fichier `.hea` et son fichier `.mat` correspondant.  \n"
            "Nous lirons les 12 dérivations du fichier MATLAB, vous pourrez choisir quelle(s) dérivation(s) afficher, "
            "puis classifierons l’enregistrement en 9 types d’arythmie."
        ),
        "ar": (
            "قم بتحميل ملفين بالضبط (نفس الاسم الأساسي): ملف `.hea` وملف `.mat` المقابل.  \n"
            "سنقوم بقراءة جميع القنوات الــ 12 من ملف MATLAB، مما يتيح لك اختيار أي قناة/قنوات للمشاهدة، "
            "ثم سنصنف التسجيل إلى 9 أنواع من اضطرابات النظم القلبي."
        )
    },
    "upload_hint": {
        "en": "Please upload exactly two files (`.hea` + `.mat`) with the same basename.",
        "fr": "Veuillez uploader exactement deux fichiers (`.hea` + `.mat`) portant le même nom de base.",
        "ar": "يرجى تحميل ملفين بالضبط (`.hea` + `.mat`) بنفس الاسم الأساسي."
    },
    "lead_selector": {
        "en": "Select lead(s) to display:",
        "fr": "Choisir la/les dérivation(s) à afficher :",
        "ar": "اختر القناة/القنوات للعرض:"
    },
    "overlay_all": {
        "en": "Overlay all leads",
        "fr": "Superposer toutes les dérivations",
        "ar": "عرض جميع القنوات معاً"
    },
    "plot_style": {
        "en": "Plot style:",
        "fr": "Style du tracé :",
        "ar": "نمط الرسم :"
    },
    "style_interactive": {
        "en": "Interactive (Plotly)",
        "fr": "Interactif (Plotly)",
        "ar": "تفاعلي (Plotly)"
    },
    "style_printed": {
        "en": "Printed ECG style",
        "fr": "Style ECG imprimé",
        "ar": "مظهر ECG مطبوع"
    },
    "prob_title": {
        "en": "Predicted Class Probabilities",
        "fr": "Probabilités prédites par classe",
        "ar": "احتمالات الفئات المتوقعة"
    },
    "prob_explain": {
        "en": (
            "Below are the nine arrhythmia classes with their respective probabilities (0–1).  \n"
            "A higher probability means the model is more confident that the rhythm is present."
        ),
        "fr": (
            "Voici les neuf classes d’arythmie avec leurs probabilités respectives (0–1).  \n"
            "Une probabilité élevée signifie que le modèle est plus confiant que le rythme est présent."
        ),
        "ar": (
            "فيما يلي تسع فئات لاضطرابات النظم الكهربائي مع احتمالاتها (من 0 إلى 1).  \n"
            "كلما زادت الاحتمالية، زادت ثقة النموذج بأن الاضطراب موجود."
        )
    },
    "abnormal_hint": {
        "en": (
            "If the model or a future explainer provides time ranges of abnormality, they will be overlaid in red."
        ),
        "fr": (
            "Si le modèle ou un futur module d’explication fournit des plages horaires d’anomalie, elles seront surlignées en rouge."
        ),
        "ar": (
            "إذا قدم النموذج أو أيّ أداة تفسيرية مستقبلية نطاقات زمنية للشذوذ، فسيتم تمييزها باللون الأحمر."
        )
    },
    "class_explanation_title": {
        "en": "Class Explanations",
        "fr": "Explications des classes",
        "ar": "تفسيرات الفئات"
    },
    "class_brief_label": {
        "en": "Brief (general public):",
        "fr": "Bref (grand public) :",
        "ar": "نبذة (للعامة):"
    },
    "class_detailed_label": {
        "en": "Details for clinicians:",
        "fr": "Détails pour cliniciens :",
        "ar": "تفاصيل للأطباء :"
    }
}

# Full class names and their translations
CLASSES = {
    "SNR": {
        "en": "Sinus Rhythm",
        "fr": "Rythme sinusal normal",
        "ar": "إيقاع جيبي طبيعي"
    },
    "AF": {
        "en": "Atrial Fibrillation",
        "fr": "Fibrillation atriale",
        "ar": "رجفان أذيني"
    },
    "IAVB": {
        "en": "First‐Degree AV Block",
        "fr": "Bloc AV du premier degré",
        "ar": "حجب أذيني بطيني من الدرجة الأولى"
    },
    "LBBB": {
        "en": "Left Bundle Branch Block",
        "fr": "Bloc de branche gauche",
        "ar": "حجب حزمة فرع أيسر"
    },
    "RBBB": {
        "en": "Right Bundle Branch Block",
        "fr": "Bloc de branche droite",
        "ar": "حجب حزمة فرع أيمن"
    },
    "PAC": {
        "en": "Premature Atrial Contraction",
        "fr": "Contraction auriculaire prématurée",
        "ar": "تقلص أذيني مبكر"
    },
    "PVC": {
        "en": "Premature Ventricular Contraction",
        "fr": "Contraction ventriculaire prématurée",
        "ar": "تقلص بطيني مبكر"
    },
    "STD": {
        "en": "ST‐Segment Depression",
        "fr": "Dépression du segment ST",
        "ar": "هبوط مقطع ST"
    },
    "STE": {
        "en": "ST‐Segment Elevation",
        "fr": "Élévation du segment ST",
        "ar": "ارتفاع مقطع ST"
    }
}

# Brief explanations for each class (for general public/patient)
BRIEF_EXPLANATIONS = {
    "SNR": {
        "en": "The heart’s natural pacemaker is producing a normal, regular rhythm.",
        "fr": "Le stimulateur naturel du cœur produit un rythme régulier normal.",
        "ar": "المُحفّز الطبيعي للقلب ينتج إيقاعًا طبيعيًا ومنتظمًا."
    },
    "AF": {
        "en": "The atria are quivering instead of contracting normally, causing irregular heartbeat.",
        "fr": "Les oreillettes tremblent au lieu de se contracter normalement, provoquant un rythme irrégulier.",
        "ar": "الأذينين يرتعشان بدلاً من الانقباض بشكل طبيعي، مما يسبب إيقاعًا غير منتظم."
    },
    "IAVB": {
        "en": "Electrical signals from the atria to the ventricles are slowed but still pass through.",
        "fr": "Les signaux électriques des oreillettes aux ventricules sont ralentis mais passent toujours.",
        "ar": "الإشارات الكهربائية من الأذينين إلى البطينين تبطؤ لكنها لا تزال تمر."
    },
    "LBBB": {
        "en": "A blockage or delay in the left side conduction system, changing ECG shape.",
        "fr": "Blocage ou retard dans le système de conduction gauche, modifiant la forme de l’ECG.",
        "ar": "انسداد أو تأخير في نظام التوصيل الأيسر، مما يغير شكل تخطيط القلب."
    },
    "RBBB": {
        "en": "A blockage or delay in the right side conduction system, changing ECG shape.",
        "fr": "Blocage ou retard dans le système de conduction droit, modifiant la forme de l’ECG.",
        "ar": "انسداد أو تأخير في نظام التوصيل الأيمن، مما يغير شكل تخطيط القلب."
    },
    "PAC": {
        "en": "An early heartbeat originating in the atria that interrupts the normal rhythm.",
        "fr": "Un battement précoce provenant des oreillettes qui interrompt le rythme normal.",
        "ar": "نبضة قلب مبكرة تنشأ من الأذينين وتقطع الإيقاع الطبيعي."
    },
    "PVC": {
        "en": "An early heartbeat originating in the ventricles that interrupts the normal rhythm.",
        "fr": "Un battement précoce provenant des ventricules qui interrompt le rythme normal.",
        "ar": "نبضة قلب مبكرة تنشأ من البطينين وتقطع الإيقاع الطبيعي."
    },
    "STD": {
        "en": "The ST segment is lower than normal, often indicating reduced blood flow to heart muscle.",
        "fr": "Le segment ST est plus bas que la normale, indiquant souvent une réduction du flux sanguin vers le muscle cardiaque.",
        "ar": "مقطع ST أقل من الطبيعي، مما يدل غالبًا على تدفق دم منخفض إلى عضلة القلب."
    },
    "STE": {
        "en": "The ST segment is higher than normal, often indicating an acute heart attack.",
        "fr": "Le segment ST est plus haut que la normale, indiquant souvent un infarctus aigu.",
        "ar": "مقطع ST أعلى من الطبيعي، مما يدل غالبًا على نوبة قلبية حادة."
    }
}

# Detailed explanations for each class (for clinicians)
DETAILED_EXPLANATIONS = {
    "SNR": {
        "en": (
            "- **What it means:** The sinoatrial node is firing normally, producing a regular cardiac rhythm.\n"
            "- **Clinical note:** Normal variant. No treatment needed unless other signs present."
        ),
        "fr": (
            "- **Signification :** Le nœud sinusal fonctionne normalement, produisant un rythme cardiaque régulier.\n"
            "- **Note clinique :** Variante normale. Aucun traitement nécessaire sauf en présence d’autres signes."
        ),
        "ar": (
            "- **المعنى:** العقدة الجيبية تنبض بشكل طبيعي، مما ينتج إيقاع قلب منتظم.\n"
            "- **ملاحظة سريرية:** نمط طبيعي. لا حاجة إلى علاج ما لم توجد علامات أخرى."
        )
    },
    "AF": {
        "en": (
            "- **What it means:** Atrial electrical activity is chaotic, no distinct P-waves, irregular R–R intervals.\n"
            "- **Clinical issues:** Increases risk of stroke, heart failure, palpitations. Consider anticoagulation, rate/rhythm control."
        ),
        "fr": (
            "- **Signification :** Activité électrique atriale chaotique, absence d’ondes P nettes, intervalles R–R irréguliers.\n"
            "- **Problèmes cliniques :** Augmente le risque d’AVC, d’insuffisance cardiaque, palpitations. Envisager anticoagulation, contrôle du rythme/de la fréquence."
        ),
        "ar": (
            "- **المعنى:** النشاط الكهربائي الأذيني فوضوي، لا توجد موجات P واضحة، فترات R–R غير منتظمة.\n"
            "- **ملاحظات سريرية:** يزيد خطر السكتة الدماغية وفشل القلب والخفقان. فكر في العلاج بمضادات التخثر والسيطرة على معدل/إيقاع القلب."
        )
    },
    "IAVB": {
        "en": (
            "- **What it means:** The PR interval is prolonged (>200 ms) but all impulses conduct to ventricles.\n"
            "- **Clinical issues:** Often benign. Monitor for progression to higher‐degree AV block. Evaluate electrolytes & medications."
        ),
        "fr": (
            "- **Signification :** Intervalle PR prolongé (>200 ms) mais toutes les impulsions se conduisent aux ventricules.\n"
            "- **Problèmes cliniques :** Souvent bénin. Surveiller la progression vers un bloc AV de degré supérieur. Évaluer électrolytes et médicaments."
        ),
        "ar": (
            "- **المعنى:** فترة PR ممتدة (>200 مللي ثانية) لكن جميع النبضات تنتقل إلى البطينين.\n"
            "- **ملاحظات سريرية:** غالبًا ما يكون حميدًا. راقب تطور الحالة إلى حجب أذيني بطيني أعلى. قيّم الكهارل والأدوية."
        )
    },
    "LBBB": {
        "en": (
            "- **What it means:** QRS duration ≥120 ms with broad/notched R-waves in leads V5–V6, I, aVL.\n"
            "- **Clinical issues:** May indicate underlying cardiomyopathy or ischemia. Correlate with echocardiogram findings."
        ),
        "fr": (
            "- **Signification :** Durée QRS ≥120 ms avec ondes R larges/biseautées en V5–V6, I, aVL.\n"
            "- **Problèmes cliniques :** Peut indiquer une cardiomyopathie ou ischémie sous-jacente. Mettre en corrélation avec échocardiographie."
        ),
        "ar": (
            "- **المعنى:** مدة QRS ≥120 مللي ثانية مع موجات R عريضة/منقسمة في V5–V6، I، aVL.\n"
            "- **ملاحظات سريرية:** قد تشير إلى اعتلال عضلة القلب أو نقص التروية. قارن مع نتائج تخطيط صدى القلب."
        )
    },
    "RBBB": {
        "en": (
            "- **What it means:** QRS duration ≥120 ms with rsR′ pattern in V1 and wide S in lateral leads.\n"
            "- **Clinical issues:** Often benign in young; if new, evaluate for pulmonary embolism or structural heart disease."
        ),
        "fr": (
            "- **Signification :** Durée QRS ≥120 ms avec schéma rsR′ en V1 et onde S large dans les dérivations latérales.\n"
            "- **Problèmes cliniques :** Souvent bénin chez le jeune ; si nouveau, évaluer pour embolie pulmonaire ou cardiopathie structurelle."
        ),
        "ar": (
            "- **المعنى:** مدة QRS ≥120 مللي ثانية مع نمط rsR′ في V1 وموجة S عريضة في القنوات الجانبية.\n"
            "- **ملاحظات سريرية:** غالبًا ما يكون حميدًا عند الشباب؛ إذا كان جديدًا، قُم بالتقييم بحثًا عن الانصمام الرئوي أو أمراض القلب الهيكلية."
        )
    },
    "PAC": {
        "en": (
            "- **What it means:** An early P-wave occurs before the expected sinus P-wave, often with a non-compensatory pause.\n"
            "- **Clinical issues:** Usually benign; frequent PACs may trigger atrial fibrillation. Assess for stimulants or electrolyte imbalance."
        ),
        "fr": (
            "- **Signification :** Une onde P précoce apparaît avant l’onde P sinusal attendue, souvent avec une pause non compensatoire.\n"
            "- **Problèmes cliniques :** Habituellement bénin ; des PAC fréquentes peuvent déclencher une fibrillation atriale. Rechercher stimulants ou déséquilibre électrolytique."
        ),
        "ar": (
            "- **المعنى:** تحدث موجة P مبكرة قبل موجة P الجيبية المتوقعة، غالبًا مع توقف غير تعويضي.\n"
            "- **ملاحظات سريرية:** عادةً ما يكون حميدًا؛ قد تؤدي PAC المتكررة إلى رجفان أذيني. تحقق من المنبهات أو خلل الإلكتروليت."
        )
    },
    "PVC": {
        "en": (
            "- **What it means:** A wide QRS complex appears earlier than expected, without prior P-wave.\n"
            "- **Clinical issues:** Occasional PVCs are benign; frequent/multifocal PVCs warrant evaluation for ischemia or cardiomyopathy."
        ),
        "fr": (
            "- **Signification :** Un complexe QRS large apparaît plus tôt que prévu, sans onde P antérieure.\n"
            "- **Problèmes cliniques :** Des PVC occasionnelles sont bénignes ; des PVC fréquentes/multifocales nécessitent une évaluation pour ischémie ou cardiomyopathie."
        ),
        "ar": (
            "- **المعنى:** يظهر مركب QRS عريض في وقت أبكر مما هو متوقع، بدون موجة P سابقة.\n"
            "- **ملاحظات سريرية:** غالبًا ما تكون PVC العرضية حميدة؛ تستدعي PVC المتكررة/متعددة البؤر تقييم نقص التروية أو اعتلال عضلة القلب."
        )
    },
    "STD": {
        "en": (
            "- **What it means:** ST segment is depressed ≥0.5 mm below baseline in two contiguous leads.\n"
            "- **Clinical issues:** Suggests subendocardial ischemia. Correlate with patient symptoms; consider stress testing or angiography."
        ),
        "fr": (
            "- **Signification :** Le segment ST est abaissé ≥0.5 mm sous la ligne de base dans deux dérivations contiguës.\n"
            "- **Problèmes cliniques :** Suggère une ischémie sous-endocardique. Mettre en corrélation avec les symptômes ; envisager test d’effort ou angiographie."
        ),
        "ar": (
            "- **المعنى:** مقطع ST منخفض ≥0.5 مم تحت الخط الأساسي في اثنين من القنوات المجاورة.\n"
            "- **ملاحظات سريرية:** يشير إلى نقص تروية تحت الباطنة. قارن مع أعراض المريض؛ اعتبر اختبار الإجهاد أو تصوير الأوعية."
        )
    },
    "STE": {
        "en": (
            "- **What it means:** ST segment is elevated ≥1 mm above baseline in two contiguous leads.\n"
            "- **Clinical issues:** Suggests acute transmural myocardial infarction. Requires immediate cardiology evaluation and likely reperfusion."
        ),
        "fr": (
            "- **Signification :** Le segment ST est surélevé ≥1 mm au-dessus de la ligne de base dans deux dérivations contiguës.\n"
            "- **Problèmes cliniques :** Suggère un infarctus transmural aigu. Nécessite une évaluation cardiologique immédiate et probablement une reperfusion."
        ),
        "ar": (
            "- **المعنى:** مقطع ST مرتفع ≥1 مم فوق الخط الأساسي في اثنين من القنوات المجاورة.\n"
            "- **ملاحظات سريرية:** يشير إلى احتشاء عضلة القلب الحاد الشامل. يتطلب تقييمًا عاجلاً من طبيب القلب وربما إعادة تدفق الدم."
        )
    }
}

st.title(TEXT["title"][lang])
st.write(TEXT["description"][lang])

# ------------------------------------------------------------
# 4) FILE-UPLOADER
# ------------------------------------------------------------
uploaded = st.file_uploader(
    TEXT["upload_hint"][lang],
    type=["hea", "mat"],
    accept_multiple_files=True
)

if not (uploaded and len(uploaded) == 2):
    st.info(TEXT["upload_hint"][lang])
    st.stop()

# ------------------------------------------------------------
# 5) SAVE UPLOADED FILES TO TEMPORARY DIRECTORY
# ------------------------------------------------------------
tmpdir = tempfile.mkdtemp()
for f in uploaded:
    path = os.path.join(tmpdir, f.name)
    with open(path, "wb") as out:
        out.write(f.getbuffer())

# Infer basename (must match between .hea and .mat)
base1 = os.path.splitext(uploaded[0].name)[0]
base2 = os.path.splitext(uploaded[1].name)[0]
if base1 != base2:
    st.error(
        "❌ " + {
            "en": "The two files must share the exact same basename (e.g. both start with '100').",
            "fr": "Les deux fichiers doivent partager exactement le même nom de base (par ex. tous deux commencent par '100').",
            "ar": "يجب أن يكون لملفين نفس الاسم الأساسي بالضبط (مثال: جميعهما يبدأ بـ '100')."
        }[lang]
    )
    st.stop()

record_name = base1
recpath = os.path.join(tmpdir, record_name)

# ------------------------------------------------------------
# 6) READ WFDB RECORD (MATLAB)
# ------------------------------------------------------------
try:
    record = wfdb.rdrecord(recpath)
    sig_all = record.p_signal       # shape: [n_samples, n_leads]
    lead_names = record.sig_name     # e.g. ["I","II","III",...,"V6"]
except Exception as e:
    st.error(
        "❌ " + {
            "en": f"Could not read WFDB record `{record_name}`: {e}",
            "fr": f"Impossible de lire l’enregistrement WFDB `{record_name}` : {e}",
            "ar": f"تعذر قراءة سجل WFDB `{record_name}`: {e}"
        }[lang]
    )
    st.stop()

# Validate shape
if sig_all.ndim != 2 or len(lead_names) == 0:
    st.error(
        "❌ " + {
            "en": "Loaded signal has unexpected shape; cannot proceed.",
            "fr": "Le signal chargé a une forme inattendue ; impossible de continuer.",
            "ar": "الإشارة المحملة لها شكل غير متوقع؛ لا يمكن المتابعة."
        }[lang]
    )
    st.stop()

nsteps, nleads = sig_all.shape  # usually [≥15000, 12]

# ------------------------------------------------------------
# 7) SIDEBAR CONTROLS: lead selection, overlay all, plot style
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{TEXT['lead_selector'][lang]}**")

# Multi-select for leads (just use lead names, no CLASSES lookup)
all_lead_options = lead_names.copy()
selected_leads = st.sidebar.multiselect(
    label="",
    options=all_lead_options,
    default=["II"] if "II" in all_lead_options else [all_lead_options[0]]
)

overlay_all = st.sidebar.checkbox(TEXT["overlay_all"][lang], value=False)

st.sidebar.markdown(f"**{TEXT['plot_style'][lang]}**")
plot_style = st.sidebar.radio(
    "",
    [TEXT["style_interactive"][lang], TEXT["style_printed"][lang]],
    index=0
)

st.sidebar.markdown("---")

# ------------------------------------------------------------
# 8) PLOT ECG: either overlay all leads or the selected ones
# ------------------------------------------------------------
fs = record.fs if hasattr(record, "fs") else 500
times = np.arange(nsteps) / fs

# Function to overlay abnormal ranges if provided
def add_abnormal_shapes(fig, abnormal_ranges):
    for (t0, t1) in abnormal_ranges:
        fig.add_vrect(
            x0=t0, x1=t1,
            fillcolor="red",
            opacity=0.25,
            line_width=0,
            layer="below",
            annotation_text=""
        )

abnormal_ranges = []  # ← populate with (start_s, end_s) if available

if overlay_all:
    # Overlay every lead with a distinct color
    if plot_style == TEXT["style_interactive"][lang]:
        fig = go.Figure()
        for idx, lead in enumerate(lead_names):
            fig.add_trace(go.Scatter(
                x=times,
                y=sig_all[:, idx],
                mode="lines",
                name=lead  # just the lead name
            ))
        add_abnormal_shapes(fig, abnormal_ranges)
        fig.update_layout(
            xaxis_title={
                "en": "Time (s)",
                "fr": "Temps (s)",
                "ar": "الوقت (ثواني)"
            }[lang],
            yaxis_title={
                "en": "Amplitude (µV)",
                "fr": "Amplitude (µV)",
                "ar": "السعة (ميكروفولت)"
            }[lang],
            margin=dict(l=40, r=20, t=30, b=40),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Printed ECG style: stack all leads with vertical offsets
        plt.figure(figsize=(12, 6))
        offset = 2000  # vertical offset
        for idx, lead in enumerate(lead_names):
            y = sig_all[:, idx] + offset * (len(lead_names) - idx)
            plt.plot(times, y, color="black", linewidth=0.5)
            plt.text(
                times[-1] + 0.1,
                offset * (len(lead_names) - idx),
                lead,
                verticalalignment="center"
            )
        # Light gray grid to mimic ECG paper
        plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
        plt.xlabel({
            "en": "Time (s)",
            "fr": "Temps (s)",
            "ar": "الوقت (ثواني)"
        }[lang])
        plt.yticks([])  # hide y-axis labels (we annotate next to lines)
        st.pyplot(plt.gcf())
else:
    # Plot only the selected leads
    if plot_style == TEXT["style_interactive"][lang]:
        fig = go.Figure()
        for lead in selected_leads:
            idx = lead_names.index(lead)
            fig.add_trace(go.Scatter(
                x=times,
                y=sig_all[:, idx],
                mode="lines",
                name=lead
            ))
        add_abnormal_shapes(fig, abnormal_ranges)
        fig.update_layout(
            xaxis_title={
                "en": "Time (s)",
                "fr": "Temps (s)",
                "ar": "الوقت (ثواني)"
            }[lang],
            yaxis_title={
                "en": "Amplitude (µV)",
                "fr": "Amplitude (µV)",
                "ar": "السعة (ميكروفولت)"
            }[lang],
            margin=dict(l=40, r=20, t=30, b=40),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        plt.figure(figsize=(12, 3 * len(selected_leads)))
        offset = 2000
        for i, lead in enumerate(selected_leads):
            idx = lead_names.index(lead)
            y = sig_all[:, idx] + offset * (len(selected_leads) - i)
            plt.subplot(len(selected_leads), 1, i + 1)
            plt.plot(times, y, color="black", linewidth=0.5)
            plt.text(
                times[-1] + 0.1,
                offset * (len(selected_leads) - i),
                lead,
                verticalalignment="center"
            )
            plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
            if i == len(selected_leads) - 1:
                plt.xlabel({
                    "en": "Time (s)",
                    "fr": "Temps (s)",
                    "ar": "الوقت (ثواني)"
                }[lang])
            else:
                plt.xticks([])
        st.pyplot(plt.gcf())

st.markdown(f"_{TEXT['abnormal_hint'][lang]}_", unsafe_allow_html=True)

# ------------------------------------------------------------
# 9) PREPROCESS FULL 12-LEAD SIGNAL FOR MODEL
# ------------------------------------------------------------
if nsteps >= 15000:
    clipped = sig_all[-15000:, :]
else:
    clipped = sig_all

buffered = np.zeros((15000, nleads), dtype=np.float32)
buffered[-clipped.shape[0]:, :] = clipped
x_np = buffered.transpose()                           # shape = [n_leads, 15000]
x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(device).float()  # shape = [1, n_leads, 15000]

# ------------------------------------------------------------
# 10) MODEL INFERENCE → 9 PROBABILITIES
# ------------------------------------------------------------
with torch.no_grad():
    logits = model(x_tensor)                        # shape = [1, 9]
    probs = torch.sigmoid(logits)[0].cpu().numpy()   # shape = [9]

# Map class abbreviations to full names in chosen language
class_abbrs = list(CLASSES.keys())
class_names = [CLASSES[abbr][lang] for abbr in class_abbrs]
prob_dict = {CLASSES[abbr][lang]: float(probs[i]) for i, abbr in enumerate(class_abbrs)}

st.markdown(f"## {TEXT['prob_title'][lang]}")
st.markdown(TEXT["prob_explain"][lang])

# Display probabilities as a DataFrame if pandas ≥ 1.0; else fallback to Markdown list
try:
    df_probs = pd.DataFrame({
        "Class": class_names,
        "Probability": [f"{p:.3f}" for p in probs]
    })
    st.table(df_probs)
except Exception:
    for name, p in prob_dict.items():
        st.markdown(f"- **{name}**: {p:.3f}")

# ------------------------------------------------------------
# 11) BAR CHART OF PROBABILITIES
# ------------------------------------------------------------
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=class_names,
    y=probs,
    marker_color="navy",
    text=[f"{p:.2f}" for p in probs],
    textposition="auto"
))
fig2.update_layout(
    yaxis=dict(
        title={
            "en": "Probability",
            "fr": "Probabilité",
            "ar": "احتمالية"
        }[lang],
        range=[0, 1]
    ),
    xaxis=dict(
        title={
            "en": "Class",
            "fr": "Classe",
            "ar": "فئة"
        }[lang]
    ),
    margin=dict(l=40, r=20, t=30, b=40),
    height=300
)
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------
# 12) DETAILED REPORT: class explanations for general public and clinicians
# ------------------------------------------------------------
st.markdown(f"### {TEXT['class_explanation_title'][lang]}")
for abbr in class_abbrs:
    fullname = CLASSES[abbr][lang]
    prob = float(probs[class_abbrs.index(abbr)])
    st.markdown(f"**{fullname} — {prob:.3f}**")
    # Brief explanation
    st.markdown(f"- **{TEXT['class_brief_label'][lang]}** {BRIEF_EXPLANATIONS[abbr][lang]}")
    # Detailed (clinician) explanation
    with st.expander(TEXT['class_detailed_label'][lang]):
        st.markdown(DETAILED_EXPLANATIONS[abbr][lang])
    st.markdown("---")


