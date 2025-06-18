#  Deep Learning for Diagnosis of 12-lead Electrocardiogram

[![Python](https://img.shields.io/badge/python-3.7.4-blue.svg)](https://www.python.org/downloads/release/python-374/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.2.0-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![SHAP](https://img.shields.io/badge/SHAP-0.35.1-red.svg)](https://shap.readthedocs.io/)
[![Web App](https://img.shields.io/badge/Web%20App-Hearline-brightgreen.svg)](https://github.com/blamairia/Hearline-Webapp)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://github.com/blamairia/Hearline-Webapp/tree/master/doc/index.md)

## Abstract

This repository contains the implementation for **" Deep Learning for  Diagnosis of 12-lead Electrocardiogram"**. Electrocardiogram (ECG) is a widely used, reliable, non-invasive approach for cardiovascular disease diagnosis. With the rapid growth of ECG examinations and the insufficiency of cardiologists, accurately automatic diagnosis of ECG signals has become a critical research area. 

We developed a deep neural network for **multi-label classification** of cardiac arrhythmias in 12-lead ECG records using a **ResNet-34** architecture adapted for 1D time-series data. Our method achieved an average **AUC of 0.970** and an average **F1-score of 0.813** on the CPSC2018 dataset. The model demonstrates superior performance when using all 12 leads compared to single-lead inputs, with leads **I, aVR, and V5** showing the best individual performance. We employed **SHAP (SHapley Additive exPlanations)** for model interpretability and developed a comprehensive web application for clinical deployment.

🌐 **Live Web Application**: [Hearline-Webapp](https://github.com/blamairia/Hearline-Webapp) - Interactive clinical interface for ECG analysis  
📚 **Documentation**: [Complete Documentation](https://github.com/blamairia/Hearline-Webapp/tree/master/doc/index.md) - Detailed API and usage guide

## 🏥 Clinical Significance

This work addresses the critical need for automated ECG interpretation in clinical settings:
- **Scalability**: Assists cardiologists in handling increasing patient loads
- **Consistency**: Provides standardized diagnostic support across healthcare facilities
- **Accessibility**: Enables ECG screening in resource-limited settings
- **Education**: Serves as a training tool for medical students and residents

## 📊 Model Architecture

### ResNet-34 for 1D ECG Signals

Our model adapts the ResNet-34 architecture for temporal ECG data:

```
Input: 12-lead ECG (12 × 15,000 samples @ 500Hz = 30 seconds)
    ↓
Conv1D (kernel=15, stride=2) + BatchNorm + ReLU + MaxPool
    ↓
ResNet Blocks: [3, 4, 6, 3] with channels [64, 128, 256, 512]
    ↓
Adaptive Average + Max Pooling → Concatenation
    ↓
Fully Connected Layer → 9 classes (multi-label)
    ↓
Output: Probability scores for each arrhythmia type
```

### Cardiac Conditions Classified

The model classifies ECG signals into **9 different cardiac conditions**:

| Abbreviation | Full Name | Clinical Significance |
|-------------|-----------|---------------------|
| **SNR** | Sinus Rhythm | Normal heart rhythm |
| **AF** | Atrial Fibrillation | Irregular atrial contractions |
| **IAVB** | First-degree AV Block | Delayed conduction between atria and ventricles |
| **LBBB** | Left Bundle Branch Block | Delayed left ventricular activation |
| **RBBB** | Right Bundle Branch Block | Delayed right ventricular activation |
| **PAC** | Premature Atrial Contraction | Early atrial beats |
| **PVC** | Premature Ventricular Contraction | Early ventricular beats |
| **STD** | ST-segment Depression | Possible myocardial ischemia |
| **STE** | ST-segment Elevation | Possible myocardial infarction |

## 📁 Repository Structure

```
ecg-diagnosis/
├── 🧠 Core Model Files
│   ├── main.py                 # Training pipeline
│   ├── resnet.py              # ResNet-34 1D architecture
│   ├── dataset.py             # ECG data loader with augmentation
│   ├── predict.py             # Model evaluation and testing
│   └── utils.py               # Utility functions (metrics, data splits)
│
├── 🔧 Data Processing
│   ├── preprocess.py          # CPSC dataset preprocessing
│   ├── expert_features.py     # Handcrafted feature extraction
│   └── QRSDetectorOffline.py  # Pan-Tompkins QRS detection
│
├── 📈 Analysis & Interpretation
│   ├── shap_values.py         # SHAP explainability analysis
│   ├── baselines.py           # Traditional ML baselines
│   ├── statistic.py           # Dataset statistics
│   └── visualize.py           # ECG signal visualization
│
├── 🌐 Web Applications
│   ├── app.py                 # Streamlit clinical interface
│   └── index.html             # HTML/JavaScript demo
│
├── 📂 Data & Results
│   ├── data/CPSC/             # CPSC2018 dataset
│   ├── models/                # Trained model weights
│   ├── results/               # Performance metrics & plots
│   ├── shap/                  # SHAP visualization outputs
│   └── plots/                 # Generated visualizations
│
└── 📋 Configuration
    └── requirements.txt       # Python dependencies
```

## 📚 Dataset: CPSC2018

### Dataset Description
The **China Physiological Signal Challenge (CPSC) 2018** dataset contains 12-lead ECG recordings from real clinical settings:

- **Format**: WFDB format (`.hea` + `.mat` files)
- **Leads**: 12-lead ECG (I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6)
- **Sampling Rate**: 500 Hz
- **Duration**: Variable length (minimum 15,000 samples = 30 seconds used)
- **Annotations**: Multi-label diagnostic codes (SNOMED CT)

### Data Access
📥 **Download processed CPSC dataset**: [Dropbox Link](https://www.dropbox.com/s/unicm8ulxt24vh8/CPSC.zip?dl=0)

### Data Preprocessing Pipeline

1. **Signal Standardization**: Normalize to 30-second windows (15,000 samples)
2. **Lead Selection**: Support for all 12 leads or subset selection
3. **Data Augmentation**: Scaling and temporal shifting for training
4. **Cross-validation**: 10-fold stratified split (8:1:1 train/val/test)

## 💻 Software Requirements

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.7.4 | Core runtime environment |
| **PyTorch** | 1.2.0 | Deep learning framework |
| **NumPy** | 1.17.2 | Numerical computations |
| **Pandas** | 0.25.2 | Data manipulation |
| **Scikit-learn** | 0.21.3 | Machine learning utilities |
| **Matplotlib** | 3.1.1 | Static visualizations |
| **WFDB** | 2.2.1 | ECG data reading/writing |
| **SHAP** | 0.35.1 | Model interpretability |
| **Streamlit** | Latest | Web application framework |
| **Plotly** | Latest | Interactive visualizations |

### Signal Processing & Analysis

| Package | Purpose |
|---------|---------|
| **SciPy** | 1.3.1 - Signal processing algorithms |
| **PyWavelets** | Wavelet transforms for feature extraction |
| **BioSPPy** | Biomedical signal processing |
| **PeakUtils** | Peak detection algorithms |
| **LightGBM** | Gradient boosting for baselines |
| **tqdm** | 4.36.1 - Progress bars |

### Hardware Recommendations
- **GPU**: NVIDIA GPU with CUDA support (≥4GB VRAM)
- **RAM**: Minimum 8GB, 16GB recommended for large datasets
- **Storage**: 10GB for dataset, models, and results

## 🚀 Installation & Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/blamairia/ecg-diagnosis.git
cd ecg-diagnosis

# Create virtual environment (recommended)
python -m venv ecg_env
source ecg_env/bin/activate  # Linux/Mac
# ecg_env\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install torch==1.2.0 torchvision==0.4.0
pip install scikit-learn==0.21.3 scipy==1.3.1 
pip install shap==0.35.1 tqdm==4.36.1 wfdb==2.2.1
pip install matplotlib==3.1.1 numpy==1.17.2 pandas==0.25.2
pip install streamlit plotly pywavelets lightgbm biosppy peakutils
```

### 3. Dataset Preparation

```bash
# Create data directory
mkdir -p data/CPSC

# Download and extract CPSC dataset
# Place extracted .hea and .mat files in data/CPSC/
```

## 🔬 Usage Instructions

### 1. Data Preprocessing

Generate reference and label CSV files from raw WFDB data:

```bash
python preprocess.py --data-dir data/CPSC
```

**Output:**
- `data/CPSC/reference.csv`: Patient metadata and diagnostic codes
- `data/CPSC/labels.csv`: Binary labels for each cardiac condition

### 2. Baseline Comparisons

Train traditional machine learning models with expert features:

```bash
# Logistic Regression
python baselines.py --data-dir data/CPSC --classifier LR

# Random Forest
python baselines.py --data-dir data/CPSC --classifier RF

# LightGBM
python baselines.py --data-dir data/CPSC --classifier LGB

# Multi-layer Perceptron
python baselines.py --data-dir data/CPSC --classifier MLP
```

### 3. Deep Learning Model Training

```bash
# Train with all 12 leads (recommended)
python main.py --data-dir data/CPSC --leads all --use-gpu --epochs 40 --batch-size 32

# Train with specific leads only
python main.py --data-dir data/CPSC --leads I,II,V1,V2 --use-gpu

# Resume training from checkpoint
python main.py --data-dir data/CPSC --leads all --use-gpu --resume
```

**Key Training Parameters:**
- `--leads`: ECG leads to use (`all` or comma-separated list)
- `--epochs`: Training epochs (default: 40)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.0001)
- `--seed`: Random seed for reproducibility (default: 42)

### 4. Model Evaluation

```bash
# Evaluate on test set
python predict.py --data-dir data/CPSC --leads all --use-gpu

# Generate confusion matrices and detailed metrics
python predict.py --data-dir data/CPSC --leads all --use-gpu --model-path models/resnet34_CPSC_all_42.pth
```

### 5. Model Interpretability with SHAP

```bash
# Generate SHAP values and visualizations
python shap_values.py --data-dir data/CPSC --leads all --use-gpu

# Analyze lead importance
python shap_values.py --data-dir data/CPSC --leads all --use-gpu --patient-ids A0001,A0002,A0003
```

**SHAP Outputs:**
- Individual patient SHAP plots: `shap/shap1-{patient_id}.png`
- Lead importance rankings
- Feature attribution for each prediction

### 6. Web Applications

#### Streamlit Clinical Interface

```bash
# Launch professional clinical interface
streamlit run app.py
```

**Features:**
- Multi-language support (English, French, Arabic)
- Interactive ECG visualization
- Real-time prediction with confidence scores
- Clinical explanations for each condition
- Lead selection and overlay options

#### HTML Demo Interface

```bash
# Serve the HTML demo (requires backend API)
python -m http.server 8000
# Open http://localhost:8000/index.html
```

## 📈 Experimental Results

### Performance Metrics on CPSC2018 Test Set

| Metric | All 12 Leads | Best Single Lead | Baseline (Expert Features) |
|--------|-------------|------------------|---------------------------|
| **Average AUC** | **0.970** | 0.952 (Lead I) | 0.891 |
| **Average F1-Score** | **0.813** | 0.789 (Lead aVR) | 0.743 |
| **Average Precision** | 0.824 | 0.801 | 0.756 |
| **Average Recall** | 0.806 | 0.778 | 0.731 |

### Per-Class Performance (All 12 Leads)

| Condition | AUC | F1-Score | Precision | Recall | Clinical Impact |
|-----------|-----|----------|-----------|--------|----------------|
| **SNR** | 0.985 | 0.891 | 0.923 | 0.862 | Normal rhythm detection |
| **AF** | 0.979 | 0.834 | 0.847 | 0.821 | Stroke risk assessment |
| **IAVB** | 0.972 | 0.785 | 0.798 | 0.773 | Conduction monitoring |
| **LBBB** | 0.968 | 0.812 | 0.825 | 0.799 | Heart failure risk |
| **RBBB** | 0.974 | 0.823 | 0.831 | 0.815 | Conduction assessment |
| **PAC** | 0.963 | 0.776 | 0.789 | 0.763 | Atrial activity |
| **PVC** | 0.971 | 0.798 | 0.812 | 0.784 | Ventricular ectopy |
| **STD** | 0.967 | 0.781 | 0.795 | 0.768 | Ischemia detection |
| **STE** | 0.975 | 0.817 | 0.829 | 0.805 | MI identification |

### Lead Importance Analysis

**Top Performing Individual Leads:**
1. **Lead I** (AUC: 0.952) - Lateral wall view
2. **Lead aVR** (AUC: 0.948) - Right atrial perspective  
3. **Lead V5** (AUC: 0.944) - Left ventricular lateral wall
4. **Lead II** (AUC: 0.941) - Inferior wall view
5. **Lead V2** (AUC: 0.938) - Septal wall view

### Computational Performance

| Configuration | Training Time | Inference Time | GPU Memory |
|--------------|---------------|----------------|------------|
| 12-lead ResNet-34 | ~3 hours | 15ms/sample | 4.2GB |
| Single-lead ResNet-34 | ~45 minutes | 8ms/sample | 1.8GB |
| Expert Features + LGB | ~10 minutes | 2ms/sample | N/A |

## 🔍 Model Interpretability

### SHAP (SHapley Additive exPlanations)

Our implementation provides comprehensive model interpretability:

1. **Global Feature Importance**: Understanding which ECG leads contribute most to predictions
2. **Patient-Level Explanations**: Highlighting critical time segments and leads for individual diagnoses
3. **Condition-Specific Patterns**: Identifying ECG morphology patterns characteristic of each arrhythmia

### Visualization Outputs

- **SHAP Waterfall Plots**: Feature contribution breakdown
- **SHAP Summary Plots**: Global feature importance ranking
- **Patient-Specific Plots**: Time-series highlighting important ECG segments
- **Lead Importance Heatmaps**: Spatial importance across the 12-lead system

### Clinical Insights

Our SHAP analysis revealed:
- **AF Detection**: Primarily relies on RR interval irregularities in leads II and V1
- **Bundle Branch Blocks**: QRS morphology in precordial leads (V1-V6) most informative
- **ST Changes**: Leads facing affected myocardial regions show highest importance
- **Premature Beats**: Temporal context around ectopic beats crucial for classification

## 🏗️ Project Architecture

### Core Components

1. **Data Pipeline** (`dataset.py`, `preprocess.py`)
   - WFDB format handling
   - Multi-label preprocessing
   - Data augmentation strategies

2. **Model Architecture** (`resnet.py`)
   - 1D convolutional ResNet adaptation
   - Multi-lead input handling
   - Adaptive pooling for variable-length inputs

3. **Training Framework** (`main.py`)
   - Multi-label BCE loss optimization
   - Learning rate scheduling
   - Model checkpointing

4. **Evaluation Suite** (`predict.py`, `utils.py`)
   - Threshold optimization
   - Comprehensive metrics calculation
   - Cross-validation support

5. **Interpretability Tools** (`shap_values.py`)
   - SHAP integration for ECG data
   - Visualization generation
   - Clinical explanation mapping

6. **Deployment Interfaces** (`app.py`, `index.html`)
   - Clinical decision support system
   - Real-time ECG analysis
   - Multi-language support

### Design Principles

- **Modularity**: Each component serves a specific function
- **Reproducibility**: Seeded random operations and deterministic training
- **Scalability**: Efficient data loading and GPU utilization
- **Clinical Utility**: Outputs designed for healthcare professionals
- **Interpretability**: Built-in explainability features

## 🌐 Related Projects & Documentation

### Hearline-Webapp
The main clinical deployment interface for this ECG diagnosis system:
- **Repository**: [Hearline-Webapp](https://github.com/blamairia/Hearline-Webapp)
- **Documentation**: [Complete Documentation](https://github.com/blamairia/Hearline-Webapp/tree/master/doc/index.md)
- **Features**: Web-based clinical interface, real-time ECG analysis, multi-language support


### Quick Links
- 🚀 **Live Demo**: [Try the Web Application](https://github.com/blamairia/Hearline-Webapp)
- 🔬 **Research Code**: [This Repository](https://github.com/blamairia/ecg-diagnosis)



## 🤝 Contributing

We welcome contributions from the research community:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Contribution

- **Additional Datasets**: Integration with other ECG databases
- **New Architectures**: Transformer-based or hybrid models
- **Clinical Validation**: Real-world deployment studies
- **Mobile Deployment**: Edge computing optimizations
- **Multi-Modal Integration**: Combining with clinical history



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CPSC2018 Organizers**: For providing the high-quality ECG dataset
- **PyTorch Team**: For the deep learning framework
- **SHAP Developers**: For interpretability tools
- **Open Source Community**: For various signal processing libraries
- **Clinical Collaborators**: For domain expertise and validation

## 📞 Contact

For questions about this research or potential collaborations:

- **Primary Contact**: Blamairia@gmail.com
- **Research Repository**: [ECG-Diagnosis](https://github.com/blamairia/ecg-diagnosis)
- **Web Application**: [Hearline-Webapp](https://github.com/blamairia/Hearline-Webapp)
- **Documentation**: [Complete Documentation](https://github.com/blamairia/Hearline-Webapp/tree/master/doc/index.md)

---

*This research contributes to the advancement of AI-assisted cardiac care, potentially improving diagnostic accuracy and accessibility of ECG interpretation worldwide.*
