# 🛰️ SolarSat AI — Satellite Energy Intelligence

### Satellite Solar Power Degradation Prediction and Energy Optimization (AIML Mini Project)

**SolarSat AI** is an aerospace-grade telemetry monitoring, machine learning prediction, and orbital energy allocation dashboard built using real downlinked telemetry from **NepaliSat-1** (a 1U CubeSat). The application analyzes multi-panel photovoltaic generation, evaluates session-by-session power reduction baselines, trains genuine scikit-learn regression models without target leakage to forecast solar power output, monitors electrical power subsystem (EPS) battery health, and simulates priority-based orbital load-shedding policies.

---

## 🚀 Key Features

1. **Mission Overview Console:** Real-time KPI telemetry monitoring across 3 downlinked recording passes (3,040 frames), with interactive Plotly time series of power, thermal gradients, and bus voltage.
2. **Solar Degradation Analysis:** Session-by-session 95th-percentile baseline evaluation, relative power drop distribution, panel-by-panel generation breakdown, and empirical thermal derating slope ($W/^\circ\text{C}$).
3. **AI Power Prediction Engine:** Leakage-free physical regression modeling comparing **Random Forest Regressor** ($R^2 \approx 0.99$ random, $R^2 \approx 0.71$ cross-session) against a **Linear Regression** baseline, complete with feature importance analysis and an interactive single-instance inference console.
4. **EPS Battery & Energy Subsystem:** Direct voltage ($V_{bat}$), current ($I_{batt}$), and temperature ($T_{batt}$) telemetry monitoring, user-configurable alarm thresholds, and instantaneous orbital power balance.
5. **Orbital Energy Optimization Simulator:** Transparent priority-driven power dispatch policy prioritizing critical flight hardware (OBC, ADCS, beacon) while throttling optional payloads and preserving configurable battery reserves.
6. **Telemetry Alerts & Anomaly Detection:** Dual-layer detection combining deterministic physical rule bounds with multidimensional **Isolation Forest** outlier modeling.
7. **Telemetry Data Explorer & Provenance:** Full dataset inspection, official data dictionary reference, statistical attribute profiling, and CSV export.

---

## 🛡️ Telemetry Provenance & Scientific Integrity

The application strictly adheres to aerospace data integrity standards:
- **Measured Telemetry:** Direct downlinked sensor readings (`Vpy`, `Vpx`, `Vmz`, `Vmx`, `Vpz`, `Tpy`, `Tpx`, `Tmz`, `Tmx`, `Tpz`, `Vbat`, `Ibatt`, `Tbatt`, `Time Stamp`).
- **Derived Telemetry:** Algebraically calculated physical metrics (`Ppy_Power_W`, `Ppx_Power_W`, `Pmz_Power_W`, `Pmx_Power_W`, `Ppz_Power_W`, `solar_array_power_W`, `mean_panel_temperature_C`, `session_reference_power_W`, `session_relative_power_drop_percent`).
- **Simulated Scenarios:** Algorithmic planning scenarios (`total_power_demand_W_simulated`, `critical_load_W_simulated`, `optional_load_W_simulated`, `available_energy_Wh_simulated`, `battery_capacity_Wh_assumed`).
- **Leakage Prevention:** ML models strictly exclude direct algebraic power components and target-derived session metrics during training.

---

## 📁 Project Structure

```text
SolarSat-AI/
├── app.py                      # Main landing page & mission control entry point
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation & execution guide
├── data/
│   ├── nepalese_satellite_extended.csv   # Primary dataset (3,040 records × 62 cols)
│   ├── nepalese_satellite_extended.xlsx  # Excel workbook package
│   └── data_dictionary.csv               # Official telemetry definitions & provenance
├── models/
│   ├── linear_regression.joblib          # Persisted baseline ML pipeline
│   ├── random_forest.joblib              # Persisted Random Forest ML pipeline
│   └── trained_features.joblib           # Serialized feature list
├── utils/
│   ├── __init__.py
│   ├── data_loader.py          # Cached data loading, filtering, and schema segregation
│   ├── ui_components.py        # Dark aerospace UI theme, CSS, KPI cards, and Plotly styling
│   ├── degradation_analysis.py # Session summaries, thermal derating, and panel breakdown
│   ├── prediction_model.py     # Leakage-free model training, metrics, and inference
│   ├── energy_optimizer.py     # Priority-based power dispatch simulator
│   └── anomaly_detection.py    # Physical threshold bounds and Isolation Forest detector
└── pages/
    ├── 1_Mission_Overview.py       # Page 1: Multi-metric telemetry overview
    ├── 2_Degradation_Analysis.py   # Page 2: Solar degradation & session reduction
    ├── 3_AI_Power_Prediction.py    # Page 3: ML model evaluation & inference
    ├── 4_Battery_Monitoring.py     # Page 4: EPS battery bus tracking & alerts
    ├── 5_Energy_Optimization.py    # Page 5: Interactive energy-allocation simulator
    ├── 6_Alerts.py                 # Page 6: Rule-based & Isolation Forest anomalies
    └── 7_Data_Explorer.py          # Page 7: Telemetry data viewer & CSV export
```

---

## 💻 Installation and Execution

### System Requirements
- **OS:** Windows 10/11, macOS, or Linux
- **Python:** Python 3.10 to 3.14

### 1. Setup Virtual Environment
Open your terminal (PowerShell on Windows) inside the `SolarSat-AI` project folder:

```powershell
python -m venv .venv
```

Activate the virtual environment:
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 3. Launch the Dashboard
```powershell
python -m streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 🧪 Machine Learning Methodology

### 1. Target Variable
`solar_array_power_W` (Sum of generated power across all five active panel faces).

### 2. Selected Feature Matrix (Leakage-Free)
- **Panel Voltages:** `Vpy (mV)`, `Vpx (mV)`, `Vmz (mV)`, `Vmx (mV)`, `Vpz (mV)`
- **Thermal Telemetry:** `Tpy (°C)`, `Tpx (°C)`, `Tmz (°C)`, `Tmx (°C)`, `Tpz (°C)`, `mean_panel_temperature_C`
- **Subsystem & Orbital Telemetry:** `Vbat (V)`, `Tbatt (℃)`, `Time Stamp`

### 3. Algorithms & Benchmarks
- **Linear Regression (Standard Scaled):**
  - Random Split (80/20): $R^2 \approx 0.8777$, $\text{MAE} \approx 0.2706\text{ W}$, $\text{RMSE} \approx 0.3764\text{ W}$
  - Chronological Session Split: $R^2 \approx 0.6674$, $\text{MAE} \approx 0.4597\text{ W}$, $\text{RMSE} \approx 0.5760\text{ W}$
- **Random Forest Regressor (100 Trees, Depth 12):**
  - Random Split (80/20): $R^2 \approx 0.9923$, $\text{MAE} \approx 0.0461\text{ W}$, $\text{RMSE} \approx 0.0946\text{ W}$
  - Chronological Session Split: $R^2 \approx 0.7088$, $\text{MAE} \approx 0.2421\text{ W}$, $\text{RMSE} \approx 0.5390\text{ W}$
- **Isolation Forest (Contamination 2%):**
  - Unsupervised multidimensional anomaly detection isolating coupled thermal-electrical transients.

---

## 🛠️ Troubleshooting Common Errors

1. **`UnicodeEncodeError: 'charmap' codec can't encode character '\u2103'`**  
   *Fix:* All data ingestion scripts explicitly specify `encoding='utf-8'` to cleanly handle the Unicode Celsius symbol `(℃)` present in `Tbatt (℃)`.
2. **Missing Module Errors:**  
   *Fix:* Ensure your virtual environment is active and run `pip install -r requirements.txt`.
3. **Execution Policy Error in PowerShell:**  
   *Fix:* If `.venv\Scripts\Activate.ps1` gives an execution policy error, run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   ```

---

## 🎓 AIML Mini-Project Presentation Guide

When presenting this project:
1. **Introduction (1 min):** Highlight that the project uses real downlinked telemetry from Nepal's first satellite (**NepaliSat-1**), not synthetic mock data.
2. **Data Integrity & Provenance (1 min):** Emphasize the strict segregation between measured sensor telemetry, derived physics, and simulation models.
3. **Degradation Analysis (2 mins):** Navigate to *Console 2*. Show how the 95th-percentile session reference is calculated and explain that observed reductions within a pass reflect orbital beta angle and thermal derating ($W/^\circ\text{C}$) rather than fabricated radiation degradation.
4. **Machine Learning Predictor (2 mins):** Navigate to *Console 3*. Point out how data leakage was systematically avoided (omitting component powers). Demonstrate the live slider inference console.
5. **EPS Battery & Optimization (2 mins):** Navigate to *Console 4 & 5*. Show the dual-axis voltage/current telemetry and interact with the energy optimization simulator to demonstrate priority load-shedding while protecting battery safety margins.
6. **Anomaly Detection (1 min):** Navigate to *Console 6*. Show how Isolation Forest flags anomalous telemetry transients on the timeline.
