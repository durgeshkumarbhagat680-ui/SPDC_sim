# BBO SPDC Ring Simulator

An interactive Streamlit web application to simulate and tune Type-I and Type-II collinear/non-collinear degenerate and non-degenerate Spontaneous Parametric Down-Conversion (SPDC) in BBO crystals.

---

## 🌟 Features

* **Interactive Ring Visualization**: Overlay signal and idler rings in position space. Tune propagation distance ($z$) and crystal tilt offset to see rings open, close, and separate.
* **Classic (Viridis) & Color-coded Themes**: Toggle between single-intensity Viridis plots (useful for capturing Type-II double-ring intersections) and color-coded Blue/Red overlays (to distinguish signal and idler wavelengths).
* **Physical Exit Angle Calculations**: Computes the exact emission angles of signal and idler photons in air.
* **Wave-Optics Singles Simulation**: Integrates the two-photon amplitude to compute the detailed singles spatial distribution on a camera.

---

## 🚀 How to Run Locally

### 1. Install Dependencies
Make sure you have Python installed, then install the required packages:
```bash
pip install streamlit numpy matplotlib scipy
```

### 2. Run the App
Navigate to the directory and launch the Streamlit app:
```bash
streamlit run spdc_app.py
```

The app will automatically open in your browser at `http://localhost:8501`.

---

## 🛠️ Deployment (Run in Cloud)
You can deploy this application for free to **Streamlit Community Cloud** so that anyone can access it directly via a link:
1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **New app**.
3. Select your repository `durgeshkumarbhagat680-ui/SPDC_sim` and branch `main`.
4. Set the main file path to `spdc_app.py`.
5. Click **Deploy!**
