import subprocess
import sys

# Auto-install dependencies on Streamlit Cloud if requirements.txt fails
try:
    import matplotlib
    import scipy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "scipy"])

import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Set page config for a premium look
st.set_page_config(
    page_title="BBO SPDC Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 BBO SPDC Ring Simulator")
st.markdown("Interactive tool to simulate Type-I and Type-II collinear/non-collinear degenerate and non-degenerate SPDC in BBO.")

# BBO Refractive Indices (Refined Model)
def noo(n):
    n2 = 2.7405 + ((0.0184) / ((n ** 2) - 0.0179)) - (0.0155 * n ** 2)
    rio = round(math.sqrt(n2), 4)
    return rio

def neo(n):
    n2 = 2.3730 + ((0.0128) / ((n ** 2) - 0.0156)) - (0.0044 * n ** 2)
    rio = round(math.sqrt(n2), 4)
    return rio

def n_bbo(wavelength_um, polarization):
    if polarization == 'o':
        return noo(wavelength_um)
    else:
        return neo(wavelength_um)

def n_effective_bbo(wavelength_um, theta_rad):
    no = n_bbo(wavelength_um, 'o')
    ne = n_bbo(wavelength_um, 'e')
    return 1.0 / np.sqrt((np.cos(theta_rad)**2 / no**2) + (np.sin(theta_rad)**2 / ne**2))

def spdc_parameters(lambda_um, theta_rad):
    no = n_bbo(lambda_um, 'o')
    ne = n_bbo(lambda_um, 'e')
    den = no**2 * np.sin(theta_rad)**2 + ne**2 * np.cos(theta_rad)**2
    alpha = (no**2 - ne**2) * np.sin(theta_rad) * np.cos(theta_rad) / den
    beta = no * ne / den
    gamma = no / np.sqrt(den)
    eta = ne * gamma
    return alpha, beta, gamma, eta

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Parameters Configuration")

spdc_type = st.sidebar.selectbox("SPDC Type", ["Type-I (e -> o + o)", "Type-II (e -> e + o)"])
color_theme = st.sidebar.radio("Color Theme", ["Classic (Viridis)", "Color-coded (Blues/Reds)"], index=0)

# Wavelengths
wavep = st.sidebar.number_input("Pump Wavelength (nm)", min_value=200.0, max_value=2000.0, value=405.0, step=1.0)
wavep_um = wavep * 0.001

is_degenerate = st.sidebar.checkbox("Enforce Degeneracy (λs = 2λp)", value=False)

if is_degenerate:
    waves = 2.0 * wavep
    st.sidebar.text(f"Signal Wavelength: {waves:.1f} nm")
else:
    waves = st.sidebar.number_input("Signal Wavelength (nm)", min_value=200.0, max_value=2000.0, value=810.0, step=1.0)

waves_um = waves * 0.001
if waves_um <= wavep_um:
    st.sidebar.error("Signal wavelength must be strictly greater than pump wavelength!")
    st.stop()

wavei_um = (wavep_um * waves_um) / (waves_um - wavep_um)

# Crystal and Setup
L_mm = st.sidebar.number_input("Crystal Thickness (mm)", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
L = L_mm * 1000.0  # um
wp = st.sidebar.number_input("Pump Waist (um)", min_value=5.0, max_value=1000.0, value=100.0, step=10.0)

# Sliders
st.sidebar.header("🎚️ Live Tuning Sliders")
distz_mm = st.sidebar.slider("Propagation Distance z (mm)", min_value=1.0, max_value=300.0, value=100.0, step=1.0)
z_um = distz_mm * 1000.0

angle_offset_deg = st.sidebar.slider("Tilt Angle Offset (degrees)", min_value=0.0, max_value=0.3, value=0.0, step=0.002, format="%.3f")

# --- PHASE MATCHING CALCULATIONS ---
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Calculated Angles")

try:
    if spdc_type == "Type-I (e -> o + o)":
        no_p, ne_p = n_bbo(wavep_um, 'o'), n_bbo(wavep_um, 'e')
        no_s, no_i = n_bbo(waves_um, 'o'), n_bbo(wavei_um, 'o')
        target_nep = wavep_um * (no_s / waves_um + no_i / wavei_um)
        
        sin2_theta = (ne_p**2 * (no_p**2 - target_nep**2)) / (target_nep**2 * (no_p**2 - ne_p**2))
        collinear_thetap = math.degrees(math.asin(math.sqrt(sin2_theta)))
    else:
        # Type-II collinear condition
        def type2_mismatch(theta_rad):
            nep = n_effective_bbo(wavep_um, theta_rad)
            nes = n_effective_bbo(waves_um, theta_rad)
            noi = n_bbo(wavei_um, 'o')
            kp = 2 * np.pi * nep / wavep_um
            ks = 2 * np.pi * nes / waves_um
            ki = 2 * np.pi * noi / wavei_um
            return abs(kp - ks - ki)

        res = minimize_scalar(type2_mismatch, bounds=(0, np.pi/2), method='bounded')
        collinear_thetap = math.degrees(res.x)

    st.sidebar.metric("Collinear PM Angle", f"{collinear_thetap:.4f}°")
    thetap_deg = collinear_thetap + angle_offset_deg
    st.sidebar.metric("Actual Crystal Angle (θp)", f"{thetap_deg:.4f}°")
    thetap = np.radians(thetap_deg)
except Exception as e:
    st.sidebar.error("Out of phase-matching range!")
    st.stop()

# --- INSTANT CALCULATION GRID (Plane-wave pump limit) ---
rho = 0.3
grdpnt = 250
q_range = np.linspace(-rho, rho, grdpnt)
qs_x, qs_y = np.meshgrid(q_range, q_range)

kp = (2 * np.pi) / wavep_um
nobar_s = noo(waves_um)
nobar_i = noo(wavei_um)
ks_val = 2 * np.pi * nobar_s / waves_um
ki_val = 2 * np.pi * nobar_i / wavei_um

no = noo(wavep_um)
ne = neo(wavep_um)
den = np.square(no * np.sin(thetap)) + np.square(ne * np.cos(thetap))
etap = ne * (no / np.sqrt(den))

if spdc_type == "Type-I (e -> o + o)":
    # Type-I phase mismatch: delta_kz = ksz + kiz - kpz
    ksz = ks_val - (1 / (2 * ks_val)) * (qs_x**2 + qs_y**2)
    kiz = ki_val - (1 / (2 * ki_val)) * (qs_x**2 + qs_y**2)
    kpz = kp * etap
    delta_kz = ksz + kiz - kpz
    funcmat = np.square(np.sinc(delta_kz * L / (2 * np.pi)))
else:
    # Type-II: sum of eo and oe rings
    alphap, betap, gammap, etap = spdc_parameters(wavep_um, thetap)
    alphas, betas, gammas, etas = spdc_parameters(waves_um, thetap)
    alphai, betai, gammai, etai = spdc_parameters(wavei_um, thetap)
    
    # eo term: signal=e, idler=o
    ksze = (ks_val * etas) + (alphas * qs_x) - (1 / (2 * ks_val * etas)) * (betas**2 * qs_x**2 + gammas**2 * qs_y**2)
    kiz_o = (ki_val * nobar_i) - (1 / (2 * ki_val * nobar_i)) * (qs_x**2 + qs_y**2)
    tmpeo = ksze + kiz_o - kp * etap
    
    # oe term: signal=o, idler=e (note idler is at -q_s)
    ksz_o = (ks_val * nobar_s) - (1 / (2 * ks_val * nobar_s)) * (qs_x**2 + qs_y**2)
    kize = (ki_val * etai) + (alphai * (-qs_x)) - (1 / (2 * ki_val * etai)) * (betai**2 * (-qs_x)**2 + gammai**2 * (-qs_y)**2)
    tmpoe = ksz_o + kize - kp * etap
    
    funcmat = np.square(np.sinc(tmpeo * L / (2 * np.pi))) + np.square(np.sinc(tmpoe * L / (2 * np.pi)))

funcmat /= np.max(funcmat)

# Find peak q_ring
profile = funcmat[grdpnt//2, :]
peak_idx = np.argmax(profile[grdpnt//2:]) + grdpnt//2
q_ring = q_range[peak_idx]

# Calculate exit angles in air
sin_theta_s = max(-1.0, min(1.0, q_ring * waves_um / (2 * np.pi)))
sin_theta_i = max(-1.0, min(1.0, q_ring * wavei_um / (2 * np.pi)))
exit_angle_s_deg = math.degrees(math.asin(sin_theta_s))
exit_angle_i_deg = math.degrees(math.asin(sin_theta_i))

# Create tabs for clean presentation
tab1, tab2, tab3 = st.tabs(["🖼️ Interactive Visualization", "📐 Detailed Angle Dashboard", "🧬 High-Resolution wave optics"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Plot concentric rings in position space
        k0_s = 2 * np.pi / waves_um
        pos_limit_s = rho * (z_um / k0_s)
        extent_s = (-pos_limit_s, pos_limit_s, -pos_limit_s, pos_limit_s)
        
        k0_i = 2 * np.pi / wavei_um
        pos_limit_i = rho * (z_um / k0_i)
        extent_i = (-pos_limit_i, pos_limit_i, -pos_limit_i, pos_limit_i)
        
        fig, ax = plt.subplots(figsize=(7, 7))
        if color_theme == "Classic (Viridis)":
            im = ax.imshow(funcmat, extent=extent_s, cmap='viridis', origin='lower')
        else:
            im1 = ax.imshow(funcmat, extent=extent_s, cmap='Blues', origin='lower', alpha=0.6)
            im2 = ax.imshow(funcmat, extent=extent_i, cmap='Reds', origin='lower', alpha=0.4)
        
        ax.set_xlabel('x-axis (um)', color='white')
        ax.set_ylabel('y-axis (um)', color='white')
        
        # Keep plot box fixed from -3000 to +3000 um
        ax.set_xlim(-3000, 3000)
        ax.set_ylim(-3000, 3000)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        
        # Add legend markers
        import matplotlib.patches as mpatches
        if color_theme == "Classic (Viridis)":
            green_patch = mpatches.Patch(color='#2ca02c', label='SPDC singles')
            ax.legend(handles=[green_patch], facecolor='#1e1e1e', labelcolor='white')
        else:
            blue_patch = mpatches.Patch(color='cyan', label=f'Signal ({waves:.0f} nm)')
            red_patch = mpatches.Patch(color='red', label=f'Idler ({wavei_um*1000.0:.0f} nm)')
            ax.legend(handles=[blue_patch, red_patch], facecolor='#1e1e1e', labelcolor='white')
        
        st.pyplot(fig)
        
    with col2:
        st.subheader("💡 Visual Guide")
        st.markdown(f"""
        * **Inner Ring (Blue)**: Signal photons at **{waves:.0f} nm**.
        * **Outer Ring (Red)**: Idler photons at **{wavei_um*1000.0:.0f} nm**.
        * **At 0.000° Tilt Offset**: Both rings collapse to a single spot at the center (collinear).
        * **Drag Tilt Offset Slider**: Watch the rings open up and separate.
        * **Drag Distance Slider**: Watch the rings expand outward.
        """)
        
        st.metric("Signal Ring Radius", f"{q_ring * waves_um * z_um / (2 * np.pi):.1f} μm")
        st.metric("Idler Ring Radius", f"{q_ring * wavei_um * z_um / (2 * np.pi):.1f} μm")

with tab2:
    st.subheader("📊 Geometric and Wavevector Analysis")
    
    st.markdown("### Exit Angles in Air")
    st.info("These are the physical angles outside BBO crystal relative to the pump beam axis:")
    
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Peak wavevector q_ring", f"{q_ring:.5f} μm⁻¹")
    mc2.metric("Signal Exit Angle in Air", f"{exit_angle_s_deg:.4f}°")
    mc3.metric("Idler Exit Angle in Air", f"{exit_angle_i_deg:.4f}°")
    
    st.markdown("### BBO Crystal Parameters")
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("no (pump)", f"{no:.4f}")
    pc2.metric("ne (pump)", f"{ne:.4f}")
    pc3.metric("n_effective (pump)", f"{etap:.4f}")

with tab3:
    st.subheader("🔬 Run High-Resolution Wave Optics Simulation")
    st.write("This tab calculates the full 2D singles spatial distribution (including pump focus and idler integration) which takes a few seconds.")
    
    if st.button("Run Simulation"):
        with st.spinner("Calculating Singles profile (double loop)..."):
            # Use a dynamically scaled integration step size based on the pump beam waist
            # to prevent numerical under-sampling and Moire/aliasing stripes.
            dq = 2.0 / wp
            step = dq / 5.0
            xs_h, ys_h = np.meshgrid(np.linspace(-rho, rho, 100), np.linspace(-rho, rho, 100))
            funcmat_h = np.zeros([100, 100])
            
            # Use Type-I or Type-II wavefunction depending on type
            if spdc_type == "Type-I (e -> o + o)":
                def two_photon_wavefunction(xs, ys, xi, yi):
                    ss = (xs**2+ys**2) + (xi**2+yi**2) + 2*np.sqrt((xs**2+ys**2)*(xi**2+yi**2))*np.cos(np.arctan2(ys,xs)-np.arctan2(yi,xi))
                    ks_val = (2 * np.pi) / waves_um
                    ki_val = (2 * np.pi) / wavei_um
                    ksz = (ks_val * nobar_s) - (1 / (2 * ks_val * nobar_s)) * (xs**2 + ys**2)
                    kiz = (ki_val * nobar_i) - (1 / (2 * ki_val * nobar_i)) * (xi**2 + yi**2)
                    kpz = kp * etap
                    tmp = ((ksz + kiz - kpz) * (L / 2))
                    pumpfield = np.exp(-wp**2 * ss / 4.0)
                    return pumpfield * np.sinc(tmp / np.pi)
                
                for k in np.arange(-5, 5):
                    for m in np.arange(-5, 5):
                        wf = two_photon_wavefunction(xs_h, ys_h, -xs_h + k*step, -ys_h + m*step)
                        funcmat_h += np.square(np.abs(wf)) * step * step
            else:
                # Type-II
                def two_photon_wavefunction_typeII(xs, ys, xi, yi):
                    ss = (xs**2+ys**2) + (xi**2+yi**2) + 2*np.sqrt((xs**2+ys**2)*(xi**2+yi**2))*np.cos(np.arctan2(ys,xs)-np.arctan2(yi,xi))
                    kpz = (kp * etap) + (alphap * (xs + xi)) - ((1 / (2 * kp * etap)) * (betap**2 * (xs + xi)**2 + gammap**2 * (ys + yi)**2))
                    ksz = (ks_val * nobar_s) - (1 / (2 * ks_val * nobar_s)) * (xs**2 + ys**2)
                    kiz = (ki_val * nobar_i) - (1 / (2 * ki_val * nobar_i)) * (xi**2 + yi**2)
                    ksze = (ks_val * etas) + (alphas * xs) - (1 / (2 * ks_val * etas)) * (betas**2 * xs**2 + gammas**2 * ys**2)
                    kize = (ki_val * etai) + (alphai * xi) - (1 / (2 * ki_val * etai)) * (betai**2 * xi**2 + gammai**2 * yi**2)
                    tmpeo = ((ksze + kiz - kpz) * (L / 2))
                    tmpoe = ((ksz + kize - kpz) * (L / 2))
                    pumpfield = np.exp(-(wp**2 + ((2j) * (z_um / (kp * etap)))) * (ss / 4.0))
                    
                    wf_eo = pumpfield * np.sinc(tmpeo / np.pi) * np.exp(-1.0j * tmpeo)
                    wf_oe = pumpfield * np.sinc(tmpoe / np.pi) * np.exp(-1.0j * tmpoe)
                    return wf_eo, wf_oe
                
                for k in np.arange(-5, 5):
                    for m in np.arange(-5, 5):
                        wf_eo, wf_oe = two_photon_wavefunction_typeII(xs_h, ys_h, -xs_h + k*step, -ys_h + m*step)
                        funcmat_h += (np.square(np.abs(wf_eo)) + np.square(np.abs(wf_oe))) * step * step
                
            funcmat_h /= np.max(funcmat_h)
            
            # Coordinate scaling for High-Res Signal and Idler in air
            k0_s = 2 * np.pi / waves_um
            pos_limit_s = rho * (z_um / k0_s)
            extent_s = (-pos_limit_s, pos_limit_s, -pos_limit_s, pos_limit_s)
            
            k0_i = 2 * np.pi / wavei_um
            pos_limit_i = rho * (z_um / k0_i)
            extent_i = (-pos_limit_i, pos_limit_i, -pos_limit_i, pos_limit_i)
            
            # Plot
            fig_h, ax_h = plt.subplots(figsize=(6, 6))
            if color_theme == "Classic (Viridis)":
                ax_h.imshow(funcmat_h, extent=extent_s, cmap='viridis', origin='lower')
            else:
                ax_h.imshow(funcmat_h, extent=extent_s, cmap='Blues', origin='lower', alpha=0.6)
                ax_h.imshow(funcmat_h, extent=extent_i, cmap='Reds', origin='lower', alpha=0.4)
            
            ax_h.set_title("Wave Optics Joint Distribution on Camera")
            ax_h.set_facecolor('#1e1e1e')
            fig_h.patch.set_facecolor('#1e1e1e')
            ax_h.tick_params(colors='white')
            ax_h.title.set_color('white')
            ax_h.xaxis.label.set_color('white')
            ax_h.yaxis.label.set_color('white')
            ax_h.set_xlabel("x-axis (um)")
            ax_h.set_ylabel("y-axis (um)")
            ax_h.set_xlim(-3000, 3000)
            ax_h.set_ylim(-3000, 3000)
            ax_h.grid(True, alpha=0.2)
            
            # Add legend markers
            import matplotlib.patches as mpatches
            if color_theme == "Classic (Viridis)":
                green_patch = mpatches.Patch(color='#2ca02c', label='SPDC singles')
                ax_h.legend(handles=[green_patch], facecolor='#1e1e1e', labelcolor='white')
            else:
                blue_patch = mpatches.Patch(color='cyan', label=f'Signal ({waves:.0f} nm)')
                red_patch = mpatches.Patch(color='red', label=f'Idler ({wavei_um*1000.0:.0f} nm)')
                ax_h.legend(handles=[blue_patch, red_patch], facecolor='#1e1e1e', labelcolor='white')
            
            st.pyplot(fig_h)
st.markdown("---")
st.caption("Developed for BBO SPDC Simulation & Phase-Matching Tuning.")
