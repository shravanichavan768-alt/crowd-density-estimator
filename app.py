import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.cm as cm
from density_map import generate_density_map, get_count_from_density
from zone_monitor import get_all_zone_alerts, get_overall_risk

st.set_page_config(
    page_title="CrowdSense AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e6f0;
}

section[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e2e;
}

section[data-testid="stSidebar"] * {
    color: #a0a0b8 !important;
}

.stat-card {
    background: #12121f;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

.zone-card {
    background: #12121f;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid #1e1e2e;
    transition: all 0.2s;
}

.zone-safe { border-left: 3px solid #22c55e; }
.zone-warning { border-left: 3px solid #eab308; }
.zone-danger { border-left: 3px solid #f97316; }
.zone-critical {
    border-left: 3px solid #ef4444;
    background: #1a0f0f;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.85; }
}

.zone-label {
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b6b8a;
    margin-bottom: 6px;
}

.zone-count {
    font-family: 'DM Mono', monospace;
    font-size: 28px;
    font-weight: 500;
    color: #e8e6f0;
    line-height: 1;
    margin-bottom: 4px;
}

.zone-density {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: #6b6b8a;
    margin-bottom: 8px;
}

.status-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 20px;
}

.badge-safe { background: #052e16; color: #22c55e; }
.badge-warning { background: #1c1917; color: #eab308; }
.badge-danger { background: #1c0a00; color: #f97316; }
.badge-critical { background: #1a0000; color: #ef4444; }

.alert-banner {
    padding: 16px 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    font-weight: 500;
    font-size: 15px;
}

.alert-critical {
    background: #1a0a0a;
    border: 1px solid #7f1d1d;
    color: #fca5a5;
}

.alert-danger {
    background: #1a1000;
    border: 1px solid #7c2d12;
    color: #fdba74;
}

.alert-warning {
    background: #1a1a00;
    border: 1px solid #713f12;
    color: #fde047;
}

.alert-safe {
    background: #0a1a0a;
    border: 1px solid #14532d;
    color: #86efac;
}

.header-title {
    font-size: 32px;
    font-weight: 600;
    color: #e8e6f0;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}

.header-sub {
    font-size: 14px;
    color: #6b6b8a;
    margin-bottom: 0;
}

.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
}

.metric-box {
    flex: 1;
    background: #12121f;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 16px;
}

.metric-label {
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b6b8a;
    margin-bottom: 6px;
}

.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 24px;
    font-weight: 500;
    color: #e8e6f0;
}

.divider {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 24px 0;
}

.section-label {
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b6b8a;
    margin-bottom: 16px;
}

.india-stat {
    background: #12121f;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 20px;
    border-top: 2px solid #ef4444;
}

.india-stat-num {
    font-family: 'DM Mono', monospace;
    font-size: 32px;
    font-weight: 500;
    color: #ef4444;
}

.india-stat-label {
    font-size: 13px;
    color: #6b6b8a;
    margin-top: 4px;
}

stFileUploader, [data-testid="stFileUploader"] {
    background: #12121f !important;
}
</style>
""", unsafe_allow_html=True)



with st.sidebar:
    st.markdown('<p style="font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:#6b6b8a;margin-bottom:8px;">Venue</p>', unsafe_allow_html=True)
    venue = st.selectbox("Venue", ["Kumbh Mela Ghat", "Mumbai CST Station", "Delhi Metro", "Custom"], label_visibility="collapsed")

    st.markdown('<hr style="border:none;border-top:1px solid #1e1e2e;margin:16px 0">', unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:#6b6b8a;margin-bottom:8px;">Upload Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    st.markdown('<hr style="border:none;border-top:1px solid #1e1e2e;margin:16px 0">', unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:#6b6b8a;margin-bottom:8px;">Parameters</p>', unsafe_allow_html=True)
    sigma = st.slider("Gaussian Sigma", 5, 30, 15)
    zone_area = st.slider("Zone Area m²", 5.0, 50.0, 10.0)

    st.markdown('<hr style="border:none;border-top:1px solid #1e1e2e;margin:16px 0">', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:#6b6b8a;margin-bottom:12px;">Thresholds</p>
    <div style="display:flex;flex-direction:column;gap:8px;font-size:13px;">
        <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#22c55e;flex-shrink:0"></div>
            <span style="color:#a0a0b8">Safe — under 2/m²</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#eab308;flex-shrink:0"></div>
            <span style="color:#a0a0b8">Warning — 2 to 4/m²</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#f97316;flex-shrink:0"></div>
            <span style="color:#a0a0b8">Danger — 4 to 7/m²</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#ef4444;flex-shrink:0"></div>
            <span style="color:#a0a0b8">Critical — above 7/m²</span>
        </div>
    </div>
    """, unsafe_allow_html=True)



st.markdown("""
<div style="margin-bottom:32px">
    <p class="header-title">CrowdSense AI</p>
    <p class="header-sub">Real-time crowd density estimation and stampede prevention</p>
</div>
""", unsafe_allow_html=True)



if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    gray = np.array(image.convert("L"))
    h, w = gray.shape

    # Real CSRNet model inference
    import torch
    import torchvision.transforms as transforms
    from model import CSRNet

    @st.cache_resource
    def load_model():
        device = torch.device('cpu')
        model = CSRNet(load_weights=False)
        model.load_state_dict(torch.load('models/csrnet_best.pth', map_location=device))
        model.eval()
        return model, device

    model_csrnet, device = load_model()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model_csrnet(img_tensor)

    density = output.squeeze().cpu().numpy()
    density = cv2.resize(density, (w, h))
    density = np.maximum(density, 0)
    
    # Heatmap overlay
    density_norm = density / (density.max() + 1e-8)
    heatmap = cm.jet(density_norm)[:, :, :3]
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    heatmap_bgr = cv2.cvtColor(heatmap_resized, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_bgr, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    total_count = int(get_count_from_density(density))
    total_area_m2 = 1000.0
    zone_area_m2 = total_area_m2 / 9

    alerts = get_all_zone_alerts(density, grid=(3, 3), zone_area=zone_area_m2)

    overall = get_overall_risk(alerts)

    # Alert banner
    alert_map = {
        "CRITICAL": ("alert-critical", "Critical density detected — open emergency gates immediately"),
        "DANGER":   ("alert-danger",   "High density detected — begin crowd redirection now"),
        "WARNING":  ("alert-warning",  "Crowd building up — monitor closely"),
        "SAFE":     ("alert-safe",     "All zones within safe limits"),
    }
    cls, msg = alert_map[overall]
    st.markdown(f'<div class="alert-banner {cls}">{venue} — {msg}</div>', unsafe_allow_html=True)

    # Top metrics
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="metric-label">Total Count</div>
            <div class="metric-value">{total_count:,}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Overall Risk</div>
            <div class="metric-value" style="color:{'#ef4444' if overall=='CRITICAL' else '#f97316' if overall=='DANGER' else '#eab308' if overall=='WARNING' else '#22c55e'}">{overall}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Zones Monitored</div>
            <div class="metric-value">9</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Venue</div>
            <div class="metric-value" style="font-size:16px;padding-top:6px">{venue}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Image columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-label">Live Feed</p>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    with col2:
        st.markdown('<p class="section-label">Density Heatmap</p>', unsafe_allow_html=True)
        st.image(overlay_rgb, use_container_width=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

    
    st.markdown('<p class="section-label">Zone Breakdown</p>', unsafe_allow_html=True)

    color_map = {
        "SAFE": ("#22c55e", "badge-safe", "zone-safe"),
        "WARNING": ("#eab308", "badge-warning", "zone-warning"),
        "DANGER": ("#f97316", "badge-danger", "zone-danger"),
        "CRITICAL": ("#ef4444", "badge-critical", "zone-critical"),
    }

    for row in range(3):
        cols = st.columns(3)
        for col_idx in range(3):
            zone = alerts[row * 3 + col_idx]
            status = zone["status"]
            color, badge_cls, card_cls = color_map[status]
            with cols[col_idx]:
                st.markdown(f"""
                <div class="zone-card {card_cls}">
                    <div class="zone-label">{zone['zone']}</div>
                    <div class="zone-count">{zone['count']:,}</div>
                    <div class="zone-density">{zone['density']} per m²</div>
                    <span class="status-badge {badge_cls}">{status}</span>
                </div>
                """, unsafe_allow_html=True)

else:
    
    st.markdown('<p class="section-label">Why This Matters</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    stats = [
        ("121", "Deaths at Kumbh Mela 2024"),
        ("23", "Deaths at Mumbai Elphinstone 2017"),
        ("1,500+", "Stampede deaths in India since 2000"),
    ]
    for col, (num, label) in zip([c1, c2, c3], stats):
        with col:
            st.markdown(f"""
            <div class="india-stat">
                <div class="india-stat-num">{num}</div>
                <div class="india-stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Upload an image from the sidebar to begin</p>', unsafe_allow_html=True)