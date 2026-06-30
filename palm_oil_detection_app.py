import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from translations import LANG_DICT
from utils import get_base64_bin

# --- 1. CONFIG ---
st.set_page_config(page_title="PalmVision AI", layout="wide")

# --- 2. CSS STYLES (Fixed Background Visibility & Image Sizes) ---
def apply_styles():
    try:
        bin_str = get_base64_bin('assets/background_image.jpg')
        bg_img = f'data:image/jpg;base64,{bin_str}'
    except:
        bg_img = ""

    st.markdown(f'''
        <style>
        .stApp {{
            /* High visibility background */
            background: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)), url("{bg_img}");
            background-size: cover;
            background-attachment: fixed;
        }}
        
        /* Removed column width constraints to make images cover the left half */
        .stImage img {{
            max-height: 480px !important;
            width: 100% !important;
            object-fit: contain !important;
            border-radius: 15px;
            border: 2px solid rgba(255,255,255,0.2);
            box-shadow: 0 15px 30px rgba(0,0,0,0.5);
        }}

        .diag-card {{
            background: rgba(0, 0, 0, 0.75);
            backdrop-filter: blur(15px);
            border-radius: 25px;
            padding: 40px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: white;
            min-height: 500px;
        }}

        .diagnosis-header {{
            color: #2ecc71;
            font-size: 3.2rem;
            font-weight: 800;
            margin: 0;
            line-height: 1.1;
        }}
        
        #MainMenu, footer, header {{visibility: hidden;}}
        </style>
    ''', unsafe_allow_html=True)

apply_styles()

# --- 3. MODEL ---
@st.cache_resource
def load_yolo():
    return YOLO("models/best.pt")

model = load_yolo()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:white;'>AI Controls</h2>", unsafe_allow_html=True)
    selected_lang = st.selectbox("Language", options=list(LANG_DICT.keys()))
    t = LANG_DICT[selected_lang]
    st.divider()
    st.success("Diagnostic Engine Online")

# --- 5. MAIN DASHBOARD ---
st.markdown(f'<h1 style="color:white; text-align:center; text-shadow: 2px 2px 10px black; font-size:3.5rem;">{t["title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="color:#2ecc71; text-align:center; font-weight:bold; text-shadow: 1px 1px 5px black; font-size:1.2rem;">{t["subtitle"]}</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    results = model.predict(img_bgr, conf=0.35)
    res_img = results[0].plot()
    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

    # LAYOUT: 50/50 Split
    st.markdown("<br>", unsafe_allow_html=True)
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        # Side-by-side images spanning the full left half
        st.markdown(f"<p style='text-align:center; color:white; font-weight:bold; font-size:1.1rem;'>{t['orig_header']} &nbsp; | &nbsp; {t['det_header']}</p>", unsafe_allow_html=True)
        sub_l, sub_r = st.columns(2)
        with sub_l:
            st.image(img_rgb, use_column_width=True)
        with sub_r:
            st.image(res_img_rgb, use_column_width=True)

    with right_col:
        if len(results[0].boxes) > 0:
            raw_label = results[0].names[int(results[0].boxes.cls[0])].lower().strip()
            conf = float(results[0].boxes.conf[0]) * 100
        else:
            raw_label = "healthy"
            conf = 100.0

        display_name = t['conditions'].get(raw_label, t['conditions']['healthy'])
        det_details = t['details'].get(raw_label, t['details']['healthy'])

        # FIX: Ensure HTML is NOT indented to prevent Streamlit from showing it as code
        html_card = f"""
<div class="diag-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="opacity:0.6; font-size:0.9rem; letter-spacing:1px; font-weight:bold;">{t['summary_label']}</span>
        <span style="color:#2ecc71; font-size:1rem; font-weight:bold;">{t['accuracy_label']}: {conf:.1f}%</span>
    </div>
    <h1 class="diagnosis-header">{display_name}</h1>
    <hr style="opacity:0.2; margin: 30px 0;">
    <div style="margin-bottom:30px;">
        <b style="color:#2ecc71; font-size:1.2rem;">🔍 {t['symptoms_label']}:</b><br>
        <span style="font-size:1.1rem; opacity:0.9; line-height:1.6;">{det_details['symptoms']}</span>
    </div>
    <div style="margin-bottom:30px;">
        <b style="color:#2ecc71; font-size:1.2rem;">🌱 {t['fertilizer_label']}:</b><br>
        <span style="font-size:1.1rem; opacity:0.9; line-height:1.6;">{det_details['fertilizer']}</span>
    </div>
    <div>
        <b style="color:#2ecc71; font-size:1.2rem;">💡 {t['tips_label']}:</b><br>
        <span style="font-size:1.1rem; opacity:0.9; line-height:1.6;">{det_details['tips']}</span>
    </div>
</div>
"""
        st.markdown(html_card, unsafe_allow_html=True)

else:
    st.markdown(f'''
        <div style="text-align:center; padding:120px; color:white; background:rgba(0,0,0,0.4); border-radius:30px; border:2px dashed rgba(255,255,255,0.2); margin-top:50px;">
            <h2 style="font-size:2.5rem;">{t['waiting']}</h2>
            <p style="font-size:1.2rem; opacity:0.7;">{t['awaiting']}</p>
        </div>
    ''', unsafe_allow_html=True)