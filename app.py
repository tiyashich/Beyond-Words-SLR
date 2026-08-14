import os
import sys

# 0. FORCE CPU MODE (Prevents CUDA segmentation faults on Streamlit Cloud containers)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import logging
import asyncio
import requests
import av
import cv2
import numpy as np
import streamlit as st

# 1. SUPPRESS UNHANDLED AIOICE / ASYNCIO BACKGROUND LOG NOISE
logging.getLogger("aioice").setLevel(logging.CRITICAL)
logging.getLogger("aiortc").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

def silence_aioice_exceptions(loop, context):
    exception = context.get("exception")
    message = context.get("message", "")
    if isinstance(exception, AttributeError) and "sendto" in str(exception):
        return
    if "Transaction.__retry" in message or "aioice" in str(context):
        return
    loop.default_exception_handler(context)

try:
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(silence_aioice_exceptions)
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import aiortc
import aioice
import streamlit_webrtc
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from models.gradcam import generate_gradcam
from models.loader import load_recognition_pipeline, load_yolo_detector
from models.predictor import predict_crop
from ui.session import initialize_session
from ui.styles import custom_css
from utils.constants import (
    BENGALI_CHARS,
    CENTROIDS_PATH,
    CLASS_NAMES_PATH,
    CLASS_THRESHOLDS_PATH,
    FALLBACK_CONFIDENCE_THRESHOLD,
    IMG_RESOLUTION,
    LUMINANCE_THRESHOLD,
    MODEL_PATH,
)
from utils.image_utils import apply_clahe_tf, preprocess_cropped_image
from utils.logo import get_base64_image

# 2. PAGE SETUP & STRUCTURAL CSS INJECTION
st.set_page_config(layout="wide", page_title="Beyond Words | BDSL49")
st.html(custom_css)

# Signature Branding Setup
logo_base64 = get_base64_image("assets/Signature_tc.png")
if logo_base64:
    st.html(
        f"""
        <div style="position: fixed; bottom: 24px; right: 24px; width: 110px; z-index: 999999; pointer-events: none;">
            <img src="data:image/png;base64,{logo_base64}" style="width: 100%; height: auto; opacity: 0.75;">
        </div>
        """
    )

# Load AI Engine models safely
@st.cache_resource
def init_models():
    detector = load_yolo_detector()
    feature_extractor, backbone_grad_model, centroids, class_names, class_thresholds = load_recognition_pipeline()
    return detector, feature_extractor, backbone_grad_model, centroids, class_names, class_thresholds

try:
    detector, feature_extractor, backbone_grad_model, centroids, class_names, class_thresholds = init_models()
    system_online = True
except Exception as e:
    st.error(f"System Offline - Pipeline Initialization Failure: {e}")
    system_online = False

initialize_session()

# Helper to fetch dynamic WebRTC ICE configuration (STUN + Open Relay TURN + HF / Twilio Fallbacks)
def get_ice_servers():
    """
    Fetches WebRTC ICE servers with robust firewall bypass fallbacks:
    1. Google Public STUN
    2. Open Relay TURN (Metered CA) on Ports 80 & 443 (TCP/UDP)
    3. Hugging Face / Cloudflare TURN Relay
    4. Twilio TURN Relay
    """
    ice_servers = [
        # Google Public STUN
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        
        # Free Open Relay TURN (Metered) - Bypasses firewalls over HTTP/HTTPS/TCP
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:443",
                "turn:openrelay.metered.ca:443?transport=tcp"
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]

    # Attempt Hugging Face / Cloudflare TURN Relay if secret exists
    hf_token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token:
        try:
            response = requests.get(
                "https://huggingface.co/api/turn",
                headers={"Authorization": f"Bearer {hf_token}"},
                timeout=3.0,
            )
            if response.status_code == 200:
                turn_data = response.json()
                if "iceServers" in turn_data:
                    ice_servers.extend(turn_data["iceServers"])
                elif isinstance(turn_data, list):
                    ice_servers.extend(turn_data)
        except Exception:
            pass

    # Attempt Twilio TURN Relay if secrets exist
    twilio_sid = st.secrets.get("TWILIO_ACCOUNT_SID") or os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth = st.secrets.get("TWILIO_AUTH_TOKEN") or os.getenv("TWILIO_AUTH_TOKEN")
    if twilio_sid and twilio_auth:
        try:
            from twilio.rest import Client
            client = Client(twilio_sid, twilio_auth)
            token = client.tokens.create()
            ice_servers.extend(token.ice_servers)
        except Exception:
            pass

    return ice_servers

# 3. BRAND HEADERS
st.title("Beyond Words: A Sign Language Recognition System")

with st.expander("🛠️ Environment Diagnostics (Debug Info)"):
    st.write(f"**Streamlit Version:** `{st.__version__}`")
    st.write(f"**Streamlit WebRTC:** `{streamlit_webrtc.__version__}`")
    st.write(f"**aiortc Version:** `{aiortc.__version__}`")
    st.write(f"**aioice Version:** `{aioice.__version__}`")

st.html('<span class="subtitle-text" style="font-style: italic !important;">Decoding Signs, Empowering Lives</span>')

if "gesture_history" not in st.session_state:
    st.session_state.gesture_history = []

def apply_digital_zoom(frame_bgr, zoom):
    """Crops and rescales frame centered on video coordinates based on zoom factor."""
    if zoom <= 1.0:
        return frame_bgr
    h, w, _ = frame_bgr.shape
    new_h, new_w = int(h / zoom), int(w / zoom)
    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2
    cropped_center = frame_bgr[y1 : y1 + new_h, x1 : x1 + new_w]
    return cv2.resize(cropped_center, (w, h), interpolation=cv2.INTER_LINEAR)

def process_frame_and_predict(frame_bgr):
    full_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    st.session_state.saved_full_view = full_frame_rgb
    
    mean_luminance = float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean())
    if mean_luminance < LUMINANCE_THRESHOLD:
        st.session_state.saved_prediction = {
            "low_light": True,
            "luminance": mean_luminance,
        }
    else:
        yolo_results = detector(frame_bgr, conf=0.40, verbose=False)[0]
        if len(yolo_results.boxes) == 0:
            st.session_state.saved_prediction = {
                "no_hand": True,
            }
        else:
            best_box = yolo_results.boxes[0].xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = best_box
            h_orig, w_orig = frame_bgr.shape[:2]
            crop_bgr = frame_bgr[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)]
            
            bengali_char, best_class_name, score, threshold, top3, explanation_map = predict_crop(
                crop_bgr, feature_extractor, backbone_grad_model, centroids, class_names, class_thresholds
            )
            top3_formatted = [(char, f"{sim * 100:.1f}%") for char, sim in top3]
            st.session_state.saved_prediction = {
                "char": bengali_char, 
                "class_id": best_class_name,
                "score": score, 
                "threshold": threshold, 
                "top3": top3_formatted
            }
            st.session_state.saved_hand_crop = explanation_map

# 4. CONTROL MATRIX PANEL
with st.container(border=True):
    st.html('<div class="control-matrix-marker"></div>')
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.2, 1.0, 0.8], gap="medium") 
    with col_ctrl1:
        st.html("""
        <div class="section-label">
            <span>📹</span> Input Method
        </div>
        """)
        input_mode = st.selectbox(
            label="Input Method Selection",
            options=[
                "Live Webcam Stream",
                "Upload Hand Sign Image File",
                "Native Snapshot (STUN/TURN Bypass)"
            ],
            label_visibility="collapsed"
        )
    with col_ctrl2:
        st.html("""
        <div class="section-label">
            <span>🔍</span> Optical / Digital Zoom
        </div>
        """)
        zoom_factor = st.slider(
            label="Camera Zoom Scaler",
            min_value=1.0,
            max_value=3.0,
            value=1.0,
            step=0.1,
            format="%.1fx",
            label_visibility="collapsed",
            key="zoom_scaler_slider"
        )
    with col_ctrl3:
        st.html("""<div style="margin-top: 25px;"></div>""")
        if st.session_state.get("saved_prediction") is not None:
            if st.button("🔄 Test Another Sign", use_container_width=True, key="AnotherSignBtn", type="secondary"):
                st.session_state.saved_full_view = None
                st.session_state.saved_hand_crop = None
                st.session_state.saved_prediction = None
                st.rerun()

# 5. MAIN LIVE VIEWPORT WORKSPACE
col1, col2 = st.columns([1.35, 0.65], gap="large")
with col1:
    with st.container(key="viewfinder_panel"):
        st.markdown('<div class="panel"><h3 class="panel-title">LIVE VIEWFINDER</h3>', unsafe_allow_html=True)
        st.html('<div class="viewfinder-wrapper">')
        
        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            if zoom_factor > 1.0:
                img = apply_digital_zoom(img, zoom_factor)
            
            st.session_state["webrtc_latest_frame"] = img.copy()
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if st.session_state.get("saved_prediction") is None:
            if input_mode == "Live Webcam Stream":
                webrtc_streamer(
                    key="slr-live-stream",
                    mode=WebRtcMode.SENDRECV,
                    video_frame_callback=video_frame_callback,
                    rtc_configuration={"iceServers": get_ice_servers()},
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 640, "max": 640},
                            "height": {"ideal": 360, "max": 360},
                            "frameRate": {"ideal": 15, "max": 30}
                        },
                        "audio": False
                    },
                )
            elif input_mode == "Native Snapshot (STUN/TURN Bypass)":
                camera_file = st.camera_input("Take a photo of your sign language gesture")
                if camera_file is not None:
                    file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
                    uploaded_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if zoom_factor > 1.0:
                        uploaded_bgr = apply_digital_zoom(uploaded_bgr, zoom_factor)
                    st.session_state["webrtc_latest_frame"] = uploaded_bgr
            else:
                uploaded_file = st.file_uploader("Choose a hand sign image...", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    uploaded_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if zoom_factor > 1.0:
                        uploaded_bgr = apply_digital_zoom(uploaded_bgr, zoom_factor)
                    st.session_state["webrtc_latest_frame"] = uploaded_bgr
                    st.image(cv2.cvtColor(uploaded_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded Image Preview", use_container_width=True)
        else:
            if st.session_state.get("saved_full_view") is not None:
                st.image(st.session_state.saved_full_view, caption="Captured Frame", use_container_width=True)

        st.html('</div>')
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        
        if st.session_state.get("saved_prediction") is None:
            shutter_btn = st.button(
                "📸 CAPTURE HAND GESTURE", 
                type="primary", 
                disabled=not system_online, 
                use_container_width=True,
                key="shutter_action_trigger"
            )
        else:
            shutter_btn = False

        if shutter_btn:
            if "webrtc_latest_frame" in st.session_state and st.session_state["webrtc_latest_frame"] is not None:
                frame = st.session_state["webrtc_latest_frame"]
                process_frame_and_predict(frame)
                st.rerun()
            else:
                st.warning("No image frame available. Please start the webcam stream or take a snapshot above.")

        st.markdown('</div>', unsafe_allow_html=True)

# 6. DIAGNOSTICS & RESULTS PANEL
with col2:
    with st.container(key="diagnostics_panel"):
        st.markdown('<div class="panel diagnostics-panel"><h3 class="panel-title">AI DIAGNOSTICS & RESULTS</h3>', unsafe_allow_html=True)     
        metrics_slot = st.empty()
        
        col_img1, col_img2 = st.columns(2, gap="small")
        with col_img1:
            full_output_view = st.empty()
        with col_img2:
            crop_output_view = st.empty()

        alternatives_slot = st.empty()
        reset_slot = st.empty()
        
        if st.session_state.get("saved_prediction") is not None:
            res = st.session_state.saved_prediction
            
            if st.session_state.get("saved_full_view") is not None:
                full_output_view.image(st.session_state.saved_full_view, caption="Captured Frame", use_container_width=True)
            if st.session_state.get("saved_hand_crop") is not None:
                crop_output_view.image(st.session_state.saved_hand_crop, caption="Grad-CAM Focus", use_container_width=True)

            with metrics_slot:
                if res.get("low_light"):
                    st.html("""
                    <div class="predicted-sign">
                        <p class="predicted-sign__label">Brightness is too low to capture an image</p>
                        <h2 class="predicted-sign__glyph">—</h2>
                    </div>
                    """)
                elif res.get("no_hand"):
                    st.html("""
                    <div class="predicted-sign">
                        <p class="predicted-sign__label">No hand detected in the picture</p>
                        <h2 class="predicted-sign__glyph">—</h2>
                    </div>
                    """)
                elif res['score'] < res['threshold']:
                    st.html(f"""
                    <div class="predicted-sign">
                        <p class="predicted-sign__label">Sign could not be detected, please try again.</p>
                        <h2 class="predicted-sign__glyph">—</h2>
                    </div>

                    <div class="stat-row">
                        <div class="stat-card">
                            <p class="stat-card__label">Similarity</p>
                            <h4 class="stat-card__value">{res['score']*100:.1f}%</h4>
                        </div>
                        <div class="stat-card">
                            <p class="stat-card__label">Min Threshold</p>
                            <h4 class="stat-card__value">{res['threshold']*100:.1f}%</h4>
                        </div>
                    </div>
                    """)
                else:
                    st.html(f"""
                    <div class="predicted-sign">
                        <p class="predicted-sign__label">Predicted Sign</p>
                        <h2 class="predicted-sign__glyph">{res['char']}</h2>
                    </div>

                    <div class="stat-row">
                        <div class="stat-card">
                            <p class="stat-card__label">Similarity</p>
                            <h4 class="stat-card__value">{res['score']*100:.1f}%</h4>
                        </div>
                        <div class="stat-card">
                            <p class="stat-card__label">Min Threshold</p>
                            <h4 class="stat-card__value">{res['threshold']*100:.1f}%</h4>
                        </div>
                    </div>
                    """)

            with alternatives_slot:
                if res.get("low_light"):
                    st.html("""
                    <p class="alternatives-title">RECOMMENDATIONS</p>
                    <div class="alt-list">
                        <div class="alt-bar" style="justify-content: center; gap: 10px;">
                            <span class="alt-bar__glyph" style="font-size: 1.2rem;">💡</span>
                            <span class="alt-bar__value">Increase key lighting or adjust camera exposure.</span>
                        </div>
                    </div>
                    """)
                elif res.get("no_hand"):
                    st.html("""
                    <p class="alternatives-title">RECOMMENDATIONS</p>
                    <div class="alt-list">
                        <div class="alt-bar" style="justify-content: center; gap: 10px;">
                            <span class="alt-bar__glyph" style="font-size: 1.2rem;">🖐️</span>
                            <span class="alt-bar__value">Center your hand inside the camera frame.</span>
                        </div>
                    </div>
                    """)
                else:
                    alt_html = '<p class="alternatives-title">TOP ALTERNATIVE MATCHES</p><div class="alt-list">'
                    for idx, (alt_char, alt_score) in enumerate(res['top3']):
                        bar_class = "alt-bar alt-bar--top" if idx == 0 and res['score'] >= res['threshold'] else "alt-bar"
                        alt_html += f"""
                        <div class="{bar_class}">
                            <span class="alt-bar__glyph">{alt_char}</span>
                            <span class="alt-bar__value">{alt_score}</span>
                        </div>
                        """
                    alt_html += '</div>'
                    st.html(alt_html)
            
            with reset_slot:
                st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
                if st.button("🔄 Test Another Sign", use_container_width=True, key="AnotherSignBtn", type="secondary"):
                    st.session_state.saved_full_view = None
                    st.session_state.saved_hand_crop = None
                    st.session_state.saved_prediction = None
                    st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)