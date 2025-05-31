import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from PIL import Image
import os
import tempfile
from heartbeat_utils import HeartRateEstimator
from utils import generate_pdf_report
from scipy.signal import butter, filtfilt

st.set_page_config(page_title="Heart Rate & Blood Group App")
st.title("💓 Face Scan Heartbeat & Blood Group Detector")

# --- Utility: Optional filtering ---
def apply_bandpass(signal, low=0.75, high=3.0, fs=30):
    nyq = 0.5 * fs
    b, a = butter(1, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

# --- State Management ---
if "scanning" not in st.session_state:
    st.session_state["scanning"] = False

# --- User Inputs ---
blood_group = st.selectbox("Select Blood Group", [
    "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"
])
start_button = st.button("Start Scanning")
if start_button:
    st.session_state["scanning"] = True

video_placeholder = st.empty()
bpm_placeholder = st.empty()
chart_placeholder = st.empty()
intensity_placeholder = st.empty()

estimator = HeartRateEstimator(buffer_size=150, fps=30)

# --- Live Webcam Scan ---
if st.session_state["scanning"]:
    estimator.reset()
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    history = []
    bpm = None
    no_face_count = 0

    start_time = time.time()
    duration = 20
    progress = st.progress(0)

    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera error. Please try again.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                no_face_count = 0
                (x, y, w, h) = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                green = face_roi[:, :, 1]
                mean_green = green.mean()
                estimator.update(mean_green)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                no_face_count += 1
                if no_face_count > 20:
                    st.warning("⚠️ Face not detected. Please stay still and ensure good lighting.")

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(Image.fromarray(img_rgb), caption="Live Feed", use_container_width=True)

            bpm = estimator.get_heart_rate()
            if bpm:
                bpm_placeholder.markdown(f"### 💗 Estimated BPM: **{bpm}**")
                history.append(bpm)
                chart_placeholder.line_chart(history)
            else:
                bpm_placeholder.markdown("### Measuring heartbeat...")

            intensity_placeholder.line_chart(list(estimator.buffer))
            progress.progress(min((time.time() - start_time) / duration, 1.0))



    finally:
        cap.release()
        st.session_state["scanning"] = False

    # --- After Scanning ---
    if bpm:
        st.subheader(f"✅ Final Estimated Heart Rate: **{bpm} BPM**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 Save Result"):
                record = {
                    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Blood Group": blood_group,
                    "BPM": bpm
                }
                df = pd.DataFrame([record])
                if os.path.exists("records.csv"):
                    df.to_csv("records.csv", mode='a', header=False, index=False)
                else:
                    df.to_csv("records.csv", index=False)
                st.success("Record saved!")

        with col2:
            if st.button("📄 Download PDF Report"):
                pdf_file = generate_pdf_report(bpm, blood_group)
                timestamped_name = f"report_{int(time.time())}.pdf"
                if pdf_file:
                    with open(pdf_file, "rb") as f:
                        st.download_button("Download PDF", f, file_name=timestamped_name)
                else:
                    st.error("Failed to generate PDF report.")

        if os.path.exists("records.csv"):
            df = pd.read_csv("records.csv")
            st.subheader("📄 Past Records")
            st.dataframe(df)

# --- Video Upload Section ---
st.sidebar.header("📄 Upload Video")
video_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi"])

if video_file:
    st.success("Video uploaded! Scanning...")
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    estimator.reset()
    bpm = None
    history = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            green = face_roi[:, :, 1]
            mean_green = green.mean()
            estimator.update(mean_green)

        bpm = estimator.get_heart_rate()
        if bpm:
            history.append(bpm)

    cap.release()

    if bpm:
        st.subheader(f"📊 Estimated BPM: **{bpm}**")
        st.line_chart(history)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 Save Result", key="upload_save"):
                record = {
                    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Blood Group": blood_group,
                    "BPM": bpm
                }
                df = pd.DataFrame([record])
                if os.path.exists("records.csv"):
                    df.to_csv("records.csv", mode='a', header=False, index=False)
                else:
                    df.to_csv("records.csv", index=False)
                st.success("Record saved!")

        with col2:
            if st.button("📄 Download PDF Report"):
                pdf_file = generate_pdf_report(bpm, blood_group)
                if pdf_file:
                    with open(pdf_file, "rb") as f:
                        st.download_button("Download PDF", f, file_name="report.pdf")
                else:
                    st.error("Failed to generate PDF report.")
