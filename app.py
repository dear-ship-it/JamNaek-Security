import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

st.set_page_config(page_title="จำแนก - AI Security")
st.title("🔍 จำแนก (JamNaek) - Web AI")

# โหลด Model AI แบบเบาที่สุดเพื่อให้รันบนเว็บได้
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.write("อัปโหลดรูปภาพเพื่อตรวจจับวัตถุ (คน/สัตว์/สิ่งของ)")
uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปที่อัปโหลด', use_column_width=True)
    st.write("กำลังประมวลผล...")
    
    # AI ทำงาน
    results = model(image)
    
    # แสดงผล
    res_plotted = results[0].plot()
    st.image(res_plotted, caption='ผลการตรวจจับ', use_column_width=True)
    
    # แสดงรายการที่พบ
    st.write("### วัตถุที่ตรวจพบ:")
    for result in results[0].boxes:
        label = model.names[int(result.cls[0])]
        conf = float(result.conf[0])
        st.success(f"พบ: **{label}** (ความมั่นใจ: {conf:.2f})")
