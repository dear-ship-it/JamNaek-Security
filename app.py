import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="จำแนก - Security Dashboard", layout="wide")

st.title("🔍 ระบบจำแนกวัตถุอัจฉริยะ (JamNaek)")
st.markdown("---")

# --- โหลดโมเดล AI ---
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt") 

model = load_model()

# --- ส่วนควบคุมด้านข้าง (Sidebar) ---
st.sidebar.header("⚙️ การตั้งค่า AI")
conf_threshold = st.sidebar.slider("ค่าความมั่นใจ (Confidence)", 0.0, 1.0, 0.45)
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:** วางภาพคู่กันเพื่อให้ทีม Security ตรวจสอบความถูกต้องได้ง่ายขึ้น")

# --- ส่วนอัปโหลดรูปภาพ ---
uploaded_file = st.file_uploader("📤 อัปโหลดรูปภาพจากกล้องวงจรปิด...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. เตรียมรูปภาพต้นฉบับ
    original_image = Image.open(uploaded_file)
    
    # 2. AI ประมวลผล
    results = model.predict(original_image, conf=conf_threshold)
    annotated_frame = results[0].plot() # ภาพที่ AI วาดกรอบแล้ว
    
    # 3. ส่วนการแสดงภาพแบบเปรียบเทียบ (ซ้าย-ขวา)
    st.subheader("🖼 การเปรียบเทียบ: ภาพต้นฉบับ VS ผลการตรวจจับ")
    img_col1, img_col2 = st.columns(2)
    
    with img_col1:
        st.write("⬅️ **ภาพต้นฉบับ (Original)**")
        st.image(original_image, use_column_width=True)
        
    with img_col2:
        st.write("➡️ **ผลการวิเคราะห์ (AI Detection)**")
        st.image(annotated_frame, use_column_width=True)

    st.markdown("---")

    # 4. ส่วนแสดงสถิติและข้อมูลด้านล่าง
    col_data, col_stat = st.columns([2, 1])

    with col_data:
        st.subheader("📊 ตารางวิเคราะห์รายชิ้น")
        objects_data = []
        counts = {}

        for result in results[0].boxes:
            label = model.names[int(result.cls[0])]
            conf = float(result.conf[0])
            objects_data.append({"ประเภท": label, "ความมั่นใจ": f"{conf:.2%}"})
            counts[label] = counts.get(label, 0) + 1

        if objects_data:
            df = pd.DataFrame(objects_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("⚠️ ไม่พบวัตถุที่ระบุในภาพ")

    with col_stat:
        st.subheader("📈 สรุปจำนวน")
        if counts:
            for name, count in counts.items():
                st.write(f"✅ **{name}**: {count}")
            
            # ปุ่มดาวน์โหลดรายงาน
            csv = pd.DataFrame(objects_data).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 ดาวน์โหลด CSV",
                data=csv,
                file_name='security_report.csv',
                mime='text/csv',
            )

else:
    st.info("💡 กรุณาอัปโหลดภาพเพื่อเริ่มการทำงาน")
