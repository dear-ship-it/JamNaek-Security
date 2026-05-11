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
    # ใช้ yolov8s (Small) เพื่อความแม่นยำที่เพิ่มขึ้นจากตัวเดิมนิดหน่อย
    return YOLO("yolov8s.pt") 

model = load_model()

# --- ส่วนควบคุมด้านข้าง (Sidebar) ---
st.sidebar.header("⚙️ การตั้งค่า AI")
conf_threshold = st.sidebar.slider("ค่าความมั่นใจ (Confidence)", 0.0, 1.0, 0.45)
st.sidebar.info("""
**วัตถุที่ระบบตรวจจับได้ดี:**
- 👤 คน (Person)
- 🚗 ยานพาหนะ (Car, Motorcycle, Truck)
- 🐶 สัตว์ (Dog, Cat, Bird)
- 🎒 สิ่งของ (Backpack, Suitcase, Umbrella)
- 📱 อุปกรณ์ (Cell phone, Laptop)
""")

# --- ส่วนอัปโหลดรูปภาพ ---
uploaded_file = st.file_uploader("📤 อัปโหลดรูปภาพจากกล้องวงจรปิด...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # อ่านรูปภาพ
    image = Image.open(uploaded_file)
    
    # แบ่งหน้าจอเป็น 2 คอลัมน์ (ซ้าย: รูปภาพ / ขวา: สถิติ)
    col1, col2 = st.columns([3, 2])

    # AI ประมวลผล
    results = model.predict(image, conf=conf_threshold)
    res_plotted = results[0].plot() # วาด Bounding Box

    with col1:
        st.subheader("🖼 ผลการวิเคราะห์จากภาพ")
        st.image(res_plotted, use_column_width=True)

    with col2:
        st.subheader("📊 รายละเอียดที่ตรวจพบ")
        
        # ดึงข้อมูลวัตถุทั้งหมดที่เจอ
        objects_data = []
        counts = {}

        for result in results[0].boxes:
            label = model.names[int(result.cls[0])]
            conf = float(result.conf[0])
            
            # เก็บข้อมูลลง List
            objects_data.append({"ประเภท": label, "ความมั่นใจ": f"{conf:.2%}"})
            # นับจำนวน
            counts[label] = counts.get(label, 0) + 1

        if objects_data:
            # แสดงสรุปจำนวนเป็น Badge สวยๆ
            st.write("**สรุปจำนวนวัตถุ:**")
            cols = st.columns(3)
            for i, (name, count) in enumerate(counts.items()):
                cols[i % 3].metric(label=name, value=count)
            
            st.write("---")
            
            # แสดงตารางข้อมูลดิบ
            st.write("**ตารางวิเคราะห์รายชิ้น:**")
            df = pd.DataFrame(objects_data)
            st.dataframe(df, use_container_width=True)
            
            # ปุ่มสำหรับ Export ข้อมูล
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 ดาวน์โหลดรายงาน (CSV)",
                data=csv,
                file_name='detection_report.csv',
                mime='text/csv',
            )
        else:
            st.warning("⚠️ ไม่พบวัตถุที่ระบุในภาพ กรุณาปรับค่า Confidence ลดลง")

else:
    st.info("💡 คำแนะนำ: อัปโหลดภาพนิ่งจากกล้อง Security เพื่อเริ่มการจำแนก")
