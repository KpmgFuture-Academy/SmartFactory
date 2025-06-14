import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="스마트팩토리 품질 예측 대시보드", layout="wide")

st.title("📊 스마트팩토리 품질검사 대시보드")
st.markdown("업로드된 이미지의 예측 결과를 확인하고, 분류된 이미지를 시각적으로 검토할 수 있습니다.")

# 파일 경로 설정
excel_path = r"C:\Users\Admin\Desktop\smart_factory_qa\results\results.xlsx"
sorted_dir = r"C:\Users\Admin\Desktop\smart_factory_qa\sorted"

# 🔍 결과 테이블 출력
if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
    st.subheader("📄 예측 결과 기록 (results.xlsx)")
    st.dataframe(df.tail(10), use_container_width=True)
else:
    st.warning("⚠️ 예측 결과 파일(results.xlsx)이 아직 없습니다.")

# 📸 분류된 이미지 보여주기
col1, col2 = st.columns(2)

def show_images_from_folder(folder_path, label, column):
    column.subheader(f"🖼 {label} 이미지")
    if os.path.exists(folder_path):
        images = sorted(os.listdir(folder_path), reverse=True)[-5:]
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            try:
                img = Image.open(img_path)
                column.image(img, caption=img_name, width=250)
            except:
                continue
    else:
        column.write("이미지가 없습니다.")

# 정상 / 불량 이미지 출력
show_images_from_folder(os.path.join(sorted_dir, "ok"), "정상", col1)
show_images_from_folder(os.path.join(sorted_dir, "defect"), "불량", col2)