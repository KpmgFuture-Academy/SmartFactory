#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

user_session = {
    "company_id": "11111111-1111-1111-1111-111111111111",
    "user_id": "22222222-2222-2222-2222-222222222222"
}

st.set_page_config(layout="wide")
st.title("📦 Smart Factory Data Uploader")

# ---------------------- 상태 초기화 ----------------------
if "selected_folder" not in st.session_state:
    st.session_state.selected_folder = None
if "file_list" not in st.session_state:
    st.session_state.file_list = []
if "folder_loaded" not in st.session_state:
    st.session_state.folder_loaded = False
if "upload_id" not in st.session_state:
    st.session_state.upload_id = None
    
# ---------------------- 폴더 선택 ----------------------
st.markdown("### 🔍 S3 폴더 선택")
folders_res = requests.get(f"{API_URL}/list_folders")

if folders_res.status_code == 200:
    all_folders = folders_res.json()["folders"]
    selected_folders = st.multiselect("📁 여러 폴더 선택", all_folders, key="multi_folder_select")
    st.session_state.selected_folders = selected_folders
else:
    st.error("❌ 폴더 목록을 불러올 수 없습니다.")

# ---------------------- 파일 분류 ----------------------
all_csv_files = []
all_image_files = []
folder_label_map = {}

if st.session_state.get("selected_folders"):
    for folder in st.session_state.selected_folders:
        fr = requests.get(f"{API_URL}/list_files_in_folder", params={"prefix": folder + "/"})
        if fr.status_code == 200:
            files = fr.json()["files"]
            csvs = [f for f in files if f.endswith(".csv")]
            imgs = [f for f in files if f.endswith((".jpg", ".jpeg", ".png"))]
            all_csv_files.extend(csvs)
            all_image_files.extend(imgs)
            if imgs:
                folder_label_map[folder] = st.radio(f"📁 `{folder}` 폴더의 이미지 불량 여부", ["ok", "defect"], key=f"label_{folder}")

# ---------------------- 센서 CSV 업로드 ----------------------
if all_csv_files:
    st.markdown("### 🧾 센서 CSV 업로드")
    selected_csv = st.selectbox("CSV 파일 선택", all_csv_files)
    col_resp = requests.post(f"{API_URL}/load_columns", json={"file_key": selected_csv})
    if col_resp.status_code == 200:
        columns = col_resp.json()["columns"]
        classification_type = st.radio("📌 분류 방식 선택", [
            "단일 Binary Target만 존재",
            "Binary + Multi-label Target 존재",
            "Multi-label Target만 존재"
        ])

        binary_target = None
        multi_targets = []

        if classification_type == "단일 Binary Target만 존재":
            binary_target = st.selectbox("✅ Binary Target 선택", columns)
            task_type = "binary_classification"
        elif classification_type == "Binary + Multi-label Target 존재":
            binary_target = st.selectbox("✅ Binary Target 선택", columns)
            multi_targets = st.multiselect("🧩 Multi-label Target 선택", columns)
            task_type = "binary_and_multi_label"
        else:
            multi_targets = st.multiselect("🧩 Multi-label Target 선택", columns)
            task_type = "multi_label"

        if st.button("🚀 CSV 업로드"):
            payload = {
                "company_id": user_session["company_id"],
                "uploader_id": user_session["user_id"],
                "file_key": selected_csv,
                "file_type": "sensor_csv",
                "task_type": task_type,
                "binary_target_column": binary_target,
                "multilabel_target_columns": multi_targets,
            }
            res = requests.post(f"{API_URL}/ingest_csv_to_db", json=payload)
            if res.status_code == 200:
                upload_id = res.json()["upload_id"]
                st.success(f"✅ 업로드 성공 (Upload ID: {upload_id})")
                prev = requests.post(f"{API_URL}/preview_csv", json={"upload_id": upload_id})
                if prev.status_code == 200:
                    preview_df = pd.DataFrame(prev.json()["preview"])
                    target_df = pd.DataFrame(prev.json()["target"])
                    upload_df = pd.DataFrame(prev.json().get("upload", []))
                    if "machine_failure" in target_df.columns:
                        target_df["machine_failure"] = target_df["machine_failure"].astype(str)
                    st.markdown("#### 📊 센서 데이터 미리보기")
                    st.dataframe(preview_df)
                    st.markdown("#### 🎯 타겟 정보 미리보기")
                    st.dataframe(target_df)
                    if not upload_df.empty:
                        st.markdown("#### 🗂 업로드 정보 미리보기")
                        st.dataframe(upload_df)
                else:
                    st.warning("⚠️ 미리보기를 불러오지 못했습니다.")
            else:
                st.error(f"❌ 업로드 실패: {res.text}")

# ---------------------- 이미지 업로드 ----------------------
if all_image_files:
    st.markdown("### 🖼 이미지 메타 업로드")
    label_type = st.selectbox("📌 라벨 타입", ["binary", "bbox", "mask"])

    if st.button("🚀 이미지 업로드"):
        binary_labels = {}
        for folder in st.session_state.selected_folders:
            label = folder_label_map.get(folder, "ok")
            for key in all_image_files:
                if key.startswith(folder + "/"):
                    binary_labels[key] = label

        payload = {
            "company_id": user_session["company_id"],
            "uploader_id": user_session["user_id"],
            "file_keys": all_image_files,
            "label_type": label_type,
            "binary_labels": binary_labels if label_type == "binary" else {}
        }
        res = requests.post(f"{API_URL}/ingest_images", json=payload)
        if res.status_code == 200:
            upload_id = res.json()["upload_id"]
            st.success(f"✅ 이미지 메타 업로드 성공 (Upload ID: {upload_id})")
            prev = requests.post(f"{API_URL}/preview_image", json={"upload_id": upload_id})
            if prev.status_code == 200:
                image_df = pd.DataFrame(prev.json()["images"])
                upload_df = pd.DataFrame(prev.json().get("upload", []))
                st.markdown("#### 🖼 이미지 데이터 미리보기")
                # Boolean을 문자열로 변환 (True → "ok", False → "defect")
                if "is_defect" in image_df.columns:
                    image_df["is_defect"] = image_df["is_defect"].apply(lambda x: "True" if x else "False")
                    st.dataframe(image_df)
                if not upload_df.empty:
                    st.markdown("#### 🗂 업로드 정보 미리보기")
                    st.dataframe(upload_df)
            else:
                st.warning("⚠️ 미리보기를 불러오지 못했습니다.")
        else:
            st.error(f"❌ 업로드 실패: {res.text}")


