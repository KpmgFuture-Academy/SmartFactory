#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 📁 auth_streamlit.py (Streamlit 프론트엔드 - 사용자 이름, 로그인 ID 구분)

import streamlit as st
import requests
import uuid
import pandas as pd

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="SmartFactory Login", layout="centered")

if "username" not in st.session_state:
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.department = None
    st.session_state.company_id = None

def login_ui():
    st.title("🔐 로그인")
    with st.form("login_form"):
        login_id = st.text_input("로그인 ID")
        password = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("로그인")
        if submitted:
            try:
                res = requests.post(f"{API_BASE_URL}/login", data={"login_id": login_id, "password": password})
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.username = data["username"]
                    st.session_state.role = data["role"]
                    st.session_state.department = data["department"]
                    st.session_state.company_id = data["company_id"]
                    
                    # 🔑 세션 쿠키 저장
                    if "set-cookie" in res.headers:
                        session_raw = res.headers["set-cookie"]
                        session_token = session_raw.split(";")[0].split("=")[1]
                        st.session_state.session_cookie = session_token

                    st.success("로그인 성공!")
                    st.rerun()
                else:
                    st.error(res.json().get("detail", "로그인 실패"))
            except Exception as e:
                st.error(f"서버 오류: {str(e)}")
    if st.button("회원가입 하러 가기"):
        st.session_state.show_register = True

def register_ui():
    st.title("📋 회원가입")

    with st.form("register_form"):
        username = st.text_input("이름")
        login_id = st.text_input("로그인 ID")
        password = st.text_input("비밀번호", type="password")
        department = st.selectbox("부서", ["feature_engineering", "data", "현장관리", "운영"])
        role = st.selectbox("역할", ["user", "admin"])
        company_id = st.text_input("회사 ID", placeholder="회사 UUID를 입력하세요")

        submitted = st.form_submit_button("회원가입")

    if submitted:
        try:
            res = requests.post(f"{API_BASE_URL}/register", data={
                "username": username,
                "login_id": login_id,
                "password": password,
                "department": department,
                "role": role,
                "company_id": company_id
            })
            if res.status_code == 200:
                data = res.json()
                msg = "관리자 계정이 성공적으로 생성되었습니다." if role == "admin" else "회원가입 완료. 관리자 승인을 기다려주세요."
                st.success(msg)
                st.info(f"🔑 로그인 시 사용할 ID: `{login_id}`")
                st.session_state.show_register = False
            else:
                try:
                    st.error(res.json().get("detail", "회원가입 실패"))
                except:
                    st.error(f"서버 응답 오류: {res.text}")
        except Exception as e:
            st.error(f"요청 실패: {str(e)}")

    if st.button("로그인 화면으로"):
        st.session_state.show_register = False


def admin_ui():
    st.title(f"👑 관리자: {st.session_state.username}")
    st.subheader("⏳ 승인 대기 사용자")

    try:
        cookies = {"session": st.session_state.get("session_cookie")}
        res = requests.get(f"{API_BASE_URL}/pending_users", cookies=cookies)

        if res.status_code != 200:
            st.error(f"승인 대기자 목록을 불러오지 못했습니다: {res.status_code}")
            return

        pending = res.json()
        if not pending:
            st.info("승인 대기 중인 사용자가 없습니다.")
            return

        st.markdown("### 🧾 승인 요청 목록")

        # ✅ 표 형태로 시각화 + 버튼은 표 바깥에서 각 행 별로 렌더링
        for u in pending:
            col1, col2, col3, col4, col5 = st.columns([2, 2, 3, 3, 1])
            col1.write(f"**이름:** {u['username']}")
            col2.write(f"**부서:** {u['department']}")
            col3.write(f"**로그인 ID:** `{u['login_id']}`")
            col4.markdown(f"**요청 시각:** `{u['created_at']}`")

            # ✔️ 버튼이 내려가지 않도록 작은 col5 안에 배치
            with col5:
                st.markdown("")  # 공간 확보용 빈 줄
                if st.button("승인", key=f"approve_{u['login_id']}"):
                    res = requests.post(
                        f"{API_BASE_URL}/approve",
                        data={"username": u["username"]},
                        cookies=cookies
                    )
                    if res.status_code == 200:
                        st.success(f"{u['username']} 승인 완료")
                        st.rerun()
                    else:
                        st.error("승인 실패")



    except Exception as e:
        st.error(f"승인 대기자 목록을 불러오지 못했습니다: {str(e)}")


    st.subheader("🏭 공장 상태 요약 섹터")
    st.info("→ 공장 지표 시각화 예정")

    st.subheader("👥 부서별 사용자 섹터")
    st.info("→ 부서별 사용자 목록 및 현황 예정")

def worker_ui():
    st.title(f"👤 사용자: {st.session_state.username}")
    st.write(f"부서: {st.session_state.department}")

    if st.session_state.department in ["feature_engineering", "data"]:
        st.success("✅ 데이터 처리 부서용 기능 Placeholder")
    elif st.session_state.department == "현장관리":
        st.info("📊 현장관리 대시보드 Placeholder")

# 전체 흐름 제어
if not st.session_state.username:
    if st.session_state.get("show_register", False):
        register_ui()
    else:
        login_ui()
else:
    if st.session_state.role == "admin":
        admin_ui()
    else:
        worker_ui()

