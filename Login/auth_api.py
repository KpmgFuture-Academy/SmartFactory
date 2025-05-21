#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 📁 auth_api.py (FastAPI 백엔드 - 세션 검증 로그 추가 및 안정화)

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
import psycopg2
import uuid
from datetime import datetime
from config import DB_INFO

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="dev-temp-key")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 사용자 등록 ---
@app.post("/register")
def register(
    username: str = Form(...),
    login_id: str = Form(...),
    password: str = Form(...),
    department: str = Form(...),
    role: str = Form(...),
    company_id: str = Form(...)
):
    try:
        with psycopg2.connect(**DB_INFO) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM users WHERE login_id = %s", (login_id,))
                if cur.fetchone():
                    raise HTTPException(status_code=400, detail="이미 존재하는 로그인 ID입니다.")

                user_id = str(uuid.uuid4())
                is_approved = role == "admin"
                created_at = datetime.utcnow()

                cur.execute('''
                    INSERT INTO users (user_id, company_id, username, login_id, password_hash, department, role, is_approved, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (user_id, company_id, username, login_id, password, department, role, is_approved, created_at))

                return {
                    "message": "관리자 등록 완료." if is_approved else "회원가입 완료. 관리자 승인 필요",
                    "user_id": user_id
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 로그인 ---
@app.post("/login")
def login(request: Request, login_id: str = Form(...), password: str = Form(...)):
    try:
        with psycopg2.connect(**DB_INFO) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id, username, company_id, role, department, is_approved, password_hash FROM users WHERE login_id = %s", (login_id,))
                row = cur.fetchone()

                if not row:
                    raise HTTPException(status_code=401, detail="존재하지 않는 로그인 ID입니다.")

                user_id, username, company_id, role, department, is_approved, password_hash = row

                if password != password_hash:
                    raise HTTPException(status_code=401, detail="비밀번호가 일치하지 않습니다.")

                if not is_approved:
                    raise HTTPException(status_code=403, detail="관리자 승인이 필요합니다.")

                request.session["user"] = login_id
                return {
                    "message": "로그인 성공",
                    "user_id": str(user_id),
                    "username": username,
                    "company_id": str(company_id),
                    "role": role,
                    "department": department
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 현재 사용자 정보 반환 ---
@app.get("/me")
def get_me(request: Request):
    login_id = request.session.get("user")
    if not login_id:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    try:
        with psycopg2.connect(**DB_INFO) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT user_id, company_id, username, login_id, department, role, created_at
                    FROM users
                    WHERE login_id = %s
                """, (login_id,))
                user = cur.fetchone()
                if not user:
                    raise HTTPException(status_code=404, detail="사용자 정보를 찾을 수 없습니다.")

                return {
                    "user_id": str(user[0]),
                    "company_id": str(user[1]),
                    "username": user[2],
                    "login_id": user[3],
                    "department": user[4],
                    "role": user[5],
                    "created_at": user[6]
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 승인 대기 사용자 목록 ---
@app.get("/pending_users")
def get_pending_users(request: Request):
    login_id = request.session.get("user")
    print("[DEBUG] Session login_id:", login_id)

    if not login_id:
        raise HTTPException(status_code=403, detail="로그인이 필요합니다.")

    try:
        with psycopg2.connect(**DB_INFO) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT username, department, login_id, created_at
                    FROM users
                    WHERE is_approved = FALSE
                """)
                rows = cur.fetchall()
                return [{
                    "username": r[0],
                    "department": r[1],
                    "login_id": r[2],
                    "created_at": r[3].strftime("%Y-%m-%d %H:%M:%S")
                } for r in rows]
                role = cur.fetchone()
                print("[DEBUG] Role from DB:", role)

                if not role:
                    raise HTTPException(status_code=403, detail="사용자 권한 조회 실패")
                elif role[0] != "admin":
                    raise HTTPException(status_code=403, detail="관리자만 접근 가능")

                cur.execute("""
                    SELECT username, department, login_id, created_at
                    FROM users
                    WHERE is_approved = FALSE
                """)
                rows = cur.fetchall()
                return [{
                    "username": r[0],
                    "department": r[1],
                    "login_id": r[2],
                    "created_at": r[3].strftime("%Y-%m-%d %H:%M:%S")  # 보기 좋게 포맷
                } for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 승인 처리 ---
@app.post("/approve")
def approve_user(username: str = Form(...), request: Request = None):
    login_id = request.session.get("user")
    if not login_id:
        raise HTTPException(status_code=403, detail="로그인이 필요합니다.")

    try:
        with psycopg2.connect(**DB_INFO) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT role FROM users WHERE login_id = %s", (login_id,))
                role = cur.fetchone()
                if not role or role[0] != "admin":
                    raise HTTPException(status_code=403, detail="관리자 권한 필요")

                cur.execute("UPDATE users SET is_approved = TRUE WHERE username = %s", (username,))
                conn.commit()
                return {"message": f"{username} 승인 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

