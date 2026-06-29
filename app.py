import streamlit as st
import requests
import json
from typing import Optional, List
import time
from datetime import datetime
import pandas as pd
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def init_session_state():
    defaults = {
        "token": None,
        "user": None,
        "conversation_id": None,
        "messages": [],
        "doc_ids": [],
        "uploaded_files": [],
        "page": "chat"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def api_register(username: str, email: str, password: str):
    response = requests.post(
        f"{API_BASE_URL}/auth/register",
        json={"username": username, "email": email, "password": password}
    )
    return response

def api_login(email: str, password: str):
    response = requests.post(
        f"{API_BASE_URL}/auth/login",
        json={"email": email, "password": password}
    )
    return response

def api_get_user(token: str):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_BASE_URL}/auth/me", headers=headers)
    return response

def api_get_documents(token: str):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_BASE_URL}/api/v1/documents", headers=headers)
    return response

def api_ingest_document(token: str, file):
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": (file.name, file.getvalue(), file.type)}
    response = requests.post(
        f"{API_BASE_URL}/api/v1/rag/ingest",
        headers=headers,
        files=files
    )
    return response

def api_query(token: str, query: str, doc_ids: List[str], conversation_id: Optional[str] = None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = {"query": query, "doc_ids": doc_ids}
    if conversation_id:
        data["conversation_id"] = conversation_id
    response = requests.post(
        f"{API_BASE_URL}/api/v1/rag/response",
        headers=headers,
        json=data
    )
    return response

def login_page():
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1>🔐 Welcome to RAG Assistant</h1>
            <p style='color: #666;'>Login to access your documents and AI assistant</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            st.markdown("---")
            with st.form("login_form"):
                email = st.text_input("📧 Email", placeholder="your@email.com")
                password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("🚀 Login", use_container_width=True)
                
                if submit:
                    if not email or not password:
                        st.warning("Please fill in all fields")
                    else:
                        with st.spinner("Logging in..."):
                            response = api_login(email, password)
                            if response.status_code == 200:
                                data = response.json()
                                st.session_state.token = data["access_token"]
                                user_response = api_get_user(st.session_state.token)
                                if user_response.status_code == 200:
                                    st.session_state.user = user_response.json()
                                    docs_response = api_get_documents(st.session_state.token)
                                    if docs_response.status_code == 200:
                                        docs_data = docs_response.json()
                                        for doc in docs_data.get("documents", []):
                                            doc_id = str(doc["id"])
                                            if doc_id not in st.session_state.doc_ids:
                                                st.session_state.doc_ids.append(doc_id)
                                                st.session_state.uploaded_files.append({
                                                    "id": doc_id,
                                                    "name": doc["file_name"],
                                                    "chunks": doc["chunks"],
                                                    "file_type": doc["file_type"],
                                                    "file_size": doc["file_size"],
                                                    "created_at": doc["created_at"]
                                                })
                                    st.success("✅ Login successful!")
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.error("Failed to get user info")
                            else:
                                st.error("❌ Invalid email or password")
            
            st.markdown("---")
            st.markdown("Don't have an account? Click the **Register** tab above.")

def register_page():
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h1>📝 Create Account</h1>
            <p style='color: #666;'>Join and start using RAG Assistant today</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            st.markdown("---")
            with st.form("register_form"):
                username = st.text_input("👤 Username", placeholder="Choose a username")
                email = st.text_input("📧 Email", placeholder="your@email.com")
                password = st.text_input("🔑 Password", type="password", placeholder="Min 6 characters")
                confirm_password = st.text_input("🔑 Confirm Password", type="password", placeholder="Re-enter password")
                submit = st.form_submit_button("📝 Register", use_container_width=True)
                
                if submit:
                    if not all([username, email, password, confirm_password]):
                        st.warning("Please fill in all fields")
                    elif password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        with st.spinner("Creating account..."):
                            response = api_register(username, email, password)
                            if response.status_code == 200:
                                st.success("✅ Registration successful! Please login.")
                                time.sleep(1)
                                st.session_state.page = "login"
                                st.rerun()
                            else:
                                error_detail = response.json().get("detail", "Registration failed")
                                st.error(f"❌ {error_detail}")

def document_management():
    st.sidebar.header("📄 Document Management")
    
    with st.sidebar.expander("📤 Upload Document", expanded=True):
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "doc"],
            key="file_uploader",
            help="Supported formats: PDF, DOCX, DOC"
        )
        
        if uploaded_file and st.button("📤 Upload", use_container_width=True):
            with st.spinner("Processing document..."):
                response = api_ingest_document(st.session_state.token, uploaded_file)
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Upload successful!")
                    st.info(f"📄 Document ID: {result['document_id']}")
                    st.info(f"📊 Chunks: {result['num_chunks']}")
                    st.info(f"⏱️ Time: {result['elapsed_time_seconds']:.2f}s")
                    
                    doc_id = str(result['document_id'])
                    if doc_id not in st.session_state.doc_ids:
                        st.session_state.doc_ids.append(doc_id)
                        st.session_state.uploaded_files.append({
                            "id": doc_id,
                            "name": uploaded_file.name,
                            "chunks": result['num_chunks'],
                            "file_type": Path(uploaded_file.name).suffix,
                            "file_size": len(uploaded_file.getvalue()),
                            "created_at": datetime.now().isoformat()
                        })
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ Upload failed: {response.json().get('detail', 'Unknown error')}")
    
    if st.session_state.uploaded_files:
        st.sidebar.subheader("📚 My Documents")
        df_data = []
        for doc in st.session_state.uploaded_files:
            df_data.append({
                "Name": doc['name'][:30] + "..." if len(doc['name']) > 30 else doc['name'],
                "Chunks": doc['chunks'],
                "ID": doc['id'][:8] + "..."
            })
        if df_data:
            df = pd.DataFrame(df_data)
            st.sidebar.dataframe(df, use_container_width=True)
        
        if st.sidebar.button("🗑️ Clear All Documents", use_container_width=True):
            st.session_state.doc_ids = []
            st.session_state.uploaded_files = []
            st.rerun()
    else:
        st.sidebar.info("📤 No documents uploaded yet")

def chat_interface():
    st.title("💬 RAG Assistant")
    
    if st.session_state.doc_ids:
        st.caption(f"📚 Using {len(st.session_state.doc_ids)} document(s) for context")
    else:
        st.warning("⚠️ Please upload at least one document to start querying")
    
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    
    if st.session_state.doc_ids:
        query = st.chat_input("Ask a question about your documents...")
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = api_query(
                        st.session_state.token,
                        query,
                        st.session_state.doc_ids,
                        st.session_state.conversation_id
                    )
                    if response.status_code == 200:
                        result = response.json()
                        answer = result["answer"]
                        if not st.session_state.conversation_id:
                            st.session_state.conversation_id = result["conversation_id"]
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        error_msg = f"❌ Error: {response.json().get('detail', 'Unknown error')}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()
    with col2:
        if st.session_state.messages:
            if st.button("🗑️ Clear Messages", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

def documents_page():
    st.title("📄 Document Manager")
    
    if st.session_state.uploaded_files:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Your Documents")
            for i, doc in enumerate(st.session_state.uploaded_files):
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.write(f"📄 **{doc['name']}**")
                        st.caption(f"ID: {doc['id']}")
                    with col2:
                        st.write(f"Chunks: {doc['chunks']}")
                    with col3:
                        st.write(f"Type: {doc.get('file_type', 'N/A')}")
                    with col4:
                        if st.button("🗑️", key=f"del_{i}"):
                            if doc['id'] in st.session_state.doc_ids:
                                st.session_state.doc_ids.remove(doc['id'])
                            st.session_state.uploaded_files.pop(i)
                            st.rerun()
                    st.divider()
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.doc_ids = []
                st.session_state.uploaded_files = []
                st.rerun()
    else:
        st.info("No documents uploaded yet. Go to the sidebar to upload documents.")

def main():
    st.set_page_config(
        page_title="RAG Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    st.markdown("""
        <style>
        .stChatInput {
            position: fixed;
            bottom: 0;
            background: white;
            padding: 1rem;
            width: 80%;
            z-index: 999;
        }
        .main > div {
            padding-bottom: 6rem;
        }
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if not st.session_state.token:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        with tab1:
            login_page()
        with tab2:
            register_page()
    else:
        with st.sidebar:
            st.sidebar.success(f"👤 {st.session_state.user.get('username', 'User')}")
            st.sidebar.caption(f"📧 {st.session_state.user.get('email', '')}")
            st.sidebar.divider()
            
            page = st.radio(
                "Navigation",
                ["💬 Chat", "📄 Documents"],
                index=0
            )
            
            st.sidebar.divider()
            
            if page == "📄 Documents":
                document_management()
            else:
                st.sidebar.subheader("📄 Quick Actions")
                with st.sidebar.expander("📤 Upload Document"):
                    uploaded_file = st.file_uploader(
                        "Choose a file",
                        type=["pdf", "docx", "doc"],
                        key="sidebar_uploader"
                    )
                    if uploaded_file and st.button("Upload", use_container_width=True):
                        with st.spinner("Processing..."):
                            response = api_ingest_document(st.session_state.token, uploaded_file)
                            if response.status_code == 200:
                                result = response.json()
                                st.success("✅ Uploaded!")
                                doc_id = str(result['document_id'])
                                if doc_id not in st.session_state.doc_ids:
                                    st.session_state.doc_ids.append(doc_id)
                                    st.session_state.uploaded_files.append({
                                        "id": doc_id,
                                        "name": uploaded_file.name,
                                        "chunks": result['num_chunks'],
                                        "file_type": Path(uploaded_file.name).suffix,
                                        "file_size": len(uploaded_file.getvalue()),
                                        "created_at": datetime.now().isoformat()
                                    })
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error("Upload failed")
            
            st.sidebar.divider()
            
            if st.sidebar.button("🚪 Logout", use_container_width=True):
                st.session_state.token = None
                st.session_state.user = None
                st.session_state.messages = []
                st.session_state.conversation_id = None
                st.session_state.doc_ids = []
                st.session_state.uploaded_files = []
                st.rerun()
        
        if page == "💬 Chat":
            chat_interface()
        else:
            documents_page()

if __name__ == "__main__":
    main()