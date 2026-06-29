import streamlit as st
import requests
import json
from typing import Optional, List
import time
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Update with your FastAPI server URL

# Session State Initialization
def init_session_state():
    if "token" not in st.session_state:
        st.session_state.token = None
    if "user" not in st.session_state:
        st.session_state.user = None
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "doc_ids" not in st.session_state:
        st.session_state.doc_ids = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

# API Calls
def api_register(username: str, email: str, password: str):
    response = requests.post(
        f"{API_BASE_URL}/auth/register",
        json={
            "username": username,
            "email": email,
            "password": password
        }
    )
    return response

def api_login(email: str, password: str):
    response = requests.post(
        f"{API_BASE_URL}/auth/login",
        json={
            "email": email,
            "password": password
        }
    )
    return response

def api_get_user(token: str):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(
        f"{API_BASE_URL}/auth/me",
        headers=headers
    )
    return response

def api_ingest_document(token: str, file, document_id: Optional[str] = None):
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": (file.name, file.getvalue(), file.type)}
    data = {}
    if document_id:
        data["document_id"] = document_id
    
    response = requests.post(
        f"{API_BASE_URL}/api/v1/rag/ingest",
        headers=headers,
        files=files,
        data=data
    )
    return response

def api_query(token: str, query: str, doc_ids: List[str], conversation_id: Optional[str] = None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = {
        "query": query,
        "doc_ids": doc_ids
    }
    if conversation_id:
        data["conversation_id"] = conversation_id
    
    response = requests.post(
        f"{API_BASE_URL}/api/v1/rag/response",
        headers=headers,
        json=data
    )
    return response

# UI Pages
def login_page():
    st.title("🔐 Login")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            with st.spinner("Logging in..."):
                response = api_login(email, password)
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.token = data["access_token"]
                    
                    # Get user info
                    user_response = api_get_user(st.session_state.token)
                    if user_response.status_code == 200:
                        st.session_state.user = user_response.json()
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Failed to get user info")
                else:
                    st.error("Invalid email or password")

def register_page():
    st.title("📝 Register")
    
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                with st.spinner("Registering..."):
                    response = api_register(username, email, password)
                    if response.status_code == 200:
                        st.success("Registration successful! Please login.")
                        st.session_state.page = "login"
                        st.rerun()
                    else:
                        error_detail = response.json().get("detail", "Registration failed")
                        st.error(error_detail)

def dashboard_page():
    st.title("🤖 RAG Assistant")
    st.sidebar.title(f"Welcome, {st.session_state.user.get('username', 'User')}!")
    
    # Sidebar - Document Management
    with st.sidebar:
        st.subheader("📄 Document Management")
        
        # Upload Document
        with st.expander("Upload Document", expanded=True):
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["pdf", "docx", "doc"],
                key="file_uploader"
            )
            
            doc_id = st.text_input("Document ID (optional)", placeholder="Auto-generate if empty")
            
            if uploaded_file and st.button("Upload Document"):
                with st.spinner("Uploading and processing document..."):
                    response = api_ingest_document(
                        st.session_state.token,
                        uploaded_file,
                        doc_id if doc_id else None
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Document uploaded successfully!")
                        st.info(f"Document ID: {result['document_id']}")
                        st.info(f"Chunks: {result['num_chunks']}")
                        st.info(f"Time: {result['elapsed_time_seconds']:.2f}s")
                        
                        # Add to session
                        st.session_state.doc_ids.append(result['document_id'])
                        st.session_state.uploaded_files.append({
                            "id": result['document_id'],
                            "name": uploaded_file.name,
                            "chunks": result['num_chunks']
                        })
                    else:
                        st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
        
        # Document List
        if st.session_state.uploaded_files:
            st.subheader("📚 Uploaded Documents")
            for doc in st.session_state.uploaded_files:
                st.text(f"📄 {doc['name']} (ID: {doc['id'][:8]}...)")
        
        # Logout
        if st.button("🚪 Logout"):
            st.session_state.token = None
            st.session_state.user = None
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()
    
    # Main Chat Interface
    # Chat History
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
    
    # Chat Input
    if st.session_state.doc_ids:
        st.info(f"📚 Using {len(st.session_state.doc_ids)} document(s) for context")
        
        # Query Input
        query = st.chat_input("Ask a question about your documents...")
        
        if query:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.write(query)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get RAG response
                    response = api_query(
                        st.session_state.token,
                        query,
                        st.session_state.doc_ids,
                        st.session_state.conversation_id
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result["answer"]
                        
                        # Update conversation ID
                        if not st.session_state.conversation_id:
                            st.session_state.conversation_id = result["conversation_id"]
                        
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        error_msg = f"Error: {response.json().get('detail', 'Unknown error')}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        st.info("📤 Please upload at least one document to start querying.")
    
    # Conversation Controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()
    
    with col2:
        if st.session_state.messages:
            if st.button("🗑️ Clear Messages"):
                st.session_state.messages = []
                st.rerun()

def main():
    st.set_page_config(
        page_title="RAG Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session
    init_session_state()
    
    # Custom CSS
    st.markdown("""
        <style>
        .stChatInput {
            position: fixed;
            bottom: 0;
            background: white;
            padding: 1rem;
            width: 80%;
        }
        .main > div {
            padding-bottom: 6rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Authentication Check
    if not st.session_state.token:
        # Show login/register tabs
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            login_page()
        
        with tab2:
            register_page()
    else:
        # Show main dashboard
        dashboard_page()

if __name__ == "__main__":
    main()