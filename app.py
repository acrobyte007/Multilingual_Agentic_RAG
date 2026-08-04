import streamlit as st
import requests
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict
import os

# Configure page
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Change to your server URL

# Session State Initialization
def init_session_state():
    if "token" not in st.session_state:
        st.session_state.token = None
    if "user" not in st.session_state:
        st.session_state.user = None
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_doc_ids" not in st.session_state:
        st.session_state.current_doc_ids = []
    if "selected_docs" not in st.session_state:
        st.session_state.selected_docs = []

# API Calls
def api_request(method: str, endpoint: str, data: Optional[Dict] = None, files: Optional[Dict] = None):
    """Make API requests with authentication"""
    headers = {}
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            if files:
                response = requests.post(url, headers=headers, files=files, data=data)
            else:
                response = requests.post(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            return None
        
        return response
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to server. Please make sure the server is running.")
        return None

def login(email: str, password: str):
    """Login user and get token"""
    response = api_request("POST", "/auth/login", {"email": email, "password": password})
    if response and response.status_code == 200:
        data = response.json()
        st.session_state.token = data["access_token"]
        return True
    return False

def register_user(username: str, email: str, password: str):
    """Register new user"""
    response = api_request("POST", "/auth/register", {
        "username": username,
        "email": email,
        "password": password
    })
    return response and response.status_code == 200

def get_user_info():
    """Get current user info"""
    response = api_request("GET", "/auth/me")
    if response and response.status_code == 200:
        return response.json()
    return None

def get_user_documents():
    """Get user's documents"""
    response = api_request("GET", "/api/v1/documents/")
    if response and response.status_code == 200:
        data = response.json()
        return data.get("documents", [])
    return []

def upload_document(file):
    """Upload a document"""
    files = {"file": file}
    response = api_request("POST", "/api/v1/rag/ingest", files=files)
    if response and response.status_code == 200:
        return response.json()
    return None

def delete_document(document_id: str):
    """Delete a document"""
    response = api_request("DELETE", f"/api/v1/documents/{document_id}")
    return response and response.status_code == 200

def send_query(query: str, doc_ids: List[str], conversation_id: Optional[str] = None):
    """Send a query to the RAG system"""
    data = {
        "query": query,
        "doc_ids": doc_ids,
        "conversation_id": conversation_id
    }
    response = api_request("POST", "/api/v1/rag/response", data)
    if response and response.status_code == 200:
        return response.json()
    return None

# Login/Register Page
def show_login_page():
    st.title("📚 RAG Document Assistant")
    st.subheader("Login or Register")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if email and password:
                    with st.spinner("Logging in..."):
                        if login(email, password):
                            user_info = get_user_info()
                            if user_info:
                                st.session_state.user = user_info
                                st.rerun()
                        else:
                            st.error("Invalid email or password")
                else:
                    st.warning("Please fill all fields")
    
    with tab2:
        with st.form("register_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Register")
            
            if submitted:
                if username and email and password and confirm_password:
                    if password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        with st.spinner("Registering..."):
                            if register_user(username, email, password):
                                st.success("Registration successful! Please login.")
                            else:
                                st.error("Registration failed. User might already exist.")
                else:
                    st.warning("Please fill all fields")

# Main App
def main():
    init_session_state()
    
    # Check if user is logged in
    if not st.session_state.token:
        show_login_page()
        return
    
    # Get user info if not available
    if not st.session_state.user:
        user_info = get_user_info()
        if user_info:
            st.session_state.user = user_info
        else:
            # Token might be invalid
            st.session_state.token = None
            st.rerun()
    
    # Sidebar - Document Management
    with st.sidebar:
        st.header(f"👤 {st.session_state.user.get('username', 'User')}")
        st.caption(f"📧 {st.session_state.user.get('email', '')}")
        
        st.divider()
        
        # Upload Document
        st.subheader("📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "doc", "docx"],
            help="Supported formats: PDF, DOC, DOCX"
        )
        
        if uploaded_file:
            if st.button("Upload", type="primary", use_container_width=True):
                with st.spinner("Uploading document..."):
                    result = upload_document(uploaded_file)
                    if result:
                        st.success("Document uploaded successfully!")
                        st.session_state.documents = get_user_documents()
                        st.rerun()
                    else:
                        st.error("Failed to upload document")
        
        st.divider()
        
        # Document List
        st.subheader("📄 Your Documents")
        
        # Refresh button
        if st.button("🔄 Refresh List", use_container_width=True):
            st.session_state.documents = get_user_documents()
            st.rerun()
        
        # Get documents
        if not st.session_state.documents:
            st.session_state.documents = get_user_documents()
        
        if st.session_state.documents:
            for doc in st.session_state.documents:
                col1, col2 = st.columns([4, 1])
                with col1:
                    # Checkbox for document selection
                    doc_id = str(doc["id"])
                    is_selected = doc_id in st.session_state.selected_docs
                    if st.checkbox(
                        f"📎 {doc['file_name']}",
                        value=is_selected,
                        key=f"doc_{doc_id}"
                    ):
                        if doc_id not in st.session_state.selected_docs:
                            st.session_state.selected_docs.append(doc_id)
                    else:
                        if doc_id in st.session_state.selected_docs:
                            st.session_state.selected_docs.remove(doc_id)
                
                with col2:
                    # Delete button
                    if st.button("🗑️", key=f"del_{doc_id}", help="Delete document"):
                        if delete_document(doc_id):
                            st.success(f"Deleted {doc['file_name']}")
                            st.session_state.documents = get_user_documents()
                            if doc_id in st.session_state.selected_docs:
                                st.session_state.selected_docs.remove(doc_id)
                            st.rerun()
                        else:
                            st.error("Failed to delete document")
                
                # Show document details
                with st.expander(f"ℹ️ Details - {doc['file_name']}"):
                    st.caption(f"📏 Size: {doc['file_size'] / 1024:.1f} KB")
                    st.caption(f"📝 Chunks: {doc.get('chunks', 0)}")
                    st.caption(f"🌐 Language: {doc.get('primary_language', 'unknown')}")
                    st.caption(f"📅 Uploaded: {doc.get('created_at', '')[:10]}")
        else:
            st.info("No documents uploaded yet")
        
        st.divider()
        
        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    # Main Content Area
    # Header with conversation controls
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.title("💬 RAG Chat")
    with col2:
        if st.session_state.conversation_id:
            st.info(f"Conversation: {st.session_state.conversation_id[:8]}...")
    with col3:
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()
    
    # Selected Documents Info
    if st.session_state.selected_docs:
        selected_count = len(st.session_state.selected_docs)
        st.info(f"📌 Querying across {selected_count} selected document(s)")
    else:
        st.warning("⚠️ Please select at least one document to query")
    
    st.divider()
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "timestamp" in message:
                    st.caption(message["timestamp"])
    
    # Chat input at the bottom
    if st.session_state.selected_docs:
        user_input = st.chat_input("Ask a question about your documents...")
        
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            # Show user message immediately
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = send_query(
                        query=user_input,
                        doc_ids=st.session_state.selected_docs,
                        conversation_id=st.session_state.conversation_id
                    )
                    
                    if response:
                        # Update conversation ID for subsequent messages
                        if "conversation_id" in response:
                            st.session_state.conversation_id = response["conversation_id"]
                        
                        # Display response
                        st.markdown(response["answer"])
                        
                        # Store response in messages
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["answer"],
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
                    else:
                        st.error("Failed to get response. Please check if the server is running.")
    else:
        st.info("📚 Select documents from the sidebar to start asking questions")

if __name__ == "__main__":
    main()