import streamlit as st
import requests
import json

backend_url = "http://0.0.0.0:8000"

def set_openai_api_key(api_key):
    try:
        url = f"{backend_url}/set_openai_api_key"
        data = {"api_key": api_key}
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error setting OpenAI API key: {e}")
        return {"message": "Error setting API key"}
    except json.JSONDecodeError:
        st.error("Invalid response from server")
        return {"message": "Invalid response"}

def set_pinecone_api_key(api_key):
    try:
        url = f"{backend_url}/set_pinecone_api_key"
        data = {"api_key": api_key}
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error setting Pinecone API key: {e}")
        return {"message": "Error setting API key"}
    except json.JSONDecodeError:
        st.error("Invalid response from server")
        return {"message": "Invalid response"}

def upload_file(file):
    try:
        url = f"{backend_url}/upload"
        files = {"file": (file.name, file.getvalue())}
        response = requests.post(url, files=files)
        
        # Проверяем статус ответа
        if response.status_code == 200:
            try:
                return response.json()
            except json.JSONDecodeError:
                st.error("Server returned invalid JSON response")
                st.error(f"Response text: {response.text}")
                return {"message": "Error: Invalid server response"}
        else:
            st.error(f"Server error: {response.status_code}")
            st.error(f"Response: {response.text}")
            return {"message": f"Server error: {response.status_code}"}
            
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return {"message": f"Network error: {e}"}

def ask_question(question):
    try:
        url = f"{backend_url}/ask"
        data = {"question": question}
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error asking question: {e}")
        return {"answer": f"Error: {e}"}
    except json.JSONDecodeError:
        st.error("Invalid response from server")
        return {"answer": "Error: Invalid response"}

st.title("Document Question Answering System")

# Проверка подключения к backend
try:
    health_check = requests.get(f"{backend_url}/docs", timeout=5)
    if health_check.status_code == 200:
        st.success("✅ Backend connected successfully")
    else:
        st.warning("⚠️ Backend is running but may have issues")
except:
    st.error("❌ Cannot connect to backend. Make sure the container is running.")

# Step 1: Set API Key
st.sidebar.header("Step 1: Set OPENAI API Key")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password", key="openai_key")
if st.sidebar.button("Set OPENAI API Key"):
    if openai_api_key:
        with st.spinner("Setting up OPENAI API KEY..."): 
            result = set_openai_api_key(openai_api_key)
        st.sidebar.success(result.get("message", "API key set successfully"))
    else:
        st.sidebar.error("API key cannot be empty")

# Step 2: Set Pinecone API Key
st.sidebar.header("Step 2: Set Pinecone API Key")
pinecone_api_key = st.sidebar.text_input("Enter your Pinecone API key", type="password", key="pinecone_key")
if st.sidebar.button("Set Pinecone API Key"):
    if pinecone_api_key:
        with st.spinner("Setting up Pinecone API KEY and Vectorstore..."): 
            result = set_pinecone_api_key(pinecone_api_key)
        st.sidebar.success(result.get("message", "API key set successfully"))
    else:
        st.sidebar.error("API key cannot be empty")

# Step 3: Upload PDF
st.sidebar.header("Step 3: Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
if st.sidebar.button("Upload PDF"):
    if uploaded_file:
        # Проверяем, что API ключи установлены
        if not openai_api_key:
            st.sidebar.error("Please set OpenAI API key first")
        elif not pinecone_api_key:
            st.sidebar.error("Please set Pinecone API key first")
        else:
            with st.spinner("Uploading and processing PDF..."):
                result = upload_file(uploaded_file)
            if "Error" not in result.get("message", ""):
                st.sidebar.success(result.get("message", "File uploaded successfully"))
            else:
                st.sidebar.error(result.get("message", "Upload failed"))
    else:
        st.sidebar.error("Please select a PDF file to upload")

# Step 4: Ask Questions
st.header("Ask Questions about the Uploaded Document")
question = st.text_input("Enter your question")
if st.button("Ask Question"):
    if question:
        if not openai_api_key or not pinecone_api_key:
            st.error("Please set both API keys first")
        else:
            with st.spinner("Processing your question..."):
                result = ask_question(question)
            st.write("**Answer:**")
            st.write(result.get("answer", "No answer available"))
    else:
        st.error("Question cannot be empty")

# Инструкции для пользователя
st.sidebar.markdown("---")
st.sidebar.markdown("### Instructions:")
st.sidebar.markdown("1. Set your OpenAI API key")
st.sidebar.markdown("2. Set your Pinecone API key") 
st.sidebar.markdown("3. Upload a PDF document")
st.sidebar.markdown("4. Ask questions about the document")