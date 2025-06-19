import streamlit as st
import os
import tempfile
import time
from typing import List

try:
    from langchain.document_loaders import PyPDFLoader
except ImportError:
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        # Fallback to pypdf direct usage
        import pypdf
        from langchain.schema import Document
        
        class PyPDFLoader:
            def __init__(self, file_path):
                self.file_path = file_path
            
            def load_and_split(self):
                documents = []
                with open(self.file_path, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text.strip():
                            doc = Document(
                                page_content=text,
                                metadata={"source": self.file_path, "page": page_num}
                            )
                            documents.append(doc)
                return documents

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone as LangchainPinecone
import pinecone

# Инициализация состояния сессии
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'pinecone_initialized' not in st.session_state:
    st.session_state.pinecone_initialized = False

def format_docs(docs):
    """Форматирование документов для RAG"""
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_pinecone(api_key: str):
    """Инициализация Pinecone"""
    try:
        pinecone.init(
            api_key=api_key,
            environment="us-east-1-aws"  # Измените на ваш регион
        )
        
        index_name = "streamlit-qa-bot"
        
        # Проверяем существование индекса
        if index_name not in pinecone.list_indexes():
            # Создаем новый индекс
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # Размерность для text-embedding-ada-002
                metric="cosine"
            )
            
            # Ждем готовности индекса
            while index_name not in pinecone.list_indexes():
                time.sleep(1)
        
        return True, index_name
    
    except Exception as e:
        return False, str(e)

def setup_vectorstore(api_key: str):
    """Настройка векторного хранилища"""
    try:
        success, result = initialize_pinecone(api_key)
        if not success:
            return False, f"Ошибка инициализации Pinecone: {result}"
        
        index_name = result
        embeddings = OpenAIEmbeddings()
        
        # Создаем векторное хранилище
        vectorstore = LangchainPinecone.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        return True, vectorstore
    
    except Exception as e:
        return False, str(e)

def process_pdf_file(uploaded_file):
    """Обработка загруженного PDF файла"""
    try:
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Загружаем PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load_and_split()
        
        # Разделяем на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        texts = text_splitter.split_documents(documents)
        
        # Добавляем в векторное хранилище
        if st.session_state.vectorstore:
            st.session_state.vectorstore.add_documents(texts)
        
        # Удаляем временный файл
        os.unlink(tmp_file_path)
        
        return True, len(texts)
    
    except Exception as e:
        return False, str(e)

def get_answer(question: str):
    """Получение ответа на вопрос"""
    try:
        if not st.session_state.vectorstore:
            return "Векторное хранилище не настроено"
        
        # Создаем retriever
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        # Создаем промпт
        template = """Используй следующий контекст для ответа на вопрос.
        Если ты не знаешь ответ, скажи что не знаешь, не придумывай ответ.

        {context}

        Вопрос: {question}
        Ответ:"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Создаем LLM
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        
        # Создаем цепочку RAG
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        
        # Получаем ответ
        result = rag_chain.invoke(question)
        return result.content
    
    except Exception as e:
        return f"Ошибка при получении ответа: {str(e)}"

# Основной интерфейс Streamlit
st.set_page_config(
    page_title="Document QA System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Question Answering System")
st.markdown("---")

# Боковая панель для настроек
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # OpenAI API Key
    openai_key = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        help="Введите ваш OpenAI API ключ"
    )
    
    # Pinecone API Key
    pinecone_key = st.text_input(
        "🌲 Pinecone API Key",
        type="password",
        help="Введите ваш Pinecone API ключ"
    )
    
    # Кнопка инициализации
    if st.button("🚀 Инициализировать систему"):
        if not openai_key:
            st.error("Введите OpenAI API ключ")
        elif not pinecone_key:
            st.error("Введите Pinecone API ключ")
        else:
            # Устанавливаем переменные окружения
            os.environ["OPENAI_API_KEY"] = openai_key
            
            with st.spinner("Настройка системы..."):
                success, result = setup_vectorstore(pinecone_key)
                
                if success:
                    st.session_state.vectorstore = result
                    st.session_state.pinecone_initialized = True
                    st.success("✅ Система инициализирована!")
                else:
                    st.error(f"❌ Ошибка: {result}")

# Основной контент
if st.session_state.pinecone_initialized:
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📎 Загрузка документа")
        uploaded_file = st.file_uploader(
            "Выберите PDF файл",
            type="pdf",
            help="Загрузите PDF документ для анализа"
        )
        
        if uploaded_file and st.button("📄 Обработать документ"):
            with st.spinner("Обработка документа..."):
                success, result = process_pdf_file(uploaded_file)
            
            if success:
                st.success(f"✅ Документ обработан! Создано {result} фрагментов.")
                st.session_state.documents_processed = True
            else:
                st.error(f"❌ Ошибка: {result}")
    
    with col2:
        st.header("❓ Вопросы и ответы")
        
        if st.session_state.documents_processed:
            question = st.text_area(
                "Задайте вопрос о документе:",
                height=100,
                placeholder="Например: О чем этот документ?"
            )
            
            if st.button("🔍 Получить ответ") and question:
                with st.spinner("Поиск ответа..."):
                    answer = get_answer(question)
                
                st.subheader("💬 Ответ:")
                st.write(answer)
        else:
            st.info("👈 Сначала загрузите и обработайте PDF документ")

else:
    st.warning("🔐 Пожалуйста, введите API ключи и инициализируйте систему в боковой панели")
    
    with st.expander("📋 Инструкции"):
        st.markdown("""
        ### Как получить API ключи:
        
        1. **OpenAI API Key:**
           - Зайдите на https://platform.openai.com/api-keys
           - Создайте новый API ключ
           
        2. **Pinecone API Key:**
           - Зайдите на https://app.pinecone.io/
           - В настройках найдите API ключ
        
        ### Как использовать:
        1. Введите оба API ключа в боковой панели
        2. Нажмите "Инициализировать систему"
        3. Загрузите PDF документ
        4. Задавайте вопросы о содержании
        """)

# Футер
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, LangChain, OpenAI, and Pinecone")
