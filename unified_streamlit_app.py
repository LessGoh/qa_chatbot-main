import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time
from langchain_pinecone import PineconeVectorStore

# Инициализация состояния
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def setup_pinecone():
    """Настройка Pinecone векторной базы данных"""
    try:
        pc = Pinecone()
        spec = ServerlessSpec(cloud='aws', region='us-east-1')
        
        # Создание индекса
        index_name = "streamlit-qa-bot"
        existing_indexes = [item['name'] for item in pc.list_indexes().indexes]
        
        if index_name in existing_indexes:
            pc.delete_index(index_name)
            st.info("Удаляем старый индекс...")
        
        pc.create_index(
            index_name,
            dimension=1536,  # dimensionality of text-embedding-ada-002
            metric='cosine',
            spec=spec
        )
        
        # Ожидание готовности индекса
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        
        index = pc.Index(index_name)
        embeddings = OpenAIEmbeddings()
        
        vectorstore = PineconeVectorStore(index, embeddings, "text")
        return vectorstore
    
    except Exception as e:
        st.error(f"Ошибка настройки Pinecone: {str(e)}")
        return None

def process_pdf(uploaded_file):
    """Обработка загруженного PDF файла"""
    try:
        # Сохранение временного файла
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Загрузка и обработка PDF
        pdf_loader = PyPDFLoader(tmp_file_path)
        pages = pdf_loader.load_and_split()
        
        # Разделение на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(pages)
        
        # Добавление документов в векторное хранилище
        if st.session_state.vectorstore:
            st.session_state.vectorstore.add_documents(documents=splits)
            
        # Удаление временного файла
        os.unlink(tmp_file_path)
        
        return True, len(splits)
    
    except Exception as e:
        return False, str(e)

def ask_question(question):
    """Получение ответа на вопрос"""
    try:
        if not st.session_state.vectorstore:
            return "Векторная база данных не настроена"
        
        retriever = st.session_state.vectorstore.as_retriever()
        
        prompt_rag = PromptTemplate.from_template(
            "Ответь на вопрос: {question} основываясь на контексте: {context}. "
            "Укажи источники информации из метаданных."
        )
        
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_rag
            | llm
        )
        
        result = rag_chain.invoke(question)
        return result.content
    
    except Exception as e:
        return f"Ошибка при обработке вопроса: {str(e)}"

# Интерфейс Streamlit
st.title("📄 Document Question Answering System")

# Боковая панель с настройками
st.sidebar.header("⚙️ Настройки")

# Ввод API ключей
openai_key = st.sidebar.text_input(
    "🔑 OpenAI API Key", 
    type="password",
    help="Введите ваш OpenAI API ключ"
)

pinecone_key = st.sidebar.text_input(
    "🌲 Pinecone API Key", 
    type="password",
    help="Введите ваш Pinecone API ключ"
)

# Установка API ключей
if st.sidebar.button("💾 Сохранить API ключи"):
    if openai_key and pinecone_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["PINECONE_API_KEY"] = pinecone_key
        
        with st.spinner("Настройка векторной базы данных..."):
            st.session_state.vectorstore = setup_pinecone()
        
        if st.session_state.vectorstore:
            st.sidebar.success("✅ API ключи сохранены и Pinecone настроен!")
        else:
            st.sidebar.error("❌ Ошибка настройки Pinecone")
    else:
        st.sidebar.error("❌ Пожалуйста, введите оба API ключа")

# Основной интерфейс
if openai_key and pinecone_key and st.session_state.vectorstore:
    
    # Загрузка файла
    st.header("📎 Загрузка документа")
    uploaded_file = st.file_uploader(
        "Выберите PDF файл", 
        type="pdf",
        help="Загрузите PDF документ для анализа"
    )
    
    if uploaded_file and st.button("🚀 Обработать документ"):
        with st.spinner("Обработка PDF документа..."):
            success, result = process_pdf(uploaded_file)
        
        if success:
            st.success(f"✅ Документ успешно обработан! Создано {result} фрагментов текста.")
            st.session_state.documents_processed = True
        else:
            st.error(f"❌ Ошибка обработки документа: {result}")
    
    # Секция вопросов
    if st.session_state.documents_processed:
        st.header("❓ Задайте вопрос")
        
        question = st.text_input(
            "Введите ваш вопрос о документе:",
            placeholder="Например: О чем этот документ?"
        )
        
        if st.button("🔍 Получить ответ") and question:
            with st.spinner("Поиск ответа..."):
                answer = ask_question(question)
            
            st.subheader("💬 Ответ:")
            st.write(answer)
    
    else:
        st.info("👆 Загрузите и обработайте PDF документ, чтобы начать задавать вопросы.")

else:
    st.warning("🔐 Пожалуйста, введите API ключи в боковой панели для начала работы.")
    
    # Инструкции
    with st.expander("📋 Инструкции по использованию"):
        st.markdown("""
        1. **Получите API ключи:**
           - OpenAI: https://platform.openai.com/api-keys
           - Pinecone: https://app.pinecone.io/
        
        2. **Введите ключи** в боковой панели и нажмите "Сохранить"
        
        3. **Загрузите PDF документ** и нажмите "Обработать"
        
        4. **Задавайте вопросы** о содержании документа
        """)

# Информация о приложении
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 О приложении")
st.sidebar.markdown("RAG-система для анализа PDF документов с использованием OpenAI и Pinecone.")
