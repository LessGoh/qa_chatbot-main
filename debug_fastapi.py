from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
import traceback
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from fast_back import setup_pinecone, format_docs

app = FastAPI()
vectorstore = None

class APIKey(BaseModel):
    api_key: str

class Question(BaseModel):
    question: str

def create_vectorstore(file_path):
    """Create vectorstore from uploaded PDF with detailed error handling"""
    try:
        print(f"=== Starting document processing ===")
        print(f"File path: {file_path}")
        
        if file_path is None:
            print("ERROR: No file provided")
            return "No file provided"
            
        if vectorstore is None:
            print("ERROR: Vectorstore not initialized")
            return "Pinecone vectorstore not initialized. Please set Pinecone API key first."
            
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OpenAI API key not set")
            return "OpenAI API key not set. Please set it first."
        
        # Check if file exists and is readable
        if not os.path.exists(file_path):
            print(f"ERROR: File does not exist: {file_path}")
            return "File does not exist"
            
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size} bytes")
        
        if file_size == 0:
            print("ERROR: File is empty")
            return "Uploaded file is empty"
            
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            print("ERROR: File too large")
            return "File too large (max 10MB)"
        
        # Read the PDF file
        print(f"Loading PDF: {file_path}")
        try:
            pdf_loader = PyPDFLoader(file_path)
            pages = pdf_loader.load_and_split()
            print(f"PDF loader created, attempting to load pages...")
        except Exception as e:
            print(f"ERROR loading PDF: {str(e)}")
            return f"Error reading PDF file: {str(e)}"
        
        if not pages:
            print("ERROR: No content found in PDF")
            return "No content found in PDF file. The file might be corrupted or password-protected."
        
        print(f"✓ Loaded {len(pages)} pages from PDF")
        
        # Check content
        total_content = ""
        for i, page in enumerate(pages[:3]):  # Check first 3 pages
            content = page.page_content.strip()
            total_content += content
            print(f"Page {i+1} content length: {len(content)} chars")
            if len(content) > 0:
                print(f"Page {i+1} preview: {content[:100]}...")
        
        if len(total_content.strip()) == 0:
            print("ERROR: PDF contains no readable text")
            return "PDF contains no readable text content"
        
        # Split text into chunks
        print("Splitting text into chunks...")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=100
            )
            splits = text_splitter.split_documents(pages)
            print(f"✓ Created {len(splits)} text chunks")
            
            if len(splits) == 0:
                print("ERROR: No text chunks created")
                return "Could not create text chunks from document"
                
        except Exception as e:
            print(f"ERROR splitting text: {str(e)}")
            return f"Error splitting text: {str(e)}"
        
        # Test OpenAI connection
        print("Testing OpenAI embeddings...")
        try:
            embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
            # Test with a small text
            test_embedding = embeddings.embed_query("test")
            print(f"✓ OpenAI embeddings working, dimension: {len(test_embedding)}")
        except Exception as e:
            print(f"ERROR with OpenAI embeddings: {str(e)}")
            return f"Error with OpenAI API: {str(e)}"
        
        # Add documents to vectorstore in smaller batches
        print("Adding documents to vectorstore...")
        try:
            batch_size = 10  # Process in smaller batches
            for i in range(0, len(splits), batch_size):
                batch = splits[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}: {len(batch)} documents")
                vectorstore.add_documents(documents=batch, async_req=False)
                print(f"✓ Batch {i//batch_size + 1} added successfully")
            
            print("✓ All documents added to vectorstore successfully")
            
        except Exception as e:
            print(f"ERROR adding to vectorstore: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")
            return f"Error adding documents to vectorstore: {str(e)}"
        
        print("=== Document processing completed successfully ===")
        return "Data Successfully ingested"
        
    except Exception as e:
        error_msg = f"Unexpected error in create_vectorstore: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"Full traceback: {traceback.format_exc()}")
        return error_msg

def doc_qa(question):
    """Answer questions using the vectorstore"""
    try:
        print(f"=== Processing question: {question} ===")
        
        if vectorstore is None:
            print("ERROR: Vectorstore not initialized")
            return "Vectorstore not initialized. Please upload a document first."
            
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OpenAI API key not set")
            return "OpenAI API key not set. Please set it first."
        
        print("Creating retriever...")
        retriever = vectorstore.as_retriever()
        
        print("Setting up RAG chain...")
        prompt_rag = PromptTemplate.from_template(
            "Generate answers for given question: {question} based on the context {context}. "
            "In generated response provide reference to metadata from which context used in generating response"
        )
        
        llm = ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_rag
            | llm
        )
        
        print("Invoking RAG chain...")
        res = rag_chain.invoke(question)
        
        print("✓ Question processed successfully")
        return str(res.content)
        
    except Exception as e:
        error_msg = f"Error answering question: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"Full traceback: {traceback.format_exc()}")
        return error_msg

@app.get("/")
async def root():
    return {"message": "QA Chatbot API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "pinecone_key_set": bool(os.getenv("PINECONE_API_KEY")),
        "vectorstore_initialized": vectorstore is not None
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        print(f"\n=== NEW FILE UPLOAD REQUEST ===")
        print(f"Filename: {file.filename}")
        print(f"Content type: {file.content_type}")
        
        if file.filename == '':
            raise HTTPException(status_code=400, detail="No selected file")
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Create temporary file
        temp_file_path = tempfile.mkstemp(suffix='.pdf')[1]
        print(f"Temporary file path: {temp_file_path}")
        
        try:
            # Read and save file
            content = await file.read()
            print(f"Read {len(content)} bytes from upload")
            
            with open(temp_file_path, "wb") as f:
                f.write(content)
            
            print(f"File saved to: {temp_file_path}")
            
            # Process file
            result = create_vectorstore(temp_file_path)
            
            print(f"Processing result: {result}")
            return JSONResponse(content={"message": result})
            
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            print(f"ERROR: {error_msg}")
            print(f"Full traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=error_msg)
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    print(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                print(f"Warning: Could not clean up temp file: {e}")
                
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error in upload_file: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/ask")
async def ask_question(question: Question):
    try:
        if not question.question or question.question.strip() == "":
            raise HTTPException(status_code=400, detail="No question provided")
        
        answer = doc_qa(question.question.strip())
        return JSONResponse(content={"answer": answer})
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error in ask_question: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/set_openai_api_key")
async def set_openai_api_key(api_key: APIKey):
    try:
        if not api_key.api_key or api_key.api_key.strip() == "":
            raise HTTPException(status_code=400, detail="No API key provided")
        
        os.environ["OPENAI_API_KEY"] = api_key.api_key.strip()
        print("✓ OpenAI API key set successfully")
        
        return JSONResponse(content={"message": "OpenAI API key set successfully"})
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR setting OpenAI API key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting API key: {str(e)}")

@app.post("/set_pinecone_api_key")
async def set_pinecone_api_key(api_key: APIKey):
    global vectorstore
    try:
        if not api_key.api_key or api_key.api_key.strip() == "":
            raise HTTPException(status_code=400, detail="No API key provided")
        
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=400, 
                detail="Please set OpenAI API key first"
            )
        
        os.environ["PINECONE_API_KEY"] = api_key.api_key.strip()
        print("Pinecone API key set, initializing vectorstore...")
        
        vectorstore = setup_pinecone()
        print("✓ Pinecone vectorstore initialized successfully")
        
        return JSONResponse(content={"message": "Pinecone API key set and vectorstore initialized successfully"})
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error setting Pinecone API key: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")