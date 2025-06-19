from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec, PodSpec  
import time  
from langchain_pinecone import PineconeVectorStore  
from langchain_openai import OpenAIEmbeddings
use_serverless = True  

def setup_pinecone():
    
    # configure client  
    pc = Pinecone()  
    if use_serverless:  
        spec = ServerlessSpec(cloud='aws', region='us-east-1')  
    else:  
        # if not using a starter index, you should specify a pod_type too  
        spec = PodSpec()  
    # check for and delete index if already exists  
    index_name = "sarvam-chat"
    items = pc.list_indexes().indexes
    existing_indexes = [item['name'] for item in items]
    if index_name in existing_indexes:  
        pc.delete_index(index_name)
        print("deleted_old")
    # create a new index  
    pc.create_index(  
        index_name,  
        dimension=1536,  # dimensionality of text-embedding-ada-002  
        metric='cosine',  
        spec=spec  
    )  
    # wait for index to be initialized  
    while not pc.describe_index(index_name).status['ready']:  
        time.sleep(1)  

    index = pc.Index(index_name)  
    index.describe_index_stats()  

    embeddings = OpenAIEmbeddings()
    text_field = "text"  

    vectorstore = PineconeVectorStore(  
        index, embeddings, text_field )  

    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)