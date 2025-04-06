from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set Groq API Key
os.environ["GROQ_API_KEY"] = "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"

# Define request schema
class ChatRequest(BaseModel):
    query: str

# Global variables to store loaded resources
index = None
retriever = None
groq_client = None

@app.on_event("startup")
async def startup_event():
    global index, retriever, groq_client
    
    try:
        logger.info("Loading documents and initializing models...")
        
        # Initialize Groq client first (it's faster)
        groq_client = Groq(model="llama-3.1-8b-instant")
        logger.info("Groq client initialized")
        
        # Load embedding model
        embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Embedding model loaded")
        
        # Load documents with error handling
        try:
            logger.info("Starting to load PDF document...")
            reader = SimpleDirectoryReader(input_files=["The Emperor of All Maladies_ A Biography of Cancer final.pdf"])
            documents = reader.load_data()
            logger.info(f"Successfully loaded {len(documents)} document chunks")
            
            # Create vector index for documents
            logger.info("Creating vector index...")
            index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model)
            retriever = index.as_retriever(similarity_top_k=3)
            logger.info("Vector index created successfully")
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            # Even if document loading fails, we can still provide a fallback
            logger.warning("Using fallback mode without document retrieval")
        
        logger.info("Initialization complete")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # We don't re-raise to allow the API to start even with partial functionality

@app.get("/")
def read_root():
    return {"message": "Welcome to the Cancer Information Chatbot API!"}

@app.get("/health")
def health_check():
    status = {
        "groq_client": groq_client is not None,
        "index": index is not None,
        "retriever": retriever is not None
    }
    
    if not status["groq_client"]:
        raise HTTPException(status_code=503, detail="LLM service not initialized")
        
    return {
        "status": "healthy" if all(status.values()) else "degraded",
        "components": status,
        "message": "Full functionality available" if all(status.values()) else "Limited functionality available"
    }

@app.post("/chat")
def chat(request: ChatRequest):
    if groq_client is None:
        raise HTTPException(status_code=503, detail="LLM service not initialized")
    
    try:
        # Check if we have retrieval capability
        if retriever is not None and index is not None:
            # Retrieve relevant document chunks
            retrieved_docs = retriever.retrieve(request.query)
            context = "\n".join([doc.text for doc in retrieved_docs])
            
            # Create prompt for LLM with context
            prompt = f"""
            You are a helpful assistant knowledgeable about cancer. Use the following information to answer the question. 
            Be accurate, helpful, and concise.
            
            CONTEXT INFORMATION:
            {context}
            
            QUESTION: {request.query}
            
            ANSWER:
            """
        else:
            # Fallback mode without retrieval
            prompt = f"""
            You are a helpful assistant knowledgeable about cancer. Answer the following question
            to the best of your ability, being accurate, helpful, and concise.
            
            QUESTION: {request.query}
            
            ANSWER:
            """
        
        # Get response from LLM
        response = groq_client.complete(prompt).text.strip()
        
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Run using: uvicorn app:app --host 0.0.0.0 --port 8000