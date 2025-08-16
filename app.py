import streamlit as st
import os
import chromadb
import time
from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama

# --- App Configuration ---
st.set_page_config(
    page_title="Visual RAG Demo",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Visual Document Analysis RAG")

# --- Helper Functions & Model Loading ---

@st.cache_resource
def get_embedding_model():
    """Loads the embedding model from cache."""
    st.write("Loading embedding model (all-MiniLM-L6-v2)...")
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_llm():
    """Loads the LLM from cache."""
    st.write("Loading Local LLM (Phi-3)...")
    # Make sure Ollama is running with the 'phi3' model pulled
    return ChatOllama(model="phi3", temperature=0.1)

def setup_rag_pipeline(file_path, embedding_model):
    """Processes a document and sets up the RAG pipeline."""
    with st.spinner("Analyzing document and building index... ⏳"):
        # 1. Partition the document using unstructured
        elements = partition(file_path, strategy="auto")
        raw_text = "\n\n".join([str(el) for el in elements])

        # 2. Use a more effective chunking strategy
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        text_chunks = text_splitter.split_text(raw_text)

        # 3. Setup ChromaDB client and collection
        client = chromadb.Client()
        # Delete collection if it exists, to ensure a fresh start
        try:
            client.delete_collection(name="visual_rag_collection")
        except Exception:
            pass
        collection = client.get_or_create_collection(name="visual_rag_collection")

        # 4. Generate embeddings and add to the collection
        embeddings = embedding_model.encode(text_chunks, show_progress_bar=True).tolist()
        collection.add(
            embeddings=embeddings,
            documents=text_chunks,
            ids=[f"id_{i}" for i in range(len(text_chunks))]
        )

    # Store collection in session state
    st.session_state.rag_collection = collection
    st.session_state.rag_ready = True
    st.success("Document processed and indexed successfully!")

# --- Main App Logic ---
embedding_model = get_embedding_model()
llm = get_llm()

with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF, PNG, or JPG file",
        type=["pdf", "png", "jpg"],
        label_visibility="collapsed"
    )

if uploaded_file:
    # Process a new file if it's different from the one already processed
    if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        setup_rag_pipeline(file_path, embedding_model)
        st.session_state.processed_file = uploaded_file.name
        os.remove(file_path)

if "rag_ready" in st.session_state:
    st.header("2. Ask a Question")
    user_query = st.text_input("Enter your question about the document:", key="query_input")

    if user_query:
        with st.spinner("Searching for answers... 🔍"):
            start_time = time.time()

            # 1. RETRIEVAL: Embed the query and retrieve relevant chunks with scores
            query_embedding = embedding_model.encode([user_query]).tolist()
            results = st.session_state.rag_collection.query(
                query_embeddings=query_embedding,
                n_results=5,
                include=["documents", "distances"]
            )
            retrieved_context = "\n\n---\n\n".join(results['documents'][0])

            # 2. GENERATION: Create a prompt and generate an answer
            prompt = f"""
            You are a helpful assistant. Use the following context to answer the question.
            If the answer is not in the context, say "I cannot find the answer in the document."

            CONTEXT:
            {retrieved_context}

            QUESTION:
            {user_query}

            ANSWER:
            """

            response = llm.invoke(prompt)
            end_time = time.time()

            # Display the answer and metrics
            st.subheader("Answer:")
            st.markdown(response.content)
            st.info(f"Time to answer: {end_time - start_time:.2f} seconds")

            # Display relevance scores and context in an expander
            with st.expander("See Retrieved Context and Relevance Scores"):
                for doc, dist in zip(results['documents'][0], results['distances'][0]):
                    st.write(f"**Relevance Score (distance):** {dist:.4f}")
                    st.write(doc)
                    st.divider()

else:
    st.info("Please upload a document in the sidebar to get started.")