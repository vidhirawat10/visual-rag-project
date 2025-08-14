import streamlit as st
import os
import chromadb
from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer

# --- App Configuration ---
st.set_page_config(
    page_title="Visual RAG Demo",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Visual Document Analysis RAG")

# --- Helper Functions ---
@st.cache_resource
def get_embedding_model():
    """Loads the embedding model from cache."""
    st.write("Loading embedding model... (this happens once)")
    return SentenceTransformer('all-MiniLM-L6-v2')

def setup_rag_pipeline(file_path, embedding_model):
    """Processes a document and sets up the RAG pipeline."""
    with st.spinner("Analyzing document and building index... ⏳"):
        # 1. Partition the document
        elements = partition(file_path, strategy="auto")

        # 2. Extract text from elements
        text_chunks = [str(el) for el in elements]

        # 3. Setup ChromaDB
        client = chromadb.Client()
        collection = client.get_or_create_collection(name="visual_rag_collection")

        # 4. Generate embeddings and add to ChromaDB
        embeddings = embedding_model.encode(text_chunks, show_progress_bar=True).tolist()
        collection.add(
            embeddings=embeddings,
            documents=text_chunks,
            ids=[f"id_{i}" for i in range(len(text_chunks))]
        )

    # Store the collection in session state for later use
    st.session_state.rag_collection = collection
    st.session_state.rag_ready = True
    st.success("Document processed and indexed successfully!")


# --- Main App Logic ---

# Load the embedding model once
embedding_model = get_embedding_model()

# File uploader in the sidebar
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF, PNG, or JPG file", type=["pdf", "png", "jpg"])

if uploaded_file is not None:
    # Check if the file has been processed already
    if "rag_ready" not in st.session_state:
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        setup_rag_pipeline(file_path, embedding_model)
        os.remove(file_path) # Clean up the temp file

# --- Query Interface (only shows up if RAG is ready) ---

if "rag_ready" in st.session_state:
    st.header("Ask a Question")
    user_query = st.text_input("Enter your question about the document:")

    if user_query:
        # This is where we will add the final retrieval and generation logic
        st.info("Query received! The final step is to retrieve context and generate an answer.")
        st.write("Coming up next...")

else:
    st.info("Please upload a document in the sidebar to get started.")