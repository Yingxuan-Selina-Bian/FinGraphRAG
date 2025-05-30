import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
import textwrap

# Load environment variables
load_dotenv()

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or "neo4j"

# Constants
VECTOR_INDEX_NAME = 'form_10k_chunks'
VECTOR_NODE_LABEL = 'TextChunk'
VECTOR_SOURCE_PROPERTY = 'text'
VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'

# Initialize session state for API key
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

# Initialize the vector store
@st.cache_resource
def initialize_vector_store(api_key):
    try:
        vector_store = Neo4jVector.from_existing_graph(
            embedding=OpenAIEmbeddings(api_key=api_key),
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=VECTOR_INDEX_NAME,
            node_label=VECTOR_NODE_LABEL,
            text_node_properties=[VECTOR_SOURCE_PROPERTY],
            embedding_node_property=VECTOR_EMBEDDING_PROPERTY,
        )
        return vector_store
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return None

# Initialize the QA chain
@st.cache_resource
def initialize_qa_chain(_vector_store, api_key):
    try:
        retriever = _vector_store.as_retriever()
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=ChatOpenAI(temperature=0, api_key=api_key),
            chain_type="stuff",
            retriever=retriever
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error initializing QA chain: {str(e)}")
        return None

# Function to get answer from the QA chain
def get_answer(question, qa_chain):
    try:
        response = qa_chain({"question": question}, return_only_outputs=True)
        return response["answer"]
    except Exception as e:
        st.error(f"Error getting answer: {str(e)}")
        return None

# Set up the Streamlit interface
st.set_page_config(
    page_title="Financial Document Q&A System",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Financial Document Q&A System")
st.markdown("""
This application allows you to ask questions about companies based on their SEC 10-K filings.
The system uses a knowledge graph built from multiple companies' filings to provide accurate and contextual answers.
""")

# API Key input
api_key = st.text_input("Enter your OpenAI API Key:", 
                       type="password",
                       value=st.session_state.openai_api_key,
                       help="You can find your API key at https://platform.openai.com/api-keys")

if api_key:
    st.session_state.openai_api_key = api_key
    
    # Initialize the vector store and QA chain
    vector_store = initialize_vector_store(api_key)
    if vector_store is None:
        st.stop()

    qa_chain = initialize_qa_chain(vector_store, api_key)
    if qa_chain is None:
        st.stop()

    # Create a text input for the question
    question = st.text_input("Enter your question about any company:", 
                            placeholder="e.g., What is Apple's business model? or What risks are mentioned by Tesla?")

    # Add some example questions
    st.markdown("""
    ### Example Questions:
    - What is Netflix's primary business?
    - Where is Apple headquartered?
    - What are the top risks mentioned in Johnson & Johnson's 10-K?
    - Where are the primary suppliers for Tesla?
    - How is ExxonMobil addressing climate change and the energy transition?
    """)

    # Process the question when submitted
    if question:
        with st.spinner("Searching for relevant information..."):
            answer = get_answer(question, qa_chain)
            if answer:
                st.markdown("### Answer:")
                st.write(textwrap.fill(answer, width=80))
else:
    st.warning("Please enter your OpenAI API key to continue.")

# Add a footer
st.markdown("---")
st.markdown("Built with Streamlit, Neo4j, and OpenAI") 