import streamlit as st
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.converters.txt import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.utils import Secret
from pathlib import Path
import openai
from haystack.components.retrievers.in_memory import (
    InMemoryEmbeddingRetriever,
    InMemoryBM25Retriever,
)
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator
import concurrent.futures
import os
from utils.custom_converters import DocxToTextConverter

@st.cache_resource()
def get_doc_store():
    """Get the document store for indexing and retrieval."""
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    return document_store

def write_documents(file):
    """Convert and write the documents to the document store."""
    pipeline = Pipeline()

    if file.name.endswith(".docx"):
        pipeline.add_component("converter", DocxToTextConverter())
    elif file.name.endswith(".txt"):
        pipeline.add_component("converter", TextFileToDocument())
    else:
        pipeline.add_component("converter", PyPDFToDocument())

    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component(
        "splitter", DocumentSplitter(split_by="word", split_length=350)
    )
    pipeline.add_component(
        "embedder", OpenAIDocumentEmbedder(api_key=Secret.from_token(openai.api_key))
    )
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder.documents", "writer")

    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_path = os.path.join("uploads", file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    pipeline.run({"converter": {"sources": [Path(file_path)]}})
    st.success("Indexed Document!")

def chunk_documents(file):
    """Chunk the documents for analysis."""
    pipeline = Pipeline()

    if file.name.endswith(".docx"):
        pipeline.add_component("converter", DocxToTextConverter())
    elif file.name.endswith(".txt"):
        pipeline.add_component("converter", TextFileToDocument())
    else:
        pipeline.add_component("converter", PyPDFToDocument())

    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component(
        "splitter", DocumentSplitter(split_by="word", split_length=3000)
    )

    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    file_path = os.path.join("uploads", file.name)
    docs = pipeline.run({"converter": {"sources": [file_path]}})
    return [d.content for d in docs["splitter"]["documents"]]

def query_pipeline(query):
    """Query the pipeline for context using hybrid retrieval and reciprocal rank fusion."""
    query_pipeline = Pipeline()
    query_pipeline.add_component(
        "text_embedder", OpenAITextEmbedder(Secret.from_token(openai.api_key))
    )
    query_pipeline.add_component(
        "retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=4)
    )
    query_pipeline.add_component(
        "bm25_retriever", InMemoryBM25Retriever(document_store=document_store, top_k=4)
    )
    query_pipeline.add_component(
        "joiner",
        DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=4, sort_by_score=True),
    )
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("bm25_retriever", "joiner")
    query_pipeline.connect("retriever", "joiner")

    result = query_pipeline.run(
        {"text_embedder": {"text": query}, "bm25_retriever": {"query": query}}
    )
    return result["joiner"]["documents"]

def query_router(query):
    """Route the query to the appropriate choice based on the system response."""
    generator = OpenAIChatGenerator(
        api_key=Secret.from_token(openai.api_key), model="gpt-4-turbo"
    )

    system = """You are a professional decision making query router bot for a chatbot system that decides whether a user's query requires a resume review, 
    a LinkedIn profile review, or a simple greeting/gratitude/salutation response. If the query requires a resume review, you will reply with only "(1)".
    If the query requires a LinkedIn profile review, you will reply with only "(2)". If the query requires a simple greeting/gratitude/salutation or an answer 
    to a follow up question based on conversation history response, you will reply with only "(3)"."""

    instruction = f"""You are given a user's query in the <query> field. You are responsible for routing the query to the appropriate
    choice as described in the system response. <query>{query}</query> You are also given the history of the conversation in the <history>{st.session_state.messages}</history> field."""

    messages = [ChatMessage.from_system(system), ChatMessage.from_user(instruction)]
    response = generator.run(messages)
    return response

def review_resume(query, chunk):
    """Review each chunk of the resume based on a user's query."""
    generator = OpenAIChatGenerator(
        api_key=Secret.from_token(openai.api_key), model="gpt-4-turbo"
    )

    system = """You will act as a Profile Coach to help enhance resumes for better recruiter appeal, focusing on the information technology industry.
    Your role is to provide friendly and encouraging advice, along with detailed editing suggestions, to make resumes stand out in the IT sector. 
    Pay special attention to improving professional summaries, experiences, skills, and recommendations, ensuring they are clear, engaging, and tailored to the IT sector.
    Additionally, offer tips on how LinkedIn profiles can complement resumes, maximizing visibility and connections. Avoid technical jargon and buzzwords, 
    and make educated assumptions based on common practices in the IT industry to provide advice without needing further clarification."""

    instruction = f"""
    When reviewing the resume, please consider the following tips:
    - Use a professional email.
    - Double-check contact information.
    - Highlight achievements over responsibilities where possible.
    - Mention only relevant work experience.
    - Include extracurricular activities and personal projects if work experience is limited.
    - Back up experiences with data and numbers.
    - Mention career transitions in the resume summary.
    - Tailor the resume to the job being applied for.
    - Keep the resume to one page unless highly experienced.
    - Avoid fluff and generic buzzwords.
    - Proof-read the resume using tools like Grammarly or by seeking a second opinion.
    - Include a hobbies section if relevant and space allows.
    - Follow up on applications if necessary.
    - Be aware that resume preferences can be opinion-based and vary among recruiters.

    You are given a user's query in the <query> field and a chunk of resume text in the <chunk> field. Provide detailed feedback and suggestions:
    <query>{query}</query>\n <chunk>{chunk}</chunk>"""

    messages = [ChatMessage.from_system(system), ChatMessage.from_user(instruction)]
    response = generator.run(messages)
    return response

def review_linkedin(query, profile):
    """Review the LinkedIn profile based on a user's query."""
    generator = OpenAIChatGenerator(
        api_key=Secret.from_token(openai.api_key), model="gpt-4-turbo"
    )

    system = """You are a professional LinkedIn profile reviewer for a chatbot system. 
    You are responsible for reviewing a LinkedIn profile according to a user's query."""

    instruction = f"""You are given a user's query in the <query> field. Respond appropriately to the user's input
    using the provided LinkedIn profile in the <profile> tags: <query>{query}</query>\n <profile>{profile}</profile>"""

    messages = [ChatMessage.from_system(system), ChatMessage.from_user(instruction)]
    response = generator.run(messages)
    return response

def summary_tool(query, file):
    """Review the resume based on a user's query."""
    chunks = chunk_documents(file)
    # write async function to call chat generator using concurrent futures
    futures = []
    reviews = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for chunk in chunks:
            futures.append(executor.submit(review_resume, query, chunk))
        for future in concurrent.futures.as_completed(futures):
            reviews.append(future.result())
        return reviews

def linkedin_tool(query, profile):
    """Review the LinkedIn profile based on a user's query."""
    return review_linkedin(query, profile)

def simple_responder(query):
    """Respond to a user's query based on a simple follow up response."""
    generator = OpenAIChatGenerator(
        api_key=Secret.from_token(openai.api_key), model="gpt-4-turbo"
    )

    system = """You are a professional greeting/gratitude/salutation/ follow up responder for a chatbot system. 
    You are responsible for responding politely to a user's query."""

    instruction = f"""You are given a user's query in the <query> field. Respond appropriately to the user's input: <query>{query}</query>"""

    messages = []
    history = st.session_state.messages
    messages.append(ChatMessage.from_system(system))
    for i in range(0, len(history) - 1, 2):
        messages.append(ChatMessage.from_user(history[i]["content"]))
        messages.append(ChatMessage.from_assistant(history[i + 1]["content"]))
    messages.append(ChatMessage.from_user(instruction))
    response = generator.run(messages)
    return response

class RAGAgent:
    """The RAG Agent class that routes a user query to the appropriate choice based on the system response."""

    def __init__(self):
        self.loops = 0

    def invoke_agent(self, query, file=None, profile=None):
        intent = query_router(query)["replies"][0].content.strip()

        if intent == "(1)":
            st.success("Reviewing Resume...")
            response = summary_tool(query, file)
        elif intent == "(2)":
            st.success("Reviewing LinkedIn Profile...")
            response = linkedin_tool(query, profile)
        elif intent == "(3)":
            st.success("Retrieving Simple Response...")
            response = simple_responder(query)
        return response

def clear_convo():
    st.session_state["messages"] = []

def init():
    st.set_page_config(page_title="Resume & LinkedIn Coach", page_icon=":robot_face: ")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

if __name__ == "__main__":
    init()

    document_store = get_doc_store()

    agent = RAGAgent()

    # streamlit components
    st.title("Resume & LinkedIn Coach :robot_face:")

    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    st.sidebar.markdown(
        """This app helps you enhance your resume and LinkedIn profile for better recruiter appeal. It is capable of reviewing your resume or LinkedIn profile,
        providing detailed feedback and suggestions."""
    )
    openai.api_key = api_key
    clear_button = st.sidebar.button(
        "Clear Conversation", key="clear", on_click=clear_convo
    )

    file = st.file_uploader("Choose a resume file to review...", type=["docx", "pdf", "txt"])
    linkedin_profile = st.text_area("Paste your LinkedIn profile content here for review")

    if st.button("Upload File", key="Upload") and file:
        with st.spinner("Wait for it..."):
            write_documents(file)

    user_input = st.chat_input("Ask your question")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        res = agent.invoke_agent(user_input, file=file, profile=linkedin_profile)
        st.session_state.messages.append({"role": "assistant", "content": res})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
