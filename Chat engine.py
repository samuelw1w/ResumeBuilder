import os
import logging
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
os.environ['OPENAI_API_KEY'] = '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'

# Load OpenAI API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    logging.error("OPENAI_API_KEY not found in environment variables.")
    exit(1)

# Ensure the 'Data' directory exists and load data
data_dir = 'Data'
if not os.path.isdir(data_dir):
    logging.error(f"Directory '{data_dir}' not found.")
    exit(1)

try:
    data = SimpleDirectoryReader(input_dir=data_dir).load_data()
    logging.info("Data has been loaded")
    logging.debug(f"Loaded data: {data}")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    exit(1)

try:
    index = VectorStoreIndex.from_documents(data)
    index.storage_context.persist()
    logging.info("Data has been indexed")
except Exception as e:
    logging.error(f"Error indexing data: {e}")
    exit(1)

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "You are a coach and editor specifically tailored for creating and refining LinkedIn profiles and resumes "
        "for individuals. Provide friendly and encouraging advice and editing suggestions to enhance a user's LinkedIn "
        "and general presence for companies, making it more appealing to potential recruiters. "
        "You will focus on professional summaries, experiences, skills, and recommendations, ensuring the profile is "
        "comprehensive, engaging, and tailored to the IT sector. It offers tips on networking through LinkedIn to maximize visibility and connections, "
        "while avoiding technical jargon to ensure clarity for all audiences. You will make educated assumptions based on common practices "
        "in the IT industry to provide advice without needing further clarification. Avoid jargon and buzzwords. Improve text for clarity."
    ),
)

try:
    response = chat_engine.chat("What do you know? What resumes do you currently have on file? Use this file to review my resume for recruiter appeal and make recommendations and feedback for my resume")
    print(response)
except Exception as e:
    logging.error(f"Error during chat interaction: {e}")
    exit(1)
