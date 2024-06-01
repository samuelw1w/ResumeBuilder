# import os
# import logging
# import sys
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage, PromptTemplate
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.llms.openai import OpenAI
# from llama_index.core.memory import ChatMemoryBuffer
# from llama_index.readers.file import PDFReader

# from llama_index.core.query_pipeline import QueryPipeline

# # Setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# os.environ['OPENAI_API_KEY'] = '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'

# # Load OpenAI API key from environment variable
# api_key = os.getenv('OPENAI_API_KEY')
# if not api_key:
#     logging.error("OPENAI_API_KEY not found in environment variables.")
#     exit(1)

# # Ensure the 'Data' directory exists and load data
# data_dir = 'Data'
# if not os.path.isdir(data_dir):
#     logging.error(f"Directory '{data_dir}' not found.")
#     exit(1)

# llm = Settings.llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0125")
# Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
# Settings.embed_model = OpenAIEmbedding()
# data = SimpleDirectoryReader(input_dir=data_dir).load_data()

# def construct_index(directory_path):
#     try:        
#         storage_context = StorageContext.from_defaults(persist_dir="./storage")
#         index = load_index_from_storage(storage_context)
#     except:
#         documents = SimpleDirectoryReader(directory_path).load_data()
#         index = VectorStoreIndex.from_documents(documents, show_progress=True)
#         index.storage_context.persist()
#     return index

# index = construct_index("Data")

# memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

# prompt_str = """
# You are a coach and editor specifically tailored for refining resumes for individuals. I want you to first state the current resume that you have on file.
# Provide friendly and encouraging advice and editing suggestions to enhance a user's resume and general presence for companies, making it more appealing to potential recruiters.
# You will focus on professional summaries, experiences, skills, and recommendations, ensuring the resume is
# comprehensive, engaging, and tailored to the IT sector. It offers tips on networking through LinkedIn to maximize visibility and connections,
# while avoiding technical jargon to ensure clarity for all audiences. You will make educated assumptions based on common practices 
# in the IT industry to provide advice without needing further clarification. Avoid jargon and buzzwords. Improve text for clarity. 
# I want you to cater each recommendation to each specific resume that you currently have access to. 
# """
# prompt_tmpl = PromptTemplate(prompt_str)



# # Define queries and get responses from the chatbot using the context information
# p = QueryPipeline(chain=[prompt_tmpl, llm], verbose = True)

# output = p.run()
# print(str(output))

import os
import logging
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.readers.file import PDFReader
from llama_index.core.query_pipeline import QueryPipeline

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

Settings.llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0125", system_prompt="")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.embed_model = OpenAIEmbedding()
data = SimpleDirectoryReader(input_dir=data_dir).load_data()

def construct_index(directory_path):
    try:        
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
    except:
        documents = SimpleDirectoryReader(directory_path).load_data()
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist()
    return index

index = construct_index(data_dir)

prompt_str = """
You are a coach and editor specifically tailored for refining resumes for individuals. I want you to first state the current resume that you have on file.
Provide friendly and encouraging advice and editing suggestions to enhance a user's resume and general presence for companies, making it more appealing to potential recruiters.
You will focus on professional summaries, experiences, skills, and recommendations, ensuring the resume is
comprehensive, engaging, and tailored to the IT sector. It offers tips on networking through LinkedIn to maximize visibility and connections,
while avoiding technical jargon to ensure clarity for all audiences. You will make educated assumptions based on common practices 
in the IT industry to provide advice without needing further clarification. Avoid jargon and buzzwords. Improve text for clarity. 
I want you to cater each recommendation to each specific resume that you currently have access to. 
"""

prompt_tmpl = PromptTemplate(prompt_str)

# Define a function to query the index and generate responses
def query_resume_index(prompt, query, llm):
    query_pipeline = QueryPipeline(chain=[prompt, llm], verbose=True)
    return query_pipeline.run()

# Sample query to the chatbot using the constructed index
query = "What improvements can be made to this resume?"
response = query_resume_index(prompt_tmpl, query, llm)

print(response)

