import getpass
import os
import pandas as pd
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import ollama
from langchain_openai import OpenAIEmbeddings
import streamlit as st

# Load environment variables
load_dotenv()

# Check if OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# Set up OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# Load the persisted FAISS vector store
persisted_vectorstore = FAISS.load_local("faiss_index_openai", embeddings, allow_dangerous_deserialization=True)

# Function to perform retrieval from the vector store
def retrieval(question):
    answer = persisted_vectorstore.similarity_search_with_score(question, k=3)
    answer = [{(s.split(":")[0]): (s.split(":")[1]) for s in a[0].page_content.split("\n")} for a in answer]
    return answer

# Function to build the prompt for the LLM
def build_prompt(query):
    rag_response = retrieval(query)
    suffix = """You are a helpful product assistant. Based on the following product review details, answer the user's question.
    If the given reviews are not relevant to the User Query Respond to the User as 'Review not found'

    Product Reviews:\n"""
    prompt_template = """
    'Product_title' :{product_title} \n 'rating' :{rating} \n 'review':{review} \n 'summary':{summary}
    """
    temp = PromptTemplate(template=prompt_template)
    fewshottemp = FewShotPromptTemplate(examples=rag_response, example_prompt=temp, suffix="User Query: {input} \n Your Answer:", input_variables=["input"])
    return suffix + (fewshottemp.format(input=query))

# Function to get the LLM response from Ollama
def llm_response(prompt):
    response = ollama.chat(model='llama3.1', messages=[
        {'role': 'user', 'content': prompt},
    ])
    return response

# Streamlit UI
st.title("Product Assistant - Query")

# Input field for user query
user_query = st.text_input("Ask a question about the product:")

# Display the result when the query is submitted
if user_query:
    # Build the prompt from the query
    prompt = build_prompt(user_query)
    
    # Get the response from the LLM
    response = llm_response(prompt)
    
    # Display the response from LLM
    st.write("Response from LLM:")
    st.write(response["message"]["content"])
    
    # Optionally, display relevant document excerpts retrieved from FAISS
    st.subheader("Relevant Product Reviews:")
    for item in retrieval(user_query):
        st.write(item)
