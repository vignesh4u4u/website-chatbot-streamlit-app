import os
import mysql.connector
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils.generic")

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from urllib.parse import urljoin, urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
api_key=st.secrets["token"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def format_prompt(message, history):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}  "
    prompt += f"[INST] {message} [/INST]"
    return prompt


generate_kwargs = dict(
    temperature=0.7,
    max_new_tokens=3000,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True,
    seed=42,
)

def generate_output(input_query):
    template = """
    You are an intelligent chatbot. Help the following question with brilliant answers.
    Question: {question}
    Answer:"""
    model1 = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                            model_kwargs={"temperature": 0.5,
                                          'repetition_penalty': 1.1,
                                          "max_new_tokens": 3000,
                                          "max_length": 3000})
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=model1)
    answer = llm_chain.invoke(input_query)
    response = answer["text"]
    answer_index = response.find("Answer:")
    if answer_index != -1:
        answer_text = response[answer_index + len("Answer:") + 1:].strip()
        return ( answer_text.strip())
    else:
        return ( response.strip())


def generate_text(message, history):
    prompt = format_prompt(message, history)
    output = client.text_generation(prompt, **generate_kwargs)
    return output


def apply_url_link(url,prompt):
    read = requests.get(url).content
    soup = BeautifulSoup(read, "html5lib")
    link = soup.find_all("a")
    all_links = []
    base_url = url
    for i in link:
        href = i.get('href')
        if href:
            complete_url = urljoin(base_url, href)
            all_links.append(complete_url)
    http_links = [link for link in all_links if link.startswith('http://') or link.startswith('https://')]
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=10)
    chunks = text_splitter.split_documents(docs)
    chunk_texts = [chunk.page_content for chunk in chunks]
    vector_store = FAISS.from_texts(chunk_texts, embeddings)
    docs = vector_store.similarity_search(prompt, k=6)
    page_content_list = []
    for document in docs:
        page_content = document.page_content
        page_content_list.append(page_content)
    page_content = " ".join(page_content_list)
    result = page_content + "\n\n\n\n\n" + "Please provide an answer to the following question using the text data above. Make sure the answer is relevant to the provided text." + "\n" + prompt
    return result

with st.sidebar:
    url = st.text_input("URL link", max_chars=1000, placeholder="apply URL link")

st.title("💬 Website Chatbot")
st.caption("🚀 A Streamlit chatbot specially desigened to chat with website")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("Say something")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    table_columns = st.session_state.get("table_columns", {})
    input_query = apply_url_link(url,prompt)
    answer = generate_output(input_query)
    st.session_state["messages"].append({"role": "assistant", "content": answer})

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
