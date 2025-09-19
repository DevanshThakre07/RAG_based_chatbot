import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings #to perform word embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import faiss
import streamlit as st
from pdfextractor import text_extractor_pdf


st.sidebar.title(":orange[UPLOAD YOUR DOCUMENT HERE(PDF only)]")
# file_uploaded = st.sidebar.file_uploader("Upload your file")
# file_text = text_extractor_pdf(file_uploaded)


st.title(':green[RAG Based Chatbot]')

tips = '''Follow the steps to use this application:
* Upload your pdf '''
st.write(tips)

file_uploaded = st.sidebar.file_uploader('Upload_file')
if file_uploaded:
    file_text = text_extractor_pdf(file_uploaded)
    key = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key= key)
    llm_model= genai.GenerativeModel('gemini-2.5-flash-lite')

    # configure Embedding model
    embedding_model = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

    # step 2 we create chunks (create chunks)
    splitter = RecursiveCharacterTextSplitter(chunk_size = 800,chunk_overlap = 200)
    chunks = splitter.split_text(file_text)

    # step 3 create faiss vector store
    vector_store = FAISS.from_texts(chunks,embedding_model)

    # configure retriever
    retriever = vector_store.as_retriever(search_kwargs = {'k':3})

    # Creater a function that takes query and return the generted text
    def generate_response(query):
        retrived_docs = retriever.get_relevant_documents(query=query)
        context = ' '.join([doc.page_content for doc in retrived_docs])

        prompt = f''' you are a helpful assitant using RAG
        here is the context = {context}
        The query asked by user is as follows = {query}'''
    
        content = llm_model.generate_content(prompt).text
        return content



    # while True:
    #     query = st.chat_input('user : ')
    #     if query.lower() in ['bye','exit','quit','close','end','stop']:
    #         break

    # query = st.text_input('Enter your query: ')

    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Display history
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.write(f':green[User:]{msg['text']}')    
        else:
            st.write(f':orange[Chatbot:]{msg['text']}')

    # input form
    with st.form('Chat Form',clear_on_submit=True):
        user_input = st.text_input('Enter your query here: ')
        send = st.form_submit_button('SEND')

    # start the conversation and append the output and query in history
    if user_input and send:
        st.session_state.history.append({'role':'user','text':user_input})

        model_output = generate_response(user_input)
        st.session_state.history.append({'role':'chatbot','text':model_output})

        st.rerun()


    # if query:
    #     retrived_docs = retriever.get_relevant_documents(query=query)
    #     context = ' '.join([doc.page_content for doc in retrived_docs])

    #     prompt = f''' you are a helpful assitant using RAG
    #     here is the context = {context}
    #     The query asked by user is as follows = {query}'''
    
    #     content = llm_model.generate_content(prompt).text
    #     st.write(content)

    
