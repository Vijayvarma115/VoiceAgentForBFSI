import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS #vectorstoreDB
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings #vector embeddings
from deepgramvoice import res
from dotenv import load_dotenv
import os
import time
load_dotenv()

## load GROQ And Gemini API KEY from .env
groq_api_key=os.getenv('GROQ_API_KEY')
# os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")


llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="meta-llama/llama-4-scout-17b-16e-instruct")   

# prompt=ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}

# """
# )

def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("./us_census_data") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings



if "vectors" not in st.session_state:
    vector_embedding()
    st.write("Vector Store DB Is Ready")


if res:
    # Define your prompt
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you do not know, don't try to make up an answer.
    {context}
    Question: {question}"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retriever
    retriever = st.session_state.vectors.as_retriever()

    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    st.subheader("Response from RAG:")
    start = time.process_time()
    response = retrieval_chain.invoke({'input': res})
    print("Response time:", time.process_time() - start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        st.subheader("Relevant Documents:")
        for i, doc in enumerate(response["context"]):
            st.write(f"**Document {i+1}:**")
            st.write(doc.page_content)
            st.write("--------------------------------")
else:
    st.info("Waiting for voice input...")
