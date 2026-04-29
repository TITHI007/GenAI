from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings 
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# What is Langchain?
# It is a framework which help us to work in gen ai and agentic ai models.

load_dotenv() 
  
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment/.env")

st.header("RAG Chatbot")

with st.sidebar:
    st.title("Your documents")
    file=st.file_uploader("Upload your documents here and ask questions", type= "pdf")

#Extract contents from teh uploaded file and chunck it

if file is not None:
    with pdfplumber.open(file) as pdf:
        text=""
        for page in pdf.pages:
            text+=page.extract_text() + "\n"
        # st.write(text)

        #split text into chunks
        text_splitter= RecursiveCharacterTextSplitter(
            separators=["\n\n","\n",". ", " ", ""],
            chunk_size=1000, 
            chunk_overlap=200
            )
        
        chunks=text_splitter.split_text(text)
        # st.write(chunks)

        #generating embeddings
        embeddings=OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key = api_key
        ) 

        # store the embeddings in a vector database
        vectore_store = FAISS.from_texts(chunks, embeddings)


        #Get user questions
        user_question = st.text_input("Ask a question")
                                      
        #Generate Answers
        # question -> embeddings -> similiarity search -> results to LLM -> Response(Chain)

        def format_docs(docs):
            return "\n".join([doc.page_content for doc in docs])


        retriever = vectore_store.as_retriever(
            search_type="mmr",
            # mmr is a searching technique
            search_kwargs={"k":4}
            # return 4 closet match
        )

        # define the LLM and prompts

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=1000,
            openai_api_key=api_key
        )
        
        #provide the prompt
        prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a PDF document.\n\n"
         "Guidelines:\n"
         "1. Provide complete, well-explained answers using the context below.\n"
         "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
         "3. If the context mentions related information, include it to give fuller picture.\n"
         "4. Only use information from the provided context - do not use outside knowledge.\n"
         "5. Summarize long information, ideally in bullets where needed\n"
         "6. If the information is not in the context, say so politely.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])




        chain=(
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            |StrOutputParser()
        )
        if user_question:
            response = chain.invoke(user_question)
            st.write(response)

