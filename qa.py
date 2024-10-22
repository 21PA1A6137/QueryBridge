import streamlit as st
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


st.title("Conversational RAG with PDF uploads and Chat history")
st.write("Upload PDF and chat with their content")

api_key = st.text_input("Enter your api key",type="password")

if api_key:
    llm = ChatGroq(groq_api_key = api_key,model="Gemma2-9b-It")
    session_id = st.text_input("Session ID",value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store={}

    upload_files = st.file_uploader("Upload a PDF",type="pdf",accept_multiple_files=True)
    if upload_files:
        documents = []
        for upload_file in upload_files:
            # temp = f'C:\\Users\\charan27\\OneDrive\\Desktop\\GEN_AI_COURSE\\Documents\\Attention_is_all_you_need.pdf'
            with open(upload_file.name,'wb') as file:
                file.write(upload_file.getvalue())
                file_name = upload_file.name
            loader = PyPDFLoader(upload_file.name)
            docs = loader.load()
            documents.extend(docs)
            os.remove(upload_file.name)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectordb = FAISS.from_documents(embedding=embeddings,documents=splits)
        retriever = vectordb.as_retriever()

        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)


        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        que_ans_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,que_ans_chain)

        def get_session_id(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain  = RunnableWithMessageHistory(
            rag_chain,
            get_session_history=get_session_id,
            input_messages_key="input",
            output_messages_key="answer",
            history_messages_key="chat_history"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_id(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoq API Key")